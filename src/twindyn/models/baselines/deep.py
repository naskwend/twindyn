"""The deep survival family of the main comparison.

Ref: Table 1 rows B5-B10: a DeepSurv-style network, a sparse hierarchical
transformer, an uncertainty-aware survival model, a cell-aware transformer
survival model, a linear-attention multiple-instance survival model and a dense
transformer with mixture-of-experts fusion. The manuscript's related-work section
names the families these rows stand for (a sparse and hierarchical transformer for
whole-slide survival, an uncertainty-aware head, cell-aware transformers,
linear-attention multiple-instance aggregation and a dense transformer with
mixture-of-experts fusion), so each row keeps the mechanism of its family while
sharing one protocol.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from twindyn.models.baselines.protocol import TorchRiskEstimator, feature_map


class FeatureTokenizer(nn.Module):
    """Split a flat observation into region tokens."""

    def __init__(self, features: int, token_dim: int, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        if features % token_dim != 0:
            raise ValueError("the feature count must be a multiple of the token width")
        self.token_dim = int(token_dim)
        self.tokens = features // token_dim
        self.project = nn.Linear(token_dim, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        shaped = features.view(features.shape[0], self.tokens, self.token_dim)
        return self.dropout(self.project(shaped))


class DeepSurvStyleNetwork(TorchRiskEstimator):
    """B5: a fully connected proportional-hazards network."""

    identifier = "B5_deepsurv_style"
    family = "deep survival"

    def __init__(self, features: int, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = features
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.SiLU())
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def risk_from_features(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class SparseAttentionBlock(nn.Module):
    """Multi-head attention restricted to the top-k keys of every query."""

    def __init__(self, width: int, heads: int, top_k: int) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("the width must be divisible by the head count")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = width // heads
        self.top_k = int(top_k)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def sparse_mask(self, tokens: Tensor) -> Tensor:
        similarity = tokens @ tokens.transpose(1, 2)
        keep = torch.topk(similarity, k=min(self.top_k, similarity.shape[-1]), dim=-1).indices
        return torch.zeros_like(similarity, dtype=torch.bool).scatter_(-1, keep, True)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, length, _ = tokens.shape
        mask = self.sparse_mask(tokens).unsqueeze(1)
        shape = (batch, length, self.heads, self.head_dim)
        query = self.query(tokens).view(shape).transpose(1, 2)
        key = self.key(tokens).view(shape).transpose(1, 2)
        value = self.value(tokens).view(shape).transpose(1, 2)
        scores = query @ key.transpose(-1, -2) / self.head_dim**0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, self.width)
        return self.output(attended)


class SparseHierarchicalTransformer(TorchRiskEstimator):
    """B6: sparse top-k attention over hierarchically grouped region tokens."""

    identifier = "B6_sparse_hierarchical_transformer"
    family = "deep survival"

    def __init__(
        self,
        features: int,
        token_dim: int = 17,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        top_k: int = 4,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(features, token_dim, width)
        self.blocks = nn.ModuleList(
            [SparseAttentionBlock(width, heads, top_k) for _ in range(layers)]
        )
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head = nn.Linear(width, 1)

    def forward_tokens(self, features: Tensor) -> Tensor:
        tokens = self.tokenizer(features)
        for block, norm in zip(self.blocks, self.norm, strict=True):
            tokens = norm(tokens + block(tokens))
        return tokens

    def risk_from_features(self, features: Tensor) -> Tensor:
        return self.head(self.forward_tokens(features).mean(dim=1)).squeeze(-1)

    def sparsity_report(self) -> dict[str, float]:
        return {
            "layers": float(len(self.blocks)),
            "top_k": float(getattr(self.blocks[0], "top_k", 0.0)) if self.blocks else 0.0,
            "heads": float(getattr(self.blocks[0], "heads", 0.0)) if self.blocks else 0.0,
        }


class UncertaintyAwareSurvival(TorchRiskEstimator):
    """B7: a risk head with an explicit variance channel."""

    identifier = "B7_uncertainty_aware_survival"
    family = "deep survival"

    def __init__(self, features: int, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_variance_head = nn.Linear(hidden, 1)

    def risk_from_features(self, features: Tensor) -> Tensor:
        hidden = self.trunk(features)
        return self.mean_head(hidden).squeeze(-1)

    def log_variance(self, features: Tensor) -> Tensor:
        hidden = self.trunk(features)
        return self.log_variance_head(hidden).squeeze(-1)

    def training_forward(
        self, features: Tensor, target: Tensor | None = None, group: Tensor | None = None
    ) -> Tensor:
        """The mean channel is the risk; the variance channel enters through a penalty."""
        mean = self.risk_from_features(features)
        log_variance = self.log_variance(features)
        penalty = log_variance.exp().mean()
        return mean - 0.01 * penalty


class CellAwareTransformerSurvival(TorchRiskEstimator):
    """B8: attention over region tokens weighted by the compartment fractions."""

    identifier = "B8_cell_aware_transformer"
    family = "deep survival"

    def __init__(
        self,
        features: int,
        token_dim: int = 17,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(features, token_dim, width)
        encoder_layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.awareness = nn.Linear(width, 1)
        self.head = nn.Linear(width, 1)

    def risk_from_features(self, features: Tensor) -> Tensor:
        tokens = self.encoder(self.tokenizer(features))
        weights = torch.softmax(self.awareness(tokens).squeeze(-1), dim=-1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class LinearAttentionMilSurvival(TorchRiskEstimator):
    """B9: linear-attention multiple-instance aggregation."""

    identifier = "B9_linear_attention_mil"
    family = "deep survival"

    def __init__(self, features: int, token_dim: int = 17, width: int = 64) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(features, token_dim, width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.head = nn.Linear(width, 1)

    def risk_from_features(self, features: Tensor) -> Tensor:
        tokens = self.tokenizer(features)
        query = feature_map(self.query(tokens.mean(dim=1, keepdim=True)))
        keys = feature_map(self.key(tokens))
        values = self.value(tokens)
        numerator = torch.einsum("bqk,btk,btv->bqv", query, keys, values)
        denominator = torch.einsum("bqk,btk->bq", query, keys).unsqueeze(-1).clamp_min(1e-6)
        pooled = (numerator / denominator).squeeze(1)
        return self.head(pooled).squeeze(-1)


class DenseTransformerMoeFusion(TorchRiskEstimator):
    """B10: dense attention with a mixture-of-experts feed-forward block."""

    identifier = "B10_dense_transformer_moe"
    family = "deep survival"

    def __init__(
        self,
        features: int,
        token_dim: int = 17,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        experts: int = 4,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(features, token_dim, width)
        self.blocks = nn.ModuleList(
            [nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(layers)]
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
                for _ in range(experts)
            ]
        )
        self.router = nn.Linear(width, experts)
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head = nn.Linear(width, 1)

    def risk_from_features(self, features: Tensor) -> Tensor:
        tokens = self.tokenizer(features)
        for attention, norm in zip(self.blocks, self.norm, strict=True):
            attended = attention(tokens, tokens, tokens, need_weights=False)[0]
            tokens = norm(tokens + attended)
        gate = torch.softmax(self.router(tokens), dim=-1)
        stacked = torch.stack([expert(tokens) for expert in self.experts], dim=-2)
        fused = (gate.unsqueeze(-1) * stacked).sum(dim=-2)
        return self.head(fused.mean(dim=1)).squeeze(-1)
