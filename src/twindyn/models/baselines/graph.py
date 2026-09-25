"""The graph family of the main comparison.

Ref: Table 1 row B12 (a metadata-driven dual graph). Sec. 2.2 names the graph
family that the manuscript's related work surveys (hierarchical and heterogeneous
graph learning, attributed-network embedding), and Sec. 2.2 also notes that graph
operators are the branch whose scaling ceiling motivated the linear-time operator.
The dual graph here keeps the two structures the name refers to: a
feature-similarity graph over region tokens and a metadata graph over the
clinical covariates of the same sample.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from twindyn.models.baselines.protocol import TorchRiskEstimator


class MetadataDrivenDualGraph(TorchRiskEstimator):
    """B12: convolution on a feature graph and on a metadata graph, then fused."""

    identifier = "B12_metadata_driven_dual_graph"
    family = "graph"

    def __init__(
        self,
        features: int,
        metadata: int,
        token_dim: int = 17,
        width: int = 64,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if features % token_dim != 0:
            raise ValueError("the feature count must be a multiple of the token width")
        self.token_dim = int(token_dim)
        self.tokens = features // token_dim
        self.metadata_width = max(1, int(metadata))
        self.project = nn.Linear(token_dim, width)
        self.metadata_project = nn.Linear(self.metadata_width, width)
        self.feature_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.metadata_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.cross = nn.Linear(2 * width, width)
        self.head = nn.Linear(width, 1)

    def _feature_graph(self, tokens: Tensor) -> Tensor:
        similarity = tokens @ tokens.transpose(1, 2)
        return torch.softmax(similarity, dim=-1)

    def _metadata_graph(self, metadata_tokens: Tensor) -> Tensor:
        similarity = metadata_tokens @ metadata_tokens.transpose(1, 2)
        return torch.softmax(similarity, dim=-2)

    def risk_from_features(self, features: Tensor) -> Tensor:
        return self.forward_with_metadata(
            features, torch.zeros((features.shape[0], self.metadata_width), device=features.device)
        )

    def forward_with_metadata(self, features: Tensor, metadata: Tensor) -> Tensor:
        shaped = features.view(features.shape[0], self.tokens, self.token_dim)
        tokens = self.project(shaped)
        metadata_tokens = self.metadata_project(metadata).unsqueeze(1)
        feature_adjacency = self._feature_graph(tokens)
        metadata_adjacency = self._metadata_graph(metadata_tokens)
        for feature_layer, metadata_layer, norm in zip(
            self.feature_layers, self.metadata_layers, self.norms, strict=True
        ):
            tokens = norm(
                tokens + torch.nn.functional.silu(feature_layer(feature_adjacency @ tokens))
            )
            metadata_tokens = metadata_tokens + torch.nn.functional.silu(
                metadata_layer(metadata_adjacency @ metadata_tokens)
            )
        pooled = torch.cat([tokens.mean(dim=1), metadata_tokens.mean(dim=1)], dim=-1)
        return self.head(torch.nn.functional.silu(self.cross(pooled))).squeeze(-1)

    def training_forward(
        self, features: Tensor, target: Tensor | None = None, group: Tensor | None = None
    ) -> Tensor:
        metadata = _metadata_block(group, features.shape[0], self.metadata_width, features.device)
        return self.forward_with_metadata(features, metadata)

    def predict_risk_with_metadata(self, features: Tensor, metadata: Tensor) -> Tensor:
        return self.forward_with_metadata(features, metadata)


def _metadata_block(group: Tensor | None, rows: int, width: int, device: torch.device) -> Tensor:
    block = torch.zeros((rows, width), device=device)
    if group is None:
        return block
    position = group.clamp(0, width - 1).view(-1)
    block[torch.arange(rows, device=device), position] = 1.0
    return block
