"""The fusion and alternative-representation family of the main comparison.

Ref: Table 1 rows B11, B13, B15 and B16. B16 is the strongest static composition
baseline and therefore the comparator every reported external difference is taken
against; B15 is the static shared/private disentanglement that Sec. 2.3 and
Sec. 5.3 identify as the closest alternative representation; B11 and B13 are the
graph-fusion and autoencoder entries of the fusion family.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from twindyn.models.baselines.protocol import TorchRiskEstimator


class StaticCompositionCox(TorchRiskEstimator):
    """B16: the comparator, a Cox model on the static composition vector."""

    identifier = "B16_static_composition_cox"
    family = "alternative representation"

    def __init__(self, features: int, l2: float = 1e-3) -> None:
        super().__init__()
        self.linear = nn.Linear(features, 1)
        self.l2 = float(l2)

    def risk_from_features(self, features: Tensor) -> Tensor:
        return self.linear(features).squeeze(-1)

    def training_forward(
        self, features: Tensor, target: Tensor | None = None, group: Tensor | None = None
    ) -> Tensor:
        return self.risk_from_features(features)


class GraphMultiOmicsFusion(TorchRiskEstimator):
    """B11: feature-similarity graph convolution fused with the observation."""

    identifier = "B11_graph_multiomics_fusion"
    family = "fusion"

    def __init__(
        self, features: int, token_dim: int = 17, width: int = 64, layers: int = 2
    ) -> None:
        super().__init__()
        if features % token_dim != 0:
            raise ValueError("the feature count must be a multiple of the token width")
        self.token_dim = int(token_dim)
        self.tokens = features // token_dim
        self.project = nn.Linear(token_dim, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head = nn.Linear(width, 1)

    def adjacency(self, tokens: Tensor) -> Tensor:
        similarity = tokens @ tokens.transpose(1, 2)
        adjacency = torch.softmax(similarity, dim=-1)
        return adjacency

    def risk_from_features(self, features: Tensor) -> Tensor:
        shaped = features.view(features.shape[0], self.tokens, self.token_dim)
        tokens = self.project(shaped)
        adjacency = self.adjacency(tokens)
        for linear, norm in zip(self.layers, self.norms, strict=True):
            aggregated = adjacency @ tokens
            tokens = norm(tokens + torch.nn.functional.silu(linear(aggregated)))
        return self.head(tokens.mean(dim=1)).squeeze(-1)


class MultiOmicsAutoencoder(TorchRiskEstimator):
    """B13: an autoencoder over the observation with a risk head on the latent."""

    identifier = "B13_multiomics_autoencoder"
    family = "fusion"

    def __init__(self, features: int, latent: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, features)
        )
        self.head = nn.Linear(latent, 1)

    def risk_from_features(self, features: Tensor) -> Tensor:
        return self.head(self.encoder(features)).squeeze(-1)

    def reconstruction(self, features: Tensor) -> Tensor:
        return self.decoder(self.encoder(features))

    def training_forward(
        self, features: Tensor, target: Tensor | None = None, group: Tensor | None = None
    ) -> Tensor:
        risk = self.risk_from_features(features)
        residual = (self.reconstruction(features) - features).pow(2).mean()
        return risk - 0.1 * residual


class StaticSharedPrivateDisentanglement(TorchRiskEstimator):
    """B15: a static shared and private split of the latent space."""

    identifier = "B15_static_shared_private"
    family = "alternative representation"

    def __init__(
        self,
        features: int,
        shared: int = 24,
        private: int = 8,
        hidden: int = 128,
        groups: int = 4,
    ) -> None:
        super().__init__()
        self.shared_dim = int(shared)
        self.private_dim = int(private)
        self.groups = int(groups)
        self.body = nn.Sequential(nn.Linear(features, hidden), nn.SiLU())
        self.shared_head = nn.Linear(hidden, shared)
        self.private_heads = nn.ModuleList([nn.Linear(hidden, private) for _ in range(groups)])
        self.head = nn.Linear(shared, 1)
        self.register_buffer("_group", torch.zeros(1, dtype=torch.long))

    def latent_parts(self, features: Tensor, group: Tensor | None) -> tuple[Tensor, Tensor]:
        hidden = self.body(features)
        shared = self.shared_head(hidden)
        if group is None:
            group = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        stacked = torch.stack([head(hidden) for head in self.private_heads], dim=1)
        selector = group.clamp(0, self.groups - 1).view(-1, 1, 1).expand(-1, 1, self.private_dim)
        private = torch.gather(stacked, 1, selector).squeeze(1)
        return shared, private

    def risk_from_features(self, features: Tensor) -> Tensor:
        shared, _ = self.latent_parts(features, None)
        return self.head(shared).squeeze(-1)

    def training_forward(
        self, features: Tensor, target: Tensor | None = None, group: Tensor | None = None
    ) -> Tensor:
        shared, private = self.latent_parts(features, group)
        risk = self.head(shared).squeeze(-1)
        penalty = private.pow(2).mean()
        return risk - 0.05 * penalty - 0.05 * cross_covariance(shared, private)


def cross_covariance(shared: Tensor, private: Tensor) -> Tensor:
    """Squared cross-covariance between the shared and the private latents."""
    centred_shared = shared - shared.mean(dim=0, keepdim=True)
    centred_private = private - private.mean(dim=0, keepdim=True)
    normalised_shared = torch.nn.functional.normalize(centred_shared, dim=0)
    normalised_private = torch.nn.functional.normalize(centred_private, dim=0)
    covariance = normalised_shared.t() @ normalised_private
    return covariance.pow(2).mean()
