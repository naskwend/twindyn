"""The domain-generalisation family of the main comparison.

Ref: Table 1 rows B17 and B18. Sec. 2.1 lists domain generalisation and resistance
to domain change as the leading points of assessment in the representation
literature, and Sec. 2.3 contrasts the shared and private disentanglement with a
formulation that removes the cohort's acquisition signature. B17 absorbs
cross-domain invariant features with an explicit adversarial penalty, B18 generates
the latent through learnable prompt tokens rather than through a fixed encoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from twindyn.models.baselines.protocol import TorchRiskEstimator


class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass and a sign flip in the backward pass."""

    @staticmethod
    def forward(ctx: object, value: Tensor, scale: float) -> Tensor:
        ctx.scale = scale  # type: ignore[attr-defined]
        return value.view_as(value)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.scale * gradient, None  # type: ignore[attr-defined]


def gradient_reversal(value: Tensor, scale: float = 1.0) -> Tensor:
    return _GradientReversal.apply(value, scale)


class CrossDomainInvariantAbsorption(TorchRiskEstimator):
    """B17: invariant features absorbed, domain-specific features retained."""

    identifier = "B17_cross_domain_invariant_absorption"
    family = "domain generalisation"

    def __init__(
        self,
        features: int,
        invariant: int = 32,
        specific: int = 32,
        hidden: int = 128,
        domains: int = 4,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(features, hidden), nn.SiLU())
        self.invariant_head = nn.Linear(hidden, invariant)
        self.specific_head = nn.Linear(hidden, specific)
        self.domain_head = nn.Sequential(
            nn.Linear(invariant, hidden), nn.SiLU(), nn.Linear(hidden, domains)
        )
        self.head = nn.Linear(invariant + specific, 1)

    def parts(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.trunk(features)
        return self.invariant_head(hidden), self.specific_head(hidden)

    def risk_from_features(self, features: Tensor) -> Tensor:
        invariant, specific = self.parts(features)
        return self.head(torch.cat([invariant, specific], dim=-1)).squeeze(-1)

    def training_forward(
        self, features: Tensor, target: Tensor | None = None, group: Tensor | None = None
    ) -> Tensor:
        risk = self.risk_from_features(features)
        if group is None:
            return risk
        invariant, _ = self.parts(features)
        adversarial = self.domain_head(gradient_reversal(invariant, 1.0))
        domain_loss = torch.nn.functional.cross_entropy(adversarial, group.clamp(0, 3))
        return risk - 0.1 * domain_loss


class PromptDrivenLatentGeneration(TorchRiskEstimator):
    """B18: the latent is generated from learnable prompt tokens."""

    identifier = "B18_prompt_driven_latent_generation"
    family = "domain generalisation"

    def __init__(
        self,
        features: int,
        prompts: int = 4,
        latent: int = 32,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(prompts, latent) / latent**0.5)
        self.encoder = nn.Sequential(nn.Linear(features, hidden), nn.SiLU())
        self.generator = nn.Sequential(
            nn.Linear(hidden + latent, hidden), nn.SiLU(), nn.Linear(hidden, latent)
        )
        self.scorer = nn.Linear(hidden + latent, 1)

    def generate(self, features: Tensor) -> Tensor:
        hidden = self.encoder(features)
        batch = features.shape[0]
        prompts = self.prompts.unsqueeze(0).expand(batch, -1, -1)
        context = hidden.unsqueeze(1).expand(-1, prompts.shape[1], -1)
        generated = self.generator(torch.cat([context, prompts], dim=-1))
        weights = torch.softmax(generated.norm(dim=-1), dim=-1)
        summary = (generated * weights.unsqueeze(-1)).sum(dim=1)
        return torch.cat([hidden, summary], dim=-1)

    def risk_from_features(self, features: Tensor) -> Tensor:
        return self.scorer(self.generate(features)).squeeze(-1)
