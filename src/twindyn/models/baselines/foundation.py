"""The foundation-model row of the main comparison.

Ref: Table 1 row B19 (a frozen foundation encoder with a linear head), Sec. 2.1
(the benchmark finding that downstream discrimination correlates weakly with
encoder size and training compute, which is why a frozen trunk with a linear head
is the row that isolates the representation).

No pathology foundation checkpoint ships with this repository, so the frozen
trunk here is a fixed random projection with a recorded seed. The row therefore
measures what a frozen generic encoder of the same width achieves under the shared
protocol, and it is reported with that qualification rather than as the published
encoder's value.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from twindyn.models.baselines.protocol import TorchRiskEstimator


class FrozenFoundationLinearHead(TorchRiskEstimator):
    """B19: a frozen trunk plus a single linear risk head."""

    identifier = "B19_frozen_foundation_linear_head"
    family = "foundation"

    def __init__(
        self,
        features: int,
        width: int = 128,
        trunk_seed: int = 20260101,
        nonlinear: bool = True,
    ) -> None:
        super().__init__()
        trunk = nn.Sequential(nn.Linear(features, width), nn.SiLU(), nn.Linear(width, width))
        generator = torch.Generator().manual_seed(trunk_seed)
        with torch.no_grad():
            for parameter in trunk.parameters():
                if parameter.dim() >= 2:
                    bound = 1.0 / parameter.shape[-1] ** 0.5
                    parameter.copy_(
                        torch.empty_like(parameter).uniform_(-bound, bound, generator=generator)
                    )
                else:
                    parameter.zero_()
        self.trunk = trunk
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Linear(width, 1) if nonlinear else nn.Linear(features, 1)
        self.nonlinear = bool(nonlinear)

    def frozen_features(self, features: Tensor) -> Tensor:
        with torch.no_grad():
            return self.trunk(features)

    def risk_from_features(self, features: Tensor) -> Tensor:
        if self.nonlinear:
            return self.head(self.frozen_features(features)).squeeze(-1)
        return self.head(features).squeeze(-1)

    def describe(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "frozen": True,
            "frozen_parameters": int(
                sum(parameter.numel() for parameter in self.trunk.parameters())
            ),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in self.head.parameters())
            ),
            "note": "no foundation checkpoint is bundled; the trunk is a fixed random projection",
        }
