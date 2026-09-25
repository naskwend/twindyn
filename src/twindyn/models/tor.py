"""Trajectory pooling and the shallow outcome read-out.

Ref: Sec. 3.5 (the read-out transforms a trajectory into a level of risk;
trajectory characteristics are combined along the axis and passed through a light
survival head that returns a single continuous risk; depth is kept to a minimum
because otherwise a weak state could be carried by a strong head), Sec. 3.6
(equation 6 uses g(.) as the pooled trajectory summary), Table 2 (the
trajectory-pooling row costs 0.007 and deepening the read-out from one layer to
two costs 0.002), Sec. 4.6 (the pooled representation and the head weights are
reported so the two contributions can be separated).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

POOLING_KINDS: tuple[str, ...] = ("trajectory_mean", "trajectory_last", "trajectory_peak")
POOLING_TARGETS: tuple[str, ...] = ("position_outputs", "states")


@dataclass(frozen=True)
class PoolingConfig:
    kind: str = "trajectory_mean"
    target: str = "position_outputs"


@dataclass(frozen=True)
class ReadoutConfig:
    pool_dim: int
    depth: int = 1
    hidden: int = 64
    dropout: float = 0.0


class TrajectoryPool(nn.Module):
    """Combine per-position features along the axis into one real vector."""

    def __init__(self, config: PoolingConfig) -> None:
        super().__init__()
        if config.kind not in POOLING_KINDS:
            raise ValueError(f"unknown pooling kind {config.kind!r}")
        if config.target not in POOLING_TARGETS:
            raise ValueError(f"unknown pooling target {config.target!r}")
        self.config = config

    def forward(self, features: Tensor) -> Tensor:
        """``(tau_max, batch, dim)`` to ``(batch, dim)``."""
        if features.dim() != 3:
            raise ValueError("features must be (tau_max, batch, dim)")
        if self.config.kind == "trajectory_mean":
            return features.mean(dim=0)
        if self.config.kind == "trajectory_last":
            return features[-1]
        magnitude = features.abs().sum(dim=-1)
        selected = torch.argmax(magnitude, dim=0)
        return features[selected, torch.arange(features.shape[1])]

    def describe(self) -> dict[str, str]:
        return {"kind": self.config.kind, "target": self.config.target}


def state_features(states: Tensor) -> Tensor:
    """Real feature view of a complex state path."""
    return torch.cat([torch.real(states), torch.imag(states)], dim=-1)


class TrajectoryOutcomeReadout(nn.Module):
    """A depth-1 or depth-2 head over the pooled trajectory."""

    def __init__(self, config: ReadoutConfig) -> None:
        super().__init__()
        if config.depth not in (1, 2):
            raise ValueError("read-out depth is one or two")
        self.config = config
        if config.depth == 1:
            self.head: nn.Module = nn.Linear(config.pool_dim, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(config.pool_dim, config.hidden),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden, 1),
            )

    def forward(self, pooled: Tensor) -> Tensor:
        return self.head(pooled).squeeze(-1)

    def weights_report(self) -> dict[str, float]:
        flat = torch.cat([parameter.detach().reshape(-1) for parameter in self.head.parameters()])
        return {
            "depth": float(self.config.depth),
            "parameters": float(flat.numel()),
            "weight_norm": float(flat.norm()),
            "weight_abs_mean": float(flat.abs().mean()),
        }

    def extra_repr(self) -> str:
        return f"pool_dim={self.config.pool_dim}, depth={self.config.depth}"
