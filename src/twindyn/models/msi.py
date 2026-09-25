"""Multi-source invariant encoder.

Ref: Sec. 3.3 (using the common map C, MSI links the observation to a hidden
state such that the state carries no information about the resource that produced
it; the observation map is resource-agnostic and puts different assays into one
coordinate system; each sample is a partial observation of a state that changes
along the axis), Assumption 1 (the observation map C is shared across resources
and has rank d), Sec. 3.1 (equation 1: the observation is the compartment block
followed by the morphology block), Sec. 3.6 (different resources observe at
different densities).

The shared map spans the whole observation vector, so its rank can equal the
state dimensionality even though a single axis position reveals only a few
coordinates. A position therefore writes through the columns of the
pseudo-inverse that belong to the coordinates it observes, and the reconstruction
reads back through the same rows of the shared map. A shallow residual refiner
carries assay-specific geometry into the shared coordinate system; with zero
residual layers the encoder is exactly the shared map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from twindyn.data.axis import AxisIndex


@dataclass(frozen=True)
class MsiConfig:
    observation_dim: int = 272
    state_dim: int = 64
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.0
    freeze_observation_map: bool = False


class MultiSourceInvariantEncoder(nn.Module):
    """Shared observation map plus a shallow residual refiner."""

    position_index: Tensor
    position_width: Tensor

    def __init__(self, config: MsiConfig, index: AxisIndex) -> None:
        super().__init__()
        self.config = config
        self.observation_map = nn.Parameter(
            torch.randn(config.observation_dim, config.state_dim) / config.observation_dim**0.5
        )
        self.refiner: nn.Module | None
        if config.layers > 0:
            blocks: list[nn.Module] = []
            width = index.position_dim
            for _ in range(config.layers):
                blocks.append(nn.Linear(width, config.hidden))
                blocks.append(nn.SiLU())
                if config.dropout > 0.0:
                    blocks.append(nn.Dropout(config.dropout))
                width = config.hidden
            blocks.append(nn.Linear(width, config.state_dim))
            self.refiner = nn.Sequential(*blocks)
        else:
            self.refiner = None
        self.register_buffer("position_index", index.position_index.clone(), persistent=True)
        self.register_buffer("position_width", index.position_width.clone(), persistent=True)

    def _selected_inverse(self) -> Tensor:
        matrix: Tensor = self.observation_map
        if self.config.freeze_observation_map:
            matrix = matrix.detach()
        inverse = torch.linalg.pinv(matrix)
        return inverse[:, self.position_index]

    def encode(self, observations: Tensor) -> Tensor:
        """Map padded per-position observations to the write signal of the state."""
        if observations.dim() != 3:
            raise ValueError("observations must be (batch, tau_max, position_dim)")
        selected = self._selected_inverse()
        linear = torch.einsum("dtw,btw->btd", selected, observations)
        if self.refiner is None:
            return linear
        return linear + self.refiner(observations)

    def decode_full(self, states: Tensor) -> Tensor:
        """Apply the shared map to every state of the path: (tau_max, batch, p)."""
        return torch.real(states) @ self.observation_map.t()

    def decode(self, states: Tensor) -> Tensor:
        """Reconstruct the coordinates each position observes: (tau_max, batch, position_dim)."""
        full = self.decode_full(states)
        index = self.position_index.unsqueeze(1).expand(-1, full.shape[1], -1)
        gathered = torch.gather(full, 2, index)
        validity = (self.position_width > 0).to(gathered.dtype).view(-1, 1, 1)
        return gathered * validity

    def state_map_rank(self) -> float:
        singular = torch.linalg.svdvals(self.observation_map.detach())
        tolerance = (
            singular.max() * max(self.observation_map.shape) * torch.finfo(singular.dtype).eps
        )
        return float((singular > tolerance).sum())

    def observation_map_report(self) -> dict[str, float]:
        singular = torch.linalg.svdvals(self.observation_map.detach())
        return {
            "rows": float(self.observation_map.shape[0]),
            "columns": float(self.observation_map.shape[1]),
            "rank": self.state_map_rank(),
            "condition_number": float(singular.max() / singular.min().clamp_min(1e-12)),
            "largest_singular_value": float(singular.max()),
            "smallest_singular_value": float(singular.min()),
        }

    def extra_repr(self) -> str:
        return (
            f"observation_dim={self.config.observation_dim}, state_dim={self.config.state_dim}, "
            f"refiner_layers={self.config.layers}"
        )
