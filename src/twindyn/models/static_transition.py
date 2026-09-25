"""The parameter-matched static encoder used as the level-0 ablation control.

Ref: Table 2 (the parameter-matched control replaces the selective state-space
operator with a static encoder of matched count, so the drop is attributable to
the transition law rather than to capacity; the reported parameter counts are 3.16
million for the full model and 3.14 million for the control), Sec. 4.9 (the
necessity arm of the mechanism triad is this substitution).

The control keeps the same interface as the selective operator, returns a state
path of the same shape for every position, and removes the recurrence entirely.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from twindyn.models.scan import ScanOutput


class StaticMatchedTransition(nn.Module):
    """A feed-forward map with no recurrence and a matched parameter count."""

    state_dim: int
    input_features: int
    output_channels: int

    def __init__(
        self,
        state_dim: int,
        input_features: int,
        output_channels: int,
        hidden: int | None = None,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_features = int(input_features)
        self.output_channels = int(output_channels)
        self.hidden_width = 0
        width = int(hidden) if hidden is not None else int(state_dim)
        self.hidden_width = int(width)
        blocks: list[nn.Module] = [nn.Linear(input_features, width), nn.SiLU()]
        for _ in range(max(0, layers - 1)):
            blocks.append(nn.Linear(width, width))
            blocks.append(nn.SiLU())
        self.body = nn.Sequential(*blocks)
        self.state_head = nn.Linear(width, 2 * state_dim)
        self.output_embedding = nn.Linear(state_dim, output_channels, bias=False)

    def forward(
        self,
        inputs: Tensor,
        availability: Tensor | None = None,
        initial_state: Tensor | None = None,
        inject_position: int | None = None,
        inject_delta: Tensor | None = None,
    ) -> ScanOutput:
        """Map each position's write signal straight to a state of the same shape."""
        gated = inputs
        if availability is not None:
            gated = inputs * availability.unsqueeze(-1).to(inputs.dtype)
        projected = self.state_head(self.body(gated))
        real, imaginary = projected.chunk(2, dim=-1)
        states = torch.complex(real, imaginary).transpose(0, 1)
        if inject_position is not None and inject_delta is not None:
            states = states.clone()
            states[inject_position] = states[inject_position] + inject_delta
        outputs = self.output_embedding(torch.real(states))
        step_sizes = torch.zeros(
            (states.shape[0], states.shape[1], self.state_dim),
            dtype=torch.float32,
            device=inputs.device,
        )
        return ScanOutput(states=states, outputs=outputs, step_sizes=step_sizes)

    def analytic_cost(self, tau_max: int) -> dict[str, int]:
        """A static map has no scan; the cost is the feed-forward evaluation."""
        per_step = int(self.input_features) * int(self.hidden_width) + int(self.state_dim) * int(
            self.output_channels
        )
        return {
            "state_update": 0,
            "input_projection": int(tau_max * per_step),
            "output_projection": int(tau_max * self.output_channels * self.state_dim),
            "scan": int(tau_max * per_step),
            "paper_bound_tau_max_d2": int(tau_max * self.state_dim * self.state_dim),
            "within_bound_class": 1,
        }

    def spectrum_report(self) -> dict[str, float]:
        """A static map has no spectrum to report."""
        return {
            "count": 0.0,
            "min_spectral_gap": float("nan"),
            "max_spectral_gap": float("nan"),
            "min_magnitude": float("nan"),
            "max_magnitude": float("nan"),
            "unique_count": 0.0,
        }

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, in={self.input_features}, "
            f"out={self.output_channels}, recurrent=False"
        )


def capacity_match(recurrent: nn.Module, static: nn.Module) -> dict[str, float]:
    """The two parameter counts and their relative gap, as an equal-capacity check."""
    recurrent_count = float(sum(parameter.numel() for parameter in recurrent.parameters()))
    static_count = float(sum(parameter.numel() for parameter in static.parameters()))
    reference = max(recurrent_count, static_count, 1.0)
    return {
        "recurrent_parameters": recurrent_count,
        "static_parameters": static_count,
        "absolute_gap": abs(recurrent_count - static_count),
        "relative_gap": abs(recurrent_count - static_count) / reference,
    }


def matched_transition(
    recurrent: nn.Module,
    state_dim: int,
    input_features: int,
    output_channels: int,
    max_hidden: int = 512,
) -> StaticMatchedTransition:
    """A static control whose parameter count is closest to the operator's.

    The control's width and depth are solved for rather than fixed, because the
    ablation's claim is that the drop comes from the transition law rather than
    from capacity, and that claim only holds if the two counts agree.
    """
    target = float(sum(parameter.numel() for parameter in recurrent.parameters()))
    best: tuple[float, StaticMatchedTransition] | None = None
    for layers in (1, 2, 3):
        for hidden in range(max(state_dim, 8), max_hidden + 1, 8):
            candidate = StaticMatchedTransition(
                state_dim=state_dim,
                input_features=input_features,
                output_channels=output_channels,
                hidden=hidden,
                layers=layers,
            )
            count = float(sum(parameter.numel() for parameter in candidate.parameters()))
            gap = abs(count - target) / max(count, target, 1.0)
            if best is None or gap < best[0]:
                best = (gap, candidate)
            if count > target:
                break
    if best is None:
        raise RuntimeError("no static control could be sized")
    return best[1]
