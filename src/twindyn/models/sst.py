"""Selective state-space transition operator.

Ref: Sec. 3.4 (the operator propagates the state along the ordered axis and
selects how much new observation to write and how much of the existing state to
retain), Sec. 3.8 (the scan is linear in the axis length, O(tau_max * d^2)), and
reference 54 for the discretisation, the complex-valued state and the
multi-input/multi-output form.

The continuous spectrum is diagonal and complex, so no sinusoidal basis has to be
written down for the oscillatory part; the input and output maps are dense with a
channel count that defaults to the state dimensionality, which reproduces the
stated cost bound. All selectivity is input-driven: the step size, the write gate
and the output gate are produced from the write signal at each position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from twindyn.models.scan import (
    ScanOutput,
    clamp_step_size,
    exponential_decay,
    scan_cost,
    selective_scan,
    spectrum_separation,
)


@dataclass(frozen=True)
class SstConfig:
    state_dim: int = 64
    input_features: int = 64
    input_channels: int | None = None
    output_channels: int | None = None
    selective_hidden: int | None = None
    complex_state: bool = True
    discretisation: str = "exponential_trapezoidal"
    alpha_floor: float = 0.05
    alpha_ceiling: float = 1.2

    @property
    def resolved_input_channels(self) -> int:
        return int(self.input_channels if self.input_channels is not None else self.state_dim)

    @property
    def resolved_output_channels(self) -> int:
        return int(self.output_channels if self.output_channels is not None else self.state_dim)

    @property
    def resolved_hidden(self) -> int:
        return int(self.selective_hidden if self.selective_hidden is not None else self.state_dim)


class SelectiveStateSpaceTransition(nn.Module):
    """The SST operator: one selective linear recurrence along the axis."""

    def __init__(self, config: SstConfig) -> None:
        super().__init__()
        self.config = config
        dimension = config.state_dim
        self.decay_rate = nn.Parameter(torch.linspace(0.05, 1.15, dimension))
        self.oscillation = nn.Parameter(torch.linspace(-1.0, 1.0, dimension))
        self.input_embedding_real = nn.Parameter(
            torch.randn(dimension, config.resolved_input_channels) / dimension**0.5
        )
        self.input_embedding_imag = nn.Parameter(
            torch.randn(dimension, config.resolved_input_channels) / dimension**0.5
        )
        self.output_embedding_real = nn.Parameter(
            torch.randn(config.resolved_output_channels, dimension) / dimension**0.5
        )
        self.output_embedding_imag = nn.Parameter(
            torch.randn(config.resolved_output_channels, dimension) / dimension**0.5
        )
        self.selectivity = nn.Sequential(
            nn.Linear(config.input_features, config.resolved_hidden),
            nn.SiLU(),
            nn.Linear(config.resolved_hidden, 5 * dimension),
        )
        self.step_bias = nn.Parameter(torch.zeros(dimension) - 1.0)

    @property
    def output_channels(self) -> int:
        return self.config.resolved_output_channels

    @property
    def spectrum(self) -> Tensor:
        """Complex, stable and separated: -exp(decay) + i * exp(oscillation)."""
        real = -torch.exp(self.decay_rate).clamp(
            self.config.alpha_floor, self.config.alpha_ceiling + 4.0
        )
        imaginary = torch.exp(self.oscillation) * torch.pi / 3.0
        return torch.complex(real, imaginary)

    @property
    def input_embedding(self) -> Tensor:
        return torch.complex(self.input_embedding_real, self.input_embedding_imag)

    @property
    def output_embedding(self) -> Tensor:
        return torch.complex(self.output_embedding_real, self.output_embedding_imag)

    def selectivity_heads(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Step size, write gate and output gate for a batch of positions."""
        raw = self.selectivity(features)
        split = torch.split(raw, self.config.state_dim, dim=-1)
        delta = split[0] + self.step_bias
        write = (
            torch.complex(split[1], split[2])
            if self.config.complex_state
            else torch.complex(split[1], torch.zeros_like(split[1]))
        )
        read = (
            torch.complex(split[3], split[4])
            if self.config.complex_state
            else torch.complex(split[3], torch.zeros_like(split[3]))
        )
        return delta, write, read

    def forward(
        self,
        inputs: Tensor,
        availability: Tensor | None = None,
        initial_state: Tensor | None = None,
        inject_position: int | None = None,
        inject_delta: Tensor | None = None,
    ) -> ScanOutput:
        if inputs.dim() != 3:
            raise ValueError("inputs must be (batch, tau_max, input_features)")
        if inputs.shape[-1] != self.config.input_features:
            raise ValueError(
                f"expected {self.config.input_features} input features, got {inputs.shape[-1]}"
            )
        gated = inputs
        if availability is not None:
            gated = inputs * availability.unsqueeze(-1).to(inputs.dtype)
        delta, write, read = self.selectivity_heads(gated)
        return selective_scan(
            inputs=gated,
            delta=delta,
            spectrum=self.spectrum.to(inputs.device),
            input_gate=write,
            output_gate=read,
            input_embedding=self.input_embedding.to(inputs.device),
            output_embedding=self.output_embedding.to(inputs.device),
            initial_state=initial_state,
            inject_position=inject_position,
            inject_delta=inject_delta,
        )

    def retained_fraction(self, inputs: Tensor) -> Tensor:
        """The share of the previous state kept at each position."""
        delta, _, _ = self.selectivity_heads(inputs)
        return exponential_decay(clamp_step_size(delta), self.spectrum.to(inputs.device))

    def analytic_cost(self, tau_max: int) -> dict[str, int]:
        return scan_cost(
            tau_max,
            self.config.state_dim,
            self.config.resolved_input_channels,
            self.config.resolved_output_channels,
        )

    def spectrum_report(self) -> dict[str, float]:
        return spectrum_separation(self.spectrum.detach())

    def growth_curve(self, positions: tuple[int, ...]) -> list[dict[str, float]]:
        """Scan cost against the quadratic cost of operating on spatial units."""
        curve: list[dict[str, float]] = []
        for count in positions:
            linear = float(
                scan_cost(
                    count,
                    self.config.state_dim,
                    self.config.resolved_input_channels,
                    self.config.resolved_output_channels,
                )["scan"]
            )
            curve.append(
                {
                    "positions": float(count),
                    "scan_cost": linear,
                    "quadratic_cost": float(count * count * self.config.state_dim),
                    "ratio": linear / float(count * count * self.config.state_dim),
                }
            )
        return curve

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.config.state_dim}, in={self.config.resolved_input_channels}, "
            f"out={self.config.resolved_output_channels}, complex={self.config.complex_state}"
        )
