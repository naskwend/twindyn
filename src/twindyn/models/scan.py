"""The selective state-space scan with exponential-trapezoidal discretisation.

Ref: Sec. 3.4 (the operator advances the state along the ordered axis, choosing
how much new observation is written and how much of the existing state remains;
selectivity matters because the axis is not equally informative across
positions), Sec. 3.8 (the scan cost grows linearly as O(tau_max * d^2) because
the state dimensionality is constant, and the recurrence is linear), Sec. 2.4 and
reference 54 (the exponential-trapezoidal discretisation of the continuous
transition, a complex-valued state that needs no explicit sinusoidal basis, and a
multi-input/multi-output form).

The continuous law is dz/dtau = A z + B u with A and B input-dependent. The
homogeneous part is integrated exactly over a step and the input is integrated
with the trapezoid rule, which gives

    z_tau = exp(A delta) z_{tau-1} + (delta / 2) (exp(A delta) B u_{tau-1} + B u_tau)

so the write term carries the previous input as well as the current one. The
continuous spectrum is diagonal and complex, so exp(A delta) is elementwise; the
input and output maps are dense with ``input_channels`` and ``output_channels``
columns, and both default to the state dimensionality, which is what makes the
per-step cost d^2 and the full-scan cost O(tau_max * d^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

DELTA_MIN = 1e-4
DELTA_MAX = 4.0


@dataclass(frozen=True)
class ScanOutput:
    states: Tensor
    outputs: Tensor
    step_sizes: Tensor

    @property
    def horizon(self) -> int:
        return int(self.states.shape[0])


def scan_cost(
    tau_max: int, state_dim: int, input_channels: int, output_channels: int
) -> dict[str, int]:
    """Analytic cost of one scan, in scalar multiply-add units.

    The recurrent state update costs d per position; the input and output maps
    cost d*m and n*d, where m and n are the input and output channel counts. With
    both channel counts equal to the state dimensionality the total is
    tau_max * (2 d^2 + d), which is the same linear-in-tau_max, quadratic-in-d
    class as the bound the manuscript states.
    """
    state_update = tau_max * state_dim
    input_projection = tau_max * state_dim * input_channels
    output_projection = tau_max * output_channels * state_dim
    total = state_update + input_projection + output_projection
    bound = tau_max * state_dim * state_dim
    return {
        "state_update": int(state_update),
        "input_projection": int(input_projection),
        "output_projection": int(output_projection),
        "scan": int(total),
        "paper_bound_tau_max_d2": int(bound),
        "within_bound_class": int(total <= 3 * bound),
    }


def exponential_decay(step_size: Tensor, spectrum: Tensor) -> Tensor:
    """``exp(A delta)`` for a diagonal continuous spectrum."""
    return torch.exp(step_size * spectrum)


def clamp_step_size(raw: Tensor) -> Tensor:
    return DELTA_MIN + (DELTA_MAX - DELTA_MIN) * torch.sigmoid(raw)


def exponential_trapezoidal_step(
    previous_state: Tensor,
    decay: Tensor,
    write_current: Tensor,
    write_previous: Tensor,
    step_size: Tensor,
) -> Tensor:
    """One step of the exponential-trapezoidal rule."""
    half_delta = step_size * 0.5
    return decay * previous_state + half_delta * (decay * write_previous + write_current)


def selective_scan(
    inputs: Tensor,
    delta: Tensor,
    spectrum: Tensor,
    input_gate: Tensor,
    output_gate: Tensor,
    input_embedding: Tensor,
    output_embedding: Tensor,
    initial_state: Tensor | None = None,
    inject_position: int | None = None,
    inject_delta: Tensor | None = None,
) -> ScanOutput:
    """Run the selective recurrence along the axis.

    ``inputs`` is ``(batch, tau_max, input_features)``, ``delta``, ``input_gate``
    and ``output_gate`` are ``(batch, tau_max, state_dim)``, ``spectrum`` is
    ``(state_dim,)`` complex, ``input_embedding`` is ``(state_dim, input_channels)``
    complex and ``output_embedding`` is ``(output_channels, state_dim)`` complex.

    ``inject_delta`` is added to the state the moment it reaches
    ``inject_position`` and the recurrence then continues from the perturbed
    state, which is how the perturbation-response evidence is produced.
    """
    if inputs.dim() != 3:
        raise ValueError("inputs must be (batch, tau_max, input_features)")
    batch, horizon, _ = inputs.shape
    state_dim = spectrum.shape[0]
    if delta.shape != (batch, horizon, state_dim):
        raise ValueError("delta must be (batch, tau_max, state_dim)")
    if inject_delta is not None and inject_position is None:
        raise ValueError("an injected perturbation needs a position")
    device = inputs.device
    dtype = spectrum.dtype
    state = (
        torch.zeros((batch, state_dim), dtype=dtype, device=device)
        if initial_state is None
        else initial_state.to(dtype)
    )
    write_previous = torch.zeros((batch, state_dim), dtype=dtype, device=device)
    states = torch.zeros((horizon, batch, state_dim), dtype=dtype, device=device)
    outputs = torch.zeros(
        (horizon, batch, output_embedding.shape[0]), dtype=torch.float32, device=device
    )
    step_sizes = torch.zeros((horizon, batch, state_dim), dtype=torch.float32, device=device)
    for position in range(horizon):
        step = clamp_step_size(delta[:, position])
        decay = exponential_decay(step, spectrum)
        embedded = torch.einsum("sm,bm->bs", input_embedding, inputs[:, position].to(dtype))
        write_current = input_gate[:, position].to(dtype) * embedded
        state = exponential_trapezoidal_step(state, decay, write_current, write_previous, step)
        if inject_position is not None and position == inject_position and inject_delta is not None:
            state = state + inject_delta.to(dtype)
        write_previous = write_current
        states[position] = state
        gated = output_gate[:, position].to(dtype) * state
        outputs[position] = torch.real(torch.einsum("cs,bs->bc", output_embedding, gated))
        step_sizes[position] = step
    return ScanOutput(states=states, outputs=outputs, step_sizes=step_sizes)


def discretised_transition(step_size: Tensor, spectrum: Tensor) -> Tensor:
    """The retained fraction of the previous state, exposed for the state checks."""
    return exponential_decay(step_size, spectrum)


def state_energy(states: Tensor) -> Tensor:
    """Mean squared magnitude of the complex state over the axis."""
    return states.abs().pow(2).mean(dim=(0, 1))


def spectrum_separation(spectrum: Tensor) -> dict[str, float]:
    """Minimum pairwise eigenvalue gap, the quantity Assumption 3 constrains."""
    if spectrum.dim() != 1:
        raise ValueError("spectrum must be one-dimensional")
    magnitudes = spectrum.abs()
    angles = torch.angle(spectrum)
    stacked = torch.stack([magnitudes, angles], dim=1)
    pairwise = torch.cdist(stacked, stacked)
    identity = torch.eye(spectrum.shape[0], device=spectrum.device, dtype=torch.bool)
    off_diagonal = torch.where(identity, torch.full_like(pairwise, float("inf")), pairwise)
    rounded = torch.stack([torch.round(magnitudes * 1e6), torch.round(angles * 1e6)], dim=1)
    unique = torch.unique(rounded, dim=0)
    return {
        "count": float(spectrum.shape[0]),
        "min_spectral_gap": float(off_diagonal.min()),
        "max_spectral_gap": float(off_diagonal.max()),
        "min_magnitude": float(magnitudes.min()),
        "max_magnitude": float(magnitudes.max()),
        "unique_count": float(unique.shape[0]),
    }
