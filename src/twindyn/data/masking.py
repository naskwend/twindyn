"""Masked multi-source reconstruction targets.

Ref: Sec. 3.6 (equation 4: the masked multi-source reconstruction term, where a
fraction r of the observed locations is masked and the reconstruction is taken
from the resulting state; masking is applied at different densities because the
same biological system appears at different observation densities across
resources).

Two distinct notions of missingness are kept apart. A coordinate is *unavailable*
when the resource cannot observe it at all, in which case it is never a
reconstruction target. A coordinate is *withheld* when it is available and the
masking schedule hides it, which is what the reconstruction term scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from twindyn.data.axis import MicroenvironmentAxis


@dataclass(frozen=True)
class MaskingOutcome:
    withheld: Tensor
    kept: Tensor
    available: Tensor

    @property
    def ratio(self) -> float:
        denominator = int(self.available.sum())
        if denominator == 0:
            return 0.0
        return float(self.withheld.sum()) / denominator


def position_availability(axis: MicroenvironmentAxis, availability: Tensor) -> Tensor:
    """Reduce per-column availability to per-position availability."""
    if availability.dim() != 2:
        raise ValueError("availability must be (batch, width)")
    if availability.shape[-1] != axis.width:
        raise ValueError(f"expected width {axis.width}")
    batch = availability.shape[0]
    out = torch.zeros((batch, axis.tau_max), dtype=torch.bool, device=availability.device)
    for spec in axis.positions:
        columns = [*spec.compartment_columns, *spec.morphology_columns]
        if not columns:
            continue
        out[:, spec.index] = availability[:, columns].any(dim=1)
    return out


def sample_mask(
    axis: MicroenvironmentAxis,
    availability: Tensor,
    ratio: float,
    generator: torch.Generator,
    strategy: str = "position_uniform",
    min_positions_kept: int = 2,
) -> MaskingOutcome:
    """Choose which available coordinates are withheld at the chosen density."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("mask ratio must lie in [0, 1)")
    batch = availability.shape[0]
    device = availability.device
    withheld_positions = torch.zeros((batch, axis.tau_max), dtype=torch.bool, device=device)
    observed_positions = position_availability(axis, availability)
    if strategy == "position_uniform":
        scores = torch.rand((batch, axis.tau_max), generator=generator, device=device)
        scores = torch.where(observed_positions, scores, torch.full_like(scores, 2.0))
        order = torch.argsort(scores, dim=1)
        target = torch.floor(observed_positions.sum(dim=1).to(scores.dtype) * ratio).long()
        target = torch.clamp(target, min=0)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(axis.tau_max, device=device).expand(batch, -1))
        withheld_positions = observed_positions & (ranks < target.unsqueeze(1))
    elif strategy == "coordinate_uniform":
        counts = observed_positions.sum(dim=1)
        target = torch.floor(counts.to(torch.float32) * ratio).long()
        scores = torch.rand((batch, axis.tau_max), generator=generator, device=device)
        scores = torch.where(observed_positions, scores, torch.full_like(scores, 2.0))
        order = torch.argsort(scores, dim=1)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(axis.tau_max, device=device).expand(batch, -1))
        withheld_positions = observed_positions & (ranks < target.unsqueeze(1))
    else:
        raise ValueError(f"unknown masking strategy {strategy!r}")

    if min_positions_kept > 0:
        kept_counts = observed_positions.sum(dim=1) - withheld_positions.sum(dim=1)
        deficit = (min_positions_kept - kept_counts).clamp_min(0)
        for row in range(batch):
            if int(deficit[row]) <= 0:
                continue
            candidates = torch.nonzero(withheld_positions[row], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            order = torch.randperm(candidates.numel(), generator=generator, device=device)
            restore = candidates[order[: int(deficit[row])]]
            withheld_positions[row, restore] = False

    position_mask = axis.position_mask(device=device)
    withheld = withheld_positions.unsqueeze(-1) & position_mask.unsqueeze(0)
    available = availability.new_zeros((batch, axis.tau_max, axis.position_dim))
    for spec in axis.positions:
        columns = [*spec.compartment_columns, *spec.morphology_columns]
        if not columns:
            continue
        slot = available.new_zeros((batch, len(columns)))
        for slot_index, column in enumerate(columns):
            slot[:, slot_index] = availability[:, column]
        available[:, spec.index, : len(columns)] = slot
    available = available.bool() & position_mask.unsqueeze(0)
    kept = available & ~withheld
    return MaskingOutcome(withheld=withheld, kept=kept, available=available)


def reconstruction_error(prediction: Tensor, target: Tensor, withheld: Tensor) -> Tensor:
    """Equation (4): mean squared error over the masked coordinates."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must agree in shape")
    selector = withheld.to(prediction.dtype)
    denominator = selector.sum()
    if float(denominator) == 0.0:
        return prediction.sum() * 0.0
    return ((prediction - target).pow(2) * selector).sum() / denominator
