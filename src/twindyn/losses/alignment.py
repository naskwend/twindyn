"""Cross-resource alignment loss.

Ref: Sec. 3.6, equation (5): the squared distance between the states of matched
samples drawn from different resources, taken over pairs (i, j) with r(i) != r(j).
Sec. 3.3 states that when a resource pair shares no matched samples the term
stays unspecified for that pair and contributes only to the other objectives, so
a pair set with no matches yields no term rather than a zero-distance term.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from twindyn.data.pairs import PairSet, pairs_to_index


@dataclass(frozen=True)
class AlignmentBreakdown:
    loss: Tensor
    per_pair: dict[str, float]
    available_pairs: int
    matched_pairs: int


def state_distance(left: Tensor, right: Tensor) -> Tensor:
    """Squared Euclidean distance between complex states."""
    difference = left - right
    return (difference.conj() * difference).real.sum(dim=-1)


def cross_resource_alignment_loss(
    states: Tensor,
    pairs: list[PairSet],
    weights: Tensor | None = None,
) -> AlignmentBreakdown:
    """Equation (5) over every resource pair that shares matched samples."""
    left, right, _ = pairs_to_index(pairs, states.device)
    if left.numel() == 0:
        zero = states.real.sum() * 0.0
        return AlignmentBreakdown(
            loss=zero,
            per_pair={
                f"{pair.left_resource}|{pair.right_resource}": float("nan") for pair in pairs
            },
            available_pairs=0,
            matched_pairs=0,
        )
    distance = state_distance(states[left], states[right])
    if weights is not None:
        if weights.shape != distance.shape:
            raise ValueError("pair weights must match the number of matched pairs")
        distance = distance * weights.to(distance.dtype)
    per_pair: dict[str, float] = {
        f"{pair.left_resource}|{pair.right_resource}": float("nan")
        for pair in pairs
        if not pair.available
    }
    cursor = 0
    for pair in pairs:
        if not pair.available:
            continue
        width = int(pair.left_index.numel())
        per_pair[f"{pair.left_resource}|{pair.right_resource}"] = float(
            distance[cursor : cursor + width].mean().detach()
        )
        cursor += width
    return AlignmentBreakdown(
        loss=distance.mean(),
        per_pair=per_pair,
        available_pairs=sum(1 for pair in pairs if pair.available),
        matched_pairs=int(distance.numel()),
    )


def alignment_initialisation_reference(states: Tensor, pairs: list[PairSet]) -> dict[str, float]:
    """Per-pair alignment magnitude at initialisation, for the relative readings."""
    left, right, _ = pairs_to_index(pairs, states.device)
    if left.numel() == 0:
        return {f"{pair.left_resource}|{pair.right_resource}": 0.0 for pair in pairs}
    distance = state_distance(states[left], states[right])
    reference: dict[str, float] = {}
    cursor = 0
    for pair in pairs:
        key = f"{pair.left_resource}|{pair.right_resource}"
        if not pair.available:
            reference[key] = 0.0
            continue
        width = int(pair.left_index.numel())
        reference[key] = float(distance[cursor : cursor + width].mean())
        cursor += width
    return reference


def relative_alignment(final: dict[str, float], initial: dict[str, float]) -> dict[str, float]:
    """Equation (5) magnitude per resource pair relative to its initialisation."""
    out: dict[str, float] = {}
    for key, value in final.items():
        baseline = initial.get(key, 0.0)
        out[key] = value / baseline if baseline else float("nan")
    return out


def alignment_scale(states: Tensor) -> Tensor:
    """A scale-free companion so the term does not collapse the state to zero."""
    return states.abs().pow(2).mean().clamp_min(1e-8)


def normalised_alignment_loss(
    states: Tensor,
    pairs: list[PairSet],
) -> AlignmentBreakdown:
    """Equation (5) divided by the state scale, used only for reporting."""
    breakdown = cross_resource_alignment_loss(states, pairs)
    scale = alignment_scale(states.detach())
    if torch.isnan(breakdown.loss):
        return breakdown
    return AlignmentBreakdown(
        loss=breakdown.loss / scale,
        per_pair=dict(breakdown.per_pair),
        available_pairs=breakdown.available_pairs,
        matched_pairs=breakdown.matched_pairs,
    )
