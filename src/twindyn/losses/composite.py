"""The composite label-free objective.

Ref: Sec. 3.6, equation (3): the representation-learning phase minimises three
terms, all of which are free of the survival label. Sec. 3.10 and Sec. 4.4: the
weights are among the quantities chosen on internal validation, so the values in
the configuration are engineering defaults rather than reported numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from twindyn.data.masking import MaskingOutcome
from twindyn.data.pairs import PairSet
from twindyn.losses.alignment import AlignmentBreakdown, cross_resource_alignment_loss
from twindyn.losses.reconstruction import masked_reconstruction_loss
from twindyn.losses.trajectory import TrajectoryBreakdown, trajectory_consistency_loss


@dataclass(frozen=True)
class ObjectiveWeights:
    lambda_mask: float = 1.0
    lambda_align: float = 1.0
    lambda_trajectory: float = 0.5
    trajectory_temperature: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {
            "lambda_mask": self.lambda_mask,
            "lambda_align": self.lambda_align,
            "lambda_trajectory": self.lambda_trajectory,
            "trajectory_temperature": self.trajectory_temperature,
        }


@dataclass(frozen=True)
class ObjectiveBreakdown:
    total: Tensor
    reconstruction: Tensor
    alignment: Tensor
    trajectory: Tensor
    alignment_detail: AlignmentBreakdown
    trajectory_detail: TrajectoryBreakdown

    def scalars(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "reconstruction": float(self.reconstruction.detach()),
            "alignment": float(self.alignment.detach()),
            "trajectory": float(self.trajectory.detach()),
            "trajectory_top1": float(self.trajectory_detail.accuracy.detach()),
        }


def composite_objective(
    reconstruction: Tensor,
    target: Tensor,
    withheld: Tensor,
    states: Tensor,
    pairs: list[PairSet],
    validity: Tensor,
    weights: ObjectiveWeights,
    active: frozenset[str] = frozenset({"mask", "align", "trajectory"}),
) -> ObjectiveBreakdown:
    """Equation (3) with per-term switches used by the ablation table."""
    if "mask" in active:
        reconstruction_term = masked_reconstruction_loss(reconstruction, target, withheld)
    else:
        reconstruction_term = reconstruction.sum() * 0.0
    sample_states = states.mean(dim=0)
    if "align" in active:
        alignment_detail = cross_resource_alignment_loss(sample_states, pairs)
        alignment_term = alignment_detail.loss
    else:
        alignment_detail = cross_resource_alignment_loss(sample_states, [])
        alignment_term = states.real.sum() * 0.0
    if "trajectory" in active:
        trajectory_detail = trajectory_consistency_loss(
            states, validity=validity, temperature=weights.trajectory_temperature
        )
        trajectory_term = trajectory_detail.loss
    else:
        trajectory_detail = TrajectoryBreakdown(
            loss=states.real.sum() * 0.0,
            accuracy=torch.tensor(0.0),
            candidates=0,
            temperature=weights.trajectory_temperature,
        )
        trajectory_term = states.real.sum() * 0.0
    total = (
        weights.lambda_mask * reconstruction_term
        + weights.lambda_align * alignment_term
        + weights.lambda_trajectory * trajectory_term
    )
    return ObjectiveBreakdown(
        total=total,
        reconstruction=reconstruction_term,
        alignment=alignment_term,
        trajectory=trajectory_term,
        alignment_detail=alignment_detail,
        trajectory_detail=trajectory_detail,
    )


def objective_from_masking(
    reconstruction: Tensor,
    target: Tensor,
    masking: MaskingOutcome,
    states: Tensor,
    pairs: list[PairSet],
    validity: Tensor,
    weights: ObjectiveWeights,
    active: frozenset[str] = frozenset({"mask", "align", "trajectory"}),
) -> ObjectiveBreakdown:
    """Convenience wrapper that reads the withheld mask from the masking outcome."""
    return composite_objective(
        reconstruction=reconstruction,
        target=target,
        withheld=masking.withheld,
        states=states,
        pairs=pairs,
        validity=validity,
        weights=weights,
        active=active,
    )
