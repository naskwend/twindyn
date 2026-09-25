"""Trajectory consistency loss.

Ref: Sec. 3.6, equation (6): the operator has to predict the position of objects
in the future from their past coordinates; without the term the selective scan can
imitate the signals of positions it has already encountered, which turns the
trajectory into a re-encoding rather than a prediction. ``g(.)`` is the pooled
trajectory summary and ``sim(.,.)`` is the cosine similarity.

The pool of negatives is the batch, so the term asks whether the second half of a
sample's own trajectory is closer to the first half of that same trajectory than
to the first halves of the other samples in the batch. The candidate set is
restricted to samples whose second half is populated, so a sample with no future
positions contributes no comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from twindyn.models.tor import state_features


@dataclass(frozen=True)
class TrajectoryBreakdown:
    loss: Tensor
    accuracy: Tensor
    candidates: int
    temperature: float


def split_halves(states: Tensor) -> tuple[Tensor, Tensor]:
    """Split the axis at its midpoint into a past and a future half."""
    if states.dim() != 3:
        raise ValueError("states must be (tau_max, batch, state_dim)")
    horizon = states.shape[0]
    if horizon < 2:
        raise ValueError("the trajectory needs at least two positions")
    half = horizon // 2
    return states[:half], states[half:]


def pooled_summary(half: Tensor, validity: Tensor | None = None) -> Tensor:
    """``g(.)``: mean of the real state view over the positions of one half."""
    features = state_features(half)
    if validity is None:
        return features.mean(dim=0)
    weights = validity.to(features.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=0).clamp_min(1.0)
    return (features * weights).sum(dim=0) / denominator


def trajectory_consistency_loss(
    states: Tensor,
    validity: Tensor | None = None,
    temperature: float = 0.1,
) -> TrajectoryBreakdown:
    """Equation (6) with the in-batch negatives of the other samples."""
    past, future = split_halves(states)
    past_validity = None if validity is None else validity[: past.shape[0]]
    future_validity = None if validity is None else validity[past.shape[0] :]
    past_summary = pooled_summary(past, past_validity)
    future_summary = pooled_summary(future, future_validity)
    normalised_past = torch.nn.functional.normalize(past_summary, dim=-1)
    normalised_future = torch.nn.functional.normalize(future_summary, dim=-1)
    similarity = normalised_past @ normalised_future.t()
    logits = similarity / max(temperature, 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    with torch.no_grad():
        accuracy = (logits.argmax(dim=1) == labels).to(torch.float32).mean()
    return TrajectoryBreakdown(
        loss=loss,
        accuracy=accuracy,
        candidates=int(logits.shape[0]),
        temperature=temperature,
    )


def future_prediction_gap(
    states: Tensor,
    temperature: float = 0.1,
) -> dict[str, float]:
    """Whether the operator separates its own future from the batch's futures.

    A scan that merely re-encodes the positions it already saw leaves the past and
    future summaries indistinguishable, so this gap is the quantity that separate the
    predictive reading from the re-encoding one.
    """
    with torch.no_grad():
        past, future = split_halves(states)
        past_summary = torch.nn.functional.normalize(pooled_summary(past), dim=-1)
        future_summary = torch.nn.functional.normalize(pooled_summary(future), dim=-1)
        own = torch.nn.functional.cosine_similarity(past_summary, future_summary, dim=-1)
        cross = past_summary @ future_summary.t()
        mask = 1.0 - torch.eye(cross.shape[0], device=cross.device)
        other = (cross * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return {
            "own_similarity": float(own.mean()),
            "other_similarity": float(other.mean()),
            "gap": float((own - other).mean()),
            "temperature": temperature,
        }
