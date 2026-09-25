"""Cox partial likelihood under right censoring.

Ref: Sec. 3.5 (the read-out is trained with the conventional survival objective
for a continuous risk under right censoring, and this is the only place a survival
label enters the system; reference 47 for the objective).

Ties in follow-up time use the Breslow approximation, which is the form the
standard deep survival head uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SurvivalLoss:
    loss: Tensor
    events: int
    comparables: int


def cox_partial_likelihood(risk: Tensor, time: Tensor, event: Tensor) -> SurvivalLoss:
    """Negative log partial likelihood over the risk sets at event times.

    Ties in follow-up time are handled by the Breslow approximation: every member
    of a tie block shares the risk set of the whole block, which is why the
    cumulative normaliser is taken at the last element of each block rather than at
    each element separately.
    """
    if risk.dim() != 1:
        raise ValueError("risk must be one-dimensional")
    if time.shape != risk.shape or event.shape != risk.shape:
        raise ValueError("time and event must match the risk shape")
    if risk.shape[0] == 0:
        return SurvivalLoss(risk.sum() * 0.0, 0, 0)
    order = torch.argsort(time, descending=True, stable=True)
    sorted_risk = risk[order]
    sorted_event = event[order].to(risk.dtype)
    sorted_time = time[order]
    log_normaliser = _log_risk_set_normaliser(sorted_risk, sorted_time)
    terms = (sorted_risk - log_normaliser) * sorted_event
    denominator = sorted_event.sum().clamp_min(1.0)
    return SurvivalLoss(
        loss=-terms.sum() / denominator,
        events=int(sorted_event.sum().item()),
        comparables=int(risk.shape[0]),
    )


def _log_risk_set_normaliser(sorted_risk: Tensor, sorted_time: Tensor) -> Tensor:
    """Log of the risk-set sum, shared across the members of each tie block."""
    cumulative = torch.logcumsumexp(sorted_risk, dim=0)
    count = sorted_time.shape[0]
    is_block_end = torch.ones(count, dtype=torch.bool, device=sorted_time.device)
    is_block_end[:-1] = sorted_time[1:] != sorted_time[:-1]
    positions = torch.arange(count, device=sorted_time.device)
    candidates = torch.where(is_block_end, positions, torch.full_like(positions, count))
    block_end = torch.cummin(candidates.flip(0), dim=0).values.flip(0)
    return cumulative[block_end]


def risk_set_sizes(time: Tensor, event: Tensor) -> Tensor:
    """Number of samples at risk at each event time, for the power analysis."""
    order = torch.argsort(time, descending=True, stable=True)
    sizes = torch.arange(1, time.shape[0] + 1, device=time.device, dtype=time.dtype)
    sizes = sizes.flip(0)
    return sizes[event[order].bool()]


def partial_likelihood_by_stratum(
    risk: Tensor,
    time: Tensor,
    event: Tensor,
    stratum: Tensor,
) -> dict[int, float]:
    """The same objective evaluated separately inside each stratum."""
    out: dict[int, float] = {}
    for value in torch.unique(stratum):
        selector = stratum == value
        if int(selector.sum()) < 2:
            continue
        result = cox_partial_likelihood(risk[selector], time[selector], event[selector])
        out[int(value)] = float(result.loss)
    return out
