"""Harrell's concordance index with right censoring.

Ref: Table 1 and Table 6 (C-index is Harrell's concordance with a 95% bootstrap
interval over 1,000 resamples), Sec. 4.9 (the perturbation response is read off
changes in the risk ordering).

Only comparable pairs enter: a pair is comparable when the earlier follow-up time
belongs to an event, and a tie in time between an event and a censored
observation is comparable in the direction of the event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ConcordanceResult:
    c_index: float
    comparable_pairs: int
    concordant: float
    tied_risk: int
    tied_time: int

    @property
    def somers_d(self) -> float:
        denominator = self.comparable_pairs
        if denominator == 0:
            return float("nan")
        return (2.0 * self.concordant - denominator - self.tied_risk) / denominator


def concordance_index(
    risk: Tensor | np.ndarray, time: Tensor | np.ndarray, event: Tensor | np.ndarray
) -> ConcordanceResult:
    """Harrell's C with a tie in risk contributing one half.

    A pair is comparable when the earlier follow-up belongs to an event; pairs
    with equal follow-up are comparable only when exactly one of the two is an
    event, in which case the event ranks first.
    """
    risk_array, time_array, event_array = _coerce(risk, time, event)
    count = risk_array.shape[0]
    concordant = 0.0
    comparable = 0
    tied_risk = 0
    tied_time = 0
    for left in range(count):
        for right in range(left + 1, count):
            if time_array[left] == time_array[right]:
                if event_array[left] == event_array[right]:
                    tied_time += 1
                    continue
                first, second = (left, right) if event_array[left] > 0.0 else (right, left)
            else:
                first, second = (
                    (left, right) if time_array[left] < time_array[right] else (right, left)
                )
                if event_array[first] == 0.0:
                    continue
            comparable += 1
            delta = risk_array[first] - risk_array[second]
            if delta > 0.0:
                concordant += 1.0
            elif delta == 0.0:
                tied_risk += 1
                concordant += 0.5
    return ConcordanceResult(
        c_index=concordant / comparable if comparable else float("nan"),
        comparable_pairs=comparable,
        concordant=concordant,
        tied_risk=tied_risk,
        tied_time=tied_time,
    )


def _coerce(
    risk: Tensor | np.ndarray, time: Tensor | np.ndarray, event: Tensor | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    risk_array = np.asarray(
        risk.detach().cpu().numpy() if isinstance(risk, Tensor) else risk, dtype=np.float64
    )
    time_array = np.asarray(
        time.detach().cpu().numpy() if isinstance(time, Tensor) else time, dtype=np.float64
    )
    event_array = np.asarray(
        event.detach().cpu().numpy() if isinstance(event, Tensor) else event, dtype=np.float64
    )
    if not (risk_array.shape == time_array.shape == event_array.shape):
        raise ValueError("risk, time and event must share a shape")
    if risk_array.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    return risk_array, time_array, event_array


def concordance_se(
    risk: Tensor | np.ndarray, time: Tensor | np.ndarray, event: Tensor | np.ndarray
) -> float:
    """Standard error of the concordance index from the per-sample contribution spread."""
    contributions = per_sample_concordance(risk, time, event)
    weights = contributions[:, 1].sum()
    if contributions.size == 0 or weights == 0.0:
        return float("nan")
    mean = contributions[:, 0].sum() / weights
    variance = ((contributions[:, 0] - mean * contributions[:, 1]) ** 2).sum() / max(weights, 1.0)
    return float(np.sqrt(variance / max(weights, 1.0)))


def per_sample_concordance(
    risk: Tensor | np.ndarray, time: Tensor | np.ndarray, event: Tensor | np.ndarray
) -> np.ndarray:
    """Per-sample (concordant, comparable) contributions, used for the jackknife."""
    risk_array, time_array, event_array = _coerce(risk, time, event)
    count = risk_array.shape[0]
    contributions = np.zeros((count, 2), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            if time_array[left] == time_array[right]:
                if event_array[left] == event_array[right]:
                    continue
                first, second = (left, right) if event_array[left] > 0.0 else (right, left)
            else:
                first, second = (
                    (left, right) if time_array[left] < time_array[right] else (right, left)
                )
                if event_array[first] == 0.0:
                    continue
            score = (
                1.0
                if risk_array[first] > risk_array[second]
                else (0.5 if risk_array[first] == risk_array[second] else 0.0)
            )
            contributions[first] += (score, 1.0)
            contributions[second] += (score, 1.0)
    return contributions


def pairwise_ranking_agreement(before: Tensor, after: Tensor) -> float:
    """Share of comparable ordered pairs whose relative order is unchanged."""
    left = torch.tril_indices(before.shape[0], before.shape[0], offset=-1)
    difference_before = torch.sign(before[left[0]] - before[left[1]])
    difference_after = torch.sign(after[left[0]] - after[left[1]])
    return float((difference_before == difference_after).to(torch.float32).mean())
