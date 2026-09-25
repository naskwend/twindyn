"""Ensemble and nested bootstrap reporting.

Ref: Table 1 and Table 6 (95% bootstrap intervals over 1,000 resamples; the
paired difference against the strongest static composition baseline carries its
own bootstrap interval), Sec. 4.9 and Table 5 (the probe interval comes from ten
repeated splits).

The paired difference resamples tumour samples once and evaluates both methods on
the same resample, which is what makes the interval an interval for the contrast
rather than for either method alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twindyn.metrics.concordance import concordance_index


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    resamples: int
    method: str

    @property
    def covers_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    def as_dict(self) -> dict[str, float]:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "resamples": float(self.resamples),
        }


@dataclass(frozen=True)
class PairedDifference:
    point: float
    interval: Interval
    p_value: float
    same_side: float

    def as_dict(self) -> dict[str, float]:
        payload = self.interval.as_dict()
        payload["p_value"] = self.p_value
        payload["same_side"] = self.same_side
        return payload


def percentile_interval(
    values: NDArray[np.float64], level: float = 0.95, method: str = "percentile"
) -> Interval:
    if values.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, method)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return Interval(
        point=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        resamples=int(values.size),
        method=method,
    )


def bootstrap_indices(count: int, resamples: int, seed: int) -> NDArray[np.int64]:
    generator = np.random.default_rng(seed)
    return generator.integers(0, count, size=(resamples, count), dtype=np.int64)


def bootstrap_concordance(
    risk: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    index = bootstrap_indices(risk.shape[0], resamples, seed)
    values = np.full(resamples, np.nan, dtype=np.float64)
    for row in range(resamples):
        selector = index[row]
        result = concordance_index(risk[selector], time[selector], event[selector])
        values[row] = result.c_index
    finite = values[np.isfinite(values)]
    interval = percentile_interval(finite, level)
    return Interval(
        point=concordance_index(risk, time, event).c_index,
        lower=interval.lower,
        upper=interval.upper,
        resamples=int(finite.size),
        method="percentile",
    )


def paired_bootstrap_difference(
    risk_a: NDArray[np.float64],
    risk_b: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> PairedDifference:
    """Interval and two-sided bootstrap p-value for a paired concordance contrast."""
    if not (risk_a.shape == risk_b.shape == time.shape == event.shape):
        raise ValueError("all inputs must share a shape")
    point_a = concordance_index(risk_a, time, event).c_index
    point_b = concordance_index(risk_b, time, event).c_index
    index = bootstrap_indices(risk_a.shape[0], resamples, seed)
    differences = np.full(resamples, np.nan, dtype=np.float64)
    for row in range(resamples):
        selector = index[row]
        value_a = concordance_index(risk_a[selector], time[selector], event[selector]).c_index
        value_b = concordance_index(risk_b[selector], time[selector], event[selector]).c_index
        differences[row] = value_a - value_b
    finite = differences[np.isfinite(differences)]
    interval = percentile_interval(finite, level)
    below = float((finite <= 0.0).mean())
    above = float((finite >= 0.0).mean())
    p_value = min(1.0, 2.0 * min(below, above))
    return PairedDifference(
        point=point_a - point_b,
        interval=Interval(
            point=float(finite.mean()),
            lower=interval.lower,
            upper=interval.upper,
            resamples=int(finite.size),
            method="percentile",
        ),
        p_value=float(p_value),
        same_side=max(below, above),
    )


def paired_bootstrap_contrast(
    risk_a: NDArray[np.float64],
    risk_b: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Point estimates of both arms alongside the contrast, for the report."""
    difference = paired_bootstrap_difference(risk_a, risk_b, time, event, resamples, level, seed)
    payload = difference.as_dict()
    payload["arm_a"] = concordance_index(risk_a, time, event).c_index
    payload["arm_b"] = concordance_index(risk_b, time, event).c_index
    return payload


def bootstrap_over_initialisations(
    values: NDArray[np.float64],
) -> dict[str, float]:
    """Summarise a metric computed once per random initialisation."""
    if values.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "count": 0.0}
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "count": float(values.size),
    }
