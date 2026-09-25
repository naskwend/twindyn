"""Multiplicity correction.

Ref: Sec. 4.2 (the external family of three paired comparisons and the
label-efficiency family of four comparisons are each corrected with
Holm-Bonferroni), Table 1 and Table 6 (marked significance uses the corrected
p-value).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CorrectionResult:
    labels: tuple[str, ...]
    raw: NDArray[np.float64]
    adjusted: NDArray[np.float64]
    rejected: NDArray[np.bool_]
    method: str
    alpha: float

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "entries": [
                {
                    "label": self.labels[position],
                    "p_uncorrected": float(self.raw[position]),
                    "p_corrected": float(self.adjusted[position]),
                    "rejected": bool(self.rejected[position]),
                }
                for position in range(len(self.labels))
            ],
        }


def holm_bonferroni(
    p_values: NDArray[np.float64], labels: tuple[str, ...], alpha: float = 0.05
) -> CorrectionResult:
    """Step-down Holm-Bonferroni adjustment of a family of p-values."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    if labels and len(labels) != values.shape[0]:
        raise ValueError("labels must match the p-value count")
    if not labels:
        labels = tuple(f"comparison_{index}" for index in range(values.shape[0]))
    order = np.argsort(values)
    count = values.shape[0]
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = (count - rank) * values[position]
        running = max(running, candidate)
        adjusted[position] = min(1.0, running)
    rejected = adjusted < alpha
    return CorrectionResult(
        labels=labels,
        raw=values,
        adjusted=adjusted,
        rejected=rejected,
        method="holm_bonferroni",
        alpha=alpha,
    )


def benjamini_hochberg(
    p_values: NDArray[np.float64], labels: tuple[str, ...], alpha: float = 0.05
) -> CorrectionResult:
    """Step-up false-discovery-rate control, reported alongside Holm."""
    values = np.asarray(p_values, dtype=np.float64)
    if not labels:
        labels = tuple(f"comparison_{index}" for index in range(values.shape[0]))
    order = np.argsort(values)
    count = values.shape[0]
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for rank in range(count - 1, -1, -1):
        position = order[rank]
        candidate = values[position] * count / (rank + 1)
        running = min(running, candidate)
        adjusted[position] = min(1.0, running)
    return CorrectionResult(
        labels=labels,
        raw=values,
        adjusted=adjusted,
        rejected=adjusted < alpha,
        method="benjamini_hochberg",
        alpha=alpha,
    )


def significance_marker(p_value: float, alpha: float = 0.05) -> str:
    """The marker convention used in the tables: two stars below 0.01, one below 0.05."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "**"
    if p_value < alpha:
        return "*"
    return ""


def corrected_family(
    comparisons: dict[str, float],
    alpha: float = 0.05,
) -> CorrectionResult:
    labels = tuple(comparisons.keys())
    values = np.asarray([comparisons[label] for label in labels], dtype=np.float64)
    return holm_bonferroni(values, labels, alpha=alpha)
