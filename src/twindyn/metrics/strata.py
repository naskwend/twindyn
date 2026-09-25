"""Stratum-level and modality-dropout reporting.

Ref: Table 3 (stage and grade strata pooled across cohorts with a per-cohort
breakdown, and modality-dropout rows that withhold one input family at test time
with the strongest supervised multimodal baseline shown for reference), Sec. 4.7
(grade behaves monotonically and the stage profile is non-monotone with a trough
at stage II), Sec. 4.11 (the within-stratum gain measured against a
stage-and-grade model is a different quantity from the within-segment contrast
and the two do not share an ordering).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twindyn.metrics.concordance import concordance_index

STAGE_LABELS: tuple[str, ...] = (
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_4",
)
STAGE_DEFINITIONS: dict[str, str] = {
    "stage_1": "localised",
    "stage_2": "intermediate",
    "stage_3": "locally advanced",
    "stage_4": "advanced",
}
GRADE_LABELS: tuple[str, ...] = ("grade_1", "grade_2", "grade_3", "grade_4")
GRADE_DEFINITIONS: dict[str, str] = {
    "grade_1": "lowest",
    "grade_2": "low intermediate",
    "grade_3": "high intermediate",
    "grade_4": "highest",
}


@dataclass(frozen=True)
class StratumRow:
    stratum: str
    definition: str
    c_index: float
    delta_vs_full: float
    samples: int
    comparable_pairs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "stratum": self.stratum,
            "definition": self.definition,
            "c_index": self.c_index,
            "delta_vs_full": self.delta_vs_full,
            "samples": self.samples,
            "comparable_pairs": self.comparable_pairs,
        }


def stratum_rows(
    risk: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    stratum: NDArray[np.int64],
    labels: tuple[str, ...],
    definitions: dict[str, str],
) -> list[StratumRow]:
    full = concordance_index(risk, time, event).c_index
    rows: list[StratumRow] = []
    for position, label in enumerate(labels):
        selector = stratum == (position + 1)
        if int(selector.sum()) < 2:
            continue
        result = concordance_index(risk[selector], time[selector], event[selector])
        rows.append(
            StratumRow(
                stratum=label,
                definition=definitions.get(label, ""),
                c_index=result.c_index,
                delta_vs_full=result.c_index - full,
                samples=int(selector.sum()),
                comparable_pairs=result.comparable_pairs,
            )
        )
    return rows


def full_row(
    risk: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
) -> StratumRow:
    result = concordance_index(risk, time, event)
    return StratumRow(
        stratum="full",
        definition="all external samples",
        c_index=result.c_index,
        delta_vs_full=0.0,
        samples=int(risk.shape[0]),
        comparable_pairs=result.comparable_pairs,
    )


def monotonicity(values: dict[str, float]) -> dict[str, object]:
    """Whether a stratum profile rises with the ordered stratum label."""
    ordered = sorted(values)
    series = [values[key] for key in ordered]
    differences = [series[index + 1] - series[index] for index in range(len(series) - 1)]
    increasing = all(value > 0 for value in differences)
    decreasing = all(value < 0 for value in differences)
    trough = min(ordered, key=lambda key: values[key]) if ordered else None
    peak = max(ordered, key=lambda key: values[key]) if ordered else None
    return {
        "ordered_labels": ordered,
        "series": series,
        "differences": differences,
        "monotone_increasing": increasing,
        "monotone_decreasing": decreasing,
        "trough": trough,
        "peak": peak,
        "interior_trough": bool(
            trough is not None and ordered.index(trough) not in (0, len(ordered) - 1)
        ),
    }


def dropout_rows(
    full_risk: NDArray[np.float64],
    dropout_risk: dict[str, NDArray[np.float64]],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    reference_c_index: float | None = None,
) -> list[StratumRow]:
    """The modality-dropout rows, measured against the stated reference value."""
    reference = (
        reference_c_index
        if reference_c_index is not None
        else concordance_index(full_risk, time, event).c_index
    )
    rows: list[StratumRow] = []
    for label, risk in dropout_risk.items():
        result = concordance_index(risk, time, event)
        rows.append(
            StratumRow(
                stratum=f"-{label} at test time",
                definition="dropout",
                c_index=result.c_index,
                delta_vs_full=result.c_index - reference,
                samples=int(risk.shape[0]),
                comparable_pairs=result.comparable_pairs,
            )
        )
    return rows


def within_stratum_gain(
    full_risk: NDArray[np.float64],
    covariate_risk: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    stratum: NDArray[np.int64],
    labels: tuple[str, ...],
) -> dict[str, float]:
    """Contrast against a stage-and-grade model inside each stratum."""
    out: dict[str, float] = {}
    for position, label in enumerate(labels):
        selector = stratum == (position + 1)
        if int(selector.sum()) < 2:
            continue
        full = concordance_index(full_risk[selector], time[selector], event[selector]).c_index
        covariate = concordance_index(
            covariate_risk[selector], time[selector], event[selector]
        ).c_index
        out[label] = full - covariate
    return out
