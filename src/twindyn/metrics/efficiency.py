"""Label-efficiency reporting.

Ref: Sec. 4.10 (at ten percent of the labels the margin over the strongest static
baseline is 0.076, at twenty-five percent it is 0.038 and at fifty percent it is
0.012 before parity at one hundred percent), Sec. 4.2 (the label-efficiency family
holds four comparisons at different label percentages and carries its own
correction), Sec. 4.4 (the only supervised variable is the read-out).

Supervised comparators cannot be fitted at zero labels, so the zero-label row
records the frozen representation's external value without a comparator arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twindyn.metrics.decision import decide


@dataclass(frozen=True)
class EfficiencyPoint:
    fraction: float
    method_value: float
    comparator_value: float | None
    difference: float | None
    verdict: str
    supervised_comparator_available: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "fraction": self.fraction,
            "method_value": self.method_value,
            "comparator_value": self.comparator_value,
            "difference": self.difference,
            "verdict": self.verdict,
            "supervised_comparator_available": self.supervised_comparator_available,
        }


def efficiency_curve(
    method_values: dict[float, float],
    comparator_values: dict[float, float] | None,
    margin: float = 0.005,
    threshold: float = 0.03,
) -> list[EfficiencyPoint]:
    points: list[EfficiencyPoint] = []
    for fraction in sorted(method_values):
        method_value = method_values[fraction]
        if comparator_values is None or fraction not in comparator_values:
            points.append(
                EfficiencyPoint(
                    fraction=fraction,
                    method_value=method_value,
                    comparator_value=None,
                    difference=None,
                    verdict="not_applicable",
                    supervised_comparator_available=False,
                )
            )
            continue
        comparator_value = comparator_values[fraction]
        difference = method_value - comparator_value
        points.append(
            EfficiencyPoint(
                fraction=fraction,
                method_value=method_value,
                comparator_value=comparator_value,
                difference=difference,
                verdict=decide(difference, margin=margin, threshold=threshold).verdict.value,
                supervised_comparator_available=True,
            )
        )
    return points


def margin_curve(points: list[EfficiencyPoint]) -> dict[str, float]:
    return {
        f"{point.fraction:.2f}": float(point.difference)
        for point in points
        if point.difference is not None
    }


def shrinkage_profile(points: list[EfficiencyPoint]) -> dict[str, object]:
    """Whether the margin shrinks as labels become plentiful."""
    finite = [point for point in points if point.difference is not None]
    fractions = [point.fraction for point in finite]
    differences = [float(point.difference) for point in finite if point.difference is not None]
    decreasing = all(
        differences[index + 1] <= differences[index] for index in range(len(differences) - 1)
    )
    return {
        "fractions": fractions,
        "differences": differences,
        "monotone_decreasing": decreasing,
        "span": (differences[0] - differences[-1]) if differences else float("nan"),
    }


def zero_label_transfer(
    frozen_method_value: float,
    supervised_available: bool,
    threshold: float = 0.03,
) -> dict[str, object]:
    """The no-label row: a frozen representation scored on an unseen resource."""
    return {
        "frozen_external_value": frozen_method_value,
        "supervised_comparator_available": supervised_available,
        "note": (
            "supervised comparators cannot be fitted without labels, so no paired "
            "difference is reported at zero labels"
        ),
        "exceeds_threshold_on_its_own": bool(
            np.isfinite(frozen_method_value) and frozen_method_value >= threshold
        ),
    }
