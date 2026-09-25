"""The prespecified success criteria and the decision rule.

Ref: Sec. 4.2 (the primary criterion is a paired external concordance difference
of at least 0.03 against the strongest static composition baseline, taken per
independent resource; the secondary criterion on the constraint axis requires the
margin over that comparator to exceed the decision margin at ten percent of the
labels; neither criterion was changed during the analysis), Sec. 4.4 (the
predetermined margin is 0.005 concordance units, and any result below it counts as
a lack of improvement), Table 1 (the margin over the strongest static baseline is
0.003, inside the prespecified margin, so the result is parity rather than an
advantage), Table 2 (max drop / min drop = 0.041 / 0.003 = 13.7x; the equal-budget
row retains 92.7% of the margin), Table 5 and Sec. 4.6 (the designed null falls
below the prespecified margin and is reported as an ablation with no effect).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Verdict(str, Enum):
    ADVANTAGE = "advantage"
    PARITY = "parity"
    NO_EFFECT = "no_effect"
    DEFICIT = "deficit"


@dataclass(frozen=True)
class Decision:
    difference: float
    margin: float
    threshold: float
    verdict: Verdict
    clears_thesis_threshold: bool
    inside_decision_margin: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "difference": self.difference,
            "margin": self.margin,
            "threshold": self.threshold,
            "verdict": self.verdict.value,
            "clears_thesis_threshold": self.clears_thesis_threshold,
            "inside_decision_margin": self.inside_decision_margin,
        }


def decide(
    difference: float,
    margin: float = 0.005,
    threshold: float = 0.03,
) -> Decision:
    """Classify a paired difference against the margin and the thesis threshold."""
    if not np.isfinite(difference):
        raise ValueError("difference must be finite")
    if difference < 0.0:
        verdict = Verdict.DEFICIT
    elif difference < margin:
        verdict = Verdict.PARITY if difference > 0.0 else Verdict.NO_EFFECT
    else:
        verdict = Verdict.ADVANTAGE
    return Decision(
        difference=difference,
        margin=margin,
        threshold=threshold,
        verdict=verdict,
        clears_thesis_threshold=difference >= threshold,
        inside_decision_margin=difference < margin,
    )


def primary_criterion(
    cohort_differences: dict[str, float], threshold: float = 0.03
) -> dict[str, object]:
    """The primary criterion: every external cohort must clear the threshold."""
    per_cohort = {
        key: decide(value, threshold=threshold).as_dict()
        for key, value in cohort_differences.items()
    }
    return {
        "threshold": threshold,
        "per_cohort": per_cohort,
        "all_clear": all(entry["clears_thesis_threshold"] for entry in per_cohort.values()),
        "minimum_difference": min(cohort_differences.values())
        if cohort_differences
        else float("nan"),
        "cohorts": len(cohort_differences),
    }


def secondary_criterion(
    label_efficiency_margins: dict[float, float],
    margin: float = 0.005,
) -> dict[str, object]:
    """The secondary criterion: the margin at ten percent of the labels exceeds the margin."""
    at_ten = label_efficiency_margins.get(0.10, float("nan"))
    return {
        "fraction": 0.10,
        "difference": at_ten,
        "margin": margin,
        "met": bool(np.isfinite(at_ten) and at_ten > margin),
    }


def margin_retention(full_drop: float, variant_drop: float) -> dict[str, float]:
    """Share of the full model's margin a variant retains (Table 2, equal-budget row)."""
    if full_drop == 0.0:
        return {"retention": float("nan"), "full_drop": full_drop, "variant_drop": variant_drop}
    return {
        "retention": 1.0 - variant_drop / full_drop,
        "full_drop": full_drop,
        "variant_drop": variant_drop,
    }


def drop_ratio(drops: dict[str, float]) -> dict[str, object]:
    """Largest and smallest drop with their ratio (Table 2, caption of the ablation)."""
    values: dict[str, float] = {
        key: float(value) for key, value in drops.items() if np.isfinite(value)
    }
    if not values:
        return {"max": float("nan"), "min": float("nan"), "ratio": float("nan")}
    largest = max(values.values())
    smallest = min(values.values())
    return {
        "max": largest,
        "min": smallest,
        "ratio": largest / smallest if smallest else float("nan"),
        "largest_variant": max(values, key=lambda name: values[name]),
        "smallest_variant": min(values, key=lambda name: values[name]),
    }


def descriptive_null(ablation_difference: float, margin: float = 0.005) -> dict[str, object]:
    """An ablation whose effect falls below the prespecified margin is a designed null."""
    decision = decide(ablation_difference, margin=margin)
    return {
        "difference": ablation_difference,
        "below_margin": abs(ablation_difference) < margin,
        "verdict": decision.verdict.value,
        "interpretation": "ablation with no effect",
    }


def hypothesis_families() -> dict[str, dict[str, object]]:
    """The declared families and whether each is corrected or descriptive."""
    return {
        "external": {
            "comparisons": 3,
            "correction": "holm_bonferroni",
            "descriptive": False,
        },
        "label_efficiency": {
            "comparisons": 4,
            "correction": "holm_bonferroni",
            "descriptive": False,
        },
        "level": {
            "comparisons": 9,
            "correction": None,
            "descriptive": True,
        },
        "loss": {
            "comparisons": 8,
            "correction": None,
            "descriptive": True,
        },
    }
