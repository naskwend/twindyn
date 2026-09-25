"""Resource-disjoint external validation.

Ref: Sec. 4.1 (the design keeps the resources apart: representation learning sees
the training resource and the alternate resource for the alignment task, the
read-out is fitted on the training part of the training resource, and each
external number comes from scoring a cohort once with a frozen model), Sec. 4.2
(the primary criterion is a paired external concordance difference of at least
0.03 against the strongest static composition baseline, per resource, with
Holm-Bonferroni correction across the three cohort comparisons), Table 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from twindyn.data.resources import EXTERNAL_RESOURCES
from twindyn.metrics.bootstrap import (
    Interval,
    bootstrap_concordance,
    paired_bootstrap_difference,
)
from twindyn.metrics.concordance import concordance_index
from twindyn.metrics.correction import CorrectionResult, holm_bonferroni
from twindyn.metrics.decision import primary_criterion
from twindyn.metrics.transfer import TransferTax, tax_reduction
from twindyn.utils.config import EvaluationConfig


@dataclass
class CohortResult:
    resource: str
    cohort: str
    samples: int
    method_c_index: float
    method_interval: Interval
    comparator_c_index: float
    comparator_interval: Interval
    difference: float
    difference_interval: Interval
    p_uncorrected: float
    p_corrected: float
    loss_vs_train: float

    def as_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "cohort": self.cohort,
            "samples": self.samples,
            "twindyn": self.method_c_index,
            "twindyn_interval": [self.method_interval.lower, self.method_interval.upper],
            "best_static": self.comparator_c_index,
            "best_static_interval": [
                self.comparator_interval.lower,
                self.comparator_interval.upper,
            ],
            "difference": self.difference,
            "difference_interval": [self.difference_interval.lower, self.difference_interval.upper],
            "p_uncorrected": self.p_uncorrected,
            "p_corrected": self.p_corrected,
            "loss_vs_train": self.loss_vs_train,
        }


@dataclass
class ExternalReport:
    cohorts: list[CohortResult] = field(default_factory=list)
    correction: dict[str, object] = field(default_factory=dict)
    criterion: dict[str, object] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    transfer: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "cohorts": [cohort.as_dict() for cohort in self.cohorts],
            "correction": dict(self.correction),
            "criterion": dict(self.criterion),
            "mean": dict(self.mean),
            "transfer": dict(self.transfer),
        }


def evaluate_external(
    method_risk: dict[str, NDArray[np.float64]],
    comparator_risk: dict[str, NDArray[np.float64]],
    times: dict[str, NDArray[np.float64]],
    events: dict[str, NDArray[np.float64]],
    cohort_labels: dict[str, str],
    training_value: float,
    comparator_training_value: float,
    config: EvaluationConfig,
) -> ExternalReport:
    """Score every external cohort once and correct the family of comparisons."""
    draft: list[dict[str, object]] = []
    for resource in EXTERNAL_RESOURCES:
        if resource not in method_risk:
            continue
        method = method_risk[resource]
        comparator = comparator_risk[resource]
        time = times[resource]
        event = events[resource]
        method_interval = bootstrap_concordance(
            method,
            time,
            event,
            config.bootstrap_resamples,
            config.confidence_level,
            config.random_state,
        )
        comparator_interval = bootstrap_concordance(
            comparator,
            time,
            event,
            config.bootstrap_resamples,
            config.confidence_level,
            config.random_state,
        )
        contrast = paired_bootstrap_difference(
            method,
            comparator,
            time,
            event,
            config.bootstrap_resamples,
            config.confidence_level,
            config.random_state,
        )
        draft.append(
            {
                "resource": resource,
                "cohort": cohort_labels.get(resource, resource),
                "samples": int(method.shape[0]),
                "method_interval": method_interval,
                "comparator_interval": comparator_interval,
                "contrast": contrast,
                "loss": training_value - method_interval.point,
            }
        )
    raw_p = {str(entry["resource"]): float(entry["contrast"].p_value) for entry in draft}
    correction = _correct(raw_p)
    order = {label: position for position, label in enumerate(correction.labels)}
    cohorts: list[CohortResult] = []
    for entry in draft:
        resource = str(entry["resource"])
        contrast = entry["contrast"]
        cohorts.append(
            CohortResult(
                resource=resource,
                cohort=str(entry["cohort"]),
                samples=int(entry["samples"]),
                method_c_index=entry["method_interval"].point,
                method_interval=entry["method_interval"],  # type: ignore[arg-type]
                comparator_c_index=entry["comparator_interval"].point,
                comparator_interval=entry["comparator_interval"],  # type: ignore[arg-type]
                difference=contrast.point,
                difference_interval=contrast.interval,
                p_uncorrected=contrast.p_value,
                p_corrected=float(correction.adjusted[order[resource]]),
                loss_vs_train=float(entry["loss"]),
            )
        )
    method_values = {cohort.resource: cohort.method_c_index for cohort in cohorts}
    comparator_values = {cohort.resource: cohort.comparator_c_index for cohort in cohorts}
    differences = {cohort.resource: cohort.difference for cohort in cohorts}
    mean = {
        "twindyn": float(np.mean(list(method_values.values()))) if method_values else float("nan"),
        "best_static": float(np.mean(list(comparator_values.values())))
        if comparator_values
        else float("nan"),
        "difference": float(np.mean(list(differences.values()))) if differences else float("nan"),
        "loss_vs_train": float(np.mean([cohort.loss_vs_train for cohort in cohorts]))
        if cohorts
        else float("nan"),
    }
    method_tax = TransferTax("TwinDyn", training_value, method_values)
    comparator_tax = TransferTax(
        "strongest static composition", comparator_training_value, comparator_values
    )
    return ExternalReport(
        cohorts=cohorts,
        correction=correction.as_dict(),
        criterion=primary_criterion(differences, 0.03),
        mean=mean,
        transfer={
            "method": method_tax.as_dict(),
            "comparator": comparator_tax.as_dict(),
            "reduction": tax_reduction(comparator_tax, method_tax),
        },
    )


def _correct(raw: dict[str, float]) -> CorrectionResult:
    labels = tuple(raw.keys())
    values = np.asarray([raw[label] for label in labels], dtype=np.float64)
    return holm_bonferroni(values, labels)


def single_cohort_concordance(
    risk: NDArray[np.float64], time: NDArray[np.float64], event: NDArray[np.float64]
) -> float:
    return concordance_index(risk, time, event).c_index
