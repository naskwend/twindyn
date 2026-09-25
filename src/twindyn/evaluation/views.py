"""Feature views each comparator row is fitted on.

Ref: Table 1 (the rows differ in what they are given: clinical covariates alone,
the expression block, or the static composition representation), Sec. 4.1 (the
observation is the compartment block followed by the morphology block).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from twindyn.data.schema import CohortTable
from twindyn.metrics.strata import GRADE_LABELS, STAGE_LABELS


def clinical_view(table: CohortTable) -> NDArray[np.float64]:
    """One-hot ISUP grade and TNM stage, the clinical covariates of B1."""
    width = len(STAGE_LABELS) + len(GRADE_LABELS)
    matrix = np.zeros((len(table), width), dtype=np.float64)
    for position, sample in enumerate(table.samples):
        stratum = sample.stratum
        if stratum is None:
            continue
        if stratum.stage and 1 <= stratum.stage <= len(STAGE_LABELS):
            matrix[position, stratum.stage - 1] = 1.0
        if stratum.grade and 1 <= stratum.grade <= len(GRADE_LABELS):
            matrix[position, len(STAGE_LABELS) + stratum.grade - 1] = 1.0
    return matrix


def composition_view(table: CohortTable) -> NDArray[np.float64]:
    """The static composition vector: the compartment fractions of equation (1)."""
    return np.asarray(table.observations()[:, : table.compartment_count], dtype=np.float64)


def expression_view(table: CohortTable) -> NDArray[np.float64]:
    """The full observation, the molecular readout a resource can provide."""
    return np.asarray(table.observations(), dtype=np.float64)


VIEW_BUILDERS = {
    "clinical": clinical_view,
    "composition": composition_view,
    "expression": expression_view,
}

VIEW_WIDTHS = {
    "clinical": len(STAGE_LABELS) + len(GRADE_LABELS),
    "composition": None,
    "expression": None,
}


def feature_view(table: CohortTable, name: str) -> NDArray[np.float64]:
    if name not in VIEW_BUILDERS:
        raise ValueError(f"unknown feature view {name!r}")
    return VIEW_BUILDERS[name](table)
