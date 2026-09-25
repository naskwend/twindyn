"""Shared fixtures for the unit and integration suites.

Ref: Sec. 3.1-3.2 (the observation and the ordered axis), Sec. 4.1 (the four
resources), Sec. 4.4 (the split protocol).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from twindyn.data.assembly import CohortBuild, build_synthetic_cohorts
from twindyn.data.axis import MicroenvironmentAxis
from twindyn.data.compartments import compartment_count
from twindyn.data.dataset import AxisDataset, FeatureStandardiser
from twindyn.data.splits import internal_split
from twindyn.models.twindyn import TwinDyn, TwinDynConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PER_RESOURCE = 12
MORPHOLOGY_FEATURES = 32
STATE_DIM = 16
TAU_MAX = 16


@pytest.fixture(scope="session")
def morphology_features() -> int:
    return MORPHOLOGY_FEATURES


@pytest.fixture(scope="session")
def cohort() -> CohortBuild:
    return build_synthetic_cohorts(
        tau_max=TAU_MAX,
        samples_per_resource=SAMPLES_PER_RESOURCE,
        seed=0,
        state_dim=STATE_DIM,
        resource_shift=1.0,
        morphology_features=MORPHOLOGY_FEATURES,
    )


@pytest.fixture(scope="session")
def axis(cohort: CohortBuild) -> MicroenvironmentAxis:
    return cohort.axis


@pytest.fixture(scope="session")
def standardiser(cohort: CohortBuild) -> FeatureStandardiser:
    observations = cohort.table.observations()
    availability = cohort.table.availability_matrix()
    return FeatureStandardiser.fit(observations, availability)


@pytest.fixture(scope="session")
def dataset(cohort: CohortBuild, standardiser: FeatureStandardiser) -> AxisDataset:
    return AxisDataset(cohort.table, cohort.axis, tuple(range(len(cohort.table))), standardiser)


@pytest.fixture(scope="session")
def split(cohort: CohortBuild) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return internal_split(cohort.table.samples, "tcga_kirc", 0.25, 0)


@pytest.fixture
def model(axis: MicroenvironmentAxis) -> TwinDyn:
    torch.manual_seed(0)
    return TwinDyn(TwinDynConfig(state_dim=STATE_DIM), axis.index())


@pytest.fixture(scope="session")
def compartment_total() -> int:
    return compartment_count()
