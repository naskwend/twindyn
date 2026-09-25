"""Cohort assembly, the ordered axis and the resource readers.

Ref: Sec. 3.1-3.2, Sec. 3.10, Sec. 4.1.
"""

from twindyn.data.assembly import (
    CohortBuild,
    CohortProvenance,
    assert_shared_space_no_imputation,
    build_staged_cohorts,
    build_synthetic_cohorts,
)
from twindyn.data.axis import (
    MicroenvironmentAxis,
    PositionSpec,
    collapse_to_sample_level,
    contiguous_partition,
)
from twindyn.data.compartments import (
    COMPARTMENT_ORDER,
    RAW_COMPARTMENTS,
    CompartmentFamily,
    aggregate_compartment_groups,
    aggregation_levels,
    compartment_count,
    compartment_family,
    compartment_index,
    normalise_fractions,
)
from twindyn.data.dataset import (
    RESOURCE_INDEX,
    AxisDataset,
    FeatureStandardiser,
    collate,
    materialise,
)
from twindyn.data.feature_space import (
    FeatureInventory,
    SharedFeatureSpace,
    intersect_inventories,
    project_observation,
)
from twindyn.data.masking import (
    MaskingOutcome,
    position_availability,
    reconstruction_error,
    sample_mask,
)
from twindyn.data.pairs import (
    PairSet,
    alignment_magnitude,
    build_alignment_pairs,
    mutual_nearest_pairs,
    pair_resource_tags,
    pairs_to_index,
)
from twindyn.data.resources import (
    EXTERNAL_RESOURCES,
    RESOURCES,
    TRAINING_RESOURCE,
    AssayClass,
    ResourceRole,
    ResourceSpec,
    external_specs,
    resource_spec,
    training_spec,
)
from twindyn.data.schema import ClinicalStratum, CohortTable, SampleRecord, SurvivalLabel
from twindyn.data.splits import (
    SplitIndices,
    assert_zero_overlap,
    internal_split,
    label_fraction_subset,
    modality_dropout,
    resource_partition,
    stratum_indices,
)

__all__ = [
    "COMPARTMENT_ORDER",
    "EXTERNAL_RESOURCES",
    "RAW_COMPARTMENTS",
    "RESOURCES",
    "RESOURCE_INDEX",
    "TRAINING_RESOURCE",
    "AssayClass",
    "AxisDataset",
    "ClinicalStratum",
    "CohortBuild",
    "CohortProvenance",
    "CohortTable",
    "CompartmentFamily",
    "FeatureInventory",
    "FeatureStandardiser",
    "MaskingOutcome",
    "MicroenvironmentAxis",
    "PairSet",
    "PositionSpec",
    "ResourceRole",
    "ResourceSpec",
    "SampleRecord",
    "SharedFeatureSpace",
    "SplitIndices",
    "SurvivalLabel",
    "aggregate_compartment_groups",
    "aggregation_levels",
    "alignment_magnitude",
    "assert_shared_space_no_imputation",
    "assert_zero_overlap",
    "build_alignment_pairs",
    "build_staged_cohorts",
    "build_synthetic_cohorts",
    "collapse_to_sample_level",
    "collate",
    "compartment_count",
    "compartment_family",
    "compartment_index",
    "contiguous_partition",
    "external_specs",
    "internal_split",
    "intersect_inventories",
    "label_fraction_subset",
    "materialise",
    "modality_dropout",
    "mutual_nearest_pairs",
    "normalise_fractions",
    "pair_resource_tags",
    "pairs_to_index",
    "position_availability",
    "project_observation",
    "reconstruction_error",
    "resource_partition",
    "resource_spec",
    "sample_mask",
    "stratum_indices",
    "training_spec",
]
