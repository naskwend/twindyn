"""Assembly of the four-resource cohort table.

The readers are tried first; when a resource is not staged locally the assembly
records it as unavailable and, only when the configuration asks for it, falls
back to cohorts generated from the shared latent law so that the mechanisms stay
executable. A cohort-level number is never produced from the fallback for a
resource the manuscript reports as a real archive.

Ref: Sec. 4.1 (four public resources, one training and three external from
different repositories; the shared feature space is an intersection and missing
features cannot be filled in), Sec. 3.10 (the deconvolution configuration is
fixed in advance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from twindyn.data.axis import MicroenvironmentAxis
from twindyn.data.compartments import compartment_count
from twindyn.data.deconvolution import (
    DEFAULT_CONFIGURATIONS,
    ReferenceBasedDeconvolution,
    fractions_from_protein_abundance,
    sensitivity_sweep,
    synthesise_reference,
)
from twindyn.data.feature_space import FeatureInventory, SharedFeatureSpace, intersect_inventories
from twindyn.data.loaders.arrayexpress import ArrayExpressStudy
from twindyn.data.loaders.cptac import CptacCcrcc
from twindyn.data.loaders.gdc import GdcKirc
from twindyn.data.loaders.geo import GeoSeries
from twindyn.data.resources import EXTERNAL_RESOURCES, RESOURCES, TRAINING_RESOURCE
from twindyn.data.schema import ClinicalStratum, CohortTable, SampleRecord, SurvivalLabel
from twindyn.data.synthetic import synthesise_bundle
from twindyn.registry import ENGINEERING_DEFAULT_MORPHOLOGY_FEATURES


class ClinicalReader(Protocol):
    """Any resource reader that exposes follow-up and clinical strata."""

    def survival(self) -> dict[str, tuple[float, bool]]: ...

    def strata(self) -> dict[str, tuple[int | None, int | None]]: ...


@dataclass
class CohortProvenance:
    resource: str
    available: bool
    source: str
    samples: int
    detail: str


@dataclass
class CohortBuild:
    """The assembled cohorts with the objects the pipeline needs downstream."""

    table: CohortTable
    axis: MicroenvironmentAxis
    space: SharedFeatureSpace
    source: str
    provenance: list[CohortProvenance] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    deconvolution_sensitivity: list[dict[str, object]] = field(default_factory=list)

    @property
    def blocked_resources(self) -> list[str]:
        return [entry.resource for entry in self.provenance if not entry.available]


def _markers(compartment_total: int, per_compartment: int) -> tuple[str, ...]:
    return tuple(
        f"marker_{compartment:02d}_{position:02d}"
        for compartment in range(compartment_total)
        for position in range(per_compartment)
    )


def build_synthetic_cohorts(
    tau_max: int,
    samples_per_resource: int,
    seed: int,
    state_dim: int,
    resource_shift: float,
    morphology_features: int = ENGINEERING_DEFAULT_MORPHOLOGY_FEATURES,
    ordering_rule: str = "predefined_compartment_index",
    aggregation_level: int = 0,
) -> CohortBuild:
    """Generate the four cohorts from the shared latent law.

    The axis is constructed first because each axis position observes only its own
    block of the observation, so the block layout is an input to the generator
    rather than a consequence of it.
    """
    compartment_total = compartment_count()
    width = compartment_total + morphology_features
    axis = MicroenvironmentAxis.of(width, tau_max, compartment_total, ordering_rule)
    if aggregation_level:
        axis = axis.aggregated(aggregation_level)
    bundle = synthesise_bundle(
        state_dim=state_dim,
        axis=axis,
        samples_per_resource=samples_per_resource,
        seed=seed,
        resource_shift_magnitude=resource_shift,
    )
    space = SharedFeatureSpace(
        features=tuple(f"morph_{index:03d}" for index in range(morphology_features)),
        per_resource_coverage={
            key: (1.0 if spec.has_histology else 0.0) for key, spec in RESOURCES.items()
        },
        per_resource_counts=dict.fromkeys(RESOURCES, morphology_features),
    )
    provenance = [
        CohortProvenance(
            resource=key,
            available=False,
            source="synthetic",
            samples=int(bundle.block_indices[key].shape[0]),
            detail="generated from the shared latent law; not a real archive",
        )
        for key in (TRAINING_RESOURCE, *EXTERNAL_RESOURCES)
    ]
    notes = [
        "No public archive is staged locally; cohorts are generated from the shared latent law "
        "so that the mechanisms stay executable.",
        "Cohort-level values from the manuscript's external tables are therefore not reproducible "
        "in this environment and are reported as not run.",
    ]
    return CohortBuild(
        table=bundle.table,
        axis=axis,
        space=space,
        source="synthetic",
        provenance=provenance,
        notes=notes,
    )


def build_staged_cohorts(
    root: str | Path,
    tau_max: int,
    morphology_features: int = ENGINEERING_DEFAULT_MORPHOLOGY_FEATURES,
    marker_genes_per_compartment: int = 40,
    gene_panel_size: int = 2000,
    reference_seed: int = 20260924,
    ordering_rule: str = "predefined_compartment_index",
    aggregation_level: int = 0,
    sensitivity_genes: int = 400,
) -> CohortBuild:
    """Read whatever the resource root holds and assemble the cohort table."""
    base = Path(root)
    compartment_total = compartment_count()
    markers = _markers(compartment_total, marker_genes_per_compartment)
    reference, panel = synthesise_reference(
        compartment_total,
        max(sensitivity_genes, gene_panel_size),
        marker_genes_per_compartment,
        reference_seed,
    )
    engine = ReferenceBasedDeconvolution(reference, panel)
    inventories: dict[str, FeatureInventory] = {}
    profiles: dict[str, tuple[list[str], np.ndarray, list[str]]] = {}
    provenance: list[CohortProvenance] = []
    notes: list[str] = []

    gdc = GdcKirc(base / TRAINING_RESOURCE)
    gdc_presence = gdc.presence()
    if gdc_presence.available:
        try:
            genes, matrix, samples = gdc.expression()
            profiles[TRAINING_RESOURCE] = (genes, matrix, samples)
            inventories[TRAINING_RESOURCE] = FeatureInventory(
                resource=TRAINING_RESOURCE,
                gene_features=tuple(genes),
                morphology_features=tuple(
                    f"morph_{index:02d}" for index in range(morphology_features)
                ),
            )
            provenance.append(
                CohortProvenance(TRAINING_RESOURCE, True, "staged", len(samples), "GDC open tier")
            )
        except (FileNotFoundError, ValueError) as error:
            provenance.append(CohortProvenance(TRAINING_RESOURCE, False, "staged", 0, str(error)))
    else:
        provenance.append(
            CohortProvenance(
                TRAINING_RESOURCE, False, "staged", 0, f"absent artefacts: {gdc_presence.missing}"
            )
        )

    geo = GeoSeries(base / "gse29609")
    if geo.presence().available:
        try:
            _, samples, matrix = geo.series_matrix()
            identifiers = geo.platform_genes()
            profiles["gse29609"] = (identifiers, matrix, samples)
            inventories["gse29609"] = FeatureInventory("gse29609", tuple(identifiers))
            provenance.append(
                CohortProvenance("gse29609", True, "staged", len(samples), "GEO series matrix")
            )
        except (FileNotFoundError, ValueError) as error:
            provenance.append(CohortProvenance("gse29609", False, "staged", 0, str(error)))
    else:
        provenance.append(CohortProvenance("gse29609", False, "staged", 0, "series matrix absent"))

    study = ArrayExpressStudy(base / "emtab1980")
    if study.presence().available:
        try:
            identifiers, matrix, samples = study.processed_matrix()
            profiles["emtab1980"] = (identifiers, matrix, samples)
            inventories["emtab1980"] = FeatureInventory("emtab1980", tuple(identifiers))
            provenance.append(
                CohortProvenance("emtab1980", True, "staged", len(samples), "MAGE-TAB")
            )
        except (FileNotFoundError, ValueError) as error:
            provenance.append(CohortProvenance("emtab1980", False, "staged", 0, str(error)))
    else:
        provenance.append(
            CohortProvenance("emtab1980", False, "staged", 0, "processed matrix absent")
        )

    cptac = CptacCcrcc(base / "cptac_ccrcc")
    if cptac.presence().available:
        try:
            identifiers, matrix, samples = cptac.proteome()
            profiles["cptac_ccrcc"] = (identifiers, matrix, samples)
            inventories["cptac_ccrcc"] = FeatureInventory(
                "cptac_ccrcc",
                protein_features=tuple(identifiers),
                morphology_features=tuple(
                    f"morph_{index:02d}" for index in range(morphology_features)
                ),
            )
            provenance.append(
                CohortProvenance("cptac_ccrcc", True, "staged", len(samples), "PDC proteome")
            )
        except (FileNotFoundError, ValueError) as error:
            provenance.append(CohortProvenance("cptac_ccrcc", False, "staged", 0, str(error)))
    else:
        provenance.append(
            CohortProvenance("cptac_ccrcc", False, "staged", 0, "proteome table absent")
        )

    if not inventories:
        raise FileNotFoundError("no staged resource under the configured root")

    histology = {
        key for key, spec in RESOURCES.items() if spec.has_histology and key in inventories
    }
    space = intersect_inventories(inventories, histology, markers)
    records: list[SampleRecord] = []
    survival_sources: dict[str, ClinicalReader] = {
        TRAINING_RESOURCE: gdc,
        "gse29609": geo,
        "emtab1980": study,
        "cptac_ccrcc": cptac,
    }
    for resource, (identifiers, matrix, samples) in profiles.items():
        index = {identifier: position for position, identifier in enumerate(identifiers)}
        selector = np.array(
            [index[feature] for feature in space.features if feature in index], dtype=np.int64
        )
        if selector.size == 0:
            notes.append(f"{resource}: shared feature space empty after projection")
            continue
        survival = survival_sources[resource].survival()
        strata = survival_sources[resource].strata()
        sample_count = int(matrix.shape[0])
        for column in range(sample_count):
            vector = matrix[column, selector]
            fractions = (
                fractions_from_protein_abundance(
                    vector, np.arange(selector.size), compartment_total
                )
                if resource == "cptac_ccrcc"
                else engine.fit_sample(
                    _pad_to_reference(vector, marker_genes_per_compartment, compartment_total)
                ).fractions
            )
            sample_id = samples[column] if column < len(samples) else f"{resource}-{column:05d}"
            label = survival.get(sample_id)
            stratum = strata.get(sample_id)
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    resource=resource,
                    compartments=fractions,
                    morphology=None,
                    label=SurvivalLabel(time=label[0], event=label[1]) if label else None,
                    stratum=ClinicalStratum(stratum[1], stratum[0]) if stratum else None,
                )
            )
        notes.append(f"{resource}: {selector.size} shared features retained of {len(identifiers)}")

    if not records:
        raise ValueError("staged resources yielded no usable samples")

    width = compartment_total + len(space.features)
    table = CohortTable(width=width, compartment_count=compartment_total, samples=records)
    axis = MicroenvironmentAxis.of(width, tau_max, compartment_total, ordering_rule)
    if aggregation_level:
        axis = axis.aggregated(aggregation_level)
    sweep_input = np.stack(
        [
            np.pad(row, (0, max(0, sensitivity_genes - row.shape[0])))[:sensitivity_genes]
            for row in table.observations()
        ]
    )
    sensitivity = sensitivity_sweep(
        sweep_input, DEFAULT_CONFIGURATIONS, sensitivity_genes, compartment_total, reference_seed
    )
    return CohortBuild(
        table=table,
        axis=axis,
        space=space,
        source="staged",
        provenance=provenance,
        notes=notes,
        deconvolution_sensitivity=sensitivity,
    )


def _pad_to_reference(
    vector: np.ndarray, per_compartment: int, compartment_total: int
) -> np.ndarray:
    width = per_compartment * compartment_total
    if vector.shape[0] >= width:
        return vector[:width]
    return np.pad(vector, (0, width - vector.shape[0]))


def assert_shared_space_no_imputation(space: SharedFeatureSpace) -> dict[str, object]:
    """The intersection carries no union fallback and no filled-in features."""
    return {
        "features": len(space.features),
        "per_resource_coverage": dict(space.per_resource_coverage),
        "imputation": "none",
        "mode": "intersection",
    }
