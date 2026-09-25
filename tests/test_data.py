"""The data layer: schema, feature space, deconvolution, masking, pairs and splits.

Ref: Sec. 3.1 (equation 1), Sec. 3.3 (matching), Sec. 3.6 (masking densities),
Sec. 4.1 (intersection, splitting, zero overlap).
"""

from __future__ import annotations

import gzip

import numpy as np
import pytest
import torch

from twindyn.data.compartments import (
    COMPARTMENT_ORDER,
    aggregate_compartment_groups,
    compartment_count,
    normalise_fractions,
)
from twindyn.data.deconvolution import (
    DEFAULT_CONFIGURATIONS,
    ReferenceBasedDeconvolution,
    fractions_from_protein_abundance,
    sensitivity_sweep,
    synthesise_reference,
)
from twindyn.data.feature_space import (
    FeatureInventory,
    intersect_inventories,
    project_observation,
)
from twindyn.data.loaders.arrayexpress import ArrayExpressStudy
from twindyn.data.loaders.cptac import CptacCcrcc
from twindyn.data.loaders.gdc import GdcKirc
from twindyn.data.loaders.geo import GeoSeries
from twindyn.data.loaders.tabular import parse_event, parse_grade, parse_stage, parse_survival_value
from twindyn.data.loaders.tcia import TCIA_COLLECTIONS, CancerImagingArchive
from twindyn.data.masking import position_availability, reconstruction_error, sample_mask
from twindyn.data.pairs import (
    alignment_magnitude,
    batch_alignment_pairs,
    mutual_nearest_pairs,
    pairs_to_index,
)
from twindyn.data.resources import EXTERNAL_RESOURCES, RESOURCES, TRAINING_RESOURCE
from twindyn.data.schema import ClinicalStratum, SampleRecord, SurvivalLabel
from twindyn.data.splits import (
    assert_zero_overlap,
    internal_split,
    label_fraction_subset,
    mask_labels,
    modality_dropout,
    resource_partition,
    stratum_indices,
)


def test_compartment_registry_is_ordered_and_unique() -> None:
    assert len(set(COMPARTMENT_ORDER)) == len(COMPARTMENT_ORDER) == compartment_count()
    assert len(COMPARTMENT_ORDER) == 16


@pytest.mark.parametrize(
    "row,expected",
    [
        ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0], [0.0, 0.0]),
        ([-4.0, -1.0], [0.0, 0.0]),
        ([0.5], [1.0]),
    ],
)
def test_normalise_fractions_clips_and_renormalises(row, expected) -> None:
    produced = normalise_fractions(row)
    if max(row) <= 0.0:
        assert produced == expected
    else:
        assert sum(produced) == pytest.approx(1.0)
        assert all(value >= 0.0 for value in produced)


def test_sample_record_joins_and_reports_availability(compartment_total: int) -> None:
    record = SampleRecord(
        sample_id="s",
        resource="tcga_kirc",
        compartments=np.full(compartment_total, 1.0 / compartment_total),
        morphology=np.ones(4),
    )
    width = compartment_total + 4
    observation = record.joined_observation(width)
    availability = record.joined_availability(width, compartment_total)
    assert observation.shape == (width,)
    assert availability.all()
    assert record.observation_dim == width


def test_availability_absent_for_a_resource_without_histology(cohort) -> None:
    matrix = cohort.table.availability_matrix()
    total = cohort.table.compartment_count
    for index, sample in enumerate(cohort.table.samples):
        histology = RESOURCES[sample.resource].has_histology
        assert bool(matrix[index, total:].any()) == histology


def test_survival_label_rejects_bad_times() -> None:
    with pytest.raises(ValueError):
        SurvivalLabel(time=-1.0, event=True)
    with pytest.raises(ValueError):
        SurvivalLabel(time=float("nan"), event=False)


def test_stratum_properties() -> None:
    assert ClinicalStratum(3, 2).grade_stratum == "grade_3"
    assert ClinicalStratum(3, 2).stage_stratum == "stage_2"
    assert ClinicalStratum(None, 7).stage_stratum is None


def test_shared_feature_space_is_the_intersection() -> None:
    inventories = {
        "a": FeatureInventory("a", ("g1", "g2", "g3")),
        "b": FeatureInventory("b", ("g2", "g3", "g4")),
        "c": FeatureInventory("c", ("g3", "g4", "g5")),
    }
    space = intersect_inventories(inventories, set(), ())
    assert set(space.features) == {"g3"}
    assert space.per_resource_coverage["a"] == pytest.approx(1 / 3)


def test_projection_leaves_unobserved_features_absent() -> None:
    inventories = {
        "a": FeatureInventory("a", ("g1", "g2")),
        "b": FeatureInventory("b", ("g2",)),
    }
    space = intersect_inventories(inventories, set(), ())
    values, available = project_observation({"g2": 3.0}, space)
    assert values == [3.0]
    assert available == [True]
    values, available = project_observation({}, space)
    assert values == [0.0] and available == [False]


def test_deconvolution_recovers_a_noiseless_mixture() -> None:
    reference, panel = synthesise_reference(5, 120, 12, 3)
    engine = ReferenceBasedDeconvolution(reference, panel)
    generator = np.random.default_rng(4)
    truth = generator.dirichlet(np.ones(5))
    produced = engine.fit_sample(reference @ truth).fractions
    assert np.abs(produced - truth).max() < 0.05


def test_deconvolution_matches_an_independent_nnls() -> None:
    from scipy.optimize import nnls

    reference, panel = synthesise_reference(4, 80, 10, 5)
    engine = ReferenceBasedDeconvolution(reference, panel)
    generator = np.random.default_rng(6)
    profile = reference @ np.array([0.4, 0.3, 0.2, 0.1]) + generator.normal(0, 0.01, 80)
    produced = engine.fit_sample(profile).fractions
    solution, _ = nnls(reference[panel], profile[panel])
    normalised = solution / solution.sum()
    assert np.abs(produced - normalised).max() < 0.05


def test_deconvolution_sensitivity_covers_every_configuration() -> None:
    reference, _ = synthesise_reference(3, 100, 10, 7)
    profiles = np.random.default_rng(8).dirichlet(np.ones(3), size=5) @ reference.T
    results = sensitivity_sweep(profiles, DEFAULT_CONFIGURATIONS, 100, 3, 7)
    assert [entry["configuration"] for entry in results] == [
        configuration.name for configuration in DEFAULT_CONFIGURATIONS
    ]
    assert all(np.isfinite(entry["mean_absolute_drift"]) for entry in results)


def test_protein_abundance_proxy_is_a_distribution() -> None:
    values = np.array([1.0, 3.0, 0.0, 4.0])
    produced = fractions_from_protein_abundance(values, np.arange(4), 4)
    assert produced.shape == (4,)
    assert produced.sum() == pytest.approx(1.0)


def test_masking_ratio_and_availability(cohort, axis) -> None:
    availability = torch.from_numpy(cohort.table.availability_matrix()[:6])
    for ratio in (0.1, 0.3, 0.5):
        outcome = sample_mask(axis, availability, ratio, torch.Generator().manual_seed(0))
        assert abs(outcome.ratio - ratio) < 0.2
        assert not bool((outcome.withheld & ~outcome.available).any())
        assert int(outcome.kept.sum()) >= 2


def test_masking_on_nothing_observed_yields_nothing_withheld(cohort, axis) -> None:
    availability = torch.zeros((3, axis.width), dtype=torch.bool)
    outcome = sample_mask(axis, availability, 0.5, torch.Generator().manual_seed(0))
    assert int(outcome.withheld.sum()) == 0
    assert outcome.ratio == 0.0


def test_reconstruction_error_matches_a_hand_computation() -> None:
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(prediction)
    withheld = torch.tensor([[True, False], [False, True]])
    assert float(reconstruction_error(prediction, target, withheld)) == pytest.approx(8.5)


def test_position_availability_reduces_columns(cohort, axis) -> None:
    availability = torch.from_numpy(cohort.table.availability_matrix()[:4])
    reduced = position_availability(axis, availability)
    assert reduced.shape == (4, axis.tau_max)


def test_mutual_matching_is_injective() -> None:
    left = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    right = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    left_ids, right_ids, scores = mutual_nearest_pairs(left, right, 8)
    assert len(set(right_ids.tolist())) == len(right_ids)
    assert left_ids.shape == right_ids.shape == scores.shape


def test_batch_pairs_stay_inside_one_batch_and_cross_resources(cohort, axis) -> None:
    pairs = batch_alignment_pairs(
        torch.from_numpy(cohort.table.observations()[:8, : cohort.table.compartment_count]).float(),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        4,
    )
    assert all(pair.left_resource != pair.right_resource for pair in pairs)
    for pair in pairs:
        assert int(pair.left_index.max()) < 8
        assert int(pair.right_index.max()) < 8


def test_pairs_are_unspecified_when_a_single_resource_is_present() -> None:
    pairs = batch_alignment_pairs(torch.eye(3), torch.tensor([1, 1, 1]), 4)
    assert pairs == []


def test_alignment_magnitude_reports_nan_for_missing_pairs(cohort) -> None:
    pairs = batch_alignment_pairs(
        torch.from_numpy(cohort.table.observations()[:4, : cohort.table.compartment_count]).float(),
        torch.tensor([0, 1, 0, 1]),
        4,
    )
    states = torch.randn(4, 5, dtype=torch.complex64)
    magnitude = alignment_magnitude(states, pairs)
    assert magnitude
    assert all(value >= 0.0 for value in magnitude.values() if value == value)


def test_pairs_to_index_flattens_available_pairs(cohort) -> None:
    pairs = batch_alignment_pairs(torch.eye(4), torch.tensor([0, 0, 1, 1]), 4)
    left, right, mask = pairs_to_index(pairs)
    assert left.shape == right.shape == mask.shape
    assert bool(mask.all())


def test_zero_overlap_on_disjoint_identifiers() -> None:
    records = [
        SampleRecord("a-1", "tcga_kirc", np.array([1.0])),
        SampleRecord("b-1", "gse29609", np.array([1.0])),
        SampleRecord("a-2", "tcga_kirc", np.array([1.0])),
    ]
    report = assert_zero_overlap(records)
    assert report["zero_overlap"] is True
    assert report["per_resource_counts"]["tcga_kirc"] == 2


def test_zero_overlap_detects_a_collision() -> None:
    records = [
        SampleRecord("shared", "tcga_kirc", np.array([1.0])),
        SampleRecord("shared", "cptac_ccrcc", np.array([1.0])),
    ]
    report = assert_zero_overlap(records)
    assert report["zero_overlap"] is False
    assert report["collisions"]


def test_internal_split_is_disjoint_and_complete(cohort) -> None:
    fit, validation = internal_split(cohort.table.samples, TRAINING_RESOURCE, 0.25, 0)
    assert not set(fit) & set(validation)
    assert sorted(set(fit) | set(validation)) == sorted(
        index
        for index, sample in enumerate(cohort.table.samples)
        if sample.resource == TRAINING_RESOURCE
    )


def test_label_fraction_subset_shrinks_the_labelled_part(cohort) -> None:
    fit, _ = internal_split(cohort.table.samples, TRAINING_RESOURCE, 0.25, 0)
    full = set(label_fraction_subset(cohort.table.samples, fit, 1.0, 0))
    quarter = set(label_fraction_subset(cohort.table.samples, fit, 0.25, 0))
    assert quarter.issubset(full)
    assert 0 < len(quarter) < len(full)
    assert full.issubset(set(fit))


def test_mask_labels_keeps_the_observations(cohort) -> None:
    fit, _ = internal_split(cohort.table.samples, TRAINING_RESOURCE, 0.25, 0)
    kept = set(label_fraction_subset(cohort.table.samples, fit, 0.25, 0))
    masked = mask_labels(cohort.table, set(fit) - kept)
    assert len(masked) == len(cohort.table)
    assert masked.width == cohort.table.width
    withheld = [index for index in fit if index not in kept]
    assert all(masked.samples[index].label is None for index in withheld)
    assert all(masked.samples[index].label is not None for index in kept)
    assert np.allclose(masked.observations(), cohort.table.observations())


def test_resource_partition_is_exhaustive(cohort) -> None:
    partition = resource_partition(cohort.table.samples)
    assert sorted(index for values in partition.values() for index in values) == list(
        range(len(cohort.table.samples))
    )
    assert set(partition) == {TRAINING_RESOURCE, *EXTERNAL_RESOURCES}


def test_strata_indices_are_readable(cohort) -> None:
    stage = stratum_indices(cohort.table.samples, tuple(range(len(cohort.table))), "stage")
    grade = stratum_indices(cohort.table.samples, tuple(range(len(cohort.table))), "grade")
    assert stage and grade
    assert set(stage) <= {"stage_1", "stage_2", "stage_3", "stage_4"}


def test_modality_dropout_zeroes_one_family_cohort(cohort) -> None:
    observations = cohort.table.observations()[:4]
    total = cohort.table.compartment_count
    morphology = modality_dropout(observations, total, "morphology")
    compartments = modality_dropout(observations, total, "compartments")
    assert np.abs(morphology[:, total:]).sum() == 0.0
    assert np.abs(compartments[:, :total]).sum() == 0.0
    assert np.allclose(morphology[:, :total], observations[:, :total])
    with pytest.raises(ValueError):
        modality_dropout(observations, total, "unknown")


def test_aggregation_levels_cover_the_registry() -> None:
    for level in (0, 1, 2, 3):
        groups = aggregate_compartment_groups(level)
        assert sorted(value for group in groups for value in group) == list(
            range(compartment_count())
        )


def test_parsers_read_the_reported_clinical_forms() -> None:
    assert parse_stage("Stage III") == 3
    assert parse_stage("IV") == 4
    assert parse_grade("G3") == 3
    assert parse_grade("grade 2") == 2
    assert parse_event("Dead") is True
    assert parse_event("Alive") is False
    assert parse_survival_value("24 months") == pytest.approx(24 * 30.4375)
    assert parse_survival_value("2 years") == pytest.approx(2 * 365.25)
    assert parse_survival_value("730 days") == pytest.approx(730.0)
    assert parse_survival_value("730") == pytest.approx(730.0)
    assert parse_survival_value("[Not Available]") is None


def test_readers_report_absence_without_fetching(tmp_path) -> None:
    assert not GdcKirc(tmp_path / "tcga_kirc").presence().available
    assert not GeoSeries(tmp_path / "gse29609").presence().available
    assert not ArrayExpressStudy(tmp_path / "emtab1980").presence().available
    assert not CptacCcrcc(tmp_path / "cptac_ccrcc").presence().available
    assert not CancerImagingArchive(tmp_path / "cptac_ccrcc").presence().available


def test_gdc_reader_parses_a_staged_table(tmp_path) -> None:
    root = tmp_path / "tcga_kirc"
    root.mkdir()
    (root / "clinical.tsv").write_text(
        "case_id\tvital_status\tdays_to_death\tdays_to_last_follow_up\t"
        "ajcc_pathologic_stage\tajcc_pathologic_grade\n"
        "T1\tDead\t400\t\tStage III\tG3\n"
        "T2\tAlive\t\t900\tStage I\tG2\n",
        encoding="utf-8",
    )
    reader = GdcKirc(root)
    survival = reader.survival()
    strata = reader.strata()
    assert survival["T1"] == (400.0, True)
    assert survival["T2"] == (900.0, False)
    assert strata["T1"] == (3, 3)


def test_geo_reader_parses_a_series_matrix(tmp_path) -> None:
    root = tmp_path / "gse29609"
    root.mkdir()
    body = (
        "!Series_title = Clear-cell renal cell carcinomas tumors\n"
        "!Series_geo_accession = GSE29609\n"
        "!Sample_title = tumour one\ttumour two\n"
        "!Sample_characteristics_ch1 = stage: II\tstage: III\n"
        "!Sample_characteristics_ch1 = grade: 2\tgrade: 3\n"
        "!Sample_characteristics_ch1 = survival time (months): 30\tsurvival time (months): 12\n"
        "!Sample_characteristics_ch1 = event: 1\tevent: 0\n"
        "!series_matrix_table_begin\n"
        '"ID_REF"\t"GSM1"\t"GSM2"\n'
        '"100_at"\t4.0\t5.0\n'
        "!series_matrix_table_end\n"
    )
    path = root / "GSE29609_series_matrix.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(body)
    reader = GeoSeries(root)
    _, samples, matrix = reader.series_matrix()
    assert samples == ["GSM1", "GSM2"]
    assert matrix.shape == (2, 1)
    assert reader.survival()["GSM1"][1] is True
    assert reader.strata()["GSM2"] == (3, 3)


def test_arrayexpress_reader_parses_an_sdrf(tmp_path) -> None:
    root = tmp_path / "emtab1980"
    root.mkdir()
    (root / "E-MTAB-1980.sdrf.txt").write_text(
        "Source Name\tCharacteristics[stage]\tCharacteristics[grade]\t"
        "Characteristics[survival time]\tCharacteristics[event]\n"
        "s1\tII\t2\t40\t1\n"
        "s2\tI\t1\t70\t0\n",
        encoding="utf-8",
    )
    reader = ArrayExpressStudy(root)
    assert reader.survival()["s1"] == (float(40), True)
    assert reader.strata()["s2"] == (1, 1)


def test_tcia_access_conditions_are_declared() -> None:
    access = TCIA_COLLECTIONS["cptac_ccrcc"]
    assert access.version == "14"
    assert access.licence == "CC BY 4.0"
    assert access.requires_application is False
