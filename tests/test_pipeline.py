"""The end-to-end pipeline, the selection routine and the sweep assemblers.

Ref: Sec. 4.1-4.11, Table 2, Table 3, Table 4, Sec. 3.10 (internal-validation
selection of the quantities the manuscript leaves open).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from twindyn.data.compartments import compartment_count
from twindyn.evaluation.selection import (
    declared_grids,
    merge_reports,
    sweep_values,
    validation_objective,
)
from twindyn.evaluation.sweeps import (
    ABLATION_VARIANTS,
    VARIANT_COMPONENTS,
    assemble_ablation,
    axis_length_sweep,
    state_dimension_sweep,
    summarise_sweep,
)
from twindyn.losses.composite import ObjectiveWeights
from twindyn.metrics.decision import hypothesis_families
from twindyn.pipeline import (
    build_context,
    relative_alignment,
    run_experiment,
    summarise_initialisations,
)
from twindyn.registry import SELECTED_AXIS_LENGTH, SELECTED_STATE_DIM
from twindyn.utils.config import (
    AlignmentConfig,
    AxisConfig,
    DataConfig,
    EvaluationConfig,
    FeatureSpaceConfig,
    MaskingConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimConfig,
    RunConfig,
    resolve_config,
)


def tiny_config(root, output) -> RunConfig:
    return RunConfig(
        experiment="test",
        output_dir=str(output),
        seed=0,
        axis=AxisConfig(tau_max=SELECTED_AXIS_LENGTH),
        features=FeatureSpaceConfig(morphology_features=32),
        data=DataConfig(root=str(root), synthetic_samples_per_resource=8),
        model=ModelConfig(state_dim=32),
        optim=OptimConfig(
            batch_size=8,
            representation_epochs=3,
            readout_epochs=3,
            warmup_steps=1,
            learning_rate=2e-3,
        ),
        objective=ObjectiveConfig(),
        masking=MaskingConfig(ratio=0.3),
        alignment=AlignmentConfig(max_pairs_per_resource=4),
        evaluation=EvaluationConfig(
            initialisations=1,
            bootstrap_resamples=20,
            probe_repeats=1,
            label_fractions=(1.0,),
        ),
    )


@pytest.fixture(scope="module")
def experiment(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("pipeline")
    output = root / "run"
    result = run_experiment(tiny_config(root / "data", output), output)
    return {"result": result, "output": output, "root": root}


def test_configuration_defaults_carry_the_paper_values() -> None:
    config = RunConfig()
    assert config.axis.tau_max == SELECTED_AXIS_LENGTH
    assert config.model.state_dim == SELECTED_STATE_DIM
    assert config.evaluation.initialisations == 10
    assert config.evaluation.bootstrap_resamples == 1000
    assert config.evaluation.external_threshold == pytest.approx(0.03)
    assert config.evaluation.decision_margin == pytest.approx(0.005)
    assert config.evaluation.label_fractions == (0.10, 0.25, 0.50, 1.00)
    assert config.evaluation.probe_repeats == 10


def test_context_assembles_disjoint_views(tmp_path) -> None:
    context = build_context(tiny_config(tmp_path / "data", tmp_path / "run"))
    assert not set(context.fit) & set(context.validation)
    assert set(context.external_datasets) == {"cptac_ccrcc", "gse29609", "emtab1980"}
    assert context.pair_source.max_pairs_per_resource == 4
    assert context.representation_dataset.indices == tuple(range(len(context.table)))
    assert len(context.readout_dataset) == len(context.fit)


def test_pipeline_reports_the_stages_it_ran(experiment) -> None:
    result = experiment["result"]
    assert result.cohort_source == "synthetic"
    assert result.integrity["zero_overlap"] is True
    assert result.summary["initialisations"] == 1
    assert result.summary["representation_loss_decreased"] is True
    assert result.summary["readout_loss_decreased"] is True
    assert result.comparators["available"] is True
    assert len(result.comparators["rows"]) == 18
    assert result.ablation["rows"]
    assert result.generalisation["available"] is True
    assert result.label_efficiency["available"] is True


def test_pipeline_writes_its_artefacts(experiment) -> None:
    output = experiment["output"]
    for name in ("run.json", "tables.json", "initialisations.json"):
        assert (output / name).exists(), name
    checkpoints = sorted((output / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 1


def test_pipeline_marks_the_non_distributable_tables(experiment) -> None:
    statuses = {entry["table"]: entry["status"] for entry in experiment["result"].tables["tables"]}
    assert statuses["Table 6 external validation"] == "NOT_RUN"
    assert statuses["Table 1 main comparison"] in {"PASS", "FAIL", "NOT_RUN"}
    assert experiment["result"].tables["status_counts"]


def test_pipeline_records_the_cohort_provenance(experiment) -> None:
    provenance = experiment["result"].cohort_provenance
    assert len(provenance) == 4
    assert all(entry["source"] == "synthetic" for entry in provenance)
    assert any("not a real archive" in entry["detail"] for entry in provenance)


def test_axis_and_model_report_are_recorded(experiment) -> None:
    axis = experiment["result"].axis
    assert axis["tau_max"] == SELECTED_AXIS_LENGTH
    assert axis["width"] == compartment_count() + 32
    model = experiment["result"].model
    assert model["parameter_count"]["total"] > 0
    assert model["observation_map"]["rank"] == 32


def test_summarise_initialisations_reports_alignment_and_spread(experiment) -> None:
    summary = experiment["result"].summary
    assert "alignment_relative" in summary
    assert summary["paired_differences"]
    assert summary["training"]["sd"] == 0.0
    assert summary["seed_plan"]


def test_relative_alignment_is_computed_per_pair() -> None:
    from twindyn.pipeline import InitialisationOutcome

    outcome = InitialisationOutcome(
        index=0,
        seed=0,
        representation_initial_loss=1.0,
        representation_final_loss=0.5,
        representation_steps=1,
        readout_initial_loss=1.0,
        readout_final_loss=0.5,
        training_c_index=0.5,
        external_c_index={"gse29609": 0.5},
        alignment_initial={"0|1": 2.0},
        alignment_final={"0|1": 0.5},
    )
    assert relative_alignment([outcome])["0|1"] == pytest.approx(0.25)
    assert summarise_initialisations([outcome])["initialisations"] == 1


def test_ablation_variants_cover_the_reported_rows() -> None:
    names = [variant.name for variant in ABLATION_VARIANTS]
    assert names[0] == "full"
    assert "minus_sst_static_matched" in names
    assert "equal_budget_default_hyperparameters" in names
    assert len(names) == 8
    assert VARIANT_COMPONENTS["full"] == frozenset({"mask", "align", "trajectory"})
    assert VARIANT_COMPONENTS["minus_msi_alignment"] == frozenset({"mask", "trajectory"})


def test_ablation_assembly_reports_drops_and_retention() -> None:
    means = {
        "full": 0.655,
        "minus_sst_static_matched": 0.614,
        "minus_msi_alignment": 0.631,
        "minus_masked_reconstruction": 0.645,
        "minus_trajectory_pooling": 0.648,
        "readout_depth_1_to_2": 0.653,
        "alignment_clinical_only": 0.652,
        "equal_budget_default_hyperparameters": 0.652,
    }
    report = assemble_ablation(means, {"full": 3.16})
    assert report.drops["minus_sst_static_matched"] == pytest.approx(-0.041)
    assert report.ratio["ratio"] == pytest.approx(20.5)
    assert report.retention["retention"] == pytest.approx(0.9268, abs=1e-3)
    assert len(report.rows) == 8


def test_state_property_sweeps_report_the_selected_value() -> None:
    length = axis_length_sweep({4: 0.631, 8: 0.646, 16: 0.655, 32: 0.657})
    dimension = state_dimension_sweep({16: 0.638, 32: 0.652, 64: 0.655, 128: 0.654})
    assert length.selected == 16.0
    assert length.note == "plateau after 16"
    assert dimension.selected == 64.0
    assert dimension.note == "saturation after 64"
    assert summarise_sweep({4: 0.631, 8: 0.646})["spread"] == pytest.approx(0.015)


def test_selection_grids_carry_the_table_4_grids() -> None:
    grids = declared_grids()
    assert grids["tau_max"] == (4.0, 8.0, 16.0, 32.0)
    assert grids["state_dim"] == (16.0, 32.0, 64.0, 128.0)
    assert 0.30 in grids["mask_ratio"]


def test_sweep_values_picks_the_lowest_score() -> None:
    report = sweep_values("x", (1.0, 2.0, 3.0), lambda value: abs(value - 2.0))
    assert report.selected["x"] == 2.0
    assert len(report.records) == 3
    merged = merge_reports([report, sweep_values("y", (1.0,), lambda value: value)])
    assert set(merged.selected) == {"x", "y"}


def test_sweep_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError):
        sweep_values("x", (), lambda value: value)


def test_validation_objective_is_finite(tmp_path) -> None:
    from twindyn.models.twindyn import TwinDyn, TwinDynConfig

    context = build_context(tiny_config(tmp_path / "data", tmp_path / "run"))
    model = TwinDyn(
        TwinDynConfig(state_dim=context.config.model.state_dim), context.build.axis.index()
    )
    score = validation_objective(
        model,
        context.representation_dataset,
        context.pair_source,
        ObjectiveWeights(),
        0.3,
        seed=0,
    )
    assert np.isfinite(score)


def test_hypothesis_families_match_the_declared_contrasts() -> None:
    families = hypothesis_families()
    assert families["external"]["comparisons"] == 3
    assert families["label_efficiency"]["comparisons"] == 4
    assert families["level"]["descriptive"] and families["loss"]["descriptive"]


def test_smoke_config_resolves_from_the_repository() -> None:
    config = resolve_config("configs", "_smoke")
    assert config.axis.tau_max == 16
    assert config.model.state_dim == 64
    assert config.evaluation.initialisations == 1


def test_replaced_configuration_keeps_the_other_fields() -> None:
    config = RunConfig()
    updated = replace(config, seed=11)
    assert updated.seed == 11
    assert updated.model.state_dim == config.model.state_dim
    assert updated.evaluation.bootstrap_resamples == 1000
