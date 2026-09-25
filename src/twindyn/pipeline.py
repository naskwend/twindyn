"""The end-to-end experiment pipeline.

Ref: Sec. 4.1 (resource-disjoint design), Sec. 4.4 (ten random initialisations,
each representation learned once and reused by every read-out variant), Sec. 4.5
(the main comparison), Sec. 4.9 (the mechanism triad), Sec. 4.10 and Sec. 4.11
(the external and stratum arms).

The pipeline is the single place where a run is assembled: cohorts, axis, splits,
the representation, the read-out, the comparator family, the mechanism evidence
and the reported-table comparison. Every stage records what it produced so the
verification pass reads a number back instead of recomputing a claim from prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from twindyn.data.assembly import CohortBuild, build_staged_cohorts, build_synthetic_cohorts
from twindyn.data.dataset import AxisDataset, FeatureStandardiser
from twindyn.data.resources import EXTERNAL_RESOURCES, RESOURCES
from twindyn.data.schema import CohortTable
from twindyn.data.splits import (
    assert_zero_overlap,
    internal_split,
    label_fraction_subset,
    mask_labels,
    modality_dropout,
    resource_partition,
)
from twindyn.evaluation.external import evaluate_external
from twindyn.evaluation.mechanism import collect_mechanism, summary_arrays
from twindyn.evaluation.reporting import (
    ReportingBundle,
    compare_ablation,
    compare_external,
    compare_generalisation,
    compare_label_efficiency,
    compare_main_comparison,
    compare_mechanism,
    compare_state_properties,
)
from twindyn.evaluation.sweeps import (
    ABLATION_VARIANTS,
    AblationReport,
    AblationVariant,
    assemble_ablation,
    build_variant_model,
    capacity_report,
)
from twindyn.evaluation.views import feature_view
from twindyn.metrics.concordance import concordance_index
from twindyn.metrics.decision import decide
from twindyn.metrics.efficiency import efficiency_curve, margin_curve, shrinkage_profile
from twindyn.metrics.strata import (
    GRADE_DEFINITIONS,
    GRADE_LABELS,
    STAGE_DEFINITIONS,
    STAGE_LABELS,
    dropout_rows,
    full_row,
    monotonicity,
    stratum_rows,
    within_stratum_gain,
)
from twindyn.models.baselines import BASELINES, build_baseline, run_baseline
from twindyn.models.baselines.inventory import COMPARATOR_ID
from twindyn.models.baselines.protocol import BaselineProtocol, RiskEstimator
from twindyn.models.twindyn import TwinDyn, TwinDynConfig
from twindyn.reported import PERTURBATION_POSITIONS, PERTURBATION_SCALES
from twindyn.training.checkpoint import save_checkpoint
from twindyn.training.engine import PairSource, score, train_readout, train_representation
from twindyn.training.seeding import plan_initialisations, set_seed
from twindyn.utils.config import RunConfig, config_digest, config_to_json
from twindyn.utils.io import atomic_write_json
from twindyn.utils.runtime import get_logger

LOGGER = get_logger("pipeline")

RESOURCE_CODES: dict[str, int] = {
    key: position for position, key in enumerate(("tcga_kirc", *EXTERNAL_RESOURCES))
}


@dataclass
class InitialisationOutcome:
    index: int
    seed: int
    representation_initial_loss: float
    representation_final_loss: float
    representation_steps: int
    readout_initial_loss: float
    readout_final_loss: float
    training_c_index: float
    external_c_index: dict[str, float]
    alignment_initial: dict[str, float]
    alignment_final: dict[str, float]
    risk: dict[str, np.ndarray] = field(default_factory=dict)
    checkpoint: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "seed": self.seed,
            "representation_initial_loss": self.representation_initial_loss,
            "representation_final_loss": self.representation_final_loss,
            "representation_steps": self.representation_steps,
            "readout_initial_loss": self.readout_initial_loss,
            "readout_final_loss": self.readout_final_loss,
            "training_c_index": self.training_c_index,
            "external_c_index": dict(self.external_c_index),
            "alignment_initial": dict(self.alignment_initial),
            "alignment_final": dict(self.alignment_final),
            "checkpoint": self.checkpoint,
        }


@dataclass
class PipelineResult:
    configuration: dict[str, object]
    configuration_digest: str
    cohort_source: str
    cohort_provenance: list[dict[str, object]]
    integrity: dict[str, object]
    axis: dict[str, object]
    shared_space: dict[str, object]
    model: dict[str, object]
    initialisations: list[InitialisationOutcome]
    summary: dict[str, object]
    comparators: dict[str, object]
    ablation: dict[str, object]
    generalisation: dict[str, object]
    state_properties: dict[str, object]
    mechanism: dict[str, object]
    external: dict[str, object]
    label_efficiency: dict[str, object]
    selection: dict[str, object]
    tables: dict[str, object]
    notes: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "configuration": dict(self.configuration),
            "configuration_digest": self.configuration_digest,
            "cohort_source": self.cohort_source,
            "cohort_provenance": list(self.cohort_provenance),
            "integrity": dict(self.integrity),
            "axis": dict(self.axis),
            "shared_space": dict(self.shared_space),
            "model": dict(self.model),
            "initialisations": [entry.as_dict() for entry in self.initialisations],
            "summary": dict(self.summary),
            "comparators": dict(self.comparators),
            "ablation": dict(self.ablation),
            "generalisation": dict(self.generalisation),
            "state_properties": dict(self.state_properties),
            "mechanism": dict(self.mechanism),
            "external": dict(self.external),
            "label_efficiency": dict(self.label_efficiency),
            "selection": dict(self.selection),
            "tables": dict(self.tables),
            "notes": list(self.notes),
        }


@dataclass
class RunContext:
    """Everything a stage of the run shares."""

    build: CohortBuild
    config: RunConfig
    fit: tuple[int, ...]
    validation: tuple[int, ...]
    standardiser: FeatureStandardiser
    representation_dataset: AxisDataset
    readout_dataset: AxisDataset
    external_datasets: dict[str, AxisDataset]
    partitions: dict[str, tuple[int, ...]]
    pair_source: PairSource
    device: torch.device

    @property
    def table(self) -> CohortTable:
        return self.build.table


def build_cohorts(config: RunConfig) -> CohortBuild:
    """Prefer whatever archive is staged, and only then fall back to the generator."""
    root = Path(config.data.root)
    if root.exists() and any(root.iterdir()):
        try:
            return build_staged_cohorts(
                root=root,
                tau_max=config.axis.tau_max,
                morphology_features=config.features.morphology_features,
                marker_genes_per_compartment=config.deconvolution.marker_genes_per_compartment,
                gene_panel_size=config.deconvolution.gene_panel_size,
                ordering_rule=config.axis.ordering_rule,
                aggregation_level=config.axis.aggregation_level,
            )
        except (FileNotFoundError, ValueError) as error:
            LOGGER.warning("staged resources unusable (%s); generating cohorts", error)
    return build_synthetic_cohorts(
        tau_max=config.axis.tau_max,
        samples_per_resource=config.data.synthetic_samples_per_resource,
        seed=config.data.synthetic_seed,
        state_dim=config.model.state_dim,
        resource_shift=config.data.synthetic_resource_shift,
        morphology_features=config.features.morphology_features,
        ordering_rule=config.axis.ordering_rule,
        aggregation_level=config.axis.aggregation_level,
    )


def build_context(config: RunConfig | None = None, **overrides: object) -> RunContext:
    """Assemble splits, standardisation and dataset views once."""
    resolved = config if config is not None else RunConfig(**overrides)  # type: ignore[arg-type]
    build = build_cohorts(resolved)
    table = build.table
    fit, validation = internal_split(
        table.samples,
        resolved.data.training_resource,
        resolved.data.internal_validation_fraction,
        resolved.seed,
    )
    observations = table.observations()
    availability = table.availability_matrix()
    standardiser = FeatureStandardiser.fit(observations[list(fit)], availability[list(fit)])
    partitions = resource_partition(table.samples)
    device = torch.device(resolved.device)
    alignment_rows = tuple(
        index
        for index, sample in enumerate(table.samples)
        if sample.resource in resolved.data.resources
    )
    return RunContext(
        build=build,
        config=resolved,
        fit=fit,
        validation=validation,
        standardiser=standardiser,
        representation_dataset=AxisDataset(table, build.axis, alignment_rows, standardiser),
        readout_dataset=AxisDataset(table, build.axis, fit, standardiser),
        external_datasets={
            resource: AxisDataset(table, build.axis, partitions.get(resource, ()), standardiser)
            for resource in EXTERNAL_RESOURCES
            if partitions.get(resource)
        },
        partitions=partitions,
        pair_source=PairSource(
            max_pairs_per_resource=resolved.alignment.max_pairs_per_resource,
            clinical_only=resolved.alignment.clinical_only_null,
        ),
        device=device,
    )


def model_config(config: RunConfig) -> TwinDynConfig:
    return TwinDynConfig(
        state_dim=config.model.state_dim,
        msi_hidden=config.model.msi_hidden,
        msi_layers=config.model.msi_layers,
        msi_dropout=config.model.msi_dropout,
        complex_state=config.model.complex_state,
        discretisation=config.model.discretisation,
        selective=config.model.selective,
        mimo=config.model.mimo,
        readout_depth=config.model.readout_depth,
        readout_hidden=config.model.readout_hidden,
        pooling=config.model.pooling,
    )


def labelled_rows(context: RunContext, resource: str | None = None) -> list[int]:
    rows = (
        list(context.partitions.get(resource, ()))
        if resource is not None
        else list(range(len(context.table)))
    )
    return [index for index in rows if context.table.samples[index].label is not None]


def labelled_arrays(context: RunContext, rows: list[int]) -> dict[str, np.ndarray]:
    table = context.table
    return {
        "time": np.asarray([table.samples[index].label.time for index in rows], dtype=np.float64),
        "event": np.asarray(
            [float(table.samples[index].label.event) for index in rows],
            dtype=np.float64,
        ),
    }


def external_arrays(context: RunContext, resource: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    rows = labelled_rows(context, resource)
    arrays = labelled_arrays(context, rows)
    return rows, arrays["time"], arrays["event"]


def run_experiment(config: RunConfig, output_dir: str | Path | None = None) -> PipelineResult:
    """Run one complete experiment and write its artefacts."""
    set_seed(config.seed, deterministic=config.deterministic)
    context = build_context(config)
    table = context.table
    integrity = assert_zero_overlap(table.samples)
    plan = plan_initialisations(config.seed, config.evaluation.initialisations)
    checkpoint_dir = Path(output_dir or config.output_dir) / "checkpoints"
    outcomes: list[InitialisationOutcome] = []
    model_report: dict[str, object] = {}
    for index in range(plan.count):
        seed = plan.seed(index)
        set_seed(seed, deterministic=config.deterministic)
        model = TwinDyn(model_config(config), context.build.axis.index())
        representation, alignment_initial, alignment_final = train_representation(
            model=model,
            dataset=context.representation_dataset,
            pair_source=context.pair_source,
            optim_config=config.optim,
            objective_config=config.objective,
            seed=seed,
            mask_ratio=config.masking.ratio,
            device=context.device,
            ema_decay=config.optim.ema_decay,
        )
        readout = train_readout(
            model=model,
            dataset=context.readout_dataset,
            optim_config=config.optim,
            seed=seed,
            device=context.device,
        )
        training_rows = labelled_rows(context, config.data.training_resource)
        training_scores = score(model, context.readout_dataset, context.device)
        labelled = training_scores["has_label"]
        training_c = concordance_index(
            training_scores["risk"][labelled],
            training_scores["time"][labelled],
            training_scores["event"][labelled],
        ).c_index
        external_c: dict[str, float] = {}
        risks: dict[str, np.ndarray] = {}
        for resource, dataset in context.external_datasets.items():
            rows, time, event = external_arrays(context, resource)
            if len(rows) < 2:
                continue
            scores = score(model, dataset, context.device)
            risks[resource] = scores["risk"]
            external_c[resource] = concordance_index(scores["risk"], time, event).c_index
        checkpoint_path = checkpoint_dir / f"initialisation_{index:02d}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            seed=seed,
            configuration_digest=config_digest(config),
            stage="readout",
            epoch=config.optim.readout_epochs,
            metrics={
                "training_c_index": training_c,
                "external_mean": float(np.mean(list(external_c.values())))
                if external_c
                else float("nan"),
            },
            extra={"initialisation_index": index, "training_rows": len(training_rows)},
        )
        if not model_report:
            model_report = model.report()
        outcomes.append(
            InitialisationOutcome(
                index=index,
                seed=seed,
                representation_initial_loss=representation.initial_loss,
                representation_final_loss=representation.final_loss,
                representation_steps=representation.steps,
                readout_initial_loss=readout.initial_loss,
                readout_final_loss=readout.final_loss,
                training_c_index=training_c,
                external_c_index=external_c,
                alignment_initial=alignment_initial,
                alignment_final=alignment_final,
                risk=risks,
                checkpoint=str(checkpoint_path),
            )
        )
        LOGGER.info(
            "initialisation %d/%d: training C-index %.4f, external mean %.4f",
            index + 1,
            plan.count,
            training_c,
            float(np.mean(list(external_c.values()))) if external_c else float("nan"),
        )
    summary = summarise_initialisations(outcomes)
    comparators = run_comparators(context)
    ablation = run_ablation(context)
    generalisation = run_generalisation(context)
    mechanism = run_mechanism(context)
    external = run_external(context, outcomes)
    efficiency = run_label_efficiency(context)
    tables = assemble_tables(
        summary, comparators, ablation, generalisation, mechanism, efficiency, external
    )
    result = PipelineResult(
        configuration=config_to_json(config),
        configuration_digest=config_digest(config),
        cohort_source=context.build.source,
        cohort_provenance=[entry.__dict__ for entry in context.build.provenance],
        integrity=integrity,
        axis=context.build.axis.describe(),
        shared_space={
            "features": len(context.build.space.features),
            "coverage": dict(context.build.space.per_resource_coverage),
            "imputation": "none",
            "mode": "intersection",
        },
        model=model_report,
        initialisations=outcomes,
        summary=summary,
        comparators=comparators,
        ablation=ablation.as_dict(),
        generalisation=generalisation,
        state_properties=state_properties(summary),
        mechanism=mechanism,
        external=external,
        label_efficiency=efficiency,
        selection={"initialisations": plan.as_dict()},
        tables=tables,
        notes=list(context.build.notes),
    )
    if output_dir is not None:
        write_run(result, output_dir)
    return result


def summarise_initialisations(outcomes: list[InitialisationOutcome]) -> dict[str, object]:
    """Means, standard deviations and the alignment reading across initialisations."""
    training = np.asarray([entry.training_c_index for entry in outcomes], dtype=np.float64)
    external_keys = sorted({key for entry in outcomes for key in entry.external_c_index})
    per_resource: dict[str, dict[str, float]] = {}
    for key in external_keys:
        values = np.asarray(
            [entry.external_c_index[key] for entry in outcomes if key in entry.external_c_index],
            dtype=np.float64,
        )
        per_resource[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    per_difference: dict[str, dict[str, float]] = {}
    for position, left in enumerate(external_keys):
        for right in external_keys[position + 1 :]:
            differences = np.asarray(
                [
                    entry.external_c_index[left] - entry.external_c_index[right]
                    for entry in outcomes
                    if left in entry.external_c_index and right in entry.external_c_index
                ],
                dtype=np.float64,
            )
            if differences.size == 0:
                continue
            per_difference[f"{left}-{right}"] = {
                "mean": float(differences.mean()),
                "sd": float(differences.std(ddof=1)) if differences.size > 1 else 0.0,
            }
    mean_external = (
        float(np.mean([entry["mean"] for entry in per_resource.values()]))
        if per_resource
        else float("nan")
    )
    return {
        "initialisations": len(outcomes),
        "training": {
            "mean": float(training.mean()) if training.size else float("nan"),
            "sd": float(training.std(ddof=1)) if training.size > 1 else 0.0,
            "min": float(training.min()) if training.size else float("nan"),
            "max": float(training.max()) if training.size else float("nan"),
        },
        "external": per_resource,
        "external_mean": mean_external,
        "paired_differences": per_difference,
        "alignment_relative": relative_alignment(outcomes),
        "representation_loss_decreased": all(
            entry.representation_final_loss < entry.representation_initial_loss
            for entry in outcomes
        ),
        "readout_loss_decreased": all(
            entry.readout_final_loss < entry.readout_initial_loss for entry in outcomes
        ),
        "seed_plan": [entry.seed for entry in outcomes],
    }


def relative_alignment(outcomes: list[InitialisationOutcome]) -> dict[str, float]:
    """Equation (5) at the final epoch relative to its value at initialisation."""
    keys = sorted({key for entry in outcomes for key in entry.alignment_final})
    out: dict[str, float] = {}
    for key in keys:
        ratios = [
            entry.alignment_final[key] / entry.alignment_initial[key]
            for entry in outcomes
            if key in entry.alignment_final and entry.alignment_initial.get(key)
        ]
        if ratios:
            out[key] = float(np.mean(ratios))
    return out


def state_properties(summary: dict[str, object]) -> dict[str, object]:
    """Table 4's own quantities that this run can produce."""
    return {
        "alignment_relative": summary.get("alignment_relative", {}),
        "external_mean": summary.get("external_mean", float("nan")),
        "external_per_resource": summary.get("external", {}),
    }


def run_comparators(context: RunContext) -> dict[str, object]:
    """The comparator family of Table 1, fitted on the fit split and scored held out."""
    fit_rows = list(context.fit)
    holdout_rows = list(context.validation)
    if len(fit_rows) < 4 or len(holdout_rows) < 2:
        return {"available": False, "reason": "too few labelled training samples"}
    fit_arrays = labelled_arrays(context, fit_rows)
    holdout_arrays = labelled_arrays(context, holdout_rows)
    protocol = BaselineProtocol(
        epochs=context.config.optim.representation_epochs,
        learning_rate=context.config.optim.learning_rate,
        weight_decay=context.config.optim.weight_decay,
        batch_size=context.config.optim.batch_size,
        seed=context.config.seed,
        initialisations=context.config.evaluation.initialisations,
    )
    resources = np.asarray(
        [RESOURCE_CODES[context.table.samples[index].resource] for index in fit_rows],
        dtype=np.int64,
    )
    per_baseline: dict[str, object] = {}
    for identifier, spec in BASELINES.items():
        view = feature_view(context.table, spec.view)
        estimator = build_baseline(
            identifier, view.shape[1], 4, token_dim=context.build.axis.position_dim
        )
        run_baseline(
            estimator, view[fit_rows], fit_arrays["time"], fit_arrays["event"], protocol, resources
        )
        holdout_risk = estimator.predict_risk(view[holdout_rows])
        result = concordance_index(holdout_risk, holdout_arrays["time"], holdout_arrays["event"])
        per_baseline[identifier] = {
            "method": spec.method,
            "family": spec.family,
            "view": spec.view,
            "c_index": result.c_index,
            "comparable_pairs": result.comparable_pairs,
            "reported_c_index": spec.reported_c_index,
            "reported_sd": spec.reported_sd,
            "fit_rows": len(fit_rows),
            "holdout_rows": len(holdout_rows),
        }
    return {
        "available": True,
        "protocol": dict(protocol.__dict__),
        "rows": per_baseline,
        "comparator": COMPARATOR_ID,
        "holdout": "internal validation split of the training resource",
    }


def _train_variant(context: RunContext, variant: AblationVariant, config: RunConfig) -> TwinDyn:
    model, active = build_variant_model(model_config(config), context.build.axis.index(), variant)
    set_seed(config.seed, deterministic=config.deterministic)
    source = PairSource(
        max_pairs_per_resource=context.pair_source.max_pairs_per_resource,
        clinical_only=variant.clinical_only_alignment,
    )
    train_representation(
        model=model,
        dataset=context.representation_dataset,
        pair_source=source,
        optim_config=config.optim,
        objective_config=config.objective,
        seed=config.seed,
        active=active,
        mask_ratio=config.masking.ratio,
        device=context.device,
    )
    train_readout(
        model=model,
        dataset=context.readout_dataset,
        optim_config=config.optim,
        seed=config.seed,
        device=context.device,
    )
    return model


def _mean_external(context: RunContext, model: TwinDyn) -> float:
    values: list[float] = []
    for resource in context.external_datasets:
        rows, time, event = external_arrays(context, resource)
        if len(rows) < 2:
            continue
        scores = score(model, context.external_datasets[resource], context.device)
        values.append(concordance_index(scores["risk"], time, event).c_index)
    return float(np.mean(values)) if values else float("nan")


def run_ablation(context: RunContext) -> AblationReport:
    """Table 2: substitute every component one at a time."""
    means: dict[str, float] = {}
    parameters: dict[str, float] = {}
    for variant in ABLATION_VARIANTS:
        model = _train_variant(context, variant, context.config)
        means[variant.name] = _mean_external(context, model)
        parameters[variant.name] = capacity_report(model)["total_millions"]
    return assemble_ablation(means, parameters)


def run_generalisation(context: RunContext) -> dict[str, object]:
    """Table 3: strata and modality dropout on the model fitted for the run."""
    config = context.config
    model = _train_variant(context, _full_variant(), config)
    pooled_risk: list[np.ndarray] = []
    pooled_time: list[np.ndarray] = []
    pooled_event: list[np.ndarray] = []
    pooled_stage: list[np.ndarray] = []
    pooled_grade: list[np.ndarray] = []
    pooled_rows: list[int] = []
    dropout_risk: dict[str, list[np.ndarray]] = {"morphology": [], "compartments": []}
    for resource, dataset in context.external_datasets.items():
        rows, time, event = external_arrays(context, resource)
        if len(rows) < 2:
            continue
        scores = score(model, dataset, context.device)
        pooled_risk.append(scores["risk"])
        pooled_time.append(time)
        pooled_event.append(event)
        pooled_stage.append(dataset.view.stages.numpy().astype(np.int64))
        pooled_grade.append(dataset.view.grades.numpy().astype(np.int64))
        pooled_rows.extend(rows)
        for family in dropout_risk:
            dropped = modality_dropout(
                dataset.raw_observations, context.table.compartment_count, family
            )
            dropped_dataset = AxisDataset(
                context.table, context.build.axis, dataset.indices, context.standardiser, dropped
            )
            dropout_scores = score(model, dropped_dataset, context.device)
            dropout_risk[family].append(dropout_scores["risk"])
    if not pooled_risk:
        return {"available": False, "reason": "no labelled external samples"}
    risk = np.concatenate(pooled_risk)
    time = np.concatenate(pooled_time)
    event = np.concatenate(pooled_event)
    stage = np.concatenate(pooled_stage)
    grade = np.concatenate(pooled_grade)
    full = full_row(risk, time, event)
    stage_rows = stratum_rows(risk, time, event, stage, STAGE_LABELS, STAGE_DEFINITIONS)
    grade_rows = stratum_rows(risk, time, event, grade, GRADE_LABELS, GRADE_DEFINITIONS)
    dropout_summary = dropout_rows(
        risk,
        {family: np.concatenate(values) for family, values in dropout_risk.items() if values},
        time,
        event,
    )
    covariate_risk = _covariate_risk(context, pooled_rows)
    gains = (
        within_stratum_gain(risk, covariate_risk, time, event, stage, STAGE_LABELS)
        if covariate_risk is not None
        else {}
    )
    return {
        "available": True,
        "full": full.as_dict(),
        "stage": [row.as_dict() for row in stage_rows],
        "grade": [row.as_dict() for row in grade_rows],
        "dropout": [row.as_dict() for row in dropout_summary],
        "stage_monotonicity": monotonicity({row.stratum: row.c_index for row in stage_rows}),
        "grade_monotonicity": monotonicity({row.stratum: row.c_index for row in grade_rows}),
        "within_stratum_gains": gains,
    }


def _full_variant() -> AblationVariant:
    return AblationVariant("full", "reference")


def _covariate_risk(context: RunContext, pooled_rows: list[int]) -> np.ndarray | None:
    """A stage-and-grade-only risk model, scored on exactly the pooled external rows."""
    from twindyn.models.baselines.classical import CoxClinical

    config = context.config
    training_rows = labelled_rows(context, config.data.training_resource)
    if len(training_rows) < 4:
        return None
    covariate = feature_view(context.table, "clinical")
    arrays = labelled_arrays(context, training_rows)
    protocol = BaselineProtocol(
        epochs=config.optim.representation_epochs,
        learning_rate=config.optim.learning_rate,
        seed=config.seed,
    )
    estimator = CoxClinical()
    fit = run_baseline(
        estimator, covariate[training_rows], arrays["time"], arrays["event"], protocol
    )
    del fit
    return estimator.predict_risk(covariate[pooled_rows])


def run_mechanism(context: RunContext) -> dict[str, object]:
    """Table 5: the probe and perturbation arms."""
    config = context.config
    model = _train_variant(context, _full_variant(), config)
    report = collect_mechanism(
        model,
        context.representation_dataset,
        config.evaluation,
        PERTURBATION_POSITIONS,
        PERTURBATION_SCALES,
        context.device,
    )
    payload = report.as_dict()
    payload["summary"] = summary_arrays(report)
    payload["available"] = True
    return payload


def run_external(
    context: RunContext,
    outcomes: list[InitialisationOutcome],
) -> dict[str, object]:
    """Table 6: the external arm against the strongest static composition baseline."""
    method_risk: dict[str, np.ndarray] = {}
    times: dict[str, np.ndarray] = {}
    events: dict[str, np.ndarray] = {}
    for resource in EXTERNAL_RESOURCES:
        rows, time, event = external_arrays(context, resource)
        if len(rows) < 2:
            continue
        stacked = [entry.risk[resource] for entry in outcomes if resource in entry.risk]
        if not stacked:
            continue
        method_risk[resource] = np.mean(np.stack(stacked), axis=0)
        times[resource] = time
        events[resource] = event
    if not method_risk:
        return {"available": False, "reason": "no labelled external cohorts"}
    comparator = _comparator_external(context, times)
    if comparator is None:
        return {"available": False, "reason": "the strongest static baseline could not be fitted"}
    training_values = [entry.training_c_index for entry in outcomes]
    report = evaluate_external(
        method_risk=method_risk,
        comparator_risk=comparator["risk"],
        times=times,
        events=events,
        cohort_labels={key: RESOURCES[key].cohort_label for key in method_risk},
        training_value=float(np.mean(training_values)),
        comparator_training_value=comparator["training"],
        config=context.config.evaluation,
    )
    payload = report.as_dict()
    payload["available"] = True
    payload["comparator_view"] = comparator["view"]
    payload["comparator_training_value"] = comparator["training"]
    return payload


def _comparator_external(
    context: RunContext, times: dict[str, np.ndarray]
) -> dict[str, object] | None:
    """Fit the strongest static composition baseline and score every cohort once."""
    config = context.config
    rows = labelled_rows(context, config.data.training_resource)
    if len(rows) < 4:
        return None
    composition = feature_view(context.table, "composition")
    arrays = labelled_arrays(context, rows)
    protocol = BaselineProtocol(
        epochs=config.optim.representation_epochs,
        learning_rate=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
        batch_size=config.optim.batch_size,
        seed=config.seed,
    )
    estimator = build_baseline(
        COMPARATOR_ID, composition.shape[1], 4, token_dim=context.build.axis.position_dim
    )
    run_baseline(estimator, composition[rows], arrays["time"], arrays["event"], protocol)
    risk = {
        resource: estimator.predict_risk(composition[labelled_rows(context, resource)])
        for resource in times
    }
    training_score = concordance_index(
        estimator.predict_risk(composition[rows]), arrays["time"], arrays["event"]
    ).c_index
    return {"risk": risk, "training": training_score, "view": BASELINES[COMPARATOR_ID].method}


def run_label_efficiency(context: RunContext) -> dict[str, object]:
    """Sec. 4.10: the margin over the strongest static baseline at each label fraction."""
    config = context.config
    composition = feature_view(context.table, "composition")
    per_fraction: list[dict[str, object]] = []
    method_values: dict[float, float] = {}
    comparator_values: dict[float, float] = {}
    for fraction in config.evaluation.label_fractions:
        subset_rows = list(
            label_fraction_subset(context.table.samples, context.fit, fraction, config.seed)
        )
        if len(subset_rows) < 4:
            continue
        withheld = set(context.fit) - set(subset_rows)
        masked_table = mask_labels(context.table, withheld)
        model = TwinDyn(model_config(config), context.build.axis.index())
        train_representation(
            model,
            context.representation_dataset,
            context.pair_source,
            config.optim,
            config.objective,
            config.seed,
            mask_ratio=config.masking.ratio,
            device=context.device,
        )
        train_readout(
            model,
            AxisDataset(masked_table, context.build.axis, context.fit, context.standardiser),
            config.optim,
            config.seed,
            device=context.device,
        )
        method_values[fraction] = _mean_external(context, model)
        arrays = labelled_arrays(context, subset_rows)
        protocol = BaselineProtocol(epochs=config.optim.representation_epochs, seed=config.seed)
        estimator = build_baseline(
            COMPARATOR_ID, composition.shape[1], 4, token_dim=context.build.axis.position_dim
        )
        run_baseline(estimator, composition[subset_rows], arrays["time"], arrays["event"], protocol)
        comparator_values[fraction] = _mean_external_comparator(context, estimator, composition)
        difference = method_values[fraction] - comparator_values[fraction]
        per_fraction.append(
            {
                "fraction": fraction,
                "labelled_samples": len(subset_rows),
                "method": method_values[fraction],
                "comparator": comparator_values[fraction],
                "difference": difference,
                "verdict": decide(
                    difference,
                    config.evaluation.decision_margin,
                    config.evaluation.external_threshold,
                ).verdict.value,
            }
        )
    points = efficiency_curve(method_values, comparator_values, config.evaluation.decision_margin)
    return {
        "available": bool(per_fraction),
        "fractions": per_fraction,
        "margins": margin_curve(points),
        "shrinkage": shrinkage_profile(points),
        "grid": list(config.evaluation.label_fractions),
    }


def _mean_external_comparator(
    context: RunContext, estimator: RiskEstimator, composition: np.ndarray
) -> float:
    values: list[float] = []
    for resource in context.external_datasets:
        rows, time, event = external_arrays(context, resource)
        if len(rows) < 2:
            continue
        risk = estimator.predict_risk(composition[rows])
        values.append(concordance_index(risk, time, event).c_index)
    return float(np.mean(values)) if values else float("nan")


def assemble_tables(
    summary: dict[str, object],
    comparators: dict[str, object],
    ablation: AblationReport,
    generalisation: dict[str, object],
    mechanism: dict[str, object],
    efficiency: dict[str, object],
    external: dict[str, object],
) -> dict[str, object]:
    """Compare every produced quantity against its transcribed reported counterpart."""
    produced_rows = {
        identifier: float(entry["c_index"])
        for identifier, entry in (comparators.get("rows", {}) or {}).items()
    }
    produced_rows["TwinDyn"] = float(summary.get("external_mean", float("nan")))
    generalisation_values: dict[str, float] = {}
    for key in ("stage", "grade", "dropout"):
        for row in generalisation.get(key, []) or []:
            generalisation_values[str(row["stratum"])] = float(row["c_index"])
    if generalisation.get("full"):
        generalisation_values["full"] = float(generalisation["full"]["c_index"])
    ablation_deltas = {variant: float(entry) for variant, entry in (ablation.drops or {}).items()}
    mechanism_values = {
        key: float(value)
        for key, value in (mechanism.get("summary", {}) or {}).items()
        if isinstance(value, (int, float))
    }
    bundle = ReportingBundle(
        tables=[
            compare_main_comparison(produced_rows),
            compare_ablation(ablation_deltas),
            compare_generalisation(generalisation_values),
            compare_state_properties(state_properties(summary)),
            compare_mechanism(mechanism_values),
            compare_label_efficiency(
                {
                    float(entry["fraction"]): float(entry["difference"])
                    for entry in efficiency.get("fractions", []) or []
                }
            ),
            compare_external(external.get("cohorts", {}) if isinstance(external, dict) else {}),
        ],
        environment_notes=[
            "Values are produced only from executions that ran in this environment."
        ],
    )
    return bundle.as_dict()


def write_run(result: PipelineResult, output_dir: str | Path) -> list[Path]:
    """Write a run's artefacts, with each JSON payload written atomically."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        atomic_write_json(directory / "run.json", result.as_dict()),
        atomic_write_json(directory / "tables.json", result.tables),
        atomic_write_json(
            directory / "initialisations.json",
            [entry.as_dict() for entry in result.initialisations],
        ),
    ]
