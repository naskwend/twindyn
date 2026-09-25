"""State-property sweep and internal-validation selection entry point.

Ref: Table 4 (the trajectory length and state dimension sweeps around the selected
configuration, and the shuffled-ordering ablation), Sec. 3.10 and Sec. 4.4 (the
latent dimension, the masking ratio and the objective weights are the quantities
chosen on internal validation).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from twindyn.cli.common import base_parser, compose, report_failure
from twindyn.data.axis import MicroenvironmentAxis
from twindyn.evaluation.selection import (
    declared_grids,
    merge_reports,
    sweep_values,
    validation_objective,
)
from twindyn.evaluation.sweeps import (
    axis_length_sweep,
    plateau_point,
    state_dimension_sweep,
)
from twindyn.losses.composite import ObjectiveWeights
from twindyn.metrics.concordance import concordance_index
from twindyn.models.twindyn import TwinDyn
from twindyn.pipeline import RunContext, build_context, external_arrays, model_config
from twindyn.training.engine import score, train_readout, train_representation
from twindyn.utils.config import RunConfig
from twindyn.utils.io import atomic_write_json
from twindyn.utils.runtime import get_logger, set_seed

LOGGER = get_logger("cli.sweep")


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(
        "Sweep the axis length and the state dimension and select the open quantities"
    )
    parser.add_argument(
        "--quantity",
        action="append",
        choices=("tau_max", "state_dim", "mask_ratio", "weights"),
        default=[],
        help="which sweep to run; default is the two state properties of Table 4",
    )
    return parser


def _external_mean(context: RunContext, model: TwinDyn) -> float:
    values: list[float] = []
    for resource in context.external_datasets:
        rows, time, event = external_arrays(context, resource)
        if len(rows) < 2:
            continue
        scores = score(model, context.external_datasets[resource], context.device)
        values.append(concordance_index(scores["risk"], time, event).c_index)
    return float(sum(values) / len(values)) if values else float("nan")


def _fit_and_score(
    context: RunContext, config: RunConfig, tau_max: int, state_dim: int, aggregation_level: int = 0
) -> float:
    axis = MicroenvironmentAxis.of(
        context.table.width, tau_max, context.table.compartment_count, config.axis.ordering_rule
    )
    if aggregation_level:
        axis = axis.aggregated(aggregation_level)
    from twindyn.data.dataset import AxisDataset

    representation = AxisDataset(
        context.table, axis, context.representation_dataset.indices, context.standardiser
    )
    readout = AxisDataset(context.table, axis, context.fit, context.standardiser)
    externals = {
        resource: AxisDataset(context.table, axis, values, context.standardiser)
        for resource, values in context.partitions.items()
        if resource in context.external_datasets
    }
    model = TwinDyn(replace(model_config(config), state_dim=state_dim), axis.index())
    set_seed(config.seed, deterministic=config.deterministic)
    train_representation(
        model,
        representation,
        context.pair_source,
        config.optim,
        config.objective,
        config.seed,
        mask_ratio=config.masking.ratio,
        device=context.device,
    )
    train_readout(model, readout, config.optim, config.seed, device=context.device)
    values: list[float] = []
    for resource, dataset in externals.items():
        rows, time, event = external_arrays(context, resource)
        if len(rows) < 2:
            continue
        scores = score(model, dataset, context.device)
        values.append(concordance_index(scores["risk"], time, event).c_index)
    return float(sum(values) / len(values)) if values else float("nan")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    quantities = args.quantity or ["tau_max", "state_dim"]
    try:
        config = compose(parser, args)
        context = build_context(config)
        grids = declared_grids()
        payload: dict[str, object] = {"grids": {key: list(value) for key, value in grids.items()}}
        if "tau_max" in quantities:
            values = {
                int(value): _fit_and_score(context, config, int(value), config.model.state_dim)
                for value in grids["tau_max"]
            }
            sweep = axis_length_sweep(values)
            payload["axis_length"] = sweep.as_dict()
            payload["axis_length_plateau"] = plateau_point(sweep.grid, sweep.values)
            LOGGER.info("axis length sweep: %s", values)
        if "state_dim" in quantities:
            values = {
                int(value): _fit_and_score(context, config, config.axis.tau_max, int(value))
                for value in grids["state_dim"]
            }
            sweep = state_dimension_sweep(values)
            payload["state_dimension"] = sweep.as_dict()
            payload["state_dimension_plateau"] = plateau_point(sweep.grid, sweep.values)
            LOGGER.info("state dimension sweep: %s", values)
        if "mask_ratio" in quantities:
            dataset = context.representation_dataset
            axis = context.build.axis
            model = TwinDyn(model_config(config), axis.index())
            weights = ObjectiveWeights(
                lambda_mask=config.objective.lambda_mask,
                lambda_align=config.objective.lambda_align,
                lambda_trajectory=config.objective.lambda_trajectory,
                trajectory_temperature=config.objective.trajectory_temperature,
            )
            report = sweep_values(
                "mask_ratio",
                grids["mask_ratio"],
                lambda value: validation_objective(
                    model,
                    dataset,
                    context.pair_source,
                    weights,
                    value,
                    config.seed,
                    device=context.device,
                ),
            )
            payload["mask_ratio"] = report.as_dict()
            LOGGER.info("mask ratio selection: %s", report.selected)
        if "weights" in quantities:
            dataset = context.representation_dataset
            model = TwinDyn(model_config(config), context.build.axis.index())
            reports = []
            for name in ("lambda_align", "lambda_mask", "lambda_trajectory"):
                values = grids[name]
                selected = sweep_values(
                    name,
                    values,
                    lambda value, key=name: validation_objective(
                        model,
                        dataset,
                        context.pair_source,
                        _weights_with(config, key, value),
                        config.masking.ratio,
                        config.seed,
                        device=context.device,
                    ),
                )
                reports.append(selected)
            merged = merge_reports(reports)
            payload["weights"] = merged.as_dict()
            LOGGER.info("objective weight selection: %s", merged.selected)
        destination = Path(args.output_dir or config.output_dir) / "sweeps.json"
        atomic_write_json(destination, payload)
        LOGGER.info("wrote %s", destination)
    except Exception as error:
        return report_failure("sweep", error)
    return 0


def _weights_with(config: RunConfig, key: str, value: float) -> ObjectiveWeights:
    settings = {
        "lambda_align": config.objective.lambda_align,
        "lambda_mask": config.objective.lambda_mask,
        "lambda_trajectory": config.objective.lambda_trajectory,
    }
    settings[key] = value
    return ObjectiveWeights(
        lambda_mask=settings["lambda_mask"],
        lambda_align=settings["lambda_align"],
        lambda_trajectory=settings["lambda_trajectory"],
        trajectory_temperature=config.objective.trajectory_temperature,
    )


if __name__ == "__main__":
    sys.exit(main())
