"""Evaluation entry point for a fitted checkpoint.

Ref: Sec. 4.1 (a frozen model scores each external cohort once), Sec. 3.5 (the
read-out is the only consumer of the survival label).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from twindyn.cli.common import base_parser, compose, report_failure
from twindyn.data.dataset import AxisDataset
from twindyn.metrics.concordance import concordance_index
from twindyn.models.twindyn import TwinDyn
from twindyn.pipeline import build_context, external_arrays, model_config
from twindyn.training.checkpoint import load_checkpoint
from twindyn.training.engine import score
from twindyn.utils.io import atomic_write_json
from twindyn.utils.runtime import get_logger

LOGGER = get_logger("cli.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser(
        "Score a fitted checkpoint on the training hold-out and the external cohorts"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="checkpoint written by training"
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        help="restrict the report to these resources; repeatable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = compose(parser, args)
        context = build_context(config)
        model = TwinDyn(model_config(config), context.build.axis.index())
        contents = load_checkpoint(args.checkpoint)
        model.load_state_dict(contents.model_state, strict=True)
        report: dict[str, object] = {
            "checkpoint": str(args.checkpoint),
            "seed": contents.seed,
            "stage": contents.stage,
            "configuration_digest": contents.configuration_digest,
            "cohort_source": context.build.source,
            "resources": {},
        }
    except Exception as error:
        return report_failure("evaluate", error)
    requested = set(args.resource) if args.resource else set(context.external_datasets)
    entries: dict[str, object] = {}
    holdout = AxisDataset(
        context.table, context.build.axis, context.validation, context.standardiser
    )
    holdout_scores = score(model, holdout, context.device)
    labelled = holdout_scores["has_label"]
    entries["training_holdout"] = {
        "samples": int(labelled.sum()),
        "c_index": concordance_index(
            holdout_scores["risk"][labelled],
            holdout_scores["time"][labelled],
            holdout_scores["event"][labelled],
        ).c_index,
    }
    for resource in sorted(requested):
        if resource not in context.external_datasets:
            continue
        rows, time, event = external_arrays(context, resource)
        if len(rows) < 2:
            entries[resource] = {"samples": len(rows), "c_index": float("nan")}
            continue
        scores = score(model, context.external_datasets[resource], context.device)
        entries[resource] = {
            "samples": len(rows),
            "c_index": concordance_index(scores["risk"], time, event).c_index,
            "events": int(np.sum(event)),
        }
    report["resources"] = entries
    destination = Path(args.output_dir or config.output_dir) / "evaluation.json"
    atomic_write_json(destination, report)
    LOGGER.info("wrote %s", destination)
    for name, entry in entries.items():
        LOGGER.info("%s: C-index %.4f on %d samples", name, entry["c_index"], entry["samples"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
