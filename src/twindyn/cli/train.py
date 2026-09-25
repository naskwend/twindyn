"""Training entry point.

Ref: Algorithms 1 and 2, Sec. 4.4 (ten indexed initialisations, each
representation learned once and reused by every read-out variant).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from twindyn.cli.common import base_parser, compose, report_failure
from twindyn.pipeline import run_experiment
from twindyn.utils.runtime import get_logger

LOGGER = get_logger("cli.train")


def build_parser() -> argparse.ArgumentParser:
    return base_parser("Fit the representation and the read-out and write the run artefacts")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = compose(parser, args)
        result = run_experiment(config, args.output_dir or config.output_dir)
    except Exception as error:
        return report_failure("train", error)
    summary = result.summary
    LOGGER.info(
        "initialisations=%d training C-index %.4f external mean %.4f",
        summary["initialisations"],
        summary["training"]["mean"],
        summary["external_mean"],
    )
    LOGGER.info("cohort source: %s", result.cohort_source)
    for note in result.notes:
        LOGGER.info("note: %s", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
