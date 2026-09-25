"""Ablation entry point.

Ref: Table 2 (every component substituted one at a time), Sec. 4.6 (the smallest
and largest drops and the equal-capacity control).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from twindyn.cli.common import base_parser, compose, report_failure
from twindyn.pipeline import build_context, run_ablation
from twindyn.utils.io import atomic_write_json
from twindyn.utils.runtime import get_logger

LOGGER = get_logger("cli.ablation")


def build_parser() -> argparse.ArgumentParser:
    return base_parser("Substitute each component and write the ablation table")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = compose(parser, args)
        context = build_context(config)
        report = run_ablation(context)
        destination = Path(args.output_dir or config.output_dir) / "ablation.json"
        atomic_write_json(destination, report.as_dict())
    except Exception as error:
        return report_failure("ablation", error)
    for row in report.rows:
        LOGGER.info(
            "%-42s mean %.4f delta %+.4f params %.2fM",
            row.variant,
            row.mean_external,
            row.delta,
            row.parameters_millions,
        )
    ratio = report.ratio
    if isinstance(ratio.get("ratio"), float) and math.isfinite(ratio["ratio"]):
        LOGGER.info("largest/smallest drop ratio %.2f", ratio["ratio"])
    LOGGER.info("wrote %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
