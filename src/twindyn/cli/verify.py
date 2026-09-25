"""Verification entry point: both passes and the root artefacts.

Ref: the release contract, not the manuscript. The command writes
``claim_to_code.json``, ``verification_report.json``, ``verification_summary.txt``,
``dataset_urls.txt`` and, last of all, ``integrity_manifest.json`` so that the
manifest describes the finished tree.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from twindyn.utils.runtime import get_logger
from twindyn.verification import write_artefacts

LOGGER = get_logger("cli.verify")

DEFAULT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the claim mapping and the execution checks and write the root artefacts"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="release root the artefacts describe",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="directory for the temporary checkpoints the checks write",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the summary line",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a directory")
    try:
        outcome = write_artefacts(root, args.scratch)
    except Exception as error:
        LOGGER.error("verify failed: %s", error)
        return 2
    summary = outcome["summary"]
    counts = summary["counts"]  # type: ignore[index]
    LOGGER.info(
        "checks %d: PASS %d, FAIL %d, NOT_RUN %d, BLOCKED %d",
        summary["total"],  # type: ignore[index]
        counts["PASS"],
        counts["FAIL"],
        counts["NOT_RUN"],
        counts["BLOCKED"],
    )
    LOGGER.info("overall status: %s", outcome["status"])
    LOGGER.info("claims mapped: %d", outcome["claim_count"])
    if outcome["missing_code"]:
        LOGGER.warning("claim references with no file: %s", outcome["missing_code"])
    if not args.quiet:
        LOGGER.info("elapsed %.1f s", outcome["elapsed"])
        LOGGER.info("manifest digest %s", outcome["manifest_digest"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
