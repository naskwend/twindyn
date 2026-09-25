"""Shared command-line plumbing.

Ref: Sec. 4.4 (a single schedule and an equal budget for every variant, so the
entry points differ only in which stage of the study they run).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from twindyn.utils.config import RunConfig, resolve_config
from twindyn.utils.runtime import get_logger

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
LOGGER = get_logger("cli")


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="directory holding the model, data, train and experiment configurations",
    )
    parser.add_argument("--experiment", default="main", help="experiment configuration to compose")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a configuration entry; repeatable",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="where run artefacts are written"
    )
    parser.add_argument("--verbose", action="store_true", help="log each training step")
    return parser


def compose(parser: argparse.ArgumentParser, args: argparse.Namespace) -> RunConfig:
    try:
        return resolve_config(args.config_dir, args.experiment, args.set)
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))
        raise


def report_failure(program: str, error: BaseException) -> int:
    LOGGER.error("%s failed: %s", program, error)
    return 2


def format_sequence(values: Sequence[float]) -> str:
    return ", ".join(f"{value:.4f}" for value in values)
