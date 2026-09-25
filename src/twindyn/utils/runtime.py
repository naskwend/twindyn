"""Logging, deterministic seeding helpers and small numeric utilities.

Ref: Sec. 4.4 (ten random initialisations indexed and predetermined).
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

LOGGER_NAME = "twindyn"


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every generator the pipeline touches and return nothing."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def initialisation_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    """Predetermined, indexed initialisation seeds for the paired protocol."""
    generator = np.random.default_rng(base_seed)
    return tuple(int(s) for s in generator.integers(0, 2**31 - 1, size=count))


@dataclass(frozen=True)
class Summary:
    """Accumulates running means without keeping the samples."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, weight: int = 1) -> Summary:
        return Summary(self.total + float(value) * weight, self.count + weight)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def to_builtin(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value
