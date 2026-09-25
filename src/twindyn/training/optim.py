"""Optimiser construction, learning-rate schedules and gradient accumulation.

Ref: Sec. 4.4 (optimisation uses a single schedule for all variants and all
variants have the same tuning budget), Sec. 3.5 (the representation is learned
once per initialisation and the read-out is fitted afterwards, so the two stages
receive separate parameter groups).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from twindyn.utils.config import OptimConfig


@dataclass(frozen=True)
class OptimiserBundle:
    optimiser: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    accumulation_steps: int


class WarmupCosineSchedule:
    """Linear warmup followed by a cosine decay to a floor."""

    def __init__(
        self,
        total_steps: int,
        warmup_steps: int,
        base_lr: float,
        minimum_ratio: float = 0.02,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total steps must be positive")
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, int(warmup_steps))
        self.base_lr = float(base_lr)
        self.minimum_ratio = float(minimum_ratio)

    def scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return max(1e-8, (step + 1) / self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine

    def learning_rate(self, step: int) -> float:
        return self.base_lr * self.scale(step)


class LinearWarmupConstantSchedule(WarmupCosineSchedule):
    """Linear warmup, then a constant rate."""

    def scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return max(1e-8, (step + 1) / self.warmup_steps)
        return 1.0


class CosineSchedule(WarmupCosineSchedule):
    """Cosine decay with no warmup phase."""

    def __init__(self, total_steps: int, base_lr: float, minimum_ratio: float = 0.02) -> None:
        super().__init__(total_steps, 0, base_lr, minimum_ratio)


SCHEDULE_NAMES: tuple[str, ...] = ("cosine", "warmup_cosine", "warmup_constant", "constant")


class LambdaScheduleAdapter:
    """Exposes a schedule object as a learning-rate lambda for torch's scheduler."""

    def __init__(self, schedule: WarmupCosineSchedule, base_lr: float) -> None:
        self.schedule = schedule
        self.base_lr = float(base_lr)
        self.step_count = 0

    def __call__(self, step: int) -> float:
        self.step_count = step
        return self.schedule.scale(step)

    def learning_rate(self) -> float:
        return self.schedule.learning_rate(self.step_count)


def build_schedule(
    name: str, total_steps: int, warmup_steps: int, base_lr: float
) -> WarmupCosineSchedule:
    if name == "cosine":
        return CosineSchedule(total_steps, base_lr)
    if name in ("warmup_constant", "constant"):
        return LinearWarmupConstantSchedule(total_steps, warmup_steps, base_lr)
    if name == "warmup_cosine":
        return WarmupCosineSchedule(total_steps, warmup_steps, base_lr)
    raise ValueError(f"unknown schedule {name!r}; expected one of {SCHEDULE_NAMES}")


def build_optimiser(
    parameters: list[nn.Parameter],
    config: OptimConfig,
    total_steps: int,
    learning_rate: float | None = None,
) -> OptimiserBundle:
    """AdamW with a warmup schedule and the declared gradient accumulation."""
    if not parameters:
        raise ValueError("no parameters to optimise")
    base_lr = float(config.learning_rate if learning_rate is None else learning_rate)
    if config.optimizer not in ("adamw", "adam", "sgd"):
        raise ValueError(f"unknown optimiser {config.optimizer!r}")
    if config.optimizer == "adamw":
        optimiser: torch.optim.Optimizer = torch.optim.AdamW(
            parameters, lr=base_lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "adam":
        optimiser = torch.optim.Adam(parameters, lr=base_lr, weight_decay=config.weight_decay)
    else:
        optimiser = torch.optim.SGD(
            parameters, lr=base_lr, momentum=0.9, weight_decay=config.weight_decay
        )
    schedule = build_schedule(config.scheduler, total_steps, config.warmup_steps, base_lr)
    adapter = LambdaScheduleAdapter(schedule, base_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=adapter)
    return OptimiserBundle(
        optimiser=optimiser,
        scheduler=scheduler,
        accumulation_steps=max(1, int(config.grad_accum)),
    )


def effective_batch_size(config: OptimConfig) -> int:
    """Batch times accumulation times world size, the quantity the schedule sees."""
    return int(config.batch_size) * max(1, int(config.grad_accum)) * max(1, int(config.world_size))


def parameter_groups(
    model: nn.Module, readout_learning_rate: float | None = None
) -> list[dict[str, object]]:
    """Two groups so the read-out can carry its own rate."""
    if not hasattr(model, "frozen_representation_parameters"):
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]
    representation_reader = cast(
        Callable[[], list[nn.Parameter]], model.frozen_representation_parameters
    )
    readout_reader = cast(Callable[[], list[nn.Parameter]], model.readout_parameters)
    representation = [p for p in representation_reader() if p.requires_grad]
    readout = [p for p in readout_reader() if p.requires_grad]
    groups: list[dict[str, object]] = [{"params": representation}]
    if readout:
        entry: dict[str, object] = {"params": readout}
        if readout_learning_rate is not None:
            entry["lr"] = float(readout_learning_rate)
        groups.append(entry)
    return [group for group in groups if group["params"]]


def trainable_parameter_count(model: nn.Module) -> int:
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
