"""Mixed-precision and exponential moving average helpers.

Ref: Sec. 4.4 (the optimisation protocol), Sec. 3.5 (the read-out follows the
frozen representation, so an EMA of the representation is carried across the two
stages when the configuration asks for one).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


def precision_context(
    precision: str, device: torch.device
) -> contextlib.AbstractContextManager[object]:
    """The autocast context for the requested precision."""
    if precision in ("fp32", "float32", "none"):
        return contextlib.nullcontext()
    if precision in ("bf16", "bfloat16"):
        return torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        )
    if precision in ("fp16", "float16", "mixed"):
        return torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        )
    raise ValueError(f"unknown precision {precision!r}")


def build_grad_scaler(precision: str, device: torch.device) -> torch.amp.GradScaler | None:
    """A gradient scaler is only meaningful for fp16 on an accelerator."""
    if precision not in ("fp16", "float16", "mixed"):
        return None
    if device.type != "cuda":
        return None
    return torch.amp.GradScaler(device.type, enabled=True)


@dataclass
class ExponentialMovingAverage:
    """A shadow copy of the trainable parameters."""

    decay: float
    shadow: dict[str, Tensor]
    steps: int = 0

    @classmethod
    def from_model(cls, model: nn.Module, decay: float) -> ExponentialMovingAverage:
        shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        return cls(decay=float(decay), shadow=shadow, steps=0)

    def update(self, model: nn.Module) -> None:
        if not 0.0 < self.decay < 1.0:
            raise ValueError("the decay must lie strictly between zero and one")
        self.steps += 1
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    parameter.copy_(self.shadow[name])

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "steps": self.steps, "shadow": dict(self.shadow)}

    def load_state_dict(self, payload: dict[str, object]) -> None:
        self.decay = float(payload["decay"])  # type: ignore[arg-type]
        self.steps = int(cast(int, payload["steps"]))
        shadow = payload["shadow"]
        if not isinstance(shadow, dict):
            raise TypeError("the EMA shadow must be a mapping")
        self.shadow = {str(key): value for key, value in shadow.items()}
