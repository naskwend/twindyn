"""Atomic checkpoints that carry the seed and the configuration digest.

Ref: Sec. 4.4 (ten random initialisations are indexed and predetermined, and each
representation is trained once per initialisation and reused by every read-out
variant, so a checkpoint has to be restorable to the same state).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from twindyn.utils.io import tensor_payload_digest

CHECKPOINT_FORMAT = "twindyn-checkpoint-1"


@dataclass(frozen=True)
class CheckpointContents:
    format: str
    seed: int
    configuration_digest: str
    stage: str
    epoch: int
    model_state: dict[str, object]
    optimiser_state: dict[str, object] | None
    scheduler_state: dict[str, object] | None
    metrics: dict[str, float]
    extra: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "seed": self.seed,
            "configuration_digest": self.configuration_digest,
            "stage": self.stage,
            "epoch": self.epoch,
            "metrics": dict(self.metrics),
            "extra": dict(self.extra),
        }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    seed: int,
    configuration_digest: str,
    stage: str,
    epoch: int,
    metrics: dict[str, float],
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write through a temporary file so a crash cannot leave a partial checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "seed": int(seed),
        "configuration_digest": str(configuration_digest),
        "stage": str(stage),
        "epoch": int(epoch),
        "model_state": dict(model.state_dict()),
        "optimiser_state": optimiser.state_dict() if optimiser is not None else None,
        "scheduler_state": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "extra": dict(extra or {}),
    }
    handle, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.")
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        temporary_path.chmod(0o644)
        temporary_path.replace(target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return target


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> CheckpointContents:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    return CheckpointContents(
        format=str(payload["format"]),
        seed=int(payload["seed"]),
        configuration_digest=str(payload["configuration_digest"]),
        stage=str(payload["stage"]),
        epoch=int(payload["epoch"]),
        model_state=dict(payload["model_state"]),
        optimiser_state=payload.get("optimiser_state"),
        scheduler_state=payload.get("scheduler_state"),
        metrics={str(key): float(value) for key, value in dict(payload["metrics"]).items()},
        extra=dict(payload.get("extra", {})),
    )


def restore(
    path: str | Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    strict: bool = True,
) -> CheckpointContents:
    """Load a checkpoint back into the objects it came from."""
    contents = load_checkpoint(path)
    model.load_state_dict(contents.model_state, strict=strict)
    if optimiser is not None and contents.optimiser_state is not None:
        optimiser.load_state_dict(contents.optimiser_state)
    if scheduler is not None and contents.scheduler_state is not None:
        scheduler.load_state_dict(contents.scheduler_state)  # type: ignore[attr-defined]
    return contents


def payload_digest(path: str | Path) -> str:
    """Digest of the tensor payload, which is stable across container metadata changes."""
    contents = load_checkpoint(path)
    return tensor_payload_digest(contents.model_state)


def verify_round_trip(path: str | Path, model: nn.Module) -> dict[str, object]:
    """Re-load a checkpoint and confirm the parameters match bit for bit."""
    before = tensor_payload_digest(dict(model.state_dict()))
    contents = load_checkpoint(path)
    model.load_state_dict(contents.model_state, strict=True)
    after = tensor_payload_digest(dict(model.state_dict()))
    return {
        "identical": before == after,
        "before": before,
        "after": after,
        "seed_restored": contents.seed,
        "stage": contents.stage,
    }
