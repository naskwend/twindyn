"""The two-stage training engine.

Ref: Algorithm 1 (representation learning, no survival label is used), Algorithm 2
(inference and trajectory read-out), Sec. 3.5 (the read-out uses the conventional
survival objective and is the only place a label enters), Sec. 4.4 (a single
optimisation schedule and an equal tuning budget for every variant).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from twindyn.data.dataset import AxisDataset, collate
from twindyn.data.masking import sample_mask
from twindyn.data.pairs import batch_alignment_pairs
from twindyn.losses.alignment import cross_resource_alignment_loss
from twindyn.losses.composite import ObjectiveWeights, objective_from_masking
from twindyn.losses.survival import cox_partial_likelihood
from twindyn.models.twindyn import TwinDyn
from twindyn.training.optim import build_optimiser, trainable_parameter_count
from twindyn.training.precision import (
    ExponentialMovingAverage,
    build_grad_scaler,
    precision_context,
)
from twindyn.utils.config import ObjectiveConfig, OptimConfig
from twindyn.utils.runtime import Summary, get_logger

LOGGER = get_logger("engine")


@dataclass
class StageReport:
    stage: str
    epochs: int
    steps: int
    initial_loss: float
    final_loss: float
    history: list[float] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    parameters: int = 0

    @property
    def loss_decreased(self) -> bool:
        return self.final_loss < self.initial_loss

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "epochs": self.epochs,
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_decreased": self.loss_decreased,
            "components": dict(self.components),
            "parameters": self.parameters,
        }


@dataclass
class TrainingReport:
    representation: StageReport
    readout: StageReport
    objective_weights: dict[str, float]
    alignment_initial: dict[str, float]
    alignment_final: dict[str, float]
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "representation": self.representation.as_dict(),
            "readout": self.readout.as_dict(),
            "objective_weights": dict(self.objective_weights),
            "alignment_initial": dict(self.alignment_initial),
            "alignment_final": dict(self.alignment_final),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PairSource:
    """Where the alignment matching reads its anchors from.

    Matching runs inside each batch, so the anchors and the resource labels travel
    with the dataset rather than being precomputed once for the whole cohort.
    """

    max_pairs_per_resource: int = 4096
    min_similarity: float = 0.0
    clinical_only: bool = False


def _loader(dataset: AxisDataset, config: OptimConfig, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(int(config.batch_size), max(1, len(dataset))),
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
        drop_last=False,
    )


def train_representation(
    model: TwinDyn,
    dataset: AxisDataset,
    pair_source: PairSource,
    optim_config: OptimConfig,
    objective_config: ObjectiveConfig,
    seed: int,
    active: frozenset[str] = frozenset({"mask", "align", "trajectory"}),
    mask_ratio: float = 0.30,
    device: torch.device | None = None,
    ema_decay: float = 0.0,
) -> tuple[StageReport, dict[str, float], dict[str, float]]:
    """Algorithm 1: fit MSI and SST on the label-free objective."""
    device = device or torch.device("cpu")
    model.to(device)
    loader = _loader(dataset, optim_config, shuffle=True, seed=seed)
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * max(1, optim_config.representation_epochs)
    bundle = build_optimiser(
        [p for p in model.frozen_representation_parameters() if p.requires_grad],
        optim_config,
        total_steps,
    )
    scaler = build_grad_scaler(optim_config.precision, device)
    weights = ObjectiveWeights(
        lambda_mask=objective_config.lambda_mask,
        lambda_align=objective_config.lambda_align,
        lambda_trajectory=objective_config.lambda_trajectory,
        trajectory_temperature=objective_config.trajectory_temperature,
    )
    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []
    components: dict[str, Summary] = {
        "reconstruction": Summary(),
        "alignment": Summary(),
        "trajectory": Summary(),
    }
    step = 0
    alignment_initial = _alignment_snapshot(model, dataset, pair_source, device)
    ema = ExponentialMovingAverage.from_model(model, ema_decay) if 0.0 < ema_decay < 1.0 else None
    for _ in range(max(1, optim_config.representation_epochs)):
        epoch_losses: list[float] = []
        bundle.optimiser.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            observations = batch["positions"].to(device)
            availability = batch["availability"].to(device)
            target = batch["target"].to(device)
            rows = dataset.rows_of([str(identifier) for identifier in batch["sample_id"]])
            masking = sample_mask(
                dataset.axis,
                dataset.view.availability[rows].to(device),
                mask_ratio,
                generator,
            )
            pairs = batch_alignment_pairs(
                dataset.anchors[rows].to(device),
                batch["resource"].to(device),
                pair_source.max_pairs_per_resource,
                pair_source.min_similarity,
                pair_source.clinical_only,
            )
            with precision_context(optim_config.precision, device):
                output = model.encode_representation(observations, availability)
                breakdown = objective_from_masking(
                    reconstruction=output.reconstructed,
                    target=target,
                    masking=masking,
                    states=output.states,
                    pairs=pairs,
                    validity=availability.any(dim=-1).transpose(0, 1),
                    weights=weights,
                    active=active,
                )
                loss = breakdown.total / bundle.accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_index + 1) % bundle.accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(bundle.optimiser)
                if optim_config.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.frozen_representation_parameters() if p.requires_grad],
                        optim_config.grad_clip,
                    )
                if scaler is not None:
                    scaler.step(bundle.optimiser)
                    scaler.update()
                else:
                    bundle.optimiser.step()
                bundle.optimiser.zero_grad(set_to_none=True)
                if bundle.scheduler is not None:
                    bundle.scheduler.step()
            if ema is not None:
                ema.update(model)
            observed = breakdown.scalars()
            epoch_losses.append(observed["total"])
            for key in components:
                components[key] = components[key].update(observed[key])
            step += 1
            if step % 50 == 0:
                LOGGER.info("representation step %d loss %.5f", step, observed["total"])
        history.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
    alignment_final = _alignment_snapshot(model, dataset, pair_source, device)
    report = StageReport(
        stage="representation",
        epochs=max(1, optim_config.representation_epochs),
        steps=step,
        initial_loss=history[0] if history else float("nan"),
        final_loss=history[-1] if history else float("nan"),
        history=history,
        components={key: value.mean for key, value in components.items()},
        parameters=trainable_parameter_count(model),
    )
    return report, alignment_initial, alignment_final


def train_readout(
    model: TwinDyn,
    dataset: AxisDataset,
    optim_config: OptimConfig,
    seed: int,
    device: torch.device | None = None,
    freeze_representation: bool = True,
) -> StageReport:
    """Algorithm 2: fit the shallow read-out on the internal validation split."""
    device = device or torch.device("cpu")
    model.to(device)
    if freeze_representation:
        model.freeze_representation()
    else:
        model.unfreeze_representation()
    loader = _loader(dataset, optim_config, shuffle=True, seed=seed + 1)
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * max(1, optim_config.readout_epochs)
    bundle = build_optimiser(
        [p for p in model.readout_parameters() if p.requires_grad],
        optim_config,
        total_steps,
        learning_rate=optim_config.readout_learning_rate,
    )
    scaler = build_grad_scaler(optim_config.precision, device)
    history: list[float] = []
    step = 0
    for _ in range(max(1, optim_config.readout_epochs)):
        epoch_losses: list[float] = []
        bundle.optimiser.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            observations = batch["positions"].to(device)
            availability = batch["availability"].to(device)
            time = batch["time"].to(device)
            event = batch["event"].to(device)
            labelled = batch["has_label"].to(device)
            if int(labelled.sum()) < 2 or float(event[labelled].sum()) == 0.0:
                continue
            with precision_context(optim_config.precision, device):
                output = model(observations, availability)
                loss = (
                    cox_partial_likelihood(
                        output.risk[labelled], time[labelled], event[labelled]
                    ).loss
                    / bundle.accumulation_steps
                )
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_index + 1) % bundle.accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(bundle.optimiser)
                if optim_config.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.readout_parameters() if p.requires_grad],
                        optim_config.grad_clip,
                    )
                if scaler is not None:
                    scaler.step(bundle.optimiser)
                    scaler.update()
                else:
                    bundle.optimiser.step()
                bundle.optimiser.zero_grad(set_to_none=True)
                if bundle.scheduler is not None:
                    bundle.scheduler.step()
            epoch_losses.append(float(loss.detach()) * bundle.accumulation_steps)
            step += 1
        history.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
    return StageReport(
        stage="readout",
        epochs=max(1, optim_config.readout_epochs),
        steps=step,
        initial_loss=history[0] if history else float("nan"),
        final_loss=history[-1] if history else float("nan"),
        history=history,
        parameters=trainable_parameter_count(model),
    )


def _alignment_snapshot(
    model: TwinDyn,
    dataset: AxisDataset,
    pair_source: PairSource,
    device: torch.device,
) -> dict[str, float]:
    """Equation (5) magnitude per resource pair, on the sample-level state.

    The snapshot is taken over the whole cohort rather than over a batch, which is
    the quantity Table 4 reports relative to its value at initialisation.
    """
    pairs = batch_alignment_pairs(
        dataset.anchors,
        dataset.view.resources,
        pair_source.max_pairs_per_resource,
        pair_source.min_similarity,
        pair_source.clinical_only,
    )
    model.eval()
    with torch.no_grad():
        states = model.encode_representation(
            dataset.positions.to(device), dataset.availability.to(device)
        ).states
    model.train()
    breakdown = cross_resource_alignment_loss(states.mean(dim=0), pairs)
    return dict(breakdown.per_pair)


def train_twindyn(
    model: TwinDyn,
    representation_dataset: AxisDataset,
    readout_dataset: AxisDataset,
    pair_source: PairSource,
    optim_config: OptimConfig,
    objective_config: ObjectiveConfig,
    seed: int,
    active: frozenset[str] = frozenset({"mask", "align", "trajectory"}),
    mask_ratio: float = 0.30,
    device: torch.device | None = None,
    ema_decay: float = 0.0,
) -> TrainingReport:
    """Both stages in order, returning what each produced."""
    representation, alignment_initial, alignment_final = train_representation(
        model=model,
        dataset=representation_dataset,
        pair_source=pair_source,
        optim_config=optim_config,
        objective_config=objective_config,
        seed=seed,
        active=active,
        mask_ratio=mask_ratio,
        device=device,
        ema_decay=ema_decay,
    )
    readout = train_readout(
        model=model,
        dataset=readout_dataset,
        optim_config=optim_config,
        seed=seed,
        device=device,
    )
    return TrainingReport(
        representation=representation,
        readout=readout,
        objective_weights=ObjectiveWeights(
            lambda_mask=objective_config.lambda_mask,
            lambda_align=objective_config.lambda_align,
            lambda_trajectory=objective_config.lambda_trajectory,
            trajectory_temperature=objective_config.trajectory_temperature,
        ).as_dict(),
        alignment_initial=alignment_initial,
        alignment_final=alignment_final,
        seed=seed,
    )


def score(
    model: TwinDyn, dataset: AxisDataset, device: torch.device | None = None
) -> dict[str, np.ndarray]:
    """Risk scores and follow-up for every sample of a dataset view."""
    device = device or torch.device("cpu")
    loader = DataLoader(
        dataset,
        batch_size=max(1, len(dataset)),
        shuffle=False,
        collate_fn=collate,
    )
    model.eval()
    risks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["positions"].to(device), batch["availability"].to(device))
            risks.append(output.risk.detach().cpu().numpy().astype(np.float64))
    model.train()
    return {
        "risk": np.concatenate(risks) if risks else np.zeros(0, dtype=np.float64),
        "time": dataset.view.times.numpy().astype(np.float64),
        "event": dataset.view.events.numpy().astype(np.float64),
        "resource": dataset.view.resources.numpy().astype(np.int64),
        "stage": dataset.view.stages.numpy().astype(np.int64),
        "grade": dataset.view.grades.numpy().astype(np.int64),
        "has_label": dataset.view.has_label.numpy().astype(bool),
    }
