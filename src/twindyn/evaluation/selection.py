"""Internal-validation selection of the quantities the manuscript leaves open.

Ref: Sec. 3.10 and Sec. 4.4 (the only quantities chosen during internal validation
are the latent dimension, the masking ratio and the objective weights, with the
read-out depth stated next to them; random initialisation methods are
predetermined and indexed so representation learning runs once per initialisation
and the read-out reuses it).

Selection is scored on the label-free objective evaluated on the internal
validation split of the training resource, so no outcome enters the choice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from twindyn.data.dataset import AxisDataset, collate
from twindyn.data.masking import sample_mask
from twindyn.data.pairs import batch_alignment_pairs
from twindyn.losses.composite import ObjectiveWeights, objective_from_masking
from twindyn.models.twindyn import TwinDyn
from twindyn.registry import AXIS_LENGTH_GRID, STATE_DIM_GRID
from twindyn.training.engine import PairSource


@dataclass
class SelectionRecord:
    quantity: str
    value: float
    score: float

    def as_dict(self) -> dict[str, object]:
        return {"quantity": self.quantity, "value": self.value, "score": self.score}


@dataclass
class SelectionReport:
    records: list[SelectionRecord] = field(default_factory=list)
    selected: dict[str, float] = field(default_factory=dict)
    grid: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "records": [record.as_dict() for record in self.records],
            "selected": dict(self.selected),
            "grid": {key: list(value) for key, value in self.grid.items()},
        }


def validation_objective(
    model: TwinDyn,
    dataset: AxisDataset,
    pair_source: PairSource,
    weights: ObjectiveWeights,
    mask_ratio: float,
    seed: int,
    active: frozenset[str] = frozenset({"mask", "align", "trajectory"}),
    device: torch.device | None = None,
) -> float:
    """The label-free objective on a held-out split of the training resource."""
    device = device or torch.device("cpu")
    loader = DataLoader(
        dataset,
        batch_size=min(64, max(1, len(dataset))),
        shuffle=False,
        collate_fn=collate,
    )
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            rows = dataset.rows_of([str(identifier) for identifier in batch["sample_id"]])
            masking = sample_mask(
                dataset.axis, dataset.view.availability[rows], mask_ratio, generator
            )
            pairs = batch_alignment_pairs(
                dataset.anchors[rows],
                batch["resource"],
                pair_source.max_pairs_per_resource,
                pair_source.min_similarity,
                pair_source.clinical_only,
            )
            output = model.encode_representation(
                batch["positions"].to(device), batch["availability"].to(device)
            )
            breakdown = objective_from_masking(
                reconstruction=output.reconstructed,
                target=batch["target"].to(device),
                masking=masking,
                states=output.states,
                pairs=pairs,
                validity=batch["availability"].any(dim=-1).transpose(0, 1).to(device),
                weights=weights,
                active=active,
            )
            scores.append(float(breakdown.total))
    model.train()
    return float(np.mean(scores)) if scores else float("nan")


def sweep_values(
    quantity: str,
    values: tuple[float, ...],
    evaluate: Callable[[float], float],
) -> SelectionReport:
    """Score each candidate value and keep the best."""
    if not values:
        raise ValueError("the sweep needs at least one candidate")
    records = [
        SelectionRecord(quantity=quantity, value=float(value), score=float(evaluate(value)))
        for value in values
    ]
    finite = [record for record in records if np.isfinite(record.score)]
    best = min(finite, key=lambda record: record.score) if finite else records[0]
    return SelectionReport(
        records=records,
        selected={quantity: best.value},
        grid={quantity: tuple(float(value) for value in values)},
    )


def merge_reports(reports: list[SelectionReport]) -> SelectionReport:
    records: list[SelectionRecord] = []
    selected: dict[str, float] = {}
    grid: dict[str, tuple[float, ...]] = {}
    for report in reports:
        records.extend(report.records)
        selected.update(report.selected)
        grid.update(report.grid)
    return SelectionReport(records=records, selected=selected, grid=grid)


def declared_grids() -> dict[str, tuple[float, ...]]:
    """The grids the sweeps run over, with the axis and state grids from Table 4."""
    return {
        "tau_max": tuple(float(value) for value in AXIS_LENGTH_GRID),
        "state_dim": tuple(float(value) for value in STATE_DIM_GRID),
        "mask_ratio": (0.10, 0.20, 0.30, 0.40, 0.50),
        "lambda_align": (0.25, 1.0, 4.0, 16.0, 64.0),
        "lambda_mask": (0.25, 1.0, 4.0),
        "lambda_trajectory": (0.25, 1.0, 4.0),
    }
