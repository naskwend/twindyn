"""Torch datasets over the ordered-axis view of a cohort table.

Ref: Sec. 3.1 (equation 1: the observation is the compartment block followed by
the morphology block), Sec. 3.10 (the axis is fixed before training), Sec. 4.4
(the read-out is fitted on the internal validation split of the training
resource).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from twindyn.data.axis import MicroenvironmentAxis
from twindyn.data.resources import EXTERNAL_RESOURCES
from twindyn.data.schema import CohortTable, SampleRecord

RESOURCE_INDEX: dict[str, int] = {
    key: position for position, key in enumerate(("tcga_kirc", *EXTERNAL_RESOURCES))
}


@dataclass(frozen=True)
class FeatureStandardiser:
    """Per-coordinate centring and scaling, fitted on the training split only."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, observations: np.ndarray, availability: np.ndarray) -> FeatureStandardiser:
        if observations.shape != availability.shape:
            raise ValueError("observations and availability must agree in shape")
        weights = availability.astype(np.float64)
        counts = weights.sum(axis=0)
        counts = np.where(counts > 0.0, counts, 1.0)
        mean = (observations * weights).sum(axis=0) / counts
        centred = (observations - mean) * weights
        variance = (centred**2).sum(axis=0) / counts
        scale = np.sqrt(np.maximum(variance, 1e-8))
        return cls(mean=mean, scale=scale)

    def apply(self, observations: np.ndarray) -> np.ndarray:
        return (observations - self.mean) / self.scale


@dataclass(frozen=True)
class DatasetView:
    """A materialised view of one index subset."""

    observations: Tensor
    availability: Tensor
    times: Tensor
    events: Tensor
    has_label: Tensor
    resources: Tensor
    stages: Tensor
    grades: Tensor
    sample_ids: list[str]


def materialise(
    table: CohortTable,
    indices: tuple[int, ...],
    standardiser: FeatureStandardiser | None,
    observations: np.ndarray | None = None,
) -> DatasetView:
    """Build the tensor view of one index subset.

    ``observations`` replaces the table-derived rows when a test-time intervention
    such as a modality dropout has to be evaluated; the availability mask is
    recomputed from it so a withheld family is genuinely absent rather than zero.
    """
    records = [table.samples[index] for index in indices]
    if observations is None:
        raw = np.stack([record.joined_observation(table.width) for record in records])
        availability = np.stack(
            [record.joined_availability(table.width, table.compartment_count) for record in records]
        )
    else:
        raw = np.asarray(observations, dtype=np.float64)
        if raw.shape[0] != len(records) or raw.shape[1] != table.width:
            raise ValueError("the observation override must match the subset and the table width")
        availability = np.stack(
            [record.joined_availability(table.width, table.compartment_count) for record in records]
        ) & (raw != 0.0)
    if standardiser is not None:
        raw = standardiser.apply(raw)
    raw = np.where(availability, raw, 0.0).astype(np.float32)
    times = np.asarray(
        [record.label.time if record.label is not None else 0.0 for record in records],
        dtype=np.float32,
    )
    events = np.asarray(
        [float(record.label.event) if record.label is not None else 0.0 for record in records],
        dtype=np.float32,
    )
    has_label = np.asarray([record.label is not None for record in records], dtype=bool)
    resources = np.asarray(
        [RESOURCE_INDEX.get(record.resource, -1) for record in records], dtype=np.int64
    )
    stages = np.asarray(
        [
            record.stratum.stage if record.stratum and record.stratum.stage else 0
            for record in records
        ],
        dtype=np.int64,
    )
    grades = np.asarray(
        [
            record.stratum.grade if record.stratum and record.stratum.grade else 0
            for record in records
        ],
        dtype=np.int64,
    )
    return DatasetView(
        observations=torch.from_numpy(raw),
        availability=torch.from_numpy(availability),
        times=torch.from_numpy(times),
        events=torch.from_numpy(events),
        has_label=torch.from_numpy(has_label),
        resources=torch.from_numpy(resources),
        stages=torch.from_numpy(stages),
        grades=torch.from_numpy(grades),
        sample_ids=[record.sample_id for record in records],
    )


class AxisDataset(Dataset):
    """Indexes into a cohort table and returns padded per-position tensors."""

    def __init__(
        self,
        table: CohortTable,
        axis: MicroenvironmentAxis,
        indices: tuple[int, ...],
        standardiser: FeatureStandardiser | None = None,
        observations: np.ndarray | None = None,
    ) -> None:
        self.table = table
        self.axis = axis
        self.indices = tuple(int(index) for index in indices)
        self.standardiser = standardiser
        if observations is None:
            self.raw_observations = np.stack(
                [table.samples[index].joined_observation(table.width) for index in self.indices]
            )
        else:
            self.raw_observations = np.asarray(observations, dtype=np.float64)
        self.anchors = torch.from_numpy(
            self.raw_observations[:, : table.compartment_count].astype(np.float32)
        )
        self.view = materialise(table, self.indices, standardiser, observations)
        self.positions = axis.positions_from_observation(self.view.observations)
        mask = axis.position_mask()
        self.availability = self._position_availability(mask)
        self.targets = torch.where(
            self.availability, self.positions, torch.zeros_like(self.positions)
        )

    def rows_of(self, sample_ids: list[str]) -> list[int]:
        """Local row positions of a batch's samples, in the order the loader emitted them."""
        lookup = {identifier: position for position, identifier in enumerate(self.view.sample_ids)}
        return [lookup[identifier] for identifier in sample_ids]

    def _position_availability(self, mask: Tensor) -> Tensor:
        batch = self.view.availability.shape[0]
        out = torch.zeros((batch, self.axis.tau_max, self.axis.position_dim), dtype=torch.bool)
        for spec in self.axis.positions:
            columns = [*spec.compartment_columns, *spec.morphology_columns]
            if not columns:
                continue
            slot = self.view.availability[:, list(columns)]
            out[:, spec.index, : len(columns)] = slot
        return out & mask.unsqueeze(0)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        return {
            "positions": self.positions[item],
            "availability": self.availability[item],
            "target": self.targets[item],
            "time": self.view.times[item],
            "event": self.view.events[item],
            "has_label": self.view.has_label[item],
            "resource": self.view.resources[item],
            "stage": self.view.stages[item],
            "grade": self.view.grades[item],
            "sample_id": self.view.sample_ids[item],
        }

    def records(self) -> list[SampleRecord]:
        return [self.table.samples[index] for index in self.indices]


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    positions = torch.stack([entry["positions"] for entry in batch])
    availability = torch.stack([entry["availability"] for entry in batch])
    targets = torch.stack([entry["target"] for entry in batch])
    times = torch.stack([entry["time"] for entry in batch])
    events = torch.stack([entry["event"] for entry in batch])
    has_label = torch.stack([entry["has_label"] for entry in batch])
    resources = torch.stack([entry["resource"] for entry in batch])
    stages = torch.stack([entry["stage"] for entry in batch])
    grades = torch.stack([entry["grade"] for entry in batch])
    return {
        "positions": positions,
        "availability": availability,
        "target": targets,
        "time": times,
        "event": events,
        "has_label": has_label,
        "resource": resources,
        "stage": stages,
        "grade": grades,
        "sample_id": [entry["sample_id"] for entry in batch],
    }
