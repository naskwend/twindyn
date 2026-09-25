"""Sample, cohort and survival-label records.

Ref: Sec. 3.1 (the survival label (t_i, delta_i) is available for a subset of
samples), Sec. 4.1 (splitting is at the level of tumour samples).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from twindyn.registry import ENGINEERING_DEFAULT_MORPHOLOGY_FEATURES


@dataclass(frozen=True)
class SurvivalLabel:
    """Right-censored follow-up for one tumour sample."""

    time: float
    event: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.time) or self.time < 0.0:
            raise ValueError(f"non-finite or negative follow-up time {self.time!r}")

    @property
    def observed(self) -> float:
        return float(self.time)


@dataclass(frozen=True)
class ClinicalStratum:
    """ISUP grade and TNM-derived stage grouped for the stratum analysis."""

    grade: int | None
    stage: int | None

    @property
    def grade_stratum(self) -> str | None:
        if self.grade is None or self.grade < 1 or self.grade > 4:
            return None
        return f"grade_{int(self.grade)}"

    @property
    def stage_stratum(self) -> str | None:
        if self.stage is None or self.stage < 1 or self.stage > 4:
            return None
        return f"stage_{int(self.stage)}"


@dataclass
class SampleRecord:
    """One tumour sample drawn from exactly one resource.

    ``compartments`` holds the deconvolved compartment fractions, ``morphology``
    the histology-derived features (absent when the resource has no histology),
    and ``availability`` records which observation coordinates this resource can
    provide at all, as opposed to coordinates withheld by masking.
    """

    sample_id: str
    resource: str
    compartments: np.ndarray
    morphology: np.ndarray | None = None
    availability: np.ndarray | None = None
    label: SurvivalLabel | None = None
    stratum: ClinicalStratum | None = None
    anchors: np.ndarray | None = None

    def __post_init__(self) -> None:
        fractions = np.asarray(self.compartments, dtype=np.float64)
        if fractions.ndim != 1:
            raise ValueError("compartment fractions must be one-dimensional")
        self.compartments = fractions
        if self.morphology is not None:
            morphology = np.asarray(self.morphology, dtype=np.float64)
            if morphology.ndim != 1:
                raise ValueError("morphology features must be one-dimensional")
            self.morphology = morphology

    @property
    def observation_dim(self) -> int:
        return int(self.compartments.shape[0]) + (
            int(self.morphology.shape[0]) if self.morphology is not None else 0
        )

    def joined_observation(self, width: int) -> np.ndarray:
        """Compartment fractions followed by morphology, padded where absent."""
        block = np.zeros(width, dtype=np.float64)
        block[: self.compartments.shape[0]] = self.compartments
        if self.morphology is not None:
            start = self.compartments.shape[0]
            stop = start + self.morphology.shape[0]
            block[start:stop] = self.morphology
        return block

    def joined_availability(self, width: int, compartment_count: int) -> np.ndarray:
        mask = np.zeros(width, dtype=bool)
        mask[:compartment_count] = True
        if self.morphology is not None:
            start = compartment_count
            stop = start + self.morphology.shape[0]
            mask[start:stop] = True
        if self.availability is not None:
            mask &= np.asarray(self.availability, dtype=bool)[:width]
        return mask


@dataclass
class CohortTable:
    """A resource-keyed collection of samples, with the feature-space width."""

    width: int
    compartment_count: int
    samples: list[SampleRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def by_resource(self) -> dict[str, list[SampleRecord]]:
        grouped: dict[str, list[SampleRecord]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.resource, []).append(sample)
        return grouped

    def sample_ids(self) -> list[str]:
        return [sample.sample_id for sample in self.samples]

    def with_label(self) -> list[SampleRecord]:
        return [sample for sample in self.samples if sample.label is not None]

    def observations(self) -> np.ndarray:
        if not self.samples:
            return np.zeros((0, self.width), dtype=np.float64)
        return np.stack([sample.joined_observation(self.width) for sample in self.samples])

    def availability_matrix(self) -> np.ndarray:
        if not self.samples:
            return np.zeros((0, self.width), dtype=bool)
        return np.stack(
            [
                sample.joined_availability(self.width, self.compartment_count)
                for sample in self.samples
            ]
        )


def default_morphology_features(morphology: int | None = None) -> int:
    return int(morphology if morphology is not None else ENGINEERING_DEFAULT_MORPHOLOGY_FEATURES)
