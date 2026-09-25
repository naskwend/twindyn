"""Resource-disjoint splitting and sample-level disjointness checks.

Ref: Sec. 4.1 (representation learning uses the training resource and the
alternate resource for the alignment task; the read-out is fitted on the training
part of the training resource; external numbers are produced by scoring each
external cohort once with a frozen model; the splitting is at the level of tumour
samples and zero overlap is verified rather than presumed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twindyn.data.schema import CohortTable, SampleRecord


@dataclass(frozen=True)
class SplitIndices:
    train: tuple[int, ...]
    internal_validation: tuple[int, ...]
    external: dict[str, tuple[int, ...]]

    def all_indices(self) -> tuple[int, ...]:
        combined = list(self.train) + list(self.internal_validation)
        for indices in self.external.values():
            combined.extend(indices)
        return tuple(combined)


def internal_split(
    records: list[SampleRecord],
    training_resource: str,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split the training resource's samples into a fit and validation part."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must lie strictly between 0 and 1")
    indices = [
        index for index, record in enumerate(records) if record.resource == training_resource
    ]
    if not indices:
        raise ValueError(f"no samples for training resource {training_resource!r}")
    generator = np.random.default_rng(seed)
    labels = {index: records[index].label for index in indices}
    events = np.asarray(
        [int(label.event) for label in labels.values() if label is not None], dtype=np.int64
    )
    stratify = events if events.size == len(indices) else None
    if stratify is None:
        shuffled = generator.permutation(indices)
        cut = round(len(shuffled) * validation_fraction)
        return tuple(int(i) for i in shuffled[cut:]), tuple(int(i) for i in shuffled[:cut])
    positive = [index for index, label in labels.items() if label is not None and label.event]
    negative = [index for index in indices if index not in set(positive)]
    generator.shuffle(positive)
    generator.shuffle(negative)
    val_positive = positive[: max(1, round(len(positive) * validation_fraction))]
    val_negative = negative[: max(1, round(len(negative) * validation_fraction))]
    validation = set(val_positive) | set(val_negative)
    fit = [index for index in indices if index not in validation]
    return tuple(fit), tuple(sorted(validation))


def resource_partition(records: list[SampleRecord]) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(record.resource, []).append(index)
    return {key: tuple(value) for key, value in grouped.items()}


def assert_zero_overlap(records: list[SampleRecord]) -> dict[str, object]:
    """Verify that no tumour sample identifier appears in more than one resource."""
    seen: dict[str, str] = {}
    collisions: list[dict[str, str]] = []
    for record in records:
        previous = seen.get(record.sample_id)
        if previous is not None and previous != record.resource:
            collisions.append(
                {
                    "sample_id": record.sample_id,
                    "resource_a": previous,
                    "resource_b": record.resource,
                }
            )
        seen.setdefault(record.sample_id, record.resource)
    per_resource: dict[str, int] = {}
    for record in records:
        per_resource[record.resource] = per_resource.get(record.resource, 0) + 1
    return {
        "unique_sample_ids": len(seen),
        "total_records": len(records),
        "collisions": collisions,
        "per_resource_counts": per_resource,
        "zero_overlap": not collisions,
    }


def label_fraction_subset(
    records: list[SampleRecord],
    indices: tuple[int, ...],
    fraction: float,
    seed: int,
) -> tuple[int, ...]:
    """The labelled indices retained at 10, 25, 50 and 100 percent of the labels.

    The observations of every index stay available; what shrinks is the set of
    samples whose survival label the read-out may use.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    labelled = [index for index in indices if records[index].label is not None]
    if fraction >= 1.0:
        return tuple(labelled)
    generator = np.random.default_rng(seed)
    generator.shuffle(labelled)
    keep = max(1, round(len(labelled) * fraction))
    return tuple(sorted(labelled[:keep]))


def mask_labels(table: CohortTable, withheld: set[int]) -> CohortTable:
    """A copy of the table whose listed samples carry no survival label."""
    records: list[SampleRecord] = []
    for index, sample in enumerate(table.samples):
        if index not in withheld or sample.label is None:
            records.append(sample)
            continue
        records.append(
            SampleRecord(
                sample_id=sample.sample_id,
                resource=sample.resource,
                compartments=sample.compartments,
                morphology=sample.morphology,
                availability=sample.availability,
                label=None,
                stratum=sample.stratum,
                anchors=sample.anchors,
            )
        )
    return CohortTable(
        width=table.width, compartment_count=table.compartment_count, samples=records
    )


def stratum_indices(
    records: list[SampleRecord],
    indices: tuple[int, ...],
    field: str,
) -> dict[str, tuple[int, ...]]:
    """Group indices by the ISUP grade or TNM stage stratum."""
    grouped: dict[str, list[int]] = {}
    for index in indices:
        stratum = records[index].stratum
        if stratum is None:
            continue
        key = stratum.grade_stratum if field == "grade" else stratum.stage_stratum
        if key is None:
            continue
        grouped.setdefault(key, []).append(index)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def modality_dropout(
    observations: np.ndarray,
    compartment_total: int,
    drop: str,
) -> np.ndarray:
    """Withhold one input family at test time without imputing it."""
    if drop == "none":
        return observations
    dropped = observations.copy()
    if drop == "morphology":
        dropped[:, compartment_total:] = 0.0
    elif drop == "compartments":
        dropped[:, :compartment_total] = 0.0
    else:
        raise ValueError(f"unknown dropout family {drop!r}")
    return dropped
