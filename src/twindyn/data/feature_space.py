"""The shared observable feature space across resources.

Ref: Sec. 4.1 (because the resources assay different molecules the shared feature
space is an intersection of observable features rather than a union with the
inclusion of missing information; missing features cannot be filled in; the
covered fraction of the initial feature set is reported per resource in
Supplementary Table S5), Sec. 3.3 (the intersection of common elements is a
modelling unit).

The supplementary table is not part of the manuscript file set, so the coverage
fractions are computed from the feature inventories the loaders actually read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twindyn.data.compartments import compartment_count


@dataclass
class FeatureInventory:
    """Observable feature identifiers declared by one resource."""

    resource: str
    gene_features: tuple[str, ...] = ()
    protein_features: tuple[str, ...] = ()
    morphology_features: tuple[str, ...] = ()

    def observable(self, histology: bool) -> set[str]:
        features: set[str] = set(self.gene_features) | set(self.protein_features)
        if histology:
            features |= set(self.morphology_features)
        return features


@dataclass
class SharedFeatureSpace:
    """The intersection across resources, plus the per-resource coverage."""

    features: tuple[str, ...]
    per_resource_coverage: dict[str, float] = field(default_factory=dict)
    per_resource_counts: dict[str, int] = field(default_factory=dict)
    compartment_count: int = field(default_factory=compartment_count)

    @property
    def width(self) -> int:
        return self.compartment_count + len(self.features)

    def index_of(self, feature: str) -> int:
        try:
            return self.features.index(feature)
        except ValueError as exc:
            raise KeyError(f"{feature!r} is outside the shared feature space") from exc


def intersect_inventories(
    inventories: dict[str, FeatureInventory],
    histology_resources: set[str],
    marker_genes: tuple[str, ...],
) -> SharedFeatureSpace:
    """Intersect the observable gene/protein features across every resource.

    Histology-derived features are carried only by the resources that provide
    histology, so they never enter the intersection; they are an extra block that
    is absent rather than imputed for the remaining resources.
    """
    shared: set[str] | None = None
    counts: dict[str, int] = {}
    coverage: dict[str, float] = {}
    for resource, inventory in inventories.items():
        observable = inventory.observable(resource in histology_resources)
        counts[resource] = len(observable)
        shared = observable if shared is None else (shared & observable)
    if shared is None:
        raise ValueError("no inventories supplied")
    ordered = tuple(sorted(shared))
    for resource in inventories:
        total = counts[resource]
        coverage[resource] = (len(ordered) / total) if total else 0.0
    anchor = [gene for gene in ordered if gene in set(marker_genes)]
    if anchor:
        ordered = tuple(anchor) + tuple(gene for gene in ordered if gene not in set(anchor))
    return SharedFeatureSpace(
        features=ordered,
        per_resource_coverage=coverage,
        per_resource_counts=counts,
    )


def project_observation(
    values: dict[str, float],
    space: SharedFeatureSpace,
) -> tuple[list[float], list[bool]]:
    """Project a resource's raw values into the shared space without imputation."""
    projected = [0.0] * len(space.features)
    available = [False] * len(space.features)
    for position, feature in enumerate(space.features):
        if feature in values:
            projected[position] = float(values[feature])
            available[position] = True
    return projected, available
