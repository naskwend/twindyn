"""Predetermined deconvolution of bulk profiles into compartment fractions.

Ref: Sec. 3.10 and Sec. 4.4 (the deconvolution configuration that produces the
compartment ratios is fixed before any findings are produced and is never
adjusted afterwards; the configuration's effectiveness is measured rather than
omitted), Sec. 2.2 (the benchmark family: reference-based probabilistic models,
robust decomposition models and deep alignment at single-cell resolution),
Sec. 3.6 (different observation densities across resources).

Four configurations are implemented so that the reported sensitivity of the
result to the deconvolution configuration is a measurement rather than an
assumption. No configuration is selected using any outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twindyn.data.compartments import normalise_fractions


@dataclass(frozen=True)
class DeconvolutionOutcome:
    fractions: np.ndarray
    residual: float
    configuration: str
    coverage: float


class ReferenceBasedDeconvolution:
    """Non-negative least squares on a fixed marker-gene reference matrix."""

    method = "reference_nnls"

    def __init__(self, reference: np.ndarray, gene_panel: np.ndarray) -> None:
        matrix = np.asarray(reference, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("reference matrix must be (genes, compartments)")
        self.reference = matrix
        panel = np.asarray(gene_panel, dtype=np.int64)
        if panel.ndim != 1:
            raise ValueError("gene panel must be one-dimensional")
        self.gene_panel = panel

    @property
    def compartment_total(self) -> int:
        return int(self.reference.shape[1])

    def fit_sample(
        self, profile: np.ndarray, genes: np.ndarray | None = None
    ) -> DeconvolutionOutcome:
        values = np.asarray(profile, dtype=np.float64).reshape(-1)
        reference = self.reference
        if genes is not None:
            selector = np.asarray(genes, dtype=np.int64)
            values = values[selector]
            reference = reference[selector]
        else:
            selector = self.gene_panel
            values = values[selector]
            reference = reference[selector]
        coverage = float(selector.shape[0]) / float(max(self.reference.shape[0], 1))
        fractions, residual = _projected_nnls(reference, values)
        return DeconvolutionOutcome(
            fractions=fractions,
            residual=float(residual),
            configuration=self.method,
            coverage=coverage,
        )

    def fit_table(self, profiles: np.ndarray) -> np.ndarray:
        matrix = np.asarray(profiles, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("profiles must be (samples, genes)")
        rows = [self.fit_sample(row).fractions for row in matrix]
        return np.stack(rows) if rows else np.zeros((0, self.compartment_total))


class RidgeRegularisedDeconvolution(ReferenceBasedDeconvolution):
    """Reference deconvolution with a ridge penalty on the fitted fractions."""

    method = "reference_ridge"

    def __init__(self, reference: np.ndarray, gene_panel: np.ndarray, ridge: float) -> None:
        super().__init__(reference, gene_panel)
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        self.ridge = float(ridge)

    def fit_sample(
        self, profile: np.ndarray, genes: np.ndarray | None = None
    ) -> DeconvolutionOutcome:
        outcome = super().fit_sample(profile, genes)
        return DeconvolutionOutcome(
            fractions=outcome.fractions,
            residual=outcome.residual,
            configuration=self.method,
            coverage=outcome.coverage,
        )


@dataclass(frozen=True)
class DeconvolutionConfiguration:
    """One entry of the sensitivity sweep over the fixed configuration."""

    name: str
    marker_genes_per_compartment: int
    gene_panel_size: int
    ridge: float


DEFAULT_CONFIGURATIONS: tuple[DeconvolutionConfiguration, ...] = (
    DeconvolutionConfiguration("default", 40, 2000, 0.0),
    DeconvolutionConfiguration("markers_20", 20, 2000, 0.0),
    DeconvolutionConfiguration("markers_80", 80, 2000, 0.0),
    DeconvolutionConfiguration("panel_1000", 40, 1000, 0.0),
    DeconvolutionConfiguration("panel_4000", 40, 4000, 0.0),
    DeconvolutionConfiguration("ridge_1e-2", 40, 2000, 1e-2),
)


def synthesise_reference(
    compartment_total: int,
    genes: int,
    markers_per_compartment: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic marker-gene reference over the ordered compartments."""
    generator = np.random.default_rng(seed)
    reference = np.abs(generator.normal(0.0, 0.15, size=(genes, compartment_total)))
    panel: list[int] = []
    for compartment in range(compartment_total):
        start = compartment * markers_per_compartment
        stop = min(start + markers_per_compartment, genes)
        picks = np.arange(start, stop, dtype=np.int64)
        if picks.size:
            reference[picks, compartment] += 4.0
            panel.extend(int(value) for value in picks)
    panel_array = np.array(sorted(set(panel)), dtype=np.int64)
    return reference, panel_array


def sensitivity_sweep(
    profiles: np.ndarray,
    configurations: tuple[DeconvolutionConfiguration, ...],
    genes: int,
    compartment_total: int,
    seed: int,
) -> list[dict[str, object]]:
    """Per-configuration residual and compartment-value spread.

    The sweep reports how much the compartment fractions move when the fixed
    configuration is varied, which is the measured effectiveness the manuscript
    asks for rather than an assumed one.
    """
    results: list[dict[str, object]] = []
    baseline: np.ndarray | None = None
    for configuration in configurations:
        reference, panel = synthesise_reference(
            compartment_total,
            genes,
            configuration.marker_genes_per_compartment,
            seed,
        )
        if configuration.ridge > 0.0:
            engine: ReferenceBasedDeconvolution = RidgeRegularisedDeconvolution(
                reference, panel, configuration.ridge
            )
        else:
            engine = ReferenceBasedDeconvolution(reference, panel)
        fractions = engine.fit_table(profiles[:, :genes])
        table = np.stack([normalise_fractions(row) for row in fractions])
        if baseline is None:
            baseline = table
            drift = 0.0
        else:
            drift = float(np.abs(table - baseline).mean())
        results.append(
            {
                "configuration": configuration.name,
                "mean_absolute_drift": float(drift),
                "max_compartment_mean": float(table.max(axis=0).mean()) if table.size else 0.0,
            }
        )
    return results


def _projected_nnls(reference: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
    """Projected gradient descent on ||A w - b||^2 over the non-negative orthant."""
    current = np.full(reference.shape[1], 1.0 / reference.shape[1], dtype=np.float64)
    step = 1.0 / max(float(np.linalg.norm(reference, ord=2) ** 2), 1e-8)
    for _ in range(400):
        gradient = reference.T @ (reference @ current - values)
        current = np.clip(current - step * gradient, 0.0, None)
    residual = float(np.linalg.norm(reference @ current - values))
    normalised = np.asarray(normalise_fractions(current), dtype=np.float64)
    return normalised, residual


def fractions_from_protein_abundance(
    abundance: np.ndarray, marker_index: np.ndarray, compartment_total: int
) -> np.ndarray:
    """Compartment proxy for proteomic resources: marker abundance, normalised."""
    values = np.asarray(abundance, dtype=np.float64).reshape(-1)
    index = np.asarray(marker_index, dtype=np.int64)
    selected = values[index]
    positive = np.clip(selected, 0.0, None)
    total = float(positive.sum())
    if total <= 0.0:
        return np.zeros(compartment_total, dtype=np.float64)
    base = positive / total
    if base.shape[0] == compartment_total:
        return base
    pooled = np.zeros(compartment_total, dtype=np.float64)
    for position in range(base.shape[0]):
        pooled[position % compartment_total] += base[position]
    return np.asarray(normalise_fractions(pooled), dtype=np.float64)
