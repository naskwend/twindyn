"""Histology-derived morphology features.

Ref: Sec. 3.1 (m_i holds the morphology features when histology provides the
resources), Sec. 3.10 and Sec. 4.1 (the descriptor is a compression of
compartment fractions plus morphology-derived characteristics for the resources
that supply histology), Table 3 (withholding the morphology features at test
time costs 0.004, withholding the compartment fractions costs 0.033).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from twindyn.data.axis import MicroenvironmentAxis

STATISTIC_NAMES: tuple[str, ...] = (
    "mean_intensity",
    "std_intensity",
    "p10_intensity",
    "p90_intensity",
    "high_density_proxy",
    "gradient_energy",
    "laplacian_variance",
    "spectral_low_band",
    "spectral_mid_band",
    "spectral_high_band",
    "entropy_histogram",
    "saturation_fraction",
    "edge_length_proxy",
    "object_size_proxy",
    "eccentricity_proxy",
    "texture_contrast",
)


@dataclass(frozen=True)
class MorphologyLayout:
    """How the morphology block of the observation is laid out per position."""

    statistics: tuple[str, ...] = STATISTIC_NAMES
    per_position: int = 3

    @property
    def total(self) -> int:
        return len(self.statistics)


def _normalise_tiles(tiles: Tensor) -> Tensor:
    if tiles.dim() == 4:
        tiles = tiles.mean(dim=1)
    if tiles.dim() != 3:
        raise ValueError("tiles must be (tiles, height, width) or (tiles, channels, height, width)")
    flat = tiles.flatten(1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    return ((flat - mean) / std).reshape(tiles.shape)


def extract_tile_statistics(tiles: Tensor) -> Tensor:
    """Deterministic per-tile statistics spanning intensity, texture and spectra."""
    normalised = _normalise_tiles(tiles)
    flat = normalised.flatten(1)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1)
    p10 = torch.quantile(flat, 0.10, dim=1)
    p90 = torch.quantile(flat, 0.90, dim=1)
    threshold = p90.unsqueeze(1)
    high_density = (flat >= threshold).to(flat.dtype).mean(dim=1)

    grad_rows = torch.diff(normalised, dim=1)
    grad_cols = torch.diff(normalised, dim=2)
    gradient_energy = grad_rows.pow(2).mean(dim=(1, 2)) + grad_cols.pow(2).mean(dim=(1, 2))

    laplacian = (
        4.0 * normalised[:, 1:-1, 1:-1]
        - normalised[:, :-2, 1:-1]
        - normalised[:, 2:, 1:-1]
        - normalised[:, 1:-1, :-2]
        - normalised[:, 1:-1, 2:]
    )
    laplacian_variance = laplacian.var(dim=(1, 2))

    spectrum = torch.fft.rfft2(normalised, norm="ortho")
    magnitude = spectrum.abs().pow(2)
    height, width = magnitude.shape[-2:]
    row_freq = torch.fft.fftfreq(height, device=tiles.device).abs().unsqueeze(1)
    col_freq = torch.fft.rfftfreq(width, device=tiles.device).abs().unsqueeze(0)
    radius = torch.sqrt(row_freq.pow(2) + col_freq.pow(2))
    low = magnitude * (radius < 0.15).to(magnitude.dtype)
    mid = magnitude * ((radius >= 0.15) & (radius < 0.35)).to(magnitude.dtype)
    high = magnitude * (radius >= 0.35).to(magnitude.dtype)
    total = magnitude.sum(dim=(1, 2)).clamp_min(1e-8)
    low_band = low.sum(dim=(1, 2)) / total
    mid_band = mid.sum(dim=(1, 2)) / total
    high_band = high.sum(dim=(1, 2)) / total

    histogram = _band_histogram(flat, bins=16)
    entropy = -(histogram.clamp_min(1e-8) * histogram.clamp_min(1e-8).log()).sum(dim=1)

    low_tail = torch.quantile(flat, 0.05, dim=1)
    high_tail = torch.quantile(flat, 0.95, dim=1)
    saturation = (
        ((flat <= low_tail.unsqueeze(1)) | (flat >= high_tail.unsqueeze(1)))
        .to(flat.dtype)
        .mean(dim=1)
    )

    edge_length = (grad_rows.abs() + grad_cols.abs()).mean(dim=(1, 2))
    object_size = high_density * float(flat.shape[1])
    eccentricity = (p90 - p10) / (p90 + p10).abs().clamp_min(1e-6)
    contrast = gradient_energy / (std.pow(2) + 1e-6)

    stacked = torch.stack(
        [
            mean,
            std,
            p10,
            p90,
            high_density,
            gradient_energy,
            laplacian_variance,
            low_band,
            mid_band,
            high_band,
            entropy,
            saturation,
            edge_length,
            object_size,
            eccentricity,
            contrast,
        ],
        dim=1,
    )
    return torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)


def _band_histogram(flat: Tensor, bins: int) -> Tensor:
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    scaled = (flat - lo) / (hi - lo).clamp_min(1e-6)
    index = (scaled * bins).clamp(0, bins - 1).long()
    counts = torch.zeros((flat.shape[0], bins), dtype=flat.dtype, device=flat.device)
    counts.scatter_add_(1, index, torch.ones_like(flat))
    return counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)


def region_morphology(
    tile_statistics: Tensor,
    region_assignment: Tensor,
    compartment_total: int,
    per_position: int = 3,
) -> Tensor:
    """Pool tile statistics per ordered compartment, then reduce the block width."""
    statistics = extract_tile_statistics(tile_statistics)
    assignment = region_assignment.reshape(-1).long()
    pooled = torch.zeros(
        (compartment_total, statistics.shape[1]), dtype=statistics.dtype, device=statistics.device
    )
    counts = torch.zeros((compartment_total,), dtype=statistics.dtype, device=statistics.device)
    for position in range(compartment_total):
        selector = assignment == position
        if bool(selector.any()):
            pooled[position] = statistics[selector].mean(dim=0)
            counts[position] = float(selector.sum())
    block = pooled[:, :per_position]
    if block.shape[1] < per_position:
        pad = block.new_zeros((block.shape[0], per_position - block.shape[1]))
        block = torch.cat([block, pad], dim=1)
    return block


def morphology_block_for_axis(
    per_compartment: Tensor,
    axis: MicroenvironmentAxis,
) -> Tensor:
    """Average per-compartment morphology into the axis positions' morphology slots."""
    if per_compartment.dim() != 2:
        raise ValueError("per-compartment morphology must be (compartments, features)")
    out = per_compartment.new_zeros((axis.tau_max, len(axis.positions[0].morphology_columns)))
    slots = len(axis.positions[0].morphology_columns)
    for spec in axis.positions:
        if not spec.compartment_columns or slots == 0:
            continue
        rows = per_compartment[list(spec.compartment_columns)]
        averaged = rows.mean(dim=0)[:slots]
        out[spec.index, : averaged.shape[0]] = averaged
    return out.flatten()


def zero_morphology(axis: MicroenvironmentAxis) -> np.ndarray:
    """The modality-dropout at test time: morphology withheld, not imputed."""
    return np.zeros(axis.morphology_total, dtype=np.float64)
