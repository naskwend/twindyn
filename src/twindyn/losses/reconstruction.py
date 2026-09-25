"""Masked multi-source reconstruction loss.

Ref: Sec. 3.6, equation (4): the squared reconstruction error over the locations
withheld by masking, taken from the state that the selective operator produced.

The term is evaluated only on coordinates that the resource could observe and the
mask withheld. Coordinates a resource cannot observe at all are left out of the
numerator and the denominator, so a resource that observes at a low density does
not push the state towards reproducing coordinates it never had.
"""

from __future__ import annotations

from torch import Tensor


def masked_reconstruction_loss(
    reconstruction: Tensor,
    target: Tensor,
    withheld: Tensor,
) -> Tensor:
    """Equation (4): mean squared error over the withheld coordinates."""
    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target must agree in shape")
    if withheld.shape != reconstruction.shape:
        raise ValueError("withheld mask must match the reconstruction shape")
    selector = withheld.to(reconstruction.dtype)
    denominator = selector.sum()
    if float(denominator) == 0.0:
        return reconstruction.sum() * 0.0
    return ((reconstruction - target).pow(2) * selector).sum() / denominator


def reconstruction_residual_map(
    reconstruction: Tensor,
    target: Tensor,
    withheld: Tensor,
) -> dict[str, Tensor]:
    """Per-position squared residuals, for the reconstruction diagnostics."""
    selector = withheld.to(reconstruction.dtype)
    squared = (reconstruction - target).pow(2) * selector
    per_position = squared.sum(dim=-1) / selector.sum(dim=-1).clamp_min(1.0)
    return {
        "per_position": per_position,
        "per_sample": squared.sum(dim=(1, 2)) / selector.sum(dim=(1, 2)).clamp_min(1.0),
        "coverage": selector.mean(),
    }
