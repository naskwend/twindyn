"""The predetermined compartment ordering that defines the microenvironment axis.

Ref: Sec. 3.2 (the ordered axis is a design created by the analyst, applied
prior to any training and never adjusted to outcomes), Sec. 3.10 and Sec. 4.4
(the locations on the ordered axis are assigned from the predefined compartment
ordering supplied with the supplementary material).

The supplementary material is not part of the manuscript file set, so the
ordering below is an engineering default: the compartment registry follows the
tumour-core-to-immune-margin hierarchy the manuscript cites for the ccRCC
microenvironment, and it is fixed before any training and never tuned on an
outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompartmentFamily(str, Enum):
    TUMOUR = "tumour"
    INTERFACE = "interface"
    VASCULAR = "vascular"
    STROMAL = "stromal"
    MYELOID = "myeloid"
    LYMPHOID = "lymphoid"
    GRANULOCYTIC = "granulocytic"


@dataclass(frozen=True)
class CompartmentSpec:
    name: str
    family: CompartmentFamily
    rank: int


RAW_COMPARTMENTS: tuple[CompartmentSpec, ...] = (
    CompartmentSpec("tumour_epithelial_ccA", CompartmentFamily.TUMOUR, 0),
    CompartmentSpec("tumour_epithelial_ccB", CompartmentFamily.TUMOUR, 1),
    CompartmentSpec("tumour_proximal_interface", CompartmentFamily.INTERFACE, 2),
    CompartmentSpec("proximal_tubule_remnant", CompartmentFamily.INTERFACE, 3),
    CompartmentSpec("capillary_endothelium", CompartmentFamily.VASCULAR, 4),
    CompartmentSpec("arterial_endothelium", CompartmentFamily.VASCULAR, 5),
    CompartmentSpec("pericyte_mural", CompartmentFamily.VASCULAR, 6),
    CompartmentSpec("peritumoural_fibroblast", CompartmentFamily.STROMAL, 7),
    CompartmentSpec("myofibroblast_smooth_muscle", CompartmentFamily.STROMAL, 8),
    CompartmentSpec("myeloid_dendritic", CompartmentFamily.MYELOID, 9),
    CompartmentSpec("macrophage_m1", CompartmentFamily.MYELOID, 10),
    CompartmentSpec("macrophage_m2", CompartmentFamily.MYELOID, 11),
    CompartmentSpec("t_cell_cd8", CompartmentFamily.LYMPHOID, 12),
    CompartmentSpec("t_cell_cd4", CompartmentFamily.LYMPHOID, 13),
    CompartmentSpec("b_plasma_cell", CompartmentFamily.LYMPHOID, 14),
    CompartmentSpec("mast_neutrophil_granulocyte", CompartmentFamily.GRANULOCYTIC, 15),
)

COMPARTMENT_ORDER: tuple[str, ...] = tuple(spec.name for spec in RAW_COMPARTMENTS)


def compartment_count() -> int:
    return len(COMPARTMENT_ORDER)


def compartment_index(name: str) -> int:
    try:
        return COMPARTMENT_ORDER.index(name)
    except ValueError as exc:
        raise KeyError(f"unknown compartment {name!r}") from exc


def compartment_family(name: str) -> CompartmentFamily:
    return RAW_COMPARTMENTS[compartment_index(name)].family


def family_positions(family: CompartmentFamily) -> tuple[int, ...]:
    return tuple(spec.rank for spec in RAW_COMPARTMENTS if spec.family is family)


def normalise_fractions(fractions: object) -> list[float]:
    """Clip to the simplex and renormalise; an all-zero row stays all-zero."""
    import numpy as np

    array = np.asarray(fractions, dtype=np.float64).reshape(-1)
    clipped = np.clip(array, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return [0.0] * clipped.shape[0]
    return [float(value) for value in clipped / total]


def aggregate_compartment_groups(level: int) -> tuple[tuple[int, ...], ...]:
    """Contiguous groupings of the ordered compartments at a coarser resolution.

    Level 0 is the compartment level itself. Level 1 merges neighbouring
    compartments by family, level 2 merges into core, interface, vascular,
    stromal and immune blocks, and the coarsest level is the aggregated-sample
    level with a single group.
    """
    if level <= 0:
        return tuple((index,) for index in range(len(COMPARTMENT_ORDER)))
    if level == 1:
        groups: list[tuple[int, ...]] = []
        for spec in RAW_COMPARTMENTS:
            if groups and compartment_family(COMPARTMENT_ORDER[groups[-1][0]]) is spec.family:
                groups[-1] = (*groups[-1], spec.rank)
            else:
                groups.append((spec.rank,))
        return tuple(groups)
    if level == 2:
        blocks = (
            (CompartmentFamily.TUMOUR,),
            (CompartmentFamily.INTERFACE,),
            (CompartmentFamily.VASCULAR,),
            (CompartmentFamily.STROMAL,),
            (CompartmentFamily.MYELOID,),
            (CompartmentFamily.LYMPHOID,),
            (CompartmentFamily.GRANULOCYTIC,),
        )
        out: list[tuple[int, ...]] = []
        for families in blocks:
            members = tuple(
                index for index, spec in enumerate(RAW_COMPARTMENTS) if spec.family in families
            )
            if members:
                out.append(members)
        return tuple(out)
    return (tuple(range(len(COMPARTMENT_ORDER))),)


AGGREGATION_LEVELS: tuple[int, ...] = (0, 1, 2)


def aggregation_levels() -> tuple[int, ...]:
    """The resolution levels reported alongside the compartment-level result."""
    return AGGREGATION_LEVELS
