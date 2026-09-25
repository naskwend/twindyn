"""The ordered microenvironment axis.

Ref: Sec. 3.2 (the axis is a predefined organisation of the sample's
microenvironment compartments, fixed before training and never adjusted to
outcomes), Sec. 3.10, Sec. 4.4, Table 4 (axis length swept over 4/8/16/32 with
the selected length at 16), Table 3 (the coarser aggregated resolution).

Each axis position carries a contiguous run of the predefined compartment order
together with the histology features measured in that region, so the ordering of
positions reproduces the ordering of the compartment registry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from twindyn.data.compartments import COMPARTMENT_ORDER, compartment_count


def contiguous_partition(size: int, groups: int) -> tuple[tuple[int, ...], ...]:
    """Split ``range(size)`` into ``groups`` contiguous blocks differing by <= 1."""
    if groups <= 0:
        raise ValueError("groups must be positive")
    if size < 0:
        raise ValueError("size must be non-negative")
    if groups >= size:
        blocks: list[tuple[int, ...]] = [(index,) for index in range(size)]
        blocks.extend(() for _ in range(groups - size))
        return tuple(blocks)
    base, remainder = divmod(size, groups)
    out: list[tuple[int, ...]] = []
    start = 0
    for index in range(groups):
        width = base + (1 if index < remainder else 0)
        out.append(tuple(range(start, start + width)))
        start += width
    return tuple(out)


@dataclass(frozen=True)
class AxisIndex:
    """Column indices each axis position observes, padded to a common width."""

    position_index: Tensor
    position_width: Tensor
    width: int
    tau_max: int
    compartment_total: int
    morphology_total: int

    @property
    def position_dim(self) -> int:
        return int(self.position_index.shape[1])

    @property
    def observable_columns(self) -> int:
        return int(self.position_width.sum())


@dataclass(frozen=True)
class PositionSpec:
    """Columns of the joined observation that position ``index`` observes."""

    index: int
    compartment_columns: tuple[int, ...]
    morphology_columns: tuple[int, ...]

    @property
    def width(self) -> int:
        return len(self.compartment_columns) + len(self.morphology_columns)

    @property
    def compartment_names(self) -> tuple[str, ...]:
        return tuple(COMPARTMENT_ORDER[column] for column in self.compartment_columns)


class MicroenvironmentAxis:
    """A fixed partition of the observation vector into ordered positions."""

    def __init__(
        self,
        tau_max: int,
        compartment_total: int,
        morphology_total: int,
        compartments_per_position: tuple[tuple[int, ...], ...],
        morphology_per_position: tuple[tuple[int, ...], ...],
        ordering_rule: str,
    ) -> None:
        self.tau_max = int(tau_max)
        self.compartment_total = int(compartment_total)
        self.morphology_total = int(morphology_total)
        self._compartment_blocks = compartments_per_position
        self._morphology_blocks = morphology_per_position
        self.ordering_rule = ordering_rule
        if len(compartments_per_position) != self.tau_max:
            raise ValueError("compartment blocks must match tau_max")
        if len(morphology_per_position) != self.tau_max:
            raise ValueError("morphology blocks must match tau_max")
        self.positions = tuple(
            PositionSpec(index, comp, morph)
            for index, (comp, morph) in enumerate(
                zip(compartments_per_position, morphology_per_position, strict=True)
            )
        )

    @classmethod
    def of(
        cls,
        width: int,
        tau_max: int,
        compartment_total: int | None = None,
        ordering_rule: str = "predefined_compartment_index",
    ) -> MicroenvironmentAxis:
        compartment_total = compartment_count() if compartment_total is None else compartment_total
        if compartment_total > width:
            raise ValueError("compartment count cannot exceed the observation width")
        morphology_total = width - compartment_total
        if ordering_rule == "predefined_compartment_index":
            if tau_max <= compartment_total:
                comp_blocks = contiguous_partition(compartment_total, tau_max)
            else:
                comp_blocks = tuple((index,) for index in range(compartment_total))
                comp_blocks = comp_blocks + ((),) * (tau_max - compartment_total)
            morph_blocks = contiguous_partition(morphology_total, tau_max)
        elif ordering_rule == "shuffled":
            comp_blocks = contiguous_partition(compartment_total, tau_max)
            morph_blocks = contiguous_partition(morphology_total, tau_max)
        else:
            raise ValueError(f"unknown ordering rule {ordering_rule!r}")
        return cls(
            tau_max=tau_max,
            compartment_total=compartment_total,
            morphology_total=morphology_total,
            compartments_per_position=comp_blocks,
            morphology_per_position=morph_blocks,
            ordering_rule=ordering_rule,
        )

    @property
    def width(self) -> int:
        return self.compartment_total + self.morphology_total

    @property
    def position_dim(self) -> int:
        return max(spec.width for spec in self.positions)

    def shuffled(self, seed: int) -> MicroenvironmentAxis:
        """Permute the axis positions before training (the shuffled-axis ablation)."""
        generator = np.random.default_rng(seed)
        order = generator.permutation(self.tau_max)
        comp = tuple(self._compartment_blocks[int(index)] for index in order)
        morph = tuple(self._morphology_blocks[int(index)] for index in order)
        return MicroenvironmentAxis(
            self.tau_max,
            self.compartment_total,
            self.morphology_total,
            comp,
            morph,
            ordering_rule="shuffled",
        )

    def aggregated(self, level: int) -> MicroenvironmentAxis:
        """Merge consecutive positions into coarser units (Table 3 resolution row)."""
        if level <= 0:
            return self
        factor = int(2**level)
        if factor >= self.tau_max:
            groups = 1
        else:
            groups = int(np.ceil(self.tau_max / factor))
        comp = _merge_blocks(self._compartment_blocks, groups)
        morph = _merge_blocks(self._morphology_blocks, groups)
        return MicroenvironmentAxis(
            groups,
            self.compartment_total,
            self.morphology_total,
            comp,
            morph,
            ordering_rule=f"{self.ordering_rule}|aggregated_{level}",
        )

    def positions_from_observation(self, observation: Tensor) -> Tensor:
        """Gather a padded ``(batch, tau_max, position_dim)`` position tensor."""
        if observation.dim() != 2:
            raise ValueError("observation must be (batch, width)")
        if observation.shape[-1] != self.width:
            raise ValueError(f"expected width {self.width}, got {observation.shape[-1]}")
        batch = observation.shape[0]
        padded = observation.new_zeros((batch, self.tau_max, self.position_dim))
        for spec in self.positions:
            columns = (
                *spec.compartment_columns,
                *[self.compartment_total + c for c in spec.morphology_columns],
            )
            if not columns:
                continue
            padded[:, spec.index, : len(columns)] = observation[:, list(columns)]
        return padded

    def position_mask(self, device: torch.device | None = None) -> Tensor:
        """Boolean mask of real (non-padding) coordinates per position."""
        mask = torch.zeros((self.tau_max, self.position_dim), dtype=torch.bool, device=device)
        for spec in self.positions:
            if spec.width:
                mask[spec.index, : spec.width] = True
        return mask

    def column_to_position(self) -> Tensor:
        """Map each observation column to its axis position, or -1 when unused."""
        mapping = torch.full((self.width,), -1, dtype=torch.long)
        for spec in self.positions:
            for column in spec.compartment_columns:
                mapping[column] = spec.index
            for column in spec.morphology_columns:
                mapping[self.compartment_total + column] = spec.index
        return mapping

    def compartment_position_matrix(self) -> Tensor:
        """``(tau_max, compartment_total)`` membership indicator with 1/|block| weights."""
        matrix = torch.zeros((self.tau_max, self.compartment_total))
        for spec in self.positions:
            if not spec.compartment_columns:
                continue
            weight = 1.0 / len(spec.compartment_columns)
            for column in spec.compartment_columns:
                matrix[spec.index, column] = weight
        return matrix

    def index(self) -> AxisIndex:
        """Padded column index each position observes, for the encoder and decoder."""
        columns = [
            [
                *spec.compartment_columns,
                *[self.compartment_total + column for column in spec.morphology_columns],
            ]
            for spec in self.positions
        ]
        widths = torch.tensor([len(entry) for entry in columns], dtype=torch.long)
        index = torch.zeros((self.tau_max, self.position_dim), dtype=torch.long)
        for position, entry in enumerate(columns):
            if entry:
                index[position, : len(entry)] = torch.tensor(entry, dtype=torch.long)
        return AxisIndex(
            position_index=index,
            position_width=widths,
            width=self.width,
            tau_max=self.tau_max,
            compartment_total=self.compartment_total,
            morphology_total=self.morphology_total,
        )

    def describe(self) -> dict[str, object]:
        return {
            "tau_max": self.tau_max,
            "width": self.width,
            "compartment_total": self.compartment_total,
            "morphology_total": self.morphology_total,
            "position_dim": self.position_dim,
            "ordering_rule": self.ordering_rule,
            "position_compartment_counts": [
                len(spec.compartment_columns) for spec in self.positions
            ],
            "position_morphology_counts": [len(spec.morphology_columns) for spec in self.positions],
        }


def _merge_blocks(blocks: tuple[tuple[int, ...], ...], groups: int) -> tuple[tuple[int, ...], ...]:
    if groups >= len(blocks):
        return blocks
    assignment = contiguous_partition(len(blocks), groups)
    merged: list[tuple[int, ...]] = []
    for group in assignment:
        combined: tuple[int, ...] = ()
        for index in group:
            combined = (*combined, *blocks[index])
        merged.append(combined)
    return tuple(merged)


def collapse_to_sample_level(axis: MicroenvironmentAxis) -> MicroenvironmentAxis:
    """The aggregated-sample level of Table 3: a single pooled position."""
    return MicroenvironmentAxis(
        1,
        axis.compartment_total,
        axis.morphology_total,
        (tuple(range(axis.compartment_total)),),
        (tuple(range(axis.morphology_total)),),
        ordering_rule=f"{axis.ordering_rule}|sample_level",
    )
