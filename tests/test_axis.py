"""The ordered axis: partitioning, partitioning balance, the sweeps and the ablations.

Ref: Sec. 3.2, Sec. 3.10, Table 3 (coarser resolution), Table 4 (axis length and
shuffled ordering).
"""

from __future__ import annotations

import pytest

from twindyn.data.axis import (
    MicroenvironmentAxis,
    collapse_to_sample_level,
    contiguous_partition,
)
from twindyn.data.compartments import COMPARTMENT_ORDER, aggregate_compartment_groups


@pytest.mark.parametrize("size", [0, 1, 5, 16, 17])
@pytest.mark.parametrize("groups", [1, 3, 16, 32])
def test_contiguous_partition_is_a_partition(size: int, groups: int) -> None:
    blocks = contiguous_partition(size, groups)
    assert len(blocks) == groups
    assert [value for block in blocks for value in block] == list(range(size))


@pytest.mark.parametrize("tau_max", [4, 8, 16, 32])
def test_axis_builds_the_requested_number_of_positions(
    axis: MicroenvironmentAxis, tau_max: int
) -> None:
    built = MicroenvironmentAxis.of(axis.width, tau_max, axis.compartment_total)
    assert len(built.positions) == tau_max
    assert built.width == axis.width


def test_positions_cover_every_column_exactly_once(axis: MicroenvironmentAxis) -> None:
    columns: list[int] = []
    for spec in axis.positions:
        columns.extend(spec.compartment_columns)
        columns.extend(axis.compartment_total + value for value in spec.morphology_columns)
    assert sorted(columns) == list(range(axis.width))


def test_position_order_follows_the_compartment_registry(axis: MicroenvironmentAxis) -> None:
    ordered: list[int] = []
    for spec in axis.positions:
        ordered.extend(spec.compartment_columns)
    assert ordered == sorted(ordered)
    assert [COMPARTMENT_ORDER[index] for index in ordered] == list(COMPARTMENT_ORDER)


def test_position_gather_round_trips(axis: MicroenvironmentAxis, dataset) -> None:
    gathered = axis.positions_from_observation(dataset.view.observations)
    expected = (dataset.view.observations.shape[0], axis.tau_max, axis.position_dim)
    assert gathered.shape == expected
    for spec in axis.positions:
        if not spec.compartment_columns:
            continue
        column = spec.compartment_columns[0]
        assert gathered[0, spec.index, 0].item() == pytest.approx(
            dataset.view.observations[0, column].item()
        )


def test_position_mask_marks_real_coordinates(axis: MicroenvironmentAxis) -> None:
    mask = axis.position_mask()
    assert mask.shape == (axis.tau_max, axis.position_dim)
    for spec in axis.positions:
        assert int(mask[spec.index].sum()) == spec.width


def test_shuffled_axis_permutes_without_losing_columns(axis: MicroenvironmentAxis) -> None:
    shuffled = axis.shuffled(0)
    assert [spec.compartment_columns for spec in shuffled.positions] != [
        spec.compartment_columns for spec in axis.positions
    ]
    assert sorted(
        value for spec in shuffled.positions for value in spec.compartment_columns
    ) == sorted(value for spec in axis.positions for value in spec.compartment_columns)
    assert shuffled.ordering_rule == "shuffled"


def test_aggregation_reduces_positions(axis: MicroenvironmentAxis) -> None:
    aggregated = axis.aggregated(1)
    assert 1 < len(aggregated.positions) < len(axis.positions)
    assert aggregated.width == axis.width
    total = sum(len(spec.compartment_columns) for spec in aggregated.positions)
    assert total == axis.compartment_total


def test_collapse_to_sample_level_is_one_position(axis: MicroenvironmentAxis) -> None:
    collapsed = collapse_to_sample_level(axis)
    assert len(collapsed.positions) == 1
    assert collapsed.positions[0].width == axis.width


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_compartment_groupings_cover_every_compartment(level: int) -> None:
    groups = aggregate_compartment_groups(level)
    assert sorted(value for group in groups for value in group) == list(
        range(len(COMPARTMENT_ORDER))
    )


def test_position_index_matches_the_position_specs(axis: MicroenvironmentAxis) -> None:
    index = axis.index()
    assert index.position_index.shape == (axis.tau_max, axis.position_dim)
    assert int(index.position_width.sum()) == axis.width
    for spec in axis.positions:
        expected = [
            *spec.compartment_columns,
            *[axis.compartment_total + value for value in spec.morphology_columns],
        ]
        produced = index.position_index[spec.index, : index.position_width[spec.index]].tolist()
        assert produced == expected


def test_invalid_ordering_rule_is_rejected(axis: MicroenvironmentAxis) -> None:
    with pytest.raises(ValueError):
        MicroenvironmentAxis.of(axis.width, axis.tau_max, axis.compartment_total, "unknown_rule")
