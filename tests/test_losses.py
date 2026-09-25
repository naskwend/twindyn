"""Losses: equations (3), (4), (5) and (6) and the survival objective.

Ref: Sec. 3.5 (the survival objective), Sec. 3.6 (equations 3-6).
"""

from __future__ import annotations

import math

import pytest
import torch

from twindyn.data.pairs import batch_alignment_pairs
from twindyn.losses.alignment import (
    alignment_initialisation_reference,
    cross_resource_alignment_loss,
    normalised_alignment_loss,
    relative_alignment,
    state_distance,
)
from twindyn.losses.composite import ObjectiveWeights, composite_objective
from twindyn.losses.reconstruction import masked_reconstruction_loss, reconstruction_residual_map
from twindyn.losses.survival import (
    cox_partial_likelihood,
    partial_likelihood_by_stratum,
    risk_set_sizes,
)
from twindyn.losses.trajectory import (
    future_prediction_gap,
    split_halves,
    trajectory_consistency_loss,
)


def hand_breslow(risk: list[float], time: list[float], event: list[float]) -> float:
    total = 0.0
    events = 0
    for index in range(len(risk)):
        if event[index] != 1:
            continue
        events += 1
        at_risk = [position for position in range(len(risk)) if time[position] >= time[index]]
        total -= risk[index] - math.log(sum(math.exp(risk[position]) for position in at_risk))
    return total / max(events, 1)


@pytest.mark.parametrize(
    "risk,time,event",
    [
        ([0.5, -0.2, 0.1], [10.0, 20.0, 20.0], [1.0, 1.0, 0.0]),
        ([0.3, 0.8, -0.4, 0.1], [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 0.0]),
        ([0.1, 0.1, 0.1], [5.0, 5.0, 5.0], [1.0, 0.0, 1.0]),
        ([2.0, 1.0], [3.0, 3.0], [1.0, 1.0]),
        ([-1.0, 0.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 1.0, 1.0]),
    ],
)
def test_cox_matches_a_hand_computed_breslow(risk, time, event) -> None:
    produced = float(
        cox_partial_likelihood(torch.tensor(risk), torch.tensor(time), torch.tensor(event)).loss
    )
    assert produced == pytest.approx(hand_breslow(risk, time, event), abs=1e-5)


def test_cox_is_invariant_to_a_risk_shift() -> None:
    risk = torch.tensor([0.5, -0.2, 0.1, 0.9])
    time = torch.tensor([10.0, 20.0, 20.0, 5.0])
    event = torch.tensor([1.0, 1.0, 0.0, 1.0])
    base = float(cox_partial_likelihood(risk, time, event).loss)
    shifted = float(cox_partial_likelihood(risk + 4.0, time, event).loss)
    assert base == pytest.approx(shifted, abs=1e-5)


def test_cox_without_events_is_zero() -> None:
    result = cox_partial_likelihood(
        torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0])
    )
    assert float(result.loss) == 0.0
    assert result.events == 0


def test_cox_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        cox_partial_likelihood(torch.zeros(3), torch.zeros(2), torch.zeros(3))


def test_risk_set_sizes_are_monotone_in_time() -> None:
    sizes = risk_set_sizes(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert list(sizes.numpy()) == [3, 2, 1]


def test_stratified_likelihood_runs_per_stratum() -> None:
    result = partial_likelihood_by_stratum(
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
        torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    assert set(result) == {0, 1}
    assert all(math.isfinite(value) for value in result.values())


def test_masked_reconstruction_is_a_masked_mean_square() -> None:
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(prediction)
    withheld = torch.tensor([[True, False], [False, True]])
    assert float(masked_reconstruction_loss(prediction, target, withheld)) == pytest.approx(8.5)


def test_masked_reconstruction_with_nothing_withheld_is_zero() -> None:
    prediction = torch.ones(2, 2)
    withheld = torch.zeros(2, 2, dtype=torch.bool)
    assert float(masked_reconstruction_loss(prediction, torch.zeros(2, 2), withheld)) == 0.0


def test_masked_reconstruction_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        masked_reconstruction_loss(
            torch.zeros(2, 2), torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.bool)
        )


def test_residual_map_reports_per_position_and_per_sample() -> None:
    prediction = torch.ones(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    withheld = torch.ones(2, 3, 4, dtype=torch.bool)
    report = reconstruction_residual_map(prediction, target, withheld)
    assert report["per_position"].shape == (2, 3)
    assert report["per_sample"].shape == (2,)
    assert float(report["coverage"]) == 1.0


def test_state_distance_is_the_squared_euclidean_distance() -> None:
    left = torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j]])
    right = torch.tensor([[2.0 + 0.0j, 0.0 + 0.0j]])
    assert float(state_distance(left, right)[0]) == pytest.approx(1.0)
    imaginary = torch.tensor([[1.0 + 1.0j, 0.0 + 0.0j]])
    assert float(state_distance(imaginary, torch.zeros_like(imaginary))[0]) == pytest.approx(2.0)


def test_alignment_is_zero_for_coincident_pairs() -> None:
    states = torch.randn(1, 5, dtype=torch.complex64)
    pairs = batch_alignment_pairs(torch.eye(2), torch.tensor([0, 1]), 4)
    breakdown = cross_resource_alignment_loss(states.expand(2, -1).contiguous(), pairs)
    assert float(breakdown.loss) == 0.0
    assert breakdown.matched_pairs == 1


def test_alignment_is_unspecified_without_pairs() -> None:
    breakdown = cross_resource_alignment_loss(torch.zeros(3, 4, dtype=torch.complex64), [])
    assert float(breakdown.loss) == 0.0
    assert breakdown.available_pairs == 0


def test_alignment_is_not_specific_to_the_topology_of_the_pairs() -> None:
    states = torch.randn(4, 5, dtype=torch.complex64)
    pairs = batch_alignment_pairs(torch.eye(4), torch.tensor([0, 0, 1, 1]), 4)
    breakdown = cross_resource_alignment_loss(states, pairs)
    assert float(breakdown.loss) > 0.0
    assert set(breakdown.per_pair) == {"0|1"}


def test_relative_alignment_uses_the_initialisation_value() -> None:
    initial = {"0|1": 2.0}
    final = {"0|1": 0.5}
    assert relative_alignment(final, initial)["0|1"] == pytest.approx(0.25)
    missing = relative_alignment(final, {"0|1": 0.0})["0|1"]
    assert missing != missing


def test_initialisation_reference_matches_a_direct_computation() -> None:
    states = torch.randn(2, 3, dtype=torch.complex64)
    pairs = batch_alignment_pairs(torch.eye(2), torch.tensor([0, 1]), 4)
    reference = alignment_initialisation_reference(states, pairs)
    assert reference["0|1"] == pytest.approx(float(state_distance(states[0:1], states[1:2])[0]))


def test_normalised_alignment_matches_the_unscaled_ratio() -> None:
    states = torch.randn(2, 4, dtype=torch.complex64)
    pairs = batch_alignment_pairs(torch.eye(2), torch.tensor([0, 1]), 4)
    plain = cross_resource_alignment_loss(states, pairs)
    scaled = normalised_alignment_loss(states, pairs)
    ratio = float(scaled.loss / plain.loss)
    assert ratio == pytest.approx(1.0 / float(states.abs().pow(2).mean()), rel=1e-3)


def test_trajectory_halves_split_at_the_midpoint() -> None:
    states = torch.zeros(7, 2, 3, dtype=torch.complex64)
    past, future = split_halves(states)
    assert past.shape[0] == 3
    assert future.shape[0] == 4


def test_trajectory_rejects_a_single_position() -> None:
    with pytest.raises(ValueError):
        split_halves(torch.zeros(1, 2, 3, dtype=torch.complex64))


def test_trajectory_at_chance_equals_the_log_candidate_count() -> None:
    states = torch.zeros(4, 6, 5, dtype=torch.complex64)
    breakdown = trajectory_consistency_loss(states, temperature=1.0)
    assert float(breakdown.loss) == pytest.approx(math.log(6), abs=1e-5)
    assert float(breakdown.accuracy) == pytest.approx(1.0 / 6.0, abs=0.2)


def test_trajectory_hand_computed_two_sample_case() -> None:
    states = torch.zeros(4, 2, 3, dtype=torch.complex64)
    states[0, 0, 0] = 1.0
    states[2, 0, 0] = 1.0
    states[0, 1, 1] = 1.0
    states[2, 1, 1] = -1.0
    produced = float(trajectory_consistency_loss(states, temperature=0.5).loss)

    def term(positive: float, candidates: list[float]) -> float:
        scaled = [value / 0.5 for value in candidates]
        top = max(scaled)
        return -(positive / 0.5 - top - math.log(sum(math.exp(value - top) for value in scaled)))

    expected = (term(1.0, [1.0, 0.0]) + term(-1.0, [0.0, -1.0])) / 2.0
    assert produced == pytest.approx(expected, abs=1e-4)


def test_future_prediction_gap_is_positive_when_the_path_is_distinct() -> None:
    states = torch.zeros(4, 3, 4, dtype=torch.complex64)
    states[0, 0, 0] = 3.0
    states[2, 0, 0] = 2.0
    states[0, 1, 1] = 3.0
    states[2, 1, 1] = -2.0
    report = future_prediction_gap(states)
    assert report["own_similarity"] > report["other_similarity"] - 2.0


def test_composite_objective_is_the_weighted_sum(dataset, model) -> None:
    from twindyn.data.masking import sample_mask

    observation = dataset.positions[:6]
    availability = dataset.availability[:6]
    target = dataset.targets[:6]
    withheld = sample_mask(
        dataset.axis, dataset.view.availability[:6], 0.3, torch.Generator().manual_seed(0)
    )
    weights = ObjectiveWeights(lambda_mask=2.0, lambda_align=3.0, lambda_trajectory=0.5)
    with torch.no_grad():
        output = model.encode_representation(observation, availability)
        pairs = batch_alignment_pairs(dataset.anchors[:6], dataset.view.resources[:6], 4)
        breakdown = composite_objective(
            reconstruction=output.reconstructed,
            target=target,
            withheld=withheld.withheld,
            states=output.states,
            pairs=pairs,
            validity=availability.any(dim=-1).transpose(0, 1),
            weights=weights,
        )
    expected = (
        2.0 * breakdown.reconstruction + 3.0 * breakdown.alignment + 0.5 * breakdown.trajectory
    )
    assert float(breakdown.total) == pytest.approx(float(expected), abs=1e-5)


def test_zero_weights_zero_the_objective(dataset, model) -> None:
    from twindyn.data.masking import sample_mask

    observation = dataset.positions[:4]
    availability = dataset.availability[:4]
    withheld = sample_mask(
        dataset.axis, dataset.view.availability[:4], 0.3, torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        output = model.encode_representation(observation, availability)
        zeroed = composite_objective(
            reconstruction=output.reconstructed,
            target=dataset.targets[:4],
            withheld=withheld.withheld,
            states=output.states,
            pairs=[],
            validity=availability.any(dim=-1).transpose(0, 1),
            weights=ObjectiveWeights(0.0, 0.0, 0.0),
        )
        active = composite_objective(
            reconstruction=output.reconstructed,
            target=dataset.targets[:4],
            withheld=withheld.withheld,
            states=output.states,
            pairs=[],
            validity=availability.any(dim=-1).transpose(0, 1),
            weights=ObjectiveWeights(1.0, 0.0, 0.0),
        )
    assert float(zeroed.total) == 0.0
    assert float(active.total) > 0.0


def test_objective_terms_can_be_switched_off_individually(dataset, model) -> None:
    from twindyn.data.masking import sample_mask

    observation = dataset.positions[:4]
    availability = dataset.availability[:4]
    withheld = sample_mask(
        dataset.axis, dataset.view.availability[:4], 0.3, torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        output = model.encode_representation(observation, availability)
        breakdown = composite_objective(
            reconstruction=output.reconstructed,
            target=dataset.targets[:4],
            withheld=withheld.withheld,
            states=output.states,
            pairs=[],
            validity=availability.any(dim=-1).transpose(0, 1),
            weights=ObjectiveWeights(),
            active=frozenset({"mask"}),
        )
    assert float(breakdown.alignment) == 0.0
    assert float(breakdown.trajectory) == 0.0
    assert float(breakdown.reconstruction) > 0.0
