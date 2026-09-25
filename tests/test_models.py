"""Models: the shared observation map, the selective scan, the read-out and the control.

Ref: Sec. 3.3-3.5, Sec. 3.8 (the cost bound), Assumptions 1 and 3, Table 2 (the
parameter-matched control).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from twindyn.data.axis import MicroenvironmentAxis
from twindyn.models.msi import MsiConfig, MultiSourceInvariantEncoder
from twindyn.models.scan import (
    clamp_step_size,
    exponential_decay,
    exponential_trapezoidal_step,
    scan_cost,
    selective_scan,
    spectrum_separation,
)
from twindyn.models.sst import SelectiveStateSpaceTransition, SstConfig
from twindyn.models.static_transition import StaticMatchedTransition, capacity_match
from twindyn.models.tor import (
    PoolingConfig,
    ReadoutConfig,
    TrajectoryOutcomeReadout,
    TrajectoryPool,
    state_features,
)
from twindyn.models.twindyn import TwinDyn, TwinDynConfig


def test_msi_map_has_rank_equal_to_the_state_dimension(axis: MicroenvironmentAxis) -> None:
    encoder = MultiSourceInvariantEncoder(
        MsiConfig(observation_dim=axis.width, state_dim=24, layers=0), axis.index()
    )
    report = encoder.observation_map_report()
    assert int(report["rank"]) == 24
    assert report["condition_number"] > 0.0


def test_msi_pseudo_inverse_identity(axis: MicroenvironmentAxis) -> None:
    encoder = MultiSourceInvariantEncoder(
        MsiConfig(observation_dim=axis.width, state_dim=24, layers=0), axis.index()
    )
    matrix = encoder.observation_map.detach()
    inverse = torch.linalg.pinv(matrix)
    assert float((matrix @ inverse @ matrix - matrix).abs().max()) < 1e-4


def test_msi_encode_decode_shapes(axis: MicroenvironmentAxis, dataset) -> None:
    encoder = MultiSourceInvariantEncoder(
        MsiConfig(observation_dim=axis.width, state_dim=24), axis.index()
    )
    observation = dataset.positions[:3]
    written = encoder.encode(observation)
    assert written.shape == (3, axis.tau_max, 24)
    states = torch.randn(axis.tau_max, 3, 24, dtype=torch.complex64)
    assert encoder.decode(states).shape == (axis.tau_max, 3, axis.position_dim)
    assert encoder.decode_full(states).shape == (axis.tau_max, 3, axis.width)


def test_msi_takes_no_resource_argument(axis: MicroenvironmentAxis) -> None:
    import inspect

    encoder = MultiSourceInvariantEncoder(
        MsiConfig(observation_dim=axis.width, state_dim=8), axis.index()
    )
    assert set(inspect.signature(encoder.encode).parameters) == {"observations"}


def test_exponential_decay_is_the_elementwise_exponential() -> None:
    step = torch.tensor([[0.5, 1.0]])
    spectrum = torch.complex(torch.tensor([-2.0, -0.5]), torch.tensor([1.0, 0.0]))
    produced = exponential_decay(step, spectrum)
    assert torch.allclose(produced, torch.exp(step * spectrum))
    assert float(produced.abs()[0, 0]) == pytest.approx(math.exp(-1.0))


def test_trapezoidal_step_reproduces_exact_decay_without_input() -> None:
    step = torch.tensor([[0.25]])
    spectrum = torch.complex(torch.tensor([-1.0]), torch.tensor([0.0]))
    previous = torch.complex(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    zero = torch.zeros(1, 1, dtype=torch.complex64)
    produced = exponential_trapezoidal_step(
        previous, exponential_decay(step, spectrum), zero, zero, step
    )
    assert float(produced.real) == pytest.approx(math.exp(-0.25), rel=1e-6)


def test_trapezoidal_step_carries_the_previous_input() -> None:
    step = torch.tensor([[0.1]])
    spectrum = torch.complex(torch.tensor([0.0]), torch.tensor([0.0]))
    state = torch.zeros(1, 1, dtype=torch.complex64)
    decay = exponential_decay(step, spectrum)
    current = torch.complex(torch.tensor([[0.0]]), torch.tensor([[0.0]]))
    previous = torch.complex(torch.tensor([[2.0]]), torch.tensor([[0.0]]))
    produced = exponential_trapezoidal_step(state, decay, current, previous, step)
    assert float(produced.real) == pytest.approx(0.1)


def test_clamp_step_size_stays_in_range() -> None:
    raw = torch.linspace(-50.0, 50.0, 11)
    clamped = clamp_step_size(raw)
    assert bool((clamped > 0.0).all())
    assert float(clamped.min()) > 0.0
    assert float(clamped.max()) <= 4.0


def test_scan_cost_is_linear_in_the_axis_length() -> None:
    unit = scan_cost(1, 64, 64, 64)["scan"]
    for tau_max in (1, 4, 16, 64):
        assert scan_cost(tau_max, 64, 64, 64)["scan"] == tau_max * unit
    assert scan_cost(16, 64, 64, 64)["within_bound_class"]


def test_spectrum_separation_reports_a_positive_gap() -> None:
    model = SelectiveStateSpaceTransition(SstConfig(state_dim=16, input_features=16))
    report = spectrum_separation(model.spectrum.detach())
    assert report["min_spectral_gap"] > 0.0
    assert int(report["unique_count"]) == 16


def test_sst_shapes_and_dtype(axis: MicroenvironmentAxis) -> None:
    model = SelectiveStateSpaceTransition(SstConfig(state_dim=32, input_features=32))
    inputs = torch.randn(4, axis.tau_max, 32)
    output = model(inputs, availability=torch.ones(4, axis.tau_max, dtype=torch.bool))
    assert output.states.shape == (axis.tau_max, 4, 32)
    assert torch.is_complex(output.states)
    assert output.outputs.shape == (axis.tau_max, 4, 32)


def test_sst_rejects_the_wrong_input_width() -> None:
    model = SelectiveStateSpaceTransition(SstConfig(state_dim=8, input_features=8))
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 5))


def test_sst_availability_blocks_a_position(axis: MicroenvironmentAxis) -> None:
    model = SelectiveStateSpaceTransition(SstConfig(state_dim=8, input_features=8))
    inputs = torch.zeros(1, axis.tau_max, 8)
    inputs[0, 0, 0] = 5.0
    available = torch.ones(1, axis.tau_max, dtype=torch.bool)
    with torch.no_grad():
        open_scan = model(inputs, availability=available)
        blocked = available.clone()
        blocked[0, 0] = False
        closed_scan = model(inputs, availability=blocked)
    assert float((open_scan.states - closed_scan.states).abs().max()) > 0.0


def test_sst_propagates_a_perturbation_forward(axis: MicroenvironmentAxis) -> None:
    model = SelectiveStateSpaceTransition(SstConfig(state_dim=8, input_features=8))
    inputs = torch.randn(2, axis.tau_max, 8)
    noise = torch.zeros(2, 8, dtype=torch.complex64)
    noise[:, 0] = 1.0
    with torch.no_grad():
        base = model(inputs, availability=torch.ones(2, axis.tau_max, dtype=torch.bool))
        perturbed = model(
            inputs,
            availability=torch.ones(2, axis.tau_max, dtype=torch.bool),
            inject_position=0,
            inject_delta=noise,
        )
    changed = float((base.states - perturbed.states).abs().max())
    assert changed > 0.0


def test_static_control_matches_the_interface_and_drops_the_recurrence() -> None:
    recurrent = SelectiveStateSpaceTransition(SstConfig(state_dim=32, input_features=32))
    static = StaticMatchedTransition(state_dim=32, input_features=32, output_channels=32)
    inputs = torch.randn(2, 5, 32)
    with torch.no_grad():
        recurrent_states = recurrent(inputs).states
        static_states = static(inputs).states
    assert recurrent_states.shape == static_states.shape
    assert not hasattr(static, "spectrum")
    report = capacity_match(recurrent, static)
    assert report["relative_gap"] < 1.0
    assert "recurrent_parameters" in report


def test_trajectory_pooling_kinds(axis: MicroenvironmentAxis) -> None:
    features = torch.arange(axis.tau_max * 3 * 2, dtype=torch.float32).reshape(axis.tau_max, 3, 2)
    for kind in ("trajectory_mean", "trajectory_last", "trajectory_peak"):
        pool = TrajectoryPool(PoolingConfig(kind=kind))
        pooled = pool(features)
        assert pooled.shape == (3, 2)
    mean = TrajectoryPool(PoolingConfig(kind="trajectory_mean"))(features)
    assert torch.allclose(mean, features.mean(dim=0))


def test_pooling_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError):
        TrajectoryPool(PoolingConfig(kind="unknown"))


def test_readout_depth_one_and_two(axis: MicroenvironmentAxis) -> None:
    pooled = torch.randn(5, 12)
    single = TrajectoryOutcomeReadout(ReadoutConfig(pool_dim=12, depth=1))
    double = TrajectoryOutcomeReadout(ReadoutConfig(pool_dim=12, depth=2, hidden=8))
    assert single(pooled).shape == (5,)
    assert double(pooled).shape == (5,)
    assert single.weights_report()["parameters"] < double.weights_report()["parameters"]
    with pytest.raises(ValueError):
        TrajectoryOutcomeReadout(ReadoutConfig(pool_dim=12, depth=3))


def test_state_features_concatenate_real_and_imaginary(axis: MicroenvironmentAxis) -> None:
    states = torch.randn(axis.tau_max, 4, 6, dtype=torch.complex64)
    features = state_features(states)
    assert features.shape == (axis.tau_max, 4, 12)
    assert torch.allclose(features[..., :6], torch.real(states))


def test_composed_model_shapes_and_gradients(dataset, model, axis) -> None:
    observation = dataset.positions[:4]
    availability = dataset.availability[:4]
    output = model(observation, availability)
    assert output.states.shape == (axis.tau_max, 4, 16)
    assert output.reconstructed.shape == (4, axis.tau_max, axis.position_dim)
    assert output.risk.shape == (4,)
    assert output.pooled.shape[0] == 4
    model.zero_grad(set_to_none=True)
    (output.risk.pow(2).mean() + output.reconstructed.pow(2).mean()).backward()
    assert model.msi.observation_map.grad is not None
    assert model.sst.decay_rate.grad is not None
    assert model.sst.output_embedding_real.grad is not None
    assert next(model.tor.parameters()).grad is not None


def test_model_report_carries_the_expected_sections(model) -> None:
    report = model.report()
    assert {"parameter_count", "observation_map", "spectrum", "readout", "pooling", "axis"} <= set(
        report
    )
    assert report["parameter_count"]["readout"] > 0


def test_representation_state_round_trips(dataset, model) -> None:
    payload = model.representation_state()
    other = TwinDyn(TwinDynConfig(state_dim=16), dataset.axis.index())
    other.load_representation_state(payload)
    for left, right in zip(
        model.frozen_representation_parameters(),
        other.frozen_representation_parameters(),
        strict=True,
    ):
        assert torch.equal(left, right)


def test_static_transition_variant_has_no_spectrum(dataset) -> None:
    model = TwinDyn(TwinDynConfig(state_dim=16, static_transition=True), dataset.axis.index())
    report = model.report()
    assert report["spectrum"]["unique_count"] == 0.0
    output = model(dataset.positions[:2], dataset.availability[:2])
    assert output.risk.shape == (2,)


def test_analytic_cost_matches_the_shipped_bound(model) -> None:
    cost = model.sst.analytic_cost(16)
    assert cost["scan"] <= 3 * cost["paper_bound_tau_max_d2"]
    curve = model.sst.growth_curve((1000, 4000))
    assert curve[0]["ratio"] > curve[1]["ratio"]


def test_selective_scan_rejects_a_mismatched_mask() -> None:
    inputs = torch.randn(2, 4, 3)
    delta = torch.randn(2, 5, 3)
    spectrum = torch.complex(torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError):
        selective_scan(
            inputs,
            delta,
            spectrum,
            torch.ones(2, 5, 3, dtype=torch.complex64),
            torch.ones(2, 5, 3, dtype=torch.complex64),
            torch.ones(3, 3, dtype=torch.complex64),
            torch.ones(3, 3, dtype=torch.complex64),
        )


def test_scan_requires_a_position_for_an_injection() -> None:
    inputs = torch.randn(1, 3, 2)
    delta = torch.randn(1, 3, 2)
    spectrum = torch.complex(torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError):
        selective_scan(
            inputs,
            delta,
            spectrum,
            torch.ones(1, 3, 2, dtype=torch.complex64),
            torch.ones(1, 3, 2, dtype=torch.complex64),
            torch.ones(2, 2, dtype=torch.complex64),
            torch.ones(2, 2, dtype=torch.complex64),
            inject_delta=torch.ones(1, 2, dtype=torch.complex64),
        )


def test_axis_dimensions_are_reported_by_the_model(model, axis) -> None:
    report = model.report()["axis"]
    assert report["tau_max"] == axis.tau_max
    assert report["width"] == axis.width
    assert np.isfinite(report["position_dim"])
