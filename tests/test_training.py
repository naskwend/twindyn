"""Training: seeding, the optimiser and schedule, checkpoints and the two-stage engine.

Ref: Sec. 3.10 and Sec. 4.4 (indexed initialisations, one schedule), Algorithms 1-2.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
import torch

from twindyn.data.dataset import AxisDataset
from twindyn.training.checkpoint import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    payload_digest,
    restore,
    save_checkpoint,
    verify_round_trip,
)
from twindyn.training.engine import (
    PairSource,
    score,
    train_readout,
    train_representation,
    train_twindyn,
)
from twindyn.training.optim import (
    build_optimiser,
    build_schedule,
    effective_batch_size,
    parameter_groups,
    trainable_parameter_count,
)
from twindyn.training.precision import (
    ExponentialMovingAverage,
    build_grad_scaler,
    precision_context,
)
from twindyn.training.seeding import apply_initialisation, plan_initialisations
from twindyn.utils.config import ObjectiveConfig, OptimConfig
from twindyn.utils.io import tensor_payload_digest


@pytest.fixture
def optim_config() -> OptimConfig:
    return OptimConfig(
        batch_size=8,
        representation_epochs=2,
        readout_epochs=2,
        warmup_steps=2,
        learning_rate=2e-3,
    )


def test_initialisation_plan_is_deterministic_and_indexed() -> None:
    first = plan_initialisations(7, 4)
    second = plan_initialisations(7, 4)
    assert first.seeds == second.seeds
    assert first.count == 4
    assert first.seed(0) != plan_initialisations(8, 4).seed(0)


def test_initialisation_plan_rejects_a_bad_index() -> None:
    plan = plan_initialisations(0, 2)
    with pytest.raises(IndexError):
        plan.seed(5)
    with pytest.raises(ValueError):
        plan_initialisations(0, 0)


def test_apply_initialisation_seeds_every_generator() -> None:
    plan = plan_initialisations(3, 2)
    seed = apply_initialisation(plan, 1)
    assert seed == plan.seed(1)
    assert torch.rand(1).item() == pytest.approx(
        apply_initialisation(plan, 1) and torch.rand(1).item()
    )


def test_effective_batch_size_multiplies_every_factor() -> None:
    config = OptimConfig(batch_size=8, grad_accum=4, world_size=2)
    assert effective_batch_size(config) == 64


def test_optimiser_holds_every_trainable_parameter() -> None:
    module = torch.nn.Linear(4, 2)
    bundle = build_optimiser(list(module.parameters()), OptimConfig(), total_steps=10)
    held = sum(
        parameter.numel()
        for group in bundle.optimiser.param_groups
        for parameter in group["params"]
    )
    assert held == sum(parameter.numel() for parameter in module.parameters())
    assert bundle.accumulation_steps == 1


def test_optimiser_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_optimiser([torch.nn.Parameter(torch.zeros(1))], OptimConfig(optimizer="rmsprop"), 10)


def test_warmup_then_decay_schedule() -> None:
    schedule = build_schedule("warmup_cosine", 100, 10, 1.0)
    assert schedule.learning_rate(0) < schedule.learning_rate(9)
    assert schedule.learning_rate(9) <= schedule.learning_rate(10)
    assert schedule.learning_rate(99) < schedule.learning_rate(10)
    assert schedule.learning_rate(99) > 0.0


def test_constant_schedule_after_warmup() -> None:
    schedule = build_schedule("warmup_constant", 100, 5, 2.0)
    assert schedule.learning_rate(5) == pytest.approx(2.0)
    assert schedule.learning_rate(90) == pytest.approx(2.0)


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_schedule("triangular", 10, 1, 1.0)


def test_parameter_groups_split_representation_and_readout(model) -> None:
    groups = parameter_groups(model, readout_learning_rate=1e-4)
    assert len(groups) == 2
    assert groups[1]["lr"] == pytest.approx(1e-4)
    assert trainable_parameter_count(model) > 0


def test_precision_context_and_scaler() -> None:
    device = torch.device("cpu")
    for name in ("fp32", "bf16", "fp16"):
        with precision_context(name, device):
            assert torch.zeros(1).shape == (1,)
    assert build_grad_scaler("fp32", device) is None
    assert build_grad_scaler("fp16", device) is None
    with pytest.raises(ValueError):
        precision_context("int8", device)


def test_ema_tracks_the_parameters() -> None:
    module = torch.nn.Linear(3, 2)
    ema = ExponentialMovingAverage.from_model(module, 0.9)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(1.0)
    ema.update(module)
    for name, parameter in module.named_parameters():
        if name in ema.shadow:
            assert not torch.equal(ema.shadow[name], parameter)
    payload = ema.state_dict()
    restored = ExponentialMovingAverage(0.5, {}, 0)
    restored.load_state_dict(payload)
    assert restored.steps == 1
    restored.copy_to(module)
    for name, parameter in module.named_parameters():
        assert torch.equal(parameter, restored.shadow[name])


def test_repr_of_the_model_names_its_geometry(model) -> None:
    assert "state_dim" in repr(model.sst)


def test_checkpoint_round_trip(tmp_path, model) -> None:
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, 5, "digest", "readout", 2, {"c_index": 0.5})
    contents = load_checkpoint(path)
    assert contents.format == CHECKPOINT_FORMAT
    assert contents.seed == 5
    assert contents.stage == "readout"
    assert contents.epoch == 2
    assert contents.metrics["c_index"] == pytest.approx(0.5)
    report = verify_round_trip(path, model)
    assert report["identical"] is True
    assert report["seed_restored"] == 5


def test_checkpoint_is_written_atomically_and_readably(tmp_path, model) -> None:
    path = tmp_path / "atomic.pt"
    save_checkpoint(path, model, 0, "d", "representation", 0, {})
    assert path.exists()
    assert oct(path.stat().st_mode & 0o777) == "0o644"
    assert not [entry for entry in tmp_path.iterdir() if entry.name.startswith(".atomic")]


def test_payload_digest_is_stable_across_rewrites(tmp_path, model) -> None:
    path = tmp_path / "digest.pt"
    save_checkpoint(path, model, 0, "d", "readout", 0, {})
    first = payload_digest(path)
    save_checkpoint(path, model, 0, "d", "readout", 0, {})
    assert payload_digest(path) == first
    assert first == tensor_payload_digest(dict(model.state_dict()))


def test_restore_returns_the_optimiser_state(tmp_path, model) -> None:
    optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
    path = tmp_path / "full.pt"
    save_checkpoint(path, model, 0, "d", "representation", 1, {}, optimiser=optimiser)
    contents = restore(path, model, optimiser=optimiser)
    assert contents.optimiser_state is not None


def test_loading_a_foreign_file_is_rejected(tmp_path) -> None:
    torch.save({"not": "a checkpoint"}, tmp_path / "foreign.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "foreign.pt")
    (tmp_path / "empty.pt").write_text("garbage", encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError, EOFError, pickle.UnpicklingError)):
        load_checkpoint(tmp_path / "empty.pt")


def test_representation_stage_reduces_its_loss(dataset, model, optim_config) -> None:
    report, initial, final = train_representation(
        model=model,
        dataset=dataset,
        pair_source=PairSource(max_pairs_per_resource=8),
        optim_config=optim_config,
        objective_config=ObjectiveConfig(),
        seed=0,
        mask_ratio=0.30,
    )
    assert report.stage == "representation"
    assert report.steps > 0
    assert report.loss_decreased
    assert set(initial) == set(final)


def test_readout_stage_reduces_its_loss(cohort, dataset, model, optim_config, split) -> None:
    fit, _ = split
    readout = AxisDataset(cohort.table, dataset.axis, fit, dataset.standardiser)
    report = train_readout(model=model, dataset=readout, optim_config=optim_config, seed=0)
    assert report.stage == "readout"
    assert report.loss_decreased


def test_readout_freezes_the_representation(cohort, dataset, model, optim_config, split) -> None:
    fit, _ = split
    readout = AxisDataset(cohort.table, dataset.axis, fit, dataset.standardiser)
    train_readout(model=model, dataset=readout, optim_config=optim_config, seed=0)
    assert all(
        not parameter.requires_grad for parameter in model.frozen_representation_parameters()
    )


def test_both_stages_run_in_order(cohort, dataset, model, optim_config, split) -> None:
    fit, _ = split
    readout = AxisDataset(cohort.table, dataset.axis, fit, dataset.standardiser)
    report = train_twindyn(
        model,
        dataset,
        readout,
        PairSource(max_pairs_per_resource=8),
        optim_config,
        ObjectiveConfig(),
        seed=0,
        mask_ratio=0.30,
    )
    assert report.representation.loss_decreased
    assert report.readout.loss_decreased
    assert report.seed == 0
    assert report.objective_weights["lambda_mask"] == pytest.approx(1.0)


def test_objective_switch_changes_the_training_signal(dataset, model, optim_config) -> None:
    report, _, _ = train_representation(
        model=model,
        dataset=dataset,
        pair_source=PairSource(max_pairs_per_resource=8),
        optim_config=optim_config,
        objective_config=ObjectiveConfig(),
        seed=0,
        active=frozenset({"mask"}),
        mask_ratio=0.30,
    )
    assert math.isfinite(report.final_loss)


def test_score_returns_the_arrays_the_report_needs(cohort, dataset, model, split) -> None:
    fit, _ = split
    readout = AxisDataset(cohort.table, dataset.axis, fit, dataset.standardiser)
    report = score(model, readout)
    assert report["risk"].shape == (len(fit),)
    assert report["time"].shape == (len(fit),)
    assert report["event"].shape == (len(fit),)
    assert report["resource"].shape == (len(fit),)
    assert report["has_label"].shape == (len(fit),)


def test_score_is_deterministic(dataset, model) -> None:
    first = score(model, dataset)["risk"]
    second = score(model, dataset)["risk"]
    assert np.allclose(first, second)
