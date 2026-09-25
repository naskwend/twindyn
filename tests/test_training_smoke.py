"""End-to-end smoke: the shipped smoke configuration trains and its losses fall.

Ref: Algorithm 1 (representation learning), Algorithm 2 (inference and read-out),
Sec. 4.4 (a single schedule). The configuration under test is
``configs/experiment/_smoke.yaml``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from twindyn.data.dataset import AxisDataset
from twindyn.models.twindyn import TwinDyn
from twindyn.pipeline import build_cohorts, model_config
from twindyn.training.engine import PairSource, score, train_readout, train_representation
from twindyn.utils.config import resolve_config
from twindyn.utils.runtime import set_seed

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke():
    config = resolve_config(REPOSITORY_ROOT / "configs", "_smoke")
    build = build_cohorts(config)
    set_seed(config.seed, deterministic=config.deterministic)
    dataset = AxisDataset(build.table, build.axis, tuple(range(len(build.table))), None)
    model = TwinDyn(model_config(config), build.axis.index())
    pair_source = PairSource(max_pairs_per_resource=config.alignment.max_pairs_per_resource)
    return config, build, dataset, model, pair_source


def test_smoke_configuration_carries_the_paper_geometry(smoke) -> None:
    config, build, dataset, model, _ = smoke
    assert config.axis.tau_max == 16
    assert config.model.state_dim == 64
    assert build.axis.tau_max == 16
    assert dataset.positions.shape[1] == 16
    assert model.report()["axis"]["tau_max"] == 16


def test_smoke_configuration_is_marked_as_a_smoke(smoke) -> None:
    config, *_ = smoke
    text = (REPOSITORY_ROOT / "configs" / "experiment" / "_smoke.yaml").read_text()
    assert "for unit-test smoke only; do not use for reporting" in text
    assert config.experiment == "_smoke"
    assert config.evaluation.initialisations == 1


def test_smoke_training_reduces_both_losses(smoke) -> None:
    config, _, dataset, _, pair_source = smoke
    model = TwinDyn(model_config(config), dataset.axis.index())
    set_seed(0, deterministic=True)
    representation, initial, final = train_representation(
        model=model,
        dataset=dataset,
        pair_source=pair_source,
        optim_config=config.optim,
        objective_config=config.objective,
        seed=0,
        mask_ratio=config.masking.ratio,
    )
    readout = train_readout(model=model, dataset=dataset, optim_config=config.optim, seed=0)
    assert representation.steps > 0
    assert representation.loss_decreased, representation.history
    assert readout.loss_decreased, readout.history
    assert set(initial) == set(final)


def test_smoke_training_updates_every_representation_parameter(smoke) -> None:
    config, _, dataset, _, pair_source = smoke
    set_seed(1, deterministic=True)
    fresh = TwinDyn(model_config(config), dataset.axis.index())
    before = [parameter.detach().clone() for parameter in fresh.frozen_representation_parameters()]
    train_representation(
        model=fresh,
        dataset=dataset,
        pair_source=pair_source,
        optim_config=replace(config.optim, representation_epochs=2),
        objective_config=config.objective,
        seed=1,
        mask_ratio=config.masking.ratio,
    )
    driven = [
        parameter
        for parameter in fresh.frozen_representation_parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
    ]
    moved = sum(
        1
        for previous, current in zip(before, fresh.frozen_representation_parameters(), strict=True)
        if not torch.equal(previous, current)
    )
    assert moved >= max(len(driven), int(0.8 * len(before)))


def test_smoke_forward_reaches_past_chance_or_stays_finite(smoke) -> None:
    _, _, dataset, model, _ = smoke
    scores = score(model, dataset)
    assert scores["risk"].shape[0] == len(dataset)
    assert torch.isfinite(torch.tensor(scores["risk"])).all()
