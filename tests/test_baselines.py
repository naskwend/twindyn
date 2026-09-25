"""The comparator family: the registry, the shared protocol and the fits.

Ref: Table 1 with its caption (one protocol for every row), Sec. 4.4, Sec. 4.5
(B16 is the strongest static composition baseline).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from twindyn.evaluation.views import clinical_view, composition_view, expression_view, feature_view
from twindyn.metrics.concordance import concordance_index
from twindyn.models.baselines import (
    ABSENT_IDENTIFIERS,
    BASELINES,
    COMPARATOR_ID,
    FEATURE_VIEWS,
    BaselineProtocol,
    CoxClinical,
    FrozenFoundationLinearHead,
    GradientBoostedSurvival,
    LinearAttentionMilSurvival,
    MetadataDrivenDualGraph,
    RandomSurvivalForest,
    RegressionTree,
    StaticCompositionCox,
    StaticSharedPrivateDisentanglement,
    SurvivalTree,
    build_baseline,
    families,
    identifiers,
    log_rank_statistic,
    nelson_aalen,
    run_baseline,
    standardise,
    view_of,
)
from twindyn.models.baselines.protocol import TorchRiskEstimator


def small_problem(rows: int = 40, columns: int = 17):
    generator = np.random.default_rng(0)
    features = generator.normal(size=(rows, columns))
    time = generator.uniform(1.0, 30.0, size=rows)
    event = (generator.uniform(size=rows) > 0.25).astype(np.float64)
    return features, time, event


def test_registry_holds_every_printed_row_except_the_absent_one() -> None:
    assert len(identifiers()) == 18
    assert "B14" not in BASELINES
    assert ABSENT_IDENTIFIERS == ("B14",)
    assert COMPARATOR_ID == "B16"
    assert BASELINES["B16"].comparator is True


def test_every_registered_view_is_known() -> None:
    for identifier, spec in BASELINES.items():
        assert spec.view in FEATURE_VIEWS, identifier
        assert spec.reported_c_index > 0.0
        assert spec.reported_interval[0] < spec.reported_interval[1]


def test_families_group_the_rows() -> None:
    grouped = families()
    assert "classical" in grouped
    assert "deep survival" in grouped
    assert len(grouped["deep survival"]) == 6


def test_every_row_can_be_constructed() -> None:
    for identifier in identifiers():
        estimator = build_baseline(identifier, features=17, metadata=4, token_dim=17)
        assert estimator is not None


def test_unknown_identifier_is_rejected() -> None:
    with pytest.raises(KeyError):
        build_baseline("B99", features=17, metadata=4)


@pytest.mark.parametrize("identifier", identifiers())
def test_each_row_fits_and_scores(identifier: str) -> None:
    features, time, event = small_problem()
    estimator = build_baseline(identifier, features.shape[1], metadata=4, token_dim=17)
    protocol = BaselineProtocol(epochs=3, batch_size=16, seed=0)
    resource = np.zeros(features.shape[0], dtype=np.int64)
    fit = run_baseline(estimator, features, time, event, protocol, resource)
    assert fit.risk.shape == (features.shape[0],)
    assert np.isfinite(fit.risk).all()
    assert np.isfinite(concordance_index(fit.risk, time, event).c_index)


def test_linear_cox_matches_the_sign_of_the_signal() -> None:
    generator = np.random.default_rng(1)
    features = generator.normal(size=(80, 4))
    hazard = np.exp(features[:, 0])
    time = generator.exponential(1.0 / hazard)
    event = np.ones(80)
    estimator = CoxClinical()
    protocol = BaselineProtocol(epochs=200, learning_rate=0.1, seed=0)
    fit = run_baseline(estimator, features, time, event, protocol)
    assert concordance_index(fit.risk, time, event).c_index > 0.6


def test_static_composition_cox_is_a_linear_head() -> None:
    model = StaticCompositionCox(6)
    assert isinstance(model, TorchRiskEstimator)
    output = model.risk_from_features(torch.randn(4, 6))
    assert output.shape == (4,)


def test_gradient_boosting_reduces_its_objective() -> None:
    features, time, event = small_problem(rows=60)
    model = GradientBoostedSurvival(rounds=10)
    model.fit(
        features,
        time,
        event,
        BaselineProtocol(epochs=1, seed=0),
    )
    assert len(model.history) == 10
    assert model.history[-1] <= model.history[0] + 1e-6


def test_survival_forest_averages_its_trees() -> None:
    features, time, event = small_problem(rows=50)
    forest = RandomSurvivalForest(trees=4, max_depth=3)
    forest.fit(features, time, event, BaselineProtocol(epochs=1, seed=0))
    assert len(forest.trees) == 4
    risk = forest.predict_risk(features)
    assert risk.shape == (50,)
    assert np.isfinite(risk).all()


def test_regression_tree_learns_a_single_feature_signal() -> None:
    generator = np.random.default_rng(2)
    features = generator.normal(size=(60, 3))
    target = features[:, 0] * 2.0
    tree = RegressionTree(max_depth=2, min_samples_leaf=4)
    tree.fit(features, target, generator)
    prediction = tree.predict(features)
    assert np.corrcoef(prediction, target)[0, 1] > 0.7


def test_survival_tree_reports_a_hazard_leaf() -> None:
    generator = np.random.default_rng(3)
    features = generator.normal(size=(40, 2))
    time = generator.uniform(1.0, 10.0, size=40)
    event = np.ones(40)
    tree = SurvivalTree(max_depth=2, max_features=2, min_samples_leaf=4)
    tree.fit(features, time, event, generator)
    risk = tree.predict_risk(features)
    assert risk.shape == (40,)


def test_nelson_aalen_is_monotone() -> None:
    time = np.asarray([1.0, 2.0, 3.0, 2.0])
    event = np.asarray([1.0, 1.0, 1.0, 0.0])
    hazard = nelson_aalen(time, event, np.unique(time))
    assert all(hazard[index] <= hazard[index + 1] + 1e-12 for index in range(hazard.shape[0] - 1))


def test_log_rank_statistic_separates_two_groups() -> None:
    time = np.asarray([1.0, 2.0, 3.0, 8.0, 9.0, 10.0])
    event = np.ones(6)
    feature = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert log_rank_statistic(feature, time, event, 0.5) > 0.0
    assert log_rank_statistic(feature, time, event, -1.0) == -np.inf


def test_standardise_returns_reusable_statistics() -> None:
    features = np.random.default_rng(4).normal(size=(20, 3))
    scaled, (mean, scale) = standardise(features)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    assert scaled.std(axis=0).max() <= 2.0
    assert mean.shape == scale.shape == (3,)


def test_comparator_views_are_ordered_correctly(cohort) -> None:
    clinical = clinical_view(cohort.table)
    composition = composition_view(cohort.table)
    expression = expression_view(cohort.table)
    assert clinical.shape[0] == composition.shape[0] == expression.shape[0] == len(cohort.table)
    assert composition.shape[1] == cohort.table.compartment_count
    assert expression.shape[1] == cohort.table.width
    assert clinical.shape[1] == 8
    assert np.array_equal(feature_view(cohort.table, "composition"), composition)
    with pytest.raises(ValueError):
        feature_view(cohort.table, "unknown")


def test_view_of_matches_the_registry() -> None:
    assert view_of("B1") == "clinical"
    assert view_of("B16") == "composition"
    assert view_of("B10") == "expression"


def test_uncertainty_and_moe_heads_expose_a_risk() -> None:
    moe = build_baseline("B10", features=17, metadata=4, token_dim=17)
    risk = moe.risk_from_features(torch.randn(5, 17))
    assert risk.shape == (5,)
    mil = LinearAttentionMilSurvival(17)
    assert mil.risk_from_features(torch.randn(5, 17)).shape == (5,)


def test_dual_graph_accepts_metadata() -> None:
    model = MetadataDrivenDualGraph(features=17, metadata=4)
    risk = model.forward_with_metadata(torch.randn(3, 17), torch.eye(4)[:3])
    assert risk.shape == (3,)


def test_shared_private_split_uses_a_group() -> None:
    model = StaticSharedPrivateDisentanglement(features=17, groups=4)
    risk = model.training_forward(torch.randn(6, 17), group=torch.tensor([0, 1, 2, 3, 0, 1]))
    assert risk.shape == (6,)


def test_frozen_foundation_head_keeps_its_trunk_frozen() -> None:
    model = FrozenFoundationLinearHead(features=17)
    for parameter in model.trunk.parameters():
        assert parameter.requires_grad is False
    report = model.describe()
    assert report["frozen"] is True
    assert "no foundation checkpoint" in report["note"]


def test_metadata_group_uses_the_resource_index() -> None:
    from twindyn.models.baselines.graph import _metadata_block

    block = _metadata_block(torch.tensor([0, 2]), rows=2, width=4, device=torch.device("cpu"))
    assert float(block.sum()) == 2.0
    assert float(block[0, 0]) == 1.0
    assert float(block[1, 2]) == 1.0
