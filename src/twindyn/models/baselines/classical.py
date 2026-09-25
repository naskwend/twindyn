"""The classical family of the main comparison.

Ref: Table 1 rows B1-B4: Cox on clinical covariates, penalised Cox on expression,
random survival forest and gradient-boosted survival. Table 1 caption states the
rows are this study's own implementations under one protocol, so the forest and
the boosted model are fitted here rather than imported from another package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from twindyn.losses.survival import cox_partial_likelihood
from twindyn.models.baselines.protocol import BaselineProtocol
from twindyn.models.baselines.tree import RegressionTree, SurvivalTree


class LinearCox:
    """Linear proportional-hazards model fitted by gradient descent on the partial likelihood."""

    identifier = "linear_cox"
    family = "classical"

    def __init__(self, l2: float = 0.0, epochs: int = 400, lr: float = 0.05) -> None:
        if l2 < 0.0:
            raise ValueError("the penalty must be non-negative")
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.coefficients: NDArray[np.float64] | None = None
        self.intercept: float = 0.0
        self.history: list[float] = []

    def fit(
        self,
        features: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.float64],
        protocol: BaselineProtocol,
        resource: NDArray[np.int64] | None = None,
    ) -> None:
        import torch

        torch.manual_seed(protocol.seed)
        matrix = torch.tensor(features, dtype=torch.float64)
        widths = matrix.shape[1]
        weight = torch.zeros(widths, dtype=torch.float64, requires_grad=True)
        offset = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.Adam([weight, offset], lr=self.lr)
        target_time = torch.tensor(time, dtype=torch.float64)
        target_event = torch.tensor(event, dtype=torch.float64)
        self.history = []
        for _ in range(self.epochs):
            optimiser.zero_grad()
            risk = matrix @ weight + offset
            loss = (
                cox_partial_likelihood(risk, target_time, target_event).loss
                + self.l2 * weight.pow(2).sum()
            )
            loss.backward()
            optimiser.step()
            self.history.append(float(loss.detach()))
        self.coefficients = weight.detach().numpy().astype(np.float64)
        self.intercept = float(offset.detach())

    def predict_risk(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coefficients is None:
            raise RuntimeError("the model has not been fitted")
        return np.asarray(features @ self.coefficients + self.intercept, dtype=np.float64)


class CoxClinical(LinearCox):
    """B1: Cox on the clinical covariates."""

    identifier = "B1_cox_clinical"
    family = "classical"

    def __init__(self) -> None:
        super().__init__(l2=1e-3, epochs=500, lr=0.1)


class PenalisedCoxExpression(LinearCox):
    """B2: penalised Cox on the expression block."""

    identifier = "B2_penalised_cox_expression"
    family = "classical"

    def __init__(self, l2: float = 0.05) -> None:
        super().__init__(l2=l2, epochs=600, lr=0.02)


class RandomSurvivalForest:
    """B3: an ensemble of survival trees with log-rank splits."""

    identifier = "B3_random_survival_forest"
    family = "classical"

    def __init__(self, trees: int = 40, max_depth: int = 6, max_features: int = 8) -> None:
        self.tree_count = int(trees)
        self.max_depth = int(max_depth)
        self.max_features = int(max_features)
        self.trees: list[SurvivalTree] = []

    def fit(
        self,
        features: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.float64],
        protocol: BaselineProtocol,
        resource: NDArray[np.int64] | None = None,
    ) -> None:
        generator = np.random.default_rng(protocol.seed)
        self.trees = []
        for _ in range(self.tree_count):
            selector = generator.integers(0, features.shape[0], size=features.shape[0])
            tree = SurvivalTree(
                max_depth=self.max_depth,
                max_features=min(self.max_features, features.shape[1]),
            )
            tree.fit(features[selector], time[selector], event[selector], generator)
            self.trees.append(tree)

    def predict_risk(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.trees:
            raise RuntimeError("the forest has not been fitted")
        stacked = np.stack([tree.predict_risk(features) for tree in self.trees])
        return np.asarray(stacked.mean(axis=0), dtype=np.float64)


class GradientBoostedSurvival:
    """B4: componentwise boosting of shallow trees on the Cox objective."""

    identifier = "B4_gradient_boosted_survival"
    family = "classical"

    def __init__(
        self,
        rounds: int = 60,
        learning_rate: float = 0.05,
        max_depth: int = 3,
    ) -> None:
        self.rounds = int(rounds)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.trees: list[RegressionTree] = []
        self.baseline: float = 0.0
        self.history: list[float] = []

    def fit(
        self,
        features: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.float64],
        protocol: BaselineProtocol,
        resource: NDArray[np.int64] | None = None,
    ) -> None:
        import torch

        generator = np.random.default_rng(protocol.seed)
        target_time = torch.tensor(time, dtype=torch.float64)
        target_event = torch.tensor(event, dtype=torch.float64)
        self.baseline = float(np.log(np.clip(event.sum() / max(time.shape[0], 1), 1e-4, 1.0)))
        score = torch.full((features.shape[0],), self.baseline, dtype=torch.float64)
        self.trees = []
        self.history = []
        for _ in range(self.rounds):
            differentiable = score.detach().requires_grad_(True)
            loss = cox_partial_likelihood(differentiable, target_time, target_event).loss
            gradient = torch.autograd.grad(loss, differentiable)[0]
            negative_gradient = (-gradient).detach().numpy().astype(np.float64)
            tree = RegressionTree(
                max_depth=self.max_depth, feature_fraction=min(1.0, 8.0 / features.shape[1])
            )
            tree.fit(features, negative_gradient, generator)
            step = tree.predict(features)
            score = (score.detach() + self.learning_rate * torch.tensor(step)).detach()
            self.trees.append(tree)
            self.history.append(
                float(cox_partial_likelihood(score, target_time, target_event).loss)
            )

    def predict_risk(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.trees:
            raise RuntimeError("the model has not been fitted")
        score = np.full(features.shape[0], self.baseline, dtype=np.float64)
        for tree in self.trees:
            score += self.learning_rate * tree.predict(features)
        return score
