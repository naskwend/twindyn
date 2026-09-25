"""Tree learners for the classical survival baselines.

Ref: Table 1 (the classical family: random survival forest and gradient-boosted
survival are re-implemented under the single shared protocol rather than taken
from another package, so that every row receives the same labels, the same
preprocessing and the same tuning budget).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


def log_rank_statistic(
    feature: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    threshold: float,
) -> float:
    """Two-sample log-rank statistic for a candidate split of one feature."""
    left = feature <= threshold
    right = ~left
    if left.sum() < 2 or right.sum() < 2:
        return -np.inf
    times = np.unique(time[event > 0.0])
    statistic = 0.0
    variance = 0.0
    for value in times:
        at_risk_left = float((time[left] >= value).sum())
        at_risk_right = float((time[right] >= value).sum())
        at_risk = at_risk_left + at_risk_right
        if at_risk <= 1:
            continue
        events_left = float(((time[left] == value) & (event[left] > 0.0)).sum())
        events_right = float(((time[right] == value) & (event[right] > 0.0)).sum())
        events = events_left + events_right
        if events == 0:
            continue
        expected = events * at_risk_left / at_risk
        variance += (
            events
            * (at_risk_left / at_risk)
            * (1.0 - at_risk_left / at_risk)
            * (at_risk - events)
            / max(at_risk - 1.0, 1.0)
        )
        statistic += events_left - expected
    if variance <= 0.0:
        return 0.0
    return float(statistic / np.sqrt(variance))


def nelson_aalen(
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Cumulative hazard over a fixed time grid."""
    hazard = np.zeros(grid.shape[0], dtype=np.float64)
    for position, value in enumerate(grid):
        deaths = float(((time == value) & (event > 0.0)).sum())
        at_risk = float((time >= value).sum())
        if position > 0:
            hazard[position] = hazard[position - 1]
        if at_risk > 0.0 and deaths > 0.0:
            hazard[position] += deaths / at_risk
    return hazard


@dataclass
class RegressionTree:
    """CART regression tree with variance-reduction splits, for boosting."""

    max_depth: int = 3
    min_samples_leaf: int = 8
    feature_fraction: float = 0.5
    _root: dict[str, object] | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        features: NDArray[np.float64],
        target: NDArray[np.float64],
        generator: np.random.Generator,
    ) -> RegressionTree:
        self._root = self._build(features, target, depth=0, generator=generator)
        return self

    def _build(
        self,
        features: NDArray[np.float64],
        target: NDArray[np.float64],
        depth: int,
        generator: np.random.Generator,
    ) -> dict[str, object]:
        node: dict[str, object] = {"value": float(target.mean()), "samples": target.shape[0]}
        if depth >= self.max_depth or target.shape[0] < 2 * self.min_samples_leaf:
            return node
        candidates = max(1, int(features.shape[1] * self.feature_fraction))
        chosen = generator.choice(features.shape[1], size=candidates, replace=False)
        parent = float(((target - target.mean()) ** 2).sum())
        best_gain = 0.0
        best_split: tuple[int, float] | None = None
        for column in chosen:
            values = features[:, int(column)]
            quantiles = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
            for threshold in quantiles:
                mask = values <= threshold
                if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
                    continue
                left = target[mask]
                right = target[~mask]
                sse = float(((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum())
                gain = parent - sse
                if gain > best_gain:
                    best_gain = gain
                    best_split = (int(column), float(threshold))
        if best_split is None:
            return node
        column, threshold = best_split
        mask = features[:, column] <= threshold
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return node
        node["column"] = column
        node["threshold"] = threshold
        node["left"] = self._build(features[mask], target[mask], depth + 1, generator)
        node["right"] = self._build(features[~mask], target[~mask], depth + 1, generator)
        return node

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._root is None:
            raise RuntimeError("tree has not been fitted")
        return np.asarray([self._descend(row, self._root) for row in features], dtype=np.float64)

    def _descend(self, row: NDArray[np.float64], node: dict[str, object]) -> float:
        while "column" in node:
            column = int(node["column"])
            threshold = float(node["threshold"])
            node = node["left"] if row[column] <= threshold else node["right"]  # type: ignore[assignment]
        return float(node["value"])


@dataclass
class SurvivalTree:
    """Random survival forest tree: log-rank splits and Nelson-Aalen leaves."""

    max_depth: int = 6
    min_samples_leaf: int = 10
    max_features: int = 8
    threshold_candidates: int = 8
    _root: dict[str, object] | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        features: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.float64],
        generator: np.random.Generator,
    ) -> SurvivalTree:
        self._root = self._build(features, time, event, depth=0, generator=generator)
        return self

    def _build(
        self,
        features: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.float64],
        depth: int,
        generator: np.random.Generator,
    ) -> dict[str, object]:
        grid = np.unique(time[event > 0.0])
        node: dict[str, object] = {
            "hazard": nelson_aalen(time, event, grid) if grid.size else np.zeros(0),
            "grid": grid,
            "samples": time.shape[0],
        }
        if (
            depth >= self.max_depth
            or time.shape[0] < 2 * self.min_samples_leaf
            or int(event.sum()) < 2
            or grid.size == 0
        ):
            return node
        columns = generator.choice(
            features.shape[1], size=min(self.max_features, features.shape[1]), replace=False
        )
        best_score = 0.0
        best_split: tuple[int, float] | None = None
        for column in columns:
            values = features[:, int(column)]
            quantiles = np.quantile(values, np.linspace(0.1, 0.9, self.threshold_candidates))
            for threshold in np.unique(quantiles):
                score = log_rank_statistic(values, time, event, float(threshold))
                if score > best_score:
                    best_score = score
                    best_split = (int(column), float(threshold))
        if best_split is None:
            return node
        column, threshold = best_split
        mask = features[:, column] <= threshold
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return node
        node["column"] = column
        node["threshold"] = threshold
        node["left"] = self._build(features[mask], time[mask], event[mask], depth + 1, generator)
        node["right"] = self._build(
            features[~mask], time[~mask], event[~mask], depth + 1, generator
        )
        return node

    def predict_risk(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._root is None:
            raise RuntimeError("tree has not been fitted")
        return np.asarray([self._descend(row, self._root) for row in features], dtype=np.float64)

    def _descend(self, row: NDArray[np.float64], node: dict[str, object]) -> float:
        while "column" in node:
            column = int(node["column"])
            threshold = float(node["threshold"])
            node = node["left"] if row[column] <= threshold else node["right"]  # type: ignore[assignment]
        hazard = np.asarray(node["hazard"], dtype=np.float64)
        return float(hazard.sum())
