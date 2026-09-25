"""Linear probe over frozen features.

Ref: Sec. 4.9 and Table 5 (a linear classifier predicts which resource issued a
sample; it reaches 0.612 accuracy on the raw composition parameters and 0.271 on
the learned state against a chance level of 0.25 for four resources; the split is
50/50 and repeated over ten random initialisations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ProbeResult:
    accuracy: float
    standard_error: float
    repeats: int
    chance_level: float
    classes: int
    per_repeat: tuple[float, ...]

    @property
    def interval(self) -> tuple[float, float]:
        half_width = 1.96 * self.standard_error / max(np.sqrt(self.repeats), 1.0)
        return (self.accuracy - half_width, self.accuracy + half_width)


@dataclass(frozen=True)
class ProbeConfig:
    repeats: int = 10
    test_fraction: float = 0.5
    weight_decay: float = 1e-4
    epochs: int = 300
    learning_rate: float = 0.05
    seed: int = 0


def _standardise(train: Tensor, test: Tensor) -> tuple[Tensor, Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / scale, (test - mean) / scale


def resource_probe(
    features: Tensor,
    labels: Tensor,
    classes: int,
    config: ProbeConfig,
) -> ProbeResult:
    """Multinomial logistic regression on a 50/50 split, repeated over seeds."""
    if features.dim() != 2:
        raise ValueError("features must be (samples, dim)")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels must match the feature rows")
    accuracies: list[float] = []
    for repeat in range(config.repeats):
        generator = torch.Generator().manual_seed(config.seed + repeat)
        order = torch.randperm(features.shape[0], generator=generator)
        cut = round(features.shape[0] * (1.0 - config.test_fraction))
        train_index = order[:cut]
        test_index = order[cut:]
        train_x, test_x = _standardise(features[train_index], features[test_index])
        train_y = labels[train_index]
        test_y = labels[test_index]
        classifier = torch.nn.Linear(features.shape[1], classes)
        optimiser = torch.optim.Adam(
            classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(config.epochs):
            optimiser.zero_grad()
            loss = loss_fn(classifier(train_x), train_y)
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            predictions = classifier(test_x).argmax(dim=1)
            accuracies.append(float((predictions == test_y).to(torch.float32).mean()))
    array = np.asarray(accuracies, dtype=np.float64)
    return ProbeResult(
        accuracy=float(array.mean()),
        standard_error=float(array.std(ddof=1)) if array.size > 1 else 0.0,
        repeats=config.repeats,
        chance_level=1.0 / classes,
        classes=classes,
        per_repeat=tuple(float(value) for value in array),
    )


def invariance_gap(raw: ProbeResult, learned: ProbeResult) -> dict[str, float]:
    """How far the learned state moves resource identity towards chance."""
    return {
        "raw_accuracy": raw.accuracy,
        "learned_accuracy": learned.accuracy,
        "chance_level": raw.chance_level,
        "raw_minus_chance": raw.accuracy - raw.chance_level,
        "learned_minus_chance": learned.accuracy - learned.chance_level,
        "readability_removed": raw.accuracy - learned.accuracy,
    }
