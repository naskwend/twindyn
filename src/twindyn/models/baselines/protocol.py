"""The shared protocol every baseline row is evaluated under.

Ref: Table 1 caption (all rows are evaluated on the identical held-out sample set
with identical preprocessing, and the re-implemented values are this study's own
implementations under a single protocol), Sec. 4.4 (a single optimisation
schedule for all variants and an equal tuning budget).

The protocol fixes the feature view, the split, the number of random
initialisations and the number of fitting epochs, and it exposes one risk-scoring
interface so the classical and the network baselines can be reported side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from twindyn.losses.survival import cox_partial_likelihood


@dataclass(frozen=True)
class BaselineProtocol:
    """The equal-budget settings shared by every row of the main comparison."""

    epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    seed: int = 0
    initialisations: int = 10
    standardise: bool = True
    grad_clip: float = 1.0


@dataclass
class BaselineFit:
    """The artefacts a fitted baseline leaves behind."""

    risk: NDArray[np.float64]
    history: list[float] = field(default_factory=list)
    epochs_run: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs_run": self.epochs_run,
            "final_loss": self.history[-1] if self.history else float("nan"),
            "initial_loss": self.history[0] if self.history else float("nan"),
        }


@runtime_checkable
class RiskEstimator(Protocol):
    """Anything that can be fitted on observations and scored for risk."""

    identifier: str
    family: str

    def fit(
        self,
        features: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.float64],
        protocol: BaselineProtocol,
        resource: NDArray[np.int64] | None = None,
    ) -> None: ...

    def predict_risk(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...


def standardise(
    features: NDArray[np.float64],
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Centre and scale, returning the statistics so a scoring split can reuse them."""
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (features - mean) / scale, (mean, scale)


class TorchRiskEstimator(nn.Module):
    """Network baselines: a risk head fitted on the Cox partial likelihood."""

    identifier: str = "torch_baseline"
    family: str = "deep survival"

    def __init__(self) -> None:
        super().__init__()
        self._history: list[float] = []

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def risk_from_features(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, features: Tensor) -> Tensor:
        return self.risk_from_features(features)

    def training_forward(
        self, features: Tensor, target: Tensor | None = None, group: Tensor | None = None
    ) -> Tensor:
        """The risk used inside the fitting loop; domain baselines add their penalty here."""
        return self.risk_from_features(features)

    def fit(
        self,
        features: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.float64],
        protocol: BaselineProtocol,
        resource: NDArray[np.int64] | None = None,
    ) -> None:
        torch.manual_seed(protocol.seed)
        generator = torch.Generator().manual_seed(protocol.seed)
        result = fit_torch_survival(
            model=self,
            features=torch.tensor(features, dtype=torch.float32),
            time=torch.tensor(time, dtype=torch.float32),
            event=torch.tensor(event, dtype=torch.float32),
            protocol=protocol,
            generator=generator,
            group=None if resource is None else torch.tensor(resource, dtype=torch.long),
        )
        self._history = result.history

    def predict_risk(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        with torch.no_grad():
            risk = self.risk_from_features(torch.tensor(features, dtype=torch.float32))
        return risk.detach().cpu().numpy().astype(np.float64)


def fit_torch_survival(
    model: nn.Module,
    features: Tensor,
    time: Tensor,
    event: Tensor,
    protocol: BaselineProtocol,
    generator: torch.Generator,
    group: Tensor | None = None,
) -> BaselineFit:
    """Fit any risk scorer with the Cox partial likelihood on minibatches."""
    if not isinstance(model, TorchRiskEstimator):
        raise TypeError("the fitting loop expects a TorchRiskEstimator")
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay
    )
    history: list[float] = []
    count = features.shape[0]
    batch = min(protocol.batch_size, count)
    for _ in range(protocol.epochs):
        order = torch.randperm(count, generator=generator)
        epoch_losses: list[float] = []
        for start in range(0, count, batch):
            selector = order[start : start + batch]
            if selector.numel() < 2 or float(event[selector].sum()) == 0.0:
                continue
            optimiser.zero_grad()
            batch_risk = model.training_forward(
                features[selector],
                target=time[selector],
                group=None if group is None else group[selector],
            )
            loss = cox_partial_likelihood(batch_risk, time[selector], event[selector]).loss
            loss.backward()
            if protocol.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.grad_clip)
            optimiser.step()
            epoch_losses.append(float(loss.detach()))
        if epoch_losses:
            history.append(float(np.mean(epoch_losses)))
    with torch.no_grad():
        risk = model.risk_from_features(features).detach().cpu().numpy().astype(np.float64)
    return BaselineFit(risk=risk, history=history, epochs_run=len(history))


def run_baseline(
    estimator: RiskEstimator,
    features: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    protocol: BaselineProtocol,
    resource: NDArray[np.int64] | None = None,
) -> BaselineFit:
    """Fit one estimator and return the risk scores on the same features."""
    prepared = features
    if protocol.standardise:
        prepared, _ = standardise(features)
    estimator.fit(prepared, time, event, protocol, resource)
    risk = estimator.predict_risk(prepared)
    return BaselineFit(risk=np.asarray(risk, dtype=np.float64))


def feature_map(matrix: Tensor) -> Tensor:
    """The positive random-feature map the linear-attention baseline uses."""
    return torch.nn.functional.elu(matrix) + 1.0
