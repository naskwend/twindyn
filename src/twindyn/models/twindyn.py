"""The composed digital-twin model.

Ref: Algorithms 1 and 2; Sec. 3.3-3.5 (MSI, SST and TOR in sequence), Sec. 3.6
(equation 4 needs the reconstruction from the resulting state), Sec. 4.9 and
Table 5 (the perturbation-response evidence injects noise into the state at a
stated position and follows the subsequent evolution of the risk ordering),
Sec. 4.4 (a representation is learned once per initialisation and reused by every
read-out variant, so the representation and the read-out are separate objects).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from twindyn.data.axis import AxisIndex
from twindyn.models.msi import MsiConfig, MultiSourceInvariantEncoder
from twindyn.models.sst import SelectiveStateSpaceTransition, SstConfig
from twindyn.models.static_transition import StaticMatchedTransition
from twindyn.models.tor import (
    PoolingConfig,
    ReadoutConfig,
    TrajectoryOutcomeReadout,
    TrajectoryPool,
    state_features,
)


@dataclass(frozen=True)
class TwinDynConfig:
    state_dim: int = 64
    msi_hidden: int = 256
    msi_layers: int = 1
    msi_dropout: float = 0.0
    complex_state: bool = True
    discretisation: str = "exponential_trapezoidal"
    selective: bool = True
    mimo: bool = True
    readout_depth: int = 1
    readout_hidden: int = 64
    pooling: str = "trajectory_mean"
    pooling_target: str = "position_outputs"
    static_transition: bool = False


@dataclass
class ForwardOutput:
    """``states`` is axis-major, the scan's native layout; ``reconstructed`` is batch-major."""

    states: Tensor
    position_outputs: Tensor
    reconstructed: Tensor
    pooled: Tensor
    risk: Tensor
    step_sizes: Tensor


class TwinDyn(nn.Module):
    """MSI, then SST, then the trajectory read-out."""

    def __init__(self, config: TwinDynConfig, index: AxisIndex) -> None:
        super().__init__()
        self.config = config
        self.index = index
        self.msi = MultiSourceInvariantEncoder(
            MsiConfig(
                observation_dim=index.width,
                state_dim=config.state_dim,
                hidden=config.msi_hidden,
                layers=config.msi_layers,
                dropout=config.msi_dropout,
            ),
            index,
        )
        self.sst: SelectiveStateSpaceTransition | StaticMatchedTransition
        if config.static_transition:
            self.sst = StaticMatchedTransition(
                state_dim=config.state_dim,
                input_features=config.state_dim,
                output_channels=config.state_dim if config.mimo else 1,
            )
        else:
            self.sst = SelectiveStateSpaceTransition(
                SstConfig(
                    state_dim=config.state_dim,
                    input_features=config.state_dim,
                    input_channels=config.state_dim if config.mimo else 1,
                    output_channels=config.state_dim if config.mimo else 1,
                    complex_state=config.complex_state,
                    discretisation=config.discretisation,
                )
            )
        self.pool = TrajectoryPool(PoolingConfig(kind=config.pooling, target=config.pooling_target))
        self.tor = TrajectoryOutcomeReadout(
            ReadoutConfig(
                pool_dim=self.pool_dim, depth=config.readout_depth, hidden=config.readout_hidden
            )
        )

    @property
    def output_channels(self) -> int:
        return int(self.sst.output_channels)

    @property
    def pool_dim(self) -> int:
        if self.config.pooling_target == "position_outputs":
            return self.output_channels
        return 2 * self.config.state_dim if self.config.complex_state else self.config.state_dim

    def encode_representation(
        self,
        observations: Tensor,
        availability: Tensor,
        inject_position: int | None = None,
        inject_delta: Tensor | None = None,
    ) -> ForwardOutput:
        """MSI plus SST, the label-free path of Algorithm 1."""
        write = self.msi.encode(observations)
        scan = self.sst(
            write,
            availability=availability.any(dim=-1),
            inject_position=inject_position,
            inject_delta=inject_delta,
        )
        reconstructed = self.msi.decode(scan.states).transpose(0, 1)
        pooled = self.pool(self.pooled_features(scan.states, scan.outputs))
        return ForwardOutput(
            states=scan.states,
            position_outputs=scan.outputs,
            reconstructed=reconstructed,
            pooled=pooled,
            risk=torch.zeros(pooled.shape[0], device=pooled.device),
            step_sizes=scan.step_sizes,
        )

    def pooled_features(self, states: Tensor, outputs: Tensor) -> Tensor:
        if self.config.pooling_target == "position_outputs":
            return outputs
        return state_features(states)

    def forward(self, observations: Tensor, availability: Tensor) -> ForwardOutput:
        """Algorithm 2: the representation plus the trajectory read-out."""
        output = self.encode_representation(observations, availability)
        risk = self.tor(output.pooled)
        return ForwardOutput(
            states=output.states,
            position_outputs=output.position_outputs,
            reconstructed=output.reconstructed,
            pooled=output.pooled,
            risk=risk,
            step_sizes=output.step_sizes,
        )

    def perturbed_risk(
        self,
        observations: Tensor,
        availability: Tensor,
        position: int,
        scale: float,
        generator: torch.Generator,
    ) -> Tensor:
        """Risk after injecting Gaussian noise into the state at one position."""
        noise = (
            torch.randn(
                (observations.shape[0], self.config.state_dim),
                generator=generator,
                dtype=torch.complex64,
            )
            * scale
        )
        perturbed = self.encode_representation(
            observations, availability, inject_position=position, inject_delta=noise
        )
        return self.tor(perturbed.pooled)

    def frozen_representation_parameters(self) -> list[nn.Parameter]:
        return list(self.msi.parameters()) + list(self.sst.parameters())

    def readout_parameters(self) -> list[nn.Parameter]:
        return list(self.tor.parameters())

    def freeze_representation(self) -> None:
        for parameter in self.frozen_representation_parameters():
            parameter.requires_grad_(False)

    def unfreeze_representation(self) -> None:
        for parameter in self.frozen_representation_parameters():
            parameter.requires_grad_(True)

    def representation_state(self) -> dict[str, Tensor]:
        """The label-free payload a checkpoint carries between the two stages."""
        payload = {f"msi.{key}": value for key, value in self.msi.state_dict().items()}
        payload.update({f"sst.{key}": value for key, value in self.sst.state_dict().items()})
        return payload

    def load_representation_state(self, payload: dict[str, Tensor]) -> None:
        msi_state = {
            key[len("msi.") :]: value for key, value in payload.items() if key.startswith("msi.")
        }
        sst_state = {
            key[len("sst.") :]: value for key, value in payload.items() if key.startswith("sst.")
        }
        self.msi.load_state_dict(msi_state, strict=True)
        self.sst.load_state_dict(sst_state, strict=True)

    def parameter_count(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        representation = sum(
            parameter.numel() for parameter in self.frozen_representation_parameters()
        )
        return {
            "total": int(total),
            "representation": int(representation),
            "readout": int(total - representation),
        }

    def report(self) -> dict[str, object]:
        return {
            "parameter_count": self.parameter_count(),
            "observation_map": self.msi.observation_map_report(),
            "spectrum": self.sst.spectrum_report(),
            "readout": self.tor.weights_report(),
            "pooling": self.pool.describe(),
            "pooled_dim": self.pool_dim,
            "axis": {
                "tau_max": self.index.tau_max,
                "width": self.index.width,
                "position_dim": self.index.position_dim,
                "observable_columns": self.index.observable_columns,
            },
        }
