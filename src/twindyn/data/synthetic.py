"""Schema-compatible cohort synthesis from a known latent law.

The public resources the manuscript uses are large archives that are not shipped
with this repository, so the mechanisms are exercised against cohorts generated
from the same state-space law the manuscript writes down. The generator exposes
its transition operator, observation map and true latent paths, which is what
lets the identification checks compare a recovered object against closed-form
truth instead of against itself.

Ref: Sec. 3.1 (equations 1 and 2), Sec. 3.2 (the axis organises the regions of a
sample and each position is a partial observation of the state), Assumption 1
(one observation map shared across resources with rank d), Assumption 2
(resource overlap), Assumption 3 (simple separated spectrum), Sec. 4.1 (four
resources, one training and three external from different repositories),
Sec. 4.9 (resource identity is readable from the raw composition and
near-unreadable from the learned state).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch

from twindyn.data.axis import MicroenvironmentAxis
from twindyn.data.compartments import normalise_fractions
from twindyn.data.resources import EXTERNAL_RESOURCES, RESOURCES, TRAINING_RESOURCE
from twindyn.data.schema import ClinicalStratum, CohortTable, SampleRecord, SurvivalLabel


@dataclass(frozen=True)
class LatentLaw:
    """Equation (2) with a shared observation map and a separated spectrum."""

    transition: torch.Tensor
    input_matrix: torch.Tensor
    observation_map: torch.Tensor
    noise: float

    @property
    def state_dim(self) -> int:
        return int(self.transition.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.input_matrix.shape[1])

    @property
    def observation_dim(self) -> int:
        return int(self.observation_map.shape[0])

    def states(self, inputs: torch.Tensor, initial: torch.Tensor | None = None) -> torch.Tensor:
        """Forward recurrence z(tau) = a * z(tau-1) + B u(tau) over the axis."""
        horizon = int(inputs.shape[0])
        state = (
            torch.zeros(self.state_dim, dtype=torch.complex128)
            if initial is None
            else initial.to(torch.complex128)
        )
        path = torch.zeros((horizon, self.state_dim), dtype=torch.complex128)
        for index in range(horizon):
            state = self.transition * state + self.input_matrix @ inputs[index].to(torch.complex128)
            path[index] = state
        return path

    def closed_form_states(
        self, inputs: torch.Tensor, initial: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Prefix-sum form of the same path, used as an independent reference.

        Folding the driver through the input matrix gives
        z_i = a^i * cumsum_s(a^-s * B u_s), which is evaluated by prefix summation
        rather than by the recurrence the operator uses.
        """
        horizon = int(inputs.shape[0])
        drivers = torch.stack(
            [self.input_matrix @ inputs[index].to(torch.complex128) for index in range(horizon)]
        )
        exponents = torch.arange(horizon, dtype=torch.float64)
        decay = self.transition.unsqueeze(0) ** exponents.unsqueeze(1)
        path = decay * torch.cumsum(drivers / decay, dim=0)
        if initial is not None:
            weights = (exponents + 1).unsqueeze(1)
            path = path + (self.transition.unsqueeze(0) ** weights) * initial.to(torch.complex128)
        return path

    def observe_path(self, states: torch.Tensor, noise_seed: int | None = None) -> torch.Tensor:
        """The full observation map evaluated along the path, shape (horizon, p)."""
        if states.dim() != 2:
            raise ValueError("states must be (horizon, state_dim)")
        observed = torch.real(states.to(torch.complex128) @ self.observation_map.t())
        if self.noise > 0.0:
            generator = np.random.default_rng(0 if noise_seed is None else noise_seed)
            jitter = generator.normal(0.0, self.noise, size=tuple(observed.shape))
            if not np.isfinite(jitter).all():
                raise RuntimeError("non-finite observation noise")
            observed = observed + torch.tensor(jitter)
        return observed


def build_law(
    state_dim: int,
    observation_dim: int,
    input_dim: int,
    seed: int,
    noise: float = 0.05,
    spectrum_gap: float = 1.0,
) -> LatentLaw:
    """A simple separated complex spectrum and a full-rank observation map."""
    generator = np.random.default_rng(seed)
    angles = np.linspace(-np.pi / 3.0, np.pi / 3.0, state_dim)
    radius = 0.62 + spectrum_gap * np.linspace(0.0, 0.35, state_dim)
    transition = torch.tensor(radius * np.exp(1j * angles * spectrum_gap), dtype=torch.complex128)
    input_matrix = torch.tensor(
        (
            generator.normal(0.0, 1.0, size=(state_dim, input_dim))
            + 1j * generator.normal(0.0, 1.0, size=(state_dim, input_dim))
        )
        / np.sqrt(input_dim),
        dtype=torch.complex128,
    )
    observation_map = torch.tensor(
        (
            generator.normal(0.0, 1.0, size=(observation_dim, state_dim))
            + 1j * generator.normal(0.0, 1.0, size=(observation_dim, state_dim))
        )
        / np.sqrt(state_dim),
        dtype=torch.complex128,
    )
    return LatentLaw(transition, input_matrix, observation_map, noise)


def resource_acquisition(
    resource: str, width: int, magnitude: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-resource detection gain and additive offset.

    The gain varies across observation coordinates, which is what keeps resource
    identity readable from the raw composition after the fractions are normalised
    and is the shortcut the alignment term is asked to remove. The generator is
    keyed on a stable checksum of the resource name rather than on the interpreter's
    string hash, so a rerun in a fresh process reproduces the same cohorts.
    """
    generator = np.random.default_rng(zlib.crc32(resource.encode("utf-8")))
    gain = 1.0 + magnitude * 1.5 * np.abs(generator.normal(0.0, 1.0, size=width))
    offset = magnitude * generator.normal(0.0, 1.0, size=width) / np.sqrt(width)
    return gain, offset


def trajectory_summary(states: torch.Tensor) -> torch.Tensor:
    """The pooled trajectory summary g(z(1..tau_max)) used by the read-out."""
    return states.mean(dim=0)


def risk_from_states(states: torch.Tensor, coefficients: torch.Tensor) -> float:
    summary = trajectory_summary(states)
    real = torch.cat([torch.real(summary), torch.imag(summary)])
    return float(torch.dot(real, coefficients.to(torch.float64)))


@dataclass
class SyntheticBundle:
    """Generated cohorts plus the closed-form objects the checks compare against."""

    law: LatentLaw
    table: CohortTable
    axis: MicroenvironmentAxis
    true_states: dict[str, torch.Tensor]
    true_risk: dict[str, float]
    risk_coefficients: torch.Tensor
    acquisition_shifts: dict[str, np.ndarray]
    block_indices: dict[str, np.ndarray]


def _region_observation(full: torch.Tensor, axis: MicroenvironmentAxis) -> torch.Tensor:
    """Keep, per axis position, only the coordinates that position observes."""
    width = full.shape[1]
    out = torch.zeros(width, dtype=full.dtype)
    for spec in axis.positions:
        columns = [
            *spec.compartment_columns,
            *[axis.compartment_total + column for column in spec.morphology_columns],
        ]
        for column in columns:
            if 0 <= column < width:
                out[column] = full[spec.index, column]
    return out


def synthesise_bundle(
    state_dim: int,
    axis: MicroenvironmentAxis,
    samples_per_resource: int,
    seed: int,
    resource_shift_magnitude: float = 1.0,
    noise: float = 0.05,
    censoring_scale: float = 1.6,
    time_scale: float = 100.0,
    histology_resources: tuple[str, ...] = ("tcga_kirc", "cptac_ccrcc"),
) -> SyntheticBundle:
    """Generate a four-resource cohort set from one shared latent law."""
    generator = np.random.default_rng(seed)
    input_dim = 8
    law = build_law(state_dim, axis.width, input_dim, seed, noise=noise)
    risk_coefficients = torch.tensor(
        generator.normal(0.0, 1.0, size=2 * state_dim) / np.sqrt(2 * state_dim),
        dtype=torch.float64,
    )
    if float(risk_coefficients.abs().sum()) == 0.0:
        raise RuntimeError("degenerate risk coefficients")

    shifts: dict[str, np.ndarray] = {}
    records: list[SampleRecord] = []
    true_states: dict[str, torch.Tensor] = {}
    true_risk: dict[str, float] = {}

    stage_weights = {
        "tcga_kirc": np.array([0.45, 0.22, 0.22, 0.11]),
        "cptac_ccrcc": np.array([0.40, 0.25, 0.22, 0.13]),
        "gse29609": np.array([0.36, 0.28, 0.23, 0.13]),
        "emtab1980": np.array([0.42, 0.24, 0.22, 0.12]),
    }
    grade_weights = np.array([0.18, 0.40, 0.28, 0.14])

    for resource in (TRAINING_RESOURCE, *EXTERNAL_RESOURCES):
        spec = RESOURCES[resource]
        gain, offset = resource_acquisition(resource, axis.width, resource_shift_magnitude)
        shifts[resource] = offset
        if spec.role.value == "training":
            count = samples_per_resource
        else:
            count = int(spec.reported_samples or samples_per_resource)
        for local in range(count):
            sample_id = f"{resource}-{local:04d}"
            factors = torch.tensor(
                generator.normal(0.0, 1.0, size=(axis.tau_max, input_dim)), dtype=torch.float64
            )
            states = law.states(factors)
            full = law.observe_path(states, noise_seed=seed + local)
            observation = _region_observation(full, axis).numpy() * gain + offset
            compartments = normalise_fractions(observation[: axis.compartment_total])
            morphology = None
            if resource in histology_resources and axis.morphology_total:
                morphology = observation[axis.compartment_total :].copy()
            risk = risk_from_states(states, risk_coefficients)
            hazard = float(np.exp(np.clip(risk, -4.0, 4.0)))
            event_time = float(generator.exponential(1.0 / max(hazard, 1e-6))) * time_scale
            censor_time = float(generator.exponential(censoring_scale)) * time_scale
            event = event_time <= censor_time
            observed = min(event_time, censor_time)
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    resource=resource,
                    compartments=np.asarray(compartments, dtype=np.float64),
                    morphology=morphology,
                    label=SurvivalLabel(time=max(observed, 1e-3), event=bool(event)),
                    stratum=ClinicalStratum(
                        grade=int(generator.choice([1, 2, 3, 4], p=grade_weights)),
                        stage=int(generator.choice([1, 2, 3, 4], p=stage_weights[resource])),
                    ),
                )
            )
            true_states[sample_id] = states
            true_risk[sample_id] = risk

    table = CohortTable(width=axis.width, compartment_count=axis.compartment_total, samples=records)
    block_indices = {
        resource: np.array(
            [index for index, record in enumerate(records) if record.resource == resource],
            dtype=np.int64,
        )
        for resource in (TRAINING_RESOURCE, *EXTERNAL_RESOURCES)
    }
    return SyntheticBundle(
        law=law,
        table=table,
        axis=axis,
        true_states=true_states,
        true_risk=true_risk,
        risk_coefficients=risk_coefficients,
        acquisition_shifts=shifts,
        block_indices=block_indices,
    )


def anchored_block(bundle: SyntheticBundle, resource: str) -> np.ndarray:
    """The compartment block used as the biological-equivalence anchor space."""
    indices = bundle.block_indices[resource]
    observations = bundle.table.observations()[indices]
    return observations[:, : bundle.table.compartment_count]
