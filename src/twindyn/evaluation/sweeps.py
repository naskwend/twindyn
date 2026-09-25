"""The component ablation, the state-property sweeps and the label-efficiency arm.

Ref: Table 2 (every component is substituted one at a time against the mean
external concordance; the parameter-matched control replaces the selective
state-space operator with a static encoder of matched count so the drop is
attributable to the transition law rather than to capacity; the design also
includes a clinical-covariate-only alignment variant as a designed null and an
equal-budget row), Table 4 (trajectory length and state dimension are swept around
the selected configuration; shuffling the axis before training destroys the
ordering the operator depends on; the alignment magnitude is reported per resource
pair relative to its value at initialisation), Table 3 (the aggregated resolution),
Sec. 4.10 (the label-efficiency arm).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from twindyn.data.axis import MicroenvironmentAxis
from twindyn.metrics.decision import drop_ratio, margin_retention
from twindyn.models.sst import SelectiveStateSpaceTransition, SstConfig
from twindyn.models.static_transition import matched_transition
from twindyn.models.twindyn import TwinDyn, TwinDynConfig
from twindyn.registry import SELECTED_AXIS_LENGTH, SELECTED_STATE_DIM

VARIANT_COMPONENTS: dict[str, frozenset[str]] = {
    "full": frozenset({"mask", "align", "trajectory"}),
    "minus_masked_reconstruction": frozenset({"align", "trajectory"}),
    "minus_msi_alignment": frozenset({"mask", "trajectory"}),
    "minus_trajectory_pooling": frozenset({"mask", "align"}),
}


@dataclass(frozen=True)
class AblationVariant:
    name: str
    role: str
    components: frozenset[str] = frozenset({"mask", "align", "trajectory"})
    readout_depth: int = 1
    static_encoder: bool = False
    clinical_only_alignment: bool = False
    default_hyperparameters: bool = False
    aggregation_level: int = 0
    pooling_kind: str = "trajectory_mean"


ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("full", "reference"),
    AblationVariant("minus_sst_static_matched", "level 0 innovation", static_encoder=True),
    AblationVariant(
        "minus_msi_alignment", "level 1a", components=frozenset({"mask", "trajectory"})
    ),
    AblationVariant(
        "minus_masked_reconstruction",
        "training objective",
        components=frozenset({"align", "trajectory"}),
    ),
    AblationVariant(
        "minus_trajectory_pooling",
        "read-out input",
        components=frozenset({"mask", "align"}),
        pooling_kind="trajectory_last",
    ),
    AblationVariant("readout_depth_1_to_2", "head capacity", readout_depth=2),
    AblationVariant(
        "alignment_clinical_only",
        "designed null below the margin",
        clinical_only_alignment=True,
    ),
    AblationVariant(
        "equal_budget_default_hyperparameters", "margin retention", default_hyperparameters=True
    ),
)


@dataclass
class AblationRow:
    variant: str
    role: str
    mean_external: float
    delta: float
    parameters_millions: float

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "role": self.role,
            "mean_external": self.mean_external,
            "delta": self.delta,
            "parameters_millions": self.parameters_millions,
        }


@dataclass
class AblationReport:
    rows: list[AblationRow] = field(default_factory=list)
    drops: dict[str, float] = field(default_factory=dict)
    ratio: dict[str, object] = field(default_factory=dict)
    retention: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "drops": dict(self.drops),
            "drop_ratio": dict(self.ratio),
            "margin_retention": dict(self.retention),
        }


def assemble_ablation(
    mean_external: dict[str, float], parameters: dict[str, float]
) -> AblationReport:
    """Turn per-variant external means into the ablation table."""
    if "full" not in mean_external:
        raise ValueError("the reference variant is missing")
    reference = mean_external["full"]
    rows: list[AblationRow] = []
    drops: dict[str, float] = {}
    for variant in ABLATION_VARIANTS:
        if variant.name not in mean_external:
            continue
        value = mean_external[variant.name]
        delta = value - reference
        rows.append(
            AblationRow(
                variant=variant.name,
                role=variant.role,
                mean_external=value,
                delta=delta,
                parameters_millions=parameters.get(variant.name, float("nan")),
            )
        )
        if variant.name != "full":
            drops[variant.name] = delta
    retention = (
        margin_retention(
            -drops["minus_sst_static_matched"], -drops["equal_budget_default_hyperparameters"]
        )
        if ("minus_sst_static_matched" in drops and "equal_budget_default_hyperparameters" in drops)
        else {}
    )
    return AblationReport(
        rows=rows,
        drops=drops,
        ratio=drop_ratio({key: abs(value) for key, value in drops.items()}),
        retention=retention,
    )


@dataclass(frozen=True)
class StateSweep:
    name: str
    grid: tuple[float, ...]
    values: tuple[float, ...]
    selected: float
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "grid": list(self.grid),
            "values": list(self.values),
            "selected": self.selected,
            "note": self.note,
        }


def axis_length_sweep(values: dict[int, float]) -> StateSweep:
    grid = tuple(sorted(values))
    return StateSweep(
        name="axis_length_tau_max",
        grid=tuple(float(value) for value in grid),
        values=tuple(float(values[value]) for value in grid),
        selected=float(SELECTED_AXIS_LENGTH),
        note="plateau after 16",
    )


def state_dimension_sweep(values: dict[int, float]) -> StateSweep:
    grid = tuple(sorted(values))
    return StateSweep(
        name="state_dimension_d",
        grid=tuple(float(value) for value in grid),
        values=tuple(float(values[value]) for value in grid),
        selected=float(SELECTED_STATE_DIM),
        note="saturation after 64",
    )


def plateau_point(
    grid: tuple[float, ...], values: tuple[float, ...], tolerance: float = 0.002
) -> float:
    """The first grid point past which further growth does not exceed the tolerance."""
    if len(grid) != len(values):
        raise ValueError("grid and values must align")
    for index in range(len(grid) - 1):
        if abs(values[index + 1] - values[index]) <= tolerance:
            return float(grid[index])
    return float(grid[-1])


def shuffled_axis(axis: MicroenvironmentAxis, seed: int) -> MicroenvironmentAxis:
    """The shuffled-ordering ablation: permute the axis before any training."""
    return axis.shuffled(seed)


def build_variant_model(
    base: TwinDynConfig,
    index,
    variant: AblationVariant,
) -> tuple[TwinDyn, frozenset[str]]:
    """Construct the model a variant needs and report which terms stay active."""
    from dataclasses import replace

    config = replace(
        base,
        readout_depth=variant.readout_depth,
        pooling=variant.pooling_kind,
        static_transition=variant.static_encoder,
    )
    model = TwinDyn(config, index)
    if variant.static_encoder:
        recurrent = SelectiveStateSpaceTransition(
            SstConfig(
                state_dim=config.state_dim,
                input_features=config.state_dim,
                input_channels=config.state_dim if config.mimo else 1,
                output_channels=config.state_dim if config.mimo else 1,
                complex_state=config.complex_state,
            )
        )
        model.sst = matched_transition(
            recurrent,
            state_dim=config.state_dim,
            input_features=config.state_dim,
            output_channels=config.state_dim if config.mimo else 1,
        )
    return model, variant.components


def capacity_report(model: TwinDyn) -> dict[str, float]:
    """Parameter counts of the transition law actually in use, in millions."""
    total = float(sum(parameter.numel() for parameter in model.parameters()))
    transition = float(sum(parameter.numel() for parameter in model.sst.parameters()))
    return {
        "total_millions": total / 1e6,
        "transition_millions": transition / 1e6,
    }


def label_efficiency_differences(
    method_values: dict[float, float], comparator_values: dict[float, float]
) -> dict[float, float]:
    """Per-fraction margin over the comparator, for the shrinkage profile."""
    return {
        fraction: method_values[fraction] - comparator_values[fraction]
        for fraction in sorted(method_values)
        if fraction in comparator_values
    }


def summarise_sweep(values: dict[int, float]) -> dict[str, float]:
    finite = {key: value for key, value in values.items() if np.isfinite(value)}
    if not finite:
        return {"min": float("nan"), "max": float("nan"), "spread": float("nan")}
    return {
        "min": float(min(finite.values())),
        "max": float(max(finite.values())),
        "spread": float(max(finite.values()) - min(finite.values())),
    }
