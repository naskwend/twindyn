"""Configuration dataclasses and YAML composition.

Ref: Sec. 3.10, Sec. 4.4 (the single optimisation schedule shared by all
variants and the quantities chosen on internal validation).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class AxisConfig:
    """Ordered-axis construction. ``tau_max`` is the paper's selected value."""

    tau_max: int = 16
    ordering_rule: str = "predefined_compartment_index"
    shuffle_positions: bool = False
    shuffle_seed: int = 0
    aggregation_level: int = 0


@dataclass(frozen=True)
class FeatureSpaceConfig:
    """Shared observable feature space. The intersection never imputes."""

    mode: str = "intersection"
    min_coverage: float = 0.0
    max_features: int = 512
    morphology_features: int = 48
    standardise: bool = True


@dataclass(frozen=True)
class DeconvolutionConfig:
    """Fixed deconvolution configuration, never tuned on any outcome."""

    method: str = "reference_nnls"
    reference: str = "ccrcc_compartment_reference_v1"
    marker_genes_per_compartment: int = 40
    gene_panel_size: int = 2000
    ridge: float = 0.0
    normalise_fractions: bool = True
    log_transform: bool = True


@dataclass(frozen=True)
class MaskingConfig:
    """Masked multi-source reconstruction. The ratio is an engineering default."""

    ratio: float = 0.30
    strategy: str = "position_uniform"
    min_positions_kept: int = 2


@dataclass(frozen=True)
class AlignmentConfig:
    """Cross-resource alignment pairing."""

    mode: str = "anchor_matched"
    anchor_compartments: bool = True
    clinical_only_null: bool = False
    max_pairs_per_resource: int = 4096


@dataclass(frozen=True)
class ModelConfig:
    """MSI, SST and TOR geometry."""

    state_dim: int = 64
    msi_hidden: int = 256
    msi_layers: int = 2
    msi_dropout: float = 0.0
    complex_state: bool = True
    discretisation: str = "exponential_trapezoidal"
    selective: bool = True
    mimo: bool = True
    readout_depth: int = 1
    readout_hidden: int = 64
    pooling: str = "trajectory_mean"


@dataclass(frozen=True)
class ObjectiveConfig:
    """Equation (3) weights. Values are engineering defaults."""

    lambda_mask: float = 1.0
    lambda_align: float = 1.0
    lambda_trajectory: float = 0.5
    trajectory_temperature: float = 0.1


@dataclass(frozen=True)
class DataConfig:
    """Resource roots and split protocol."""

    root: str = "data"
    resources: tuple[str, ...] = ("tcga_kirc", "cptac_ccrcc", "gse29609", "emtab1980")
    training_resource: str = "tcga_kirc"
    external_resources: tuple[str, ...] = ("cptac_ccrcc", "gse29609", "emtab1980")
    internal_validation_fraction: float = 0.20
    synthetic_samples_per_resource: int = 96
    synthetic_resource_shift: float = 1.0
    synthetic_seed: int = 0
    max_missing_fraction: float = 0.0


@dataclass(frozen=True)
class OptimConfig:
    """Optimisation schedule shared by every variant."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    readout_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_steps: int = 100
    grad_clip: float = 1.0
    precision: str = "fp32"
    batch_size: int = 64
    grad_accum: int = 1
    world_size: int = 1
    representation_epochs: int = 60
    readout_epochs: int = 40
    ema_decay: float = 0.0
    early_stopping_patience: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    """Reporting protocol."""

    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95
    initialisations: int = 10
    correction: str = "holm_bonferroni"
    external_threshold: float = 0.03
    decision_margin: float = 0.005
    probe_repeats: int = 10
    perturbation_scales: tuple[float, ...] = (0.25, 1.0)
    label_fractions: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
    random_state: int = 0


@dataclass(frozen=True)
class RunConfig:
    """Fully resolved experiment configuration."""

    experiment: str = "main"
    output_dir: str = "runs"
    seed: int = 0
    device: str = "cpu"
    deterministic: bool = True
    axis: AxisConfig = field(default_factory=AxisConfig)
    features: FeatureSpaceConfig = field(default_factory=FeatureSpaceConfig)
    deconvolution: DeconvolutionConfig = field(default_factory=DeconvolutionConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    variant: str = "full"


_CONFIG_TYPES: dict[str, type[Any]] = {
    "axis": AxisConfig,
    "features": FeatureSpaceConfig,
    "deconvolution": DeconvolutionConfig,
    "masking": MaskingConfig,
    "alignment": AlignmentConfig,
    "model": ModelConfig,
    "objective": ObjectiveConfig,
    "data": DataConfig,
    "optim": OptimConfig,
    "evaluation": EvaluationConfig,
}


def _coerce_scalar(current: Any, value: Any) -> Any:
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _merge_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
    updates: dict[str, Any] = {}
    kind = type(instance)
    known = {f.name: f for f in fields(instance)}
    for key, value in payload.items():
        if key not in known:
            raise KeyError(f"{kind.__name__} has no field {key!r}")
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            updates[key] = _merge_dataclass(current, value)
        else:
            updates[key] = _coerce_scalar(current, value)
    return kind(**{**asdict_shallow(instance), **updates})


def asdict_shallow(instance: Any) -> dict[str, Any]:
    return {f.name: getattr(instance, f.name) for f in fields(instance)}


def run_config_from_mapping(payload: dict[str, Any]) -> RunConfig:
    """Build a :class:`RunConfig` from nested plain mappings."""
    top = asdict_shallow(RunConfig())
    resolved: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _CONFIG_TYPES:
            base = _CONFIG_TYPES[key]()
            resolved[key] = _merge_dataclass(base, value)
        elif key in top:
            resolved[key] = _coerce_scalar(top[key], value)
        else:
            raise KeyError(f"RunConfig has no field {key!r}")
    merged = {**top, **resolved}
    return RunConfig(**merged)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} does not contain a mapping")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive mapping merge; the override wins on scalars."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_config(
    config_dir: str | Path,
    experiment: str,
    overrides: list[str] | None = None,
) -> RunConfig:
    """Compose model/data/train defaults with an experiment file and CLI overrides."""
    root = Path(config_dir)
    payload: dict[str, Any] = {}
    for name in ("model/base.yaml", "data/base.yaml", "train/base.yaml"):
        candidate = root / name
        if candidate.exists():
            payload = deep_merge(payload, load_yaml(candidate))
    experiment_path = root / "experiment" / f"{experiment}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(f"missing experiment config {experiment_path}")
    payload = deep_merge(payload, load_yaml(experiment_path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        dotted, raw = item.split("=", 1)
        node = payload
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    payload.setdefault("experiment", experiment)
    return run_config_from_mapping(payload)


def config_digest(config: RunConfig) -> str:
    """Stable SHA-256 over the resolved configuration."""
    blob = json.dumps(config_to_json(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def config_to_json(config: RunConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    result = convert(config)
    if not isinstance(result, dict):
        raise TypeError("RunConfig did not convert to a mapping")
    return result
