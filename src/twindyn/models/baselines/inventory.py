"""Registry of the comparator rows and their feature views.

Ref: Table 1 (the identifier, method name, family and reported concordance of
every row), Sec. 4.4 (an equal tuning budget for all variants), Sec. 4.5 (the
best value in the table is the one TwinDyn is compared against, and the strongest
static composition baseline is B16).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from twindyn.models.baselines.classical import (
    CoxClinical,
    GradientBoostedSurvival,
    PenalisedCoxExpression,
    RandomSurvivalForest,
)
from twindyn.models.baselines.deep import (
    CellAwareTransformerSurvival,
    DeepSurvStyleNetwork,
    DenseTransformerMoeFusion,
    LinearAttentionMilSurvival,
    SparseHierarchicalTransformer,
    UncertaintyAwareSurvival,
)
from twindyn.models.baselines.domain import (
    CrossDomainInvariantAbsorption,
    PromptDrivenLatentGeneration,
)
from twindyn.models.baselines.foundation import FrozenFoundationLinearHead
from twindyn.models.baselines.fusion import (
    GraphMultiOmicsFusion,
    MultiOmicsAutoencoder,
    StaticCompositionCox,
    StaticSharedPrivateDisentanglement,
)
from twindyn.models.baselines.graph import MetadataDrivenDualGraph
from twindyn.models.baselines.protocol import RiskEstimator

FEATURE_VIEWS: tuple[str, ...] = ("clinical", "expression", "composition")


@dataclass(frozen=True)
class BaselineSpec:
    identifier: str
    method: str
    family: str
    view: str
    reported_c_index: float
    reported_interval: tuple[float, float]
    reported_sd: float
    comparator: bool = False


BASELINES: dict[str, BaselineSpec] = {
    "B1": BaselineSpec(
        "B1", "Cox, clinical covariates", "classical", "clinical", 0.612, (0.585, 0.639), 0.021
    ),
    "B2": BaselineSpec(
        "B2",
        "Penalised Cox, expression",
        "classical",
        "expression",
        0.668,
        (0.643, 0.693),
        0.018,
    ),
    "B3": BaselineSpec(
        "B3",
        "Random survival forest",
        "classical",
        "expression",
        0.681,
        (0.657, 0.705),
        0.017,
    ),
    "B4": BaselineSpec(
        "B4",
        "Gradient-boosted survival",
        "classical",
        "expression",
        0.689,
        (0.665, 0.713),
        0.016,
    ),
    "B5": BaselineSpec(
        "B5", "DeepSurv-style network", "deep survival", "expression", 0.703, (0.681, 0.725), 0.015
    ),
    "B6": BaselineSpec(
        "B6",
        "Sparse hierarchical transformer",
        "deep survival",
        "expression",
        0.716,
        (0.695, 0.737),
        0.014,
    ),
    "B7": BaselineSpec(
        "B7",
        "Uncertainty-aware survival",
        "deep survival",
        "expression",
        0.719,
        (0.698, 0.740),
        0.014,
    ),
    "B8": BaselineSpec(
        "B8",
        "Cell-aware transformer survival",
        "deep survival",
        "expression",
        0.721,
        (0.701, 0.741),
        0.013,
    ),
    "B9": BaselineSpec(
        "B9",
        "Linear-attention MIL survival",
        "deep survival",
        "expression",
        0.724,
        (0.704, 0.744),
        0.013,
    ),
    "B10": BaselineSpec(
        "B10",
        "Dense transformer, MoE fusion",
        "deep survival",
        "expression",
        0.727,
        (0.707, 0.747),
        0.013,
    ),
    "B11": BaselineSpec(
        "B11", "Graph multi-omics fusion", "fusion", "expression", 0.722, (0.701, 0.743), 0.014
    ),
    "B12": BaselineSpec(
        "B12", "Metadata-driven dual graph", "graph", "expression", 0.718, (0.697, 0.739), 0.014
    ),
    "B13": BaselineSpec(
        "B13", "Multi-omics autoencoder", "fusion", "expression", 0.712, (0.690, 0.734), 0.015
    ),
    "B15": BaselineSpec(
        "B15",
        "Static shared/private disent.",
        "alternative representation",
        "composition",
        0.729,
        (0.710, 0.748),
        0.012,
    ),
    "B16": BaselineSpec(
        "B16",
        "Static composition + Cox (comparator)",
        "alternative representation",
        "composition",
        0.731,
        (0.712, 0.750),
        0.012,
        comparator=True,
    ),
    "B17": BaselineSpec(
        "B17",
        "Cross-domain invariant absorb.",
        "domain generalisation",
        "expression",
        0.726,
        (0.706, 0.746),
        0.013,
    ),
    "B18": BaselineSpec(
        "B18",
        "Prompt-driven latent gener.",
        "domain generalisation",
        "expression",
        0.728,
        (0.709, 0.747),
        0.012,
    ),
    "B19": BaselineSpec(
        "B19",
        "Frozen foundation + linear head",
        "foundation",
        "expression",
        0.707,
        (0.685, 0.729),
        0.015,
    ),
}

COMPARATOR_ID = "B16"

_TOKENISED = frozenset({"B6", "B8", "B9", "B10", "B11", "B12"})
HEADLINE_BASELINE = "B10"
ABSENT_IDENTIFIERS: tuple[str, ...] = ("B14",)


def resolve_token_dim(features: int, requested: int | None) -> int:
    """The token width a tokenising comparator splits the feature vector into.

    The comparators that tokenise the observation split it the way the axis does,
    so the token width is the axis position width when it divides the view, and the
    largest divisor that does not exceed it otherwise.
    """
    if requested is not None and requested > 0 and features % requested == 0:
        return int(requested)
    ceiling = features if requested is None else max(1, min(int(requested), features))
    for candidate in range(ceiling, 0, -1):
        if features % candidate == 0:
            return candidate
    return features


def build_baseline(
    identifier: str,
    features: int,
    metadata: int,
    token_dim: int | None = None,
    **overrides: object,
) -> RiskEstimator:
    """Construct one comparator with the width and the tokenisation its view provides."""
    if identifier not in BASELINES:
        raise KeyError(f"unknown comparator {identifier!r}")
    factory = _FACTORIES.get(identifier)
    if factory is None:
        raise KeyError(f"comparator {identifier!r} has no constructor")
    resolved = resolve_token_dim(features, token_dim)
    if identifier in _TOKENISED:
        overrides = {**overrides, "token_dim": resolved}
    return cast(RiskEstimator, factory(features, metadata, **overrides))


def _b1(features: int, metadata: int, **overrides: object) -> object:
    return CoxClinical()


def _b2(features: int, metadata: int, **overrides: object) -> object:
    return PenalisedCoxExpression(**overrides)  # type: ignore[arg-type]


def _b3(features: int, metadata: int, **overrides: object) -> object:
    return RandomSurvivalForest(**overrides)  # type: ignore[arg-type]


def _b4(features: int, metadata: int, **overrides: object) -> object:
    return GradientBoostedSurvival(**overrides)  # type: ignore[arg-type]


def _b5(features: int, metadata: int, **overrides: object) -> object:
    return DeepSurvStyleNetwork(features, **overrides)  # type: ignore[arg-type]


def _b6(features: int, metadata: int, **overrides: object) -> object:
    return SparseHierarchicalTransformer(features, **overrides)  # type: ignore[arg-type]


def _b7(features: int, metadata: int, **overrides: object) -> object:
    return UncertaintyAwareSurvival(features, **overrides)  # type: ignore[arg-type]


def _b8(features: int, metadata: int, **overrides: object) -> object:
    return CellAwareTransformerSurvival(features, **overrides)  # type: ignore[arg-type]


def _b9(features: int, metadata: int, **overrides: object) -> object:
    return LinearAttentionMilSurvival(features, **overrides)  # type: ignore[arg-type]


def _b10(features: int, metadata: int, **overrides: object) -> object:
    return DenseTransformerMoeFusion(features, **overrides)  # type: ignore[arg-type]


def _b11(features: int, metadata: int, **overrides: object) -> object:
    return GraphMultiOmicsFusion(features, **overrides)  # type: ignore[arg-type]


def _b12(features: int, metadata: int, **overrides: object) -> object:
    return MetadataDrivenDualGraph(features, metadata, **overrides)  # type: ignore[arg-type]


def _b13(features: int, metadata: int, **overrides: object) -> object:
    return MultiOmicsAutoencoder(features, **overrides)  # type: ignore[arg-type]


def _b15(features: int, metadata: int, **overrides: object) -> object:
    return StaticSharedPrivateDisentanglement(features, **overrides)  # type: ignore[arg-type]


def _b16(features: int, metadata: int, **overrides: object) -> object:
    return StaticCompositionCox(features, **overrides)  # type: ignore[arg-type]


def _b17(features: int, metadata: int, **overrides: object) -> object:
    return CrossDomainInvariantAbsorption(features, **overrides)  # type: ignore[arg-type]


def _b18(features: int, metadata: int, **overrides: object) -> object:
    return PromptDrivenLatentGeneration(features, **overrides)  # type: ignore[arg-type]


def _b19(features: int, metadata: int, **overrides: object) -> object:
    return FrozenFoundationLinearHead(features, **overrides)  # type: ignore[arg-type]


_FACTORIES = {
    "B1": _b1,
    "B2": _b2,
    "B3": _b3,
    "B4": _b4,
    "B5": _b5,
    "B6": _b6,
    "B7": _b7,
    "B8": _b8,
    "B9": _b9,
    "B10": _b10,
    "B11": _b11,
    "B12": _b12,
    "B13": _b13,
    "B15": _b15,
    "B16": _b16,
    "B17": _b17,
    "B18": _b18,
    "B19": _b19,
}


def families() -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for identifier, spec in BASELINES.items():
        grouped.setdefault(spec.family, []).append(identifier)
    return {key: tuple(value) for key, value in grouped.items()}


def view_of(identifier: str) -> str:
    return BASELINES[identifier].view


def identifiers() -> tuple[str, ...]:
    return tuple(BASELINES)
