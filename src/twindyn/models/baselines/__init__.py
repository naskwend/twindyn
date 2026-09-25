"""The comparator rows of the main comparison and their shared protocol.

Ref: Table 1 with its caption, Sec. 4.4, Sec. 4.5.
"""

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
    FeatureTokenizer,
    LinearAttentionMilSurvival,
    SparseHierarchicalTransformer,
    UncertaintyAwareSurvival,
)
from twindyn.models.baselines.domain import (
    CrossDomainInvariantAbsorption,
    PromptDrivenLatentGeneration,
    gradient_reversal,
)
from twindyn.models.baselines.foundation import FrozenFoundationLinearHead
from twindyn.models.baselines.fusion import (
    GraphMultiOmicsFusion,
    MultiOmicsAutoencoder,
    StaticCompositionCox,
    StaticSharedPrivateDisentanglement,
)
from twindyn.models.baselines.graph import MetadataDrivenDualGraph
from twindyn.models.baselines.inventory import (
    ABSENT_IDENTIFIERS,
    BASELINES,
    COMPARATOR_ID,
    FEATURE_VIEWS,
    HEADLINE_BASELINE,
    BaselineSpec,
    build_baseline,
    families,
    identifiers,
    view_of,
)
from twindyn.models.baselines.protocol import (
    BaselineFit,
    BaselineProtocol,
    RiskEstimator,
    TorchRiskEstimator,
    fit_torch_survival,
    run_baseline,
    standardise,
)
from twindyn.models.baselines.tree import (
    RegressionTree,
    SurvivalTree,
    log_rank_statistic,
    nelson_aalen,
)

__all__ = [
    "ABSENT_IDENTIFIERS",
    "BASELINES",
    "COMPARATOR_ID",
    "FEATURE_VIEWS",
    "HEADLINE_BASELINE",
    "BaselineFit",
    "BaselineProtocol",
    "BaselineSpec",
    "CellAwareTransformerSurvival",
    "CoxClinical",
    "CrossDomainInvariantAbsorption",
    "DeepSurvStyleNetwork",
    "DenseTransformerMoeFusion",
    "FeatureTokenizer",
    "FrozenFoundationLinearHead",
    "GradientBoostedSurvival",
    "GraphMultiOmicsFusion",
    "LinearAttentionMilSurvival",
    "MetadataDrivenDualGraph",
    "MultiOmicsAutoencoder",
    "PenalisedCoxExpression",
    "PromptDrivenLatentGeneration",
    "RandomSurvivalForest",
    "RegressionTree",
    "RiskEstimator",
    "SparseHierarchicalTransformer",
    "StaticCompositionCox",
    "StaticSharedPrivateDisentanglement",
    "SurvivalTree",
    "TorchRiskEstimator",
    "UncertaintyAwareSurvival",
    "build_baseline",
    "families",
    "fit_torch_survival",
    "gradient_reversal",
    "identifiers",
    "log_rank_statistic",
    "nelson_aalen",
    "run_baseline",
    "standardise",
    "view_of",
]
