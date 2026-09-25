"""Reporting metrics: concordance, intervals, correction, decision and strata.

Ref: Sec. 4.2, Sec. 4.4, Sec. 4.10, Sec. 4.11, Tables 1-6.
"""

from twindyn.metrics.bootstrap import (
    Interval,
    PairedDifference,
    bootstrap_concordance,
    bootstrap_indices,
    bootstrap_over_initialisations,
    paired_bootstrap_contrast,
    paired_bootstrap_difference,
    percentile_interval,
)
from twindyn.metrics.concordance import (
    ConcordanceResult,
    concordance_index,
    concordance_se,
    pairwise_ranking_agreement,
    per_sample_concordance,
)
from twindyn.metrics.correction import (
    CorrectionResult,
    benjamini_hochberg,
    corrected_family,
    holm_bonferroni,
    significance_marker,
)
from twindyn.metrics.decision import (
    Decision,
    Verdict,
    decide,
    descriptive_null,
    drop_ratio,
    hypothesis_families,
    margin_retention,
    primary_criterion,
    secondary_criterion,
)
from twindyn.metrics.efficiency import (
    EfficiencyPoint,
    efficiency_curve,
    margin_curve,
    shrinkage_profile,
    zero_label_transfer,
)
from twindyn.metrics.strata import (
    GRADE_DEFINITIONS,
    GRADE_LABELS,
    STAGE_DEFINITIONS,
    STAGE_LABELS,
    StratumRow,
    dropout_rows,
    full_row,
    monotonicity,
    stratum_rows,
    within_stratum_gain,
)
from twindyn.metrics.transfer import TransferTax, retained_share, tax_reduction

__all__ = [
    "GRADE_DEFINITIONS",
    "GRADE_LABELS",
    "STAGE_DEFINITIONS",
    "STAGE_LABELS",
    "ConcordanceResult",
    "CorrectionResult",
    "Decision",
    "EfficiencyPoint",
    "Interval",
    "PairedDifference",
    "StratumRow",
    "TransferTax",
    "Verdict",
    "benjamini_hochberg",
    "bootstrap_concordance",
    "bootstrap_indices",
    "bootstrap_over_initialisations",
    "concordance_index",
    "concordance_se",
    "corrected_family",
    "decide",
    "descriptive_null",
    "drop_ratio",
    "dropout_rows",
    "efficiency_curve",
    "full_row",
    "holm_bonferroni",
    "hypothesis_families",
    "margin_curve",
    "margin_retention",
    "monotonicity",
    "paired_bootstrap_contrast",
    "paired_bootstrap_difference",
    "pairwise_ranking_agreement",
    "per_sample_concordance",
    "percentile_interval",
    "primary_criterion",
    "retained_share",
    "secondary_criterion",
    "shrinkage_profile",
    "significance_marker",
    "stratum_rows",
    "tax_reduction",
    "within_stratum_gain",
    "zero_label_transfer",
]
