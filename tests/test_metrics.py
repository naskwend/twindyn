"""Metrics: concordance, intervals, correction, decision, strata, efficiency and transfer.

Ref: Sec. 4.2, Sec. 4.4, Sec. 4.10, Sec. 4.11 and Tables 1-6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from twindyn.metrics.bootstrap import (
    bootstrap_concordance,
    bootstrap_over_initialisations,
    paired_bootstrap_contrast,
    paired_bootstrap_difference,
    percentile_interval,
)
from twindyn.metrics.concordance import (
    concordance_index,
    concordance_se,
    pairwise_ranking_agreement,
    per_sample_concordance,
)
from twindyn.metrics.correction import (
    benjamini_hochberg,
    corrected_family,
    holm_bonferroni,
    significance_marker,
)
from twindyn.metrics.decision import (
    decide,
    descriptive_null,
    drop_ratio,
    hypothesis_families,
    margin_retention,
    primary_criterion,
    secondary_criterion,
)
from twindyn.metrics.efficiency import (
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
    dropout_rows,
    full_row,
    monotonicity,
    stratum_rows,
    within_stratum_gain,
)
from twindyn.metrics.transfer import TransferTax, retained_share, tax_reduction


def brute_force_concordance(risk, time, event) -> tuple[float, int]:
    concordant = 0.0
    comparable = 0
    for left in range(len(risk)):
        for right in range(len(risk)):
            if left == right:
                continue
            if (time[left] < time[right] and event[left] == 1) or (
                time[left] == time[right] and event[left] == 1 and event[right] == 0
            ):
                comparable += 1
                if risk[left] > risk[right]:
                    concordant += 1.0
                elif risk[left] == risk[right]:
                    concordant += 0.5
    return (concordant / comparable if comparable else float("nan")), comparable // 1


@pytest.mark.parametrize(
    "risk,time,event",
    [
        ([0.9, 0.1, 0.4, 0.6, 0.2], [4.0, 3.0, 2.0, 1.0, 5.0], [1.0, 1.0, 1.0, 0.0, 1.0]),
        ([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 0.0, 1.0]),
        ([0.5, 0.5, 0.5], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
        ([0.0, 5.0], [7.0, 7.0], [1.0, 0.0]),
    ],
)
def test_concordance_matches_a_brute_force_implementation(risk, time, event) -> None:
    produced = concordance_index(torch.tensor(risk), torch.tensor(time), torch.tensor(event))
    expected, comparable = brute_force_concordance(risk, time, event)
    assert produced.c_index == pytest.approx(expected)
    assert produced.comparable_pairs == comparable


def test_concordance_extremes() -> None:
    perfect = concordance_index(
        torch.tensor([3.0, 2.0, 1.0]), torch.tensor([1.0, 2.0, 3.0]), torch.ones(3)
    ).c_index
    reverse = concordance_index(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]), torch.ones(3)
    ).c_index
    constant = concordance_index(
        torch.tensor([2.0, 2.0, 2.0]), torch.tensor([1.0, 2.0, 3.0]), torch.ones(3)
    ).c_index
    assert perfect == pytest.approx(1.0)
    assert reverse == pytest.approx(0.0)
    assert constant == pytest.approx(0.5)


def test_concordance_has_no_comparable_pairs_without_events() -> None:
    result = concordance_index(torch.tensor([0.1, 0.9]), torch.tensor([1.0, 2.0]), torch.zeros(2))
    assert result.comparable_pairs == 0
    assert math.isnan(result.c_index)


def test_concordance_censoring_rule() -> None:
    result = concordance_index(
        torch.tensor([0.5, 0.9]), torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0])
    )
    assert result.comparable_pairs == 0


def test_concordance_somers_d_is_bounded() -> None:
    result = concordance_index(
        torch.tensor([0.9, 0.5, 0.1]), torch.tensor([1.0, 2.0, 3.0]), torch.ones(3)
    )
    assert -1.0 <= result.somers_d <= 1.0


def test_concordance_se_is_finite() -> None:
    error = concordance_se(
        torch.tensor([0.9, 0.5, 0.1, 0.4]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        torch.ones(4),
    )
    assert math.isfinite(error)
    assert error >= 0.0


def test_per_sample_contributions_sum_to_twice_the_concordant_count() -> None:
    risk = np.asarray([0.9, 0.5, 0.1])
    time = np.asarray([1.0, 2.0, 3.0])
    event = np.ones(3)
    contributions = per_sample_concordance(risk, time, event)
    total_concordant = contributions[:, 0].sum()
    assert total_concordant == pytest.approx(2.0 * 3)


def test_ranking_agreement_is_one_for_an_unchanged_ranking() -> None:
    before = torch.tensor([1.0, 2.0, 3.0])
    assert pairwise_ranking_agreement(before, before) == pytest.approx(1.0)
    assert pairwise_ranking_agreement(before, -before) == pytest.approx(0.0)


def test_concordance_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        concordance_index(torch.zeros(3), torch.zeros(2), torch.zeros(3))


def test_percentile_interval_matches_a_direct_quantile() -> None:
    values = np.random.default_rng(0).normal(size=500)
    interval = percentile_interval(values, 0.9)
    lower, upper = np.quantile(values, [0.05, 0.95])
    assert interval.lower == pytest.approx(float(lower))
    assert interval.upper == pytest.approx(float(upper))


def test_empty_interval_is_nan() -> None:
    interval = percentile_interval(np.zeros(0))
    assert math.isnan(interval.lower)


def test_bootstrap_concordance_brackets_the_point_estimate() -> None:
    generator = np.random.default_rng(3)
    time = generator.uniform(1.0, 20.0, size=50)
    event = (generator.uniform(size=50) > 0.25).astype(np.float64)
    risk = -time + generator.normal(0.0, 1.0, size=50)
    interval = bootstrap_concordance(risk, time, event, resamples=120, seed=0)
    assert interval.lower <= interval.point <= interval.upper
    assert interval.resamples > 0


def test_paired_bootstrap_detects_a_stronger_arm() -> None:
    generator = np.random.default_rng(4)
    time = generator.uniform(1.0, 20.0, size=60)
    event = (generator.uniform(size=60) > 0.2).astype(np.float64)
    weak = generator.normal(0.0, 1.0, size=60)
    strong = -time + generator.normal(0.0, 0.05, size=60)
    contrast = paired_bootstrap_difference(strong, weak, time, event, resamples=80, seed=0)
    assert contrast.point > 0.0
    assert contrast.interval.lower > -0.3
    assert 0.0 <= contrast.p_value <= 1.0


def test_paired_bootstrap_contrast_reports_both_arms() -> None:
    generator = np.random.default_rng(5)
    time = generator.uniform(1.0, 20.0, size=40)
    event = (generator.uniform(size=40) > 0.2).astype(np.float64)
    left = generator.normal(size=40)
    right = generator.normal(size=40)
    payload = paired_bootstrap_contrast(left, right, time, event, resamples=40, seed=0)
    assert {"arm_a", "arm_b", "point", "p_value"} <= set(payload)


def test_initialisation_summary_reports_the_spread() -> None:
    report = bootstrap_over_initialisations(np.asarray([0.7, 0.72, 0.68]))
    assert report["mean"] == pytest.approx(0.7)
    assert report["sd"] > 0.0
    assert bootstrap_over_initialisations(np.zeros(0))["count"] == 0.0


def test_holm_matches_a_hand_computed_worked_example() -> None:
    raw = [0.001, 0.02, 0.04, 0.2]
    produced = holm_bonferroni(np.asarray(raw), ("a", "b", "c", "d"))
    order = sorted(range(4), key=lambda index: raw[index])
    expected = [0.0] * 4
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (4 - rank) * raw[index])
        expected[index] = min(1.0, running)
    assert list(produced.adjusted) == pytest.approx(expected)


def test_holm_is_monotone_and_never_below_the_raw_value() -> None:
    raw = np.asarray([0.03, 0.001, 0.5, 0.02])
    produced = holm_bonferroni(raw, ("a", "b", "c", "d"))
    order = np.argsort(raw)
    ordered = produced.adjusted[order]
    assert all(ordered[index] <= ordered[index + 1] + 1e-12 for index in range(3))
    assert bool((produced.adjusted + 1e-12 >= raw).all())


def test_benjamini_hochberg_is_never_larger_than_holm() -> None:
    raw = np.asarray([0.001, 0.02, 0.04, 0.2])
    holm = holm_bonferroni(raw, ("a", "b", "c", "d"))
    bh = benjamini_hochberg(raw, ("a", "b", "c", "d"))
    assert bool((bh.adjusted <= holm.adjusted + 1e-12).all())


def test_corrected_family_labels_its_members() -> None:
    result = corrected_family({"x": 0.01, "y": 0.2})
    assert result.labels == ("x", "y")
    assert result.method == "holm_bonferroni"


def test_significance_markers() -> None:
    assert significance_marker(0.005) == "**"
    assert significance_marker(0.03) == "*"
    assert significance_marker(0.4) == ""
    assert significance_marker(float("nan")) == ""


def test_decision_rule_bands() -> None:
    assert decide(-0.01).verdict.value == "deficit"
    assert decide(0.0).verdict.value == "no_effect"
    assert decide(0.003).verdict.value == "parity"
    assert decide(0.02).verdict.value == "advantage"
    assert decide(0.02).clears_thesis_threshold is False
    assert decide(0.031).clears_thesis_threshold is True


def test_decision_rejects_a_non_finite_difference() -> None:
    with pytest.raises(ValueError):
        decide(float("nan"))


def test_primary_criterion_needs_every_cohort() -> None:
    passing = primary_criterion({"a": 0.046, "b": 0.039, "c": 0.040})
    failing = primary_criterion({"a": 0.046, "b": 0.029, "c": 0.040})
    assert passing["all_clear"] is True
    assert failing["all_clear"] is False


def test_secondary_criterion_reads_the_ten_percent_row() -> None:
    met = secondary_criterion({0.10: 0.076, 0.25: 0.038})
    unmet = secondary_criterion({0.10: 0.001})
    assert met["met"] is True
    assert unmet["met"] is False
    assert secondary_criterion({})["met"] is False


def test_margin_retention_and_drop_ratio_reproduce_the_caption() -> None:
    retention = margin_retention(0.041, 0.003)["retention"]
    assert retention == pytest.approx(0.926829, abs=1e-5)
    ratio = drop_ratio({"a": 0.041, "b": 0.003})
    assert ratio["ratio"] == pytest.approx(0.041 / 0.003)
    assert ratio["largest_variant"] == "a"


def test_descriptive_null_below_the_margin() -> None:
    report = descriptive_null(-0.003)
    assert report["below_margin"] is True
    assert report["interpretation"] == "ablation with no effect"


def test_hypothesis_families_are_declared() -> None:
    families = hypothesis_families()
    assert families["external"]["comparisons"] == 3
    assert families["label_efficiency"]["comparisons"] == 4
    assert families["level"]["descriptive"] is True


def test_stratum_rows_contrast_against_the_full_set() -> None:
    risk = np.asarray([3.0, 2.0, 1.0, 0.5, 2.5])
    time = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.ones(5)
    stage = np.asarray([1, 1, 1, 2, 2])
    rows = stratum_rows(risk, time, event, stage, STAGE_LABELS, STAGE_DEFINITIONS)
    full = full_row(risk, time, event)
    assert full.stratum == "full"
    for row in rows:
        assert row.delta_vs_full == pytest.approx(row.c_index - full.c_index)


def test_monotonicity_of_a_profile() -> None:
    increasing = monotonicity({"stage_1": 0.5, "stage_2": 0.6, "stage_3": 0.7})
    trough = monotonicity({"stage_1": 0.6, "stage_2": 0.5, "stage_3": 0.7})
    assert increasing["monotone_increasing"] is True
    assert trough["interior_trough"] is True
    assert trough["trough"] == "stage_2"


def test_dropout_rows_use_the_stated_reference() -> None:
    risk = np.asarray([2.0, 1.0, 0.5])
    dropped = np.asarray([1.0, 0.5, 2.0])
    time = np.asarray([1.0, 2.0, 3.0])
    event = np.ones(3)
    rows = dropout_rows(risk, {"morphology": dropped}, time, event, reference_c_index=0.9)
    assert rows[0].delta_vs_full == pytest.approx(rows[0].c_index - 0.9)


def test_within_stratum_gain_is_a_contrast() -> None:
    full = np.asarray([3.0, 2.0, 1.0, 0.5])
    covariate = np.asarray([1.0, 1.0, 1.0, 1.0])
    time = np.asarray([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4)
    stage = np.asarray([1, 1, 2, 2])
    gains = within_stratum_gain(full, covariate, time, event, stage, STAGE_LABELS)
    assert set(gains) == {"stage_1", "stage_2"}


def test_grade_labels_are_declared() -> None:
    assert GRADE_LABELS == ("grade_1", "grade_2", "grade_3", "grade_4")
    assert GRADE_DEFINITIONS["grade_1"] == "lowest"


def test_efficiency_curve_shrinks_the_margin() -> None:
    method = {0.10: 0.716, 0.25: 0.670, 0.50: 0.640, 1.00: 0.660}
    comparator = {0.10: 0.640, 0.25: 0.632, 0.50: 0.628, 1.00: 0.657}
    points = efficiency_curve(method, comparator)
    assert points[0].difference > points[-1].difference
    assert points[0].verdict in {"advantage", "parity"}
    assert margin_curve(points)["0.10"] == pytest.approx(0.076)
    assert shrinkage_profile(points)["monotone_decreasing"] is True


def test_efficiency_without_a_comparator_is_not_applicable() -> None:
    points = efficiency_curve({0.0: 0.658}, None)
    assert points[0].verdict == "not_applicable"
    assert points[0].supervised_comparator_available is False


def test_zero_label_transfer_records_the_absence_of_a_comparator() -> None:
    report = zero_label_transfer(0.658, supervised_available=False)
    assert report["supervised_comparator_available"] is False
    assert "cannot be fitted" in report["note"]


def test_transfer_tax_reproduces_the_printed_range() -> None:
    baseline = TransferTax("static", 0.731, {"a": 0.612, "b": 0.628, "c": 0.601})
    candidate = TransferTax("TwinDyn", 0.734, {"a": 0.658, "b": 0.667, "c": 0.641})
    assert [round(value, 3) for value in baseline.losses.values()] == [0.119, 0.103, 0.130]
    assert [round(value, 3) for value in candidate.losses.values()] == [0.076, 0.067, 0.093]
    reduction = tax_reduction(baseline, candidate)
    assert 0.25 < reduction["mean_reduction_share"] < 0.4
    assert retained_share(baseline, candidate)["a"] == pytest.approx(0.076 / 0.119)
