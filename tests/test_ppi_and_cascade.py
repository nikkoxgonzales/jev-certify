"""Checks for the prediction-powered audit and the cost model. No network.

PPI's promises are also statistical, so the tests simulate pools with known truth and
check that (a) the intervals keep their nominal coverage, (b) they are narrower than
labels-only when the model predictions are informative, and (c) they stay honest when
the predictions are wrong -- the failure mode that makes imputation-only accounting
dangerous.
"""

from __future__ import annotations

import random

import pytest

from jev_certify.cascade import CostAssumptions, build_operating_points, cost_per_1k
from jev_certify.conformal import certified_threshold
from jev_certify.ppi import (
    audit_simulation,
    brier_score,
    classical_estimate,
    expected_calibration_error,
    imputation_estimate,
    ppi_estimate,
    reliability_bins,
    required_labels,
)


# ------------------------------------------------------------------- PPI behaviour


def test_ppi_matches_labels_only_when_the_predictions_carry_no_information():
    """No free lunch: a constant f has zero variance, so PPI degenerates to the labels.

    This is worth a test because it is the boundary of the method's value: PPI's gain
    comes from predictions that *vary across rows and correlate with the truth*, not
    from having a model at all.
    """
    rng = random.Random(5)
    n_pool = 400
    y_pool = [1.0 if rng.random() < 0.72 else 0.0 for _ in range(n_pool)]
    f_pool = [0.75] * n_pool
    widths_ppi = []
    widths_classical = []
    for _ in range(120):
        idx = rng.sample(range(n_pool), 40)
        idx_set = set(idx)
        y_lab = [y_pool[i] for i in idx]
        f_unlab = [f_pool[i] for i in range(n_pool) if i not in idx_set]
        ppi = ppi_estimate(f_unlab, [f_pool[i] for i in idx], y_lab)
        classical = classical_estimate(y_lab, n_pool)
        widths_ppi.append(ppi.half_width)
        widths_classical.append(classical.half_width)
    ratio = (sum(widths_ppi) / len(widths_ppi)) / (sum(widths_classical) / len(widths_classical))
    assert ratio == pytest.approx(1.0, abs=0.02)


def test_ppi_is_narrower_and_still_covers_when_predictions_are_informative():
    rng = random.Random(17)
    n_pool = 400
    y_pool = [1.0 if rng.random() < 0.6 else 0.0 for _ in range(n_pool)]
    # A realistic audit predictor: right most of the time, wrong sometimes.
    f_pool = [
        min(1.0, max(0.0, (0.9 if y == 1.0 else 0.12) + rng.gauss(0, 0.12))) for y in y_pool
    ]
    truth = sum(y_pool) / n_pool
    covered_ppi = covered_classical = 0
    widths_ppi = []
    widths_classical = []
    repeats = 300
    for _ in range(repeats):
        idx = rng.sample(range(n_pool), 40)
        idx_set = set(idx)
        y_lab = [y_pool[i] for i in idx]
        f_lab = [f_pool[i] for i in idx]
        f_unlab = [f_pool[i] for i in range(n_pool) if i not in idx_set]
        ppi = ppi_estimate(f_unlab, f_lab, y_lab)
        classical = classical_estimate(y_lab, n_pool)
        covered_ppi += int(ppi.ci[0] <= truth <= ppi.ci[1])
        covered_classical += int(classical.ci[0] <= truth <= classical.ci[1])
        widths_ppi.append(ppi.half_width)
        widths_classical.append(classical.half_width)
    assert covered_ppi / repeats >= 0.88
    assert covered_classical / repeats >= 0.88
    assert sum(widths_ppi) / repeats < sum(widths_classical) / repeats


def test_imputation_is_confidently_wrong_when_the_model_is_wrong():
    """The failure mode PPI exists to correct: model-only accounting trusts a biased f."""
    y_pool = [0.0] * 200
    f_pool = [1.0] * 200
    imputation = imputation_estimate(f_pool, n_labeled=50)
    assert imputation.ci[0] > 0.9  # excludes the truth of 0.0, and looks certain
    ppi = ppi_estimate(f_pool, [1.0] * 50, [0.0] * 50)
    assert ppi.ci[0] <= 0.0 <= ppi.ci[1]  # PPI's correction pulls it back


def test_ppi_lambda_is_rectified_into_the_unit_interval():
    # f anti-correlates with y: the optimum lambda is negative, which rectification
    # clips to zero -- falling back to the labels-only estimator rather than inverting.
    y = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    f = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    estimate = ppi_estimate(f, f, y)
    assert estimate.lambda_ == 0.0
    assert estimate.estimate == pytest.approx(sum(y) / len(y))


def test_ppi_requires_two_labels():
    with pytest.raises(ValueError):
        ppi_estimate([0.5, 0.5], [0.5], [1.0])


def test_required_labels_falls_as_predictions_get_informative():
    rng = random.Random(9)
    n = 500
    y = [1.0 if rng.random() < 0.5 else 0.0 for _ in range(n)]
    noisy = [min(1.0, max(0.0, yv + rng.gauss(0, 0.5))) for yv in y]
    sharp = [0.5 + 0.45 * (1 if yv > 0.5 else -1) for yv in y]
    plan_noisy = required_labels(noisy, noisy[:100], y[:100], target_half_width=0.05)
    plan_sharp = required_labels(sharp, sharp[:100], y[:100], target_half_width=0.05)
    assert plan_sharp["ppi_labels"] < plan_noisy["ppi_labels"]
    assert plan_noisy["classical_labels"] > 0


def test_required_labels_is_infinite_when_the_pool_term_exceeds_the_target():
    f = [float(i % 2) for i in range(100)]
    plan = required_labels(f, f[:50], f[:50], target_half_width=1e-6)
    assert plan["ppi_labels"] == float("inf")


def test_audit_simulation_reports_three_methods_and_a_width_ratio():
    rng = random.Random(21)
    n = 300
    y = [1.0 if rng.random() < 0.8 else 0.0 for _ in range(n)]
    f = [min(1.0, max(0.0, 0.8 + rng.gauss(0, 0.05))) for _ in range(n)]
    sim = audit_simulation(f, y, n_labeled=30, repeats=50)
    assert set(sim["methods"]) == {"classical", "imputation", "ppi"}
    assert abs(sim["truth_pool_mean"] - sum(y) / n) < 1e-9
    assert sim["ppi_width_reduction_vs_classical"] is not None


# ------------------------------------------------------------------ diagnostics


def test_brier_and_ece_on_a_perfectly_calibrated_constant_predictor():
    probs = [0.5] * 1000
    outcomes = [1] * 500 + [0] * 500
    assert brier_score(probs, outcomes) == pytest.approx(0.25)
    assert expected_calibration_error(probs, outcomes, bins=10) == pytest.approx(0.0, abs=1e-9)


def test_reliability_bins_are_weighted_correctly():
    probs = [0.05, 0.15, 0.95]
    outcomes = [0, 0, 1]
    rows = reliability_bins(probs, outcomes, bins=10)
    assert len(rows) == 3
    assert rows[-1]["empirical_frequency"] == 1.0


# -------------------------------------------------------------------- cost model


def test_cost_per_1k_puts_every_escalated_query_on_the_fallback():
    assumptions = CostAssumptions(jev_usd_per_1k_queries=0.15, fallback_usd_per_1k_queries=4.0)
    assert cost_per_1k(1.0, assumptions) == pytest.approx(0.15)
    assert cost_per_1k(0.5, assumptions) == pytest.approx(0.15 + 0.5 * 4.0)
    assert cost_per_1k(0.0, assumptions) == pytest.approx(4.15)


def test_human_review_adds_to_the_escalation_cost():
    assumptions = CostAssumptions(
        jev_usd_per_1k_queries=0.15, fallback_usd_per_1k_queries=4.0, human_review_usd_per_1k_queries=10.0
    )
    assert cost_per_1k(0.0, assumptions) == pytest.approx(14.15)


def test_build_operating_points_compares_certified_and_hand_picked():
    rng = random.Random(31)
    probs = [rng.random() for _ in range(400)]
    correct = [rng.random() < p for p in probs]
    cert = certified_threshold(probs[:200], correct[:200], alpha=0.05)
    points = build_operating_points(
        cert, probs[200:], correct[200:], CostAssumptions(jev_usd_per_1k_queries=0.15)
    )
    names = [p["name"] for p in points["operating_points"]]
    assert any("certified" in name for name in names)
    assert any("hand-picked" in name for name in names)
    assert points["always_llm"]["cost_per_1k_usd"] == pytest.approx(4.20)
    assert "note" in points
