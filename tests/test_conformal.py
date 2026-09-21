"""Math checks for the conformal layer. No API calls, no network, pure simulation.

The interesting tests here are statistical rather than symbolic: split conformal
prediction and conformal risk control make probabilistic promises, so the way to test
them is to simulate exchangeable data many times and check the promise holds in the
aggregate. A test that only checked shapes would pass on a wrong quantile.
"""

from __future__ import annotations

import math
import random

import pytest

from jev_certify.conformal import (
    aps_score,
    certified_threshold,
    clopper_pearson_upper,
    conformal_quantile,
    coverage_of_sets,
    lac_score,
    prediction_set,
    probabilities,
    risk_coverage_curve,
    selective_report,
    top_probability,
    wilson_interval,
)


# ------------------------------------------------------------------- quantile basics


def test_conformal_quantile_uses_the_finite_sample_index():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # k = ceil(11 * 0.95) = 11 > n -> the whole space is required
    assert conformal_quantile(scores, 0.05) == math.inf
    # k = ceil(11 * 0.8) = 9 -> 9th smallest
    assert conformal_quantile(scores, 0.20) == 9.0
    # k = ceil(11 * 0.5) = 6
    assert conformal_quantile(scores, 0.50) == 6.0


def test_conformal_quantile_rejects_bad_alpha():
    with pytest.raises(ValueError):
        conformal_quantile([1.0], 0.0)
    with pytest.raises(ValueError):
        conformal_quantile([], 0.05)


# ------------------------------------------------------- prediction sets: coverage


def _random_choice_answer(rng: random.Random, n_labels: int, peaked: float) -> tuple[dict, str]:
    """A synthetic Jev-ish choice answer plus its true label."""
    weights = [rng.random() ** peaked for _ in range(n_labels)]
    total = sum(weights)
    probs = {f"label_{i}": w / total for i, w in enumerate(weights)}
    choice = max(probs.items(), key=lambda kv: kv[1])[0]
    # The truth follows the model's own probabilities plus noise: a partly-calibrated
    # model, which is the situation conformal prediction has to cope with.
    truth = rng.choices(
        list(probs), weights=[p ** 1.5 + 1e-6 for p in probs.values()], k=1
    )[0]
    return {"type": "choice", "choice": choice, "probabilities": probs}, truth


@pytest.mark.parametrize("score", ["aps", "lac"])
def test_prediction_sets_meet_their_coverage_promise(score: str):
    """Empirical coverage over many exchangeable trials must reach at least 1 - alpha."""
    alpha = 0.1
    trials = 60
    covered = 0
    total = 0
    rng = random.Random(11)
    for _ in range(trials):
        calibration = [_random_choice_answer(rng, 8, peaked=3.0) for _ in range(120)]
        scorer = aps_score if score == "aps" else lac_score
        cal_scores = [scorer(answer, truth) for answer, truth in calibration]
        qhat = conformal_quantile(cal_scores, alpha)
        for _ in range(60):
            answer, truth = _random_choice_answer(rng, 8, peaked=3.0)
            predicted = prediction_set(answer, qhat, score=score)
            covered += int(truth in predicted.labels)
            total += 1
    empirical = covered / total
    assert empirical >= 1 - alpha - 0.02, f"{score} coverage {empirical:.4f} below the promise"


def test_lac_set_contains_top_label_and_aps_is_non_empty():
    answer = {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}}
    for score in ("aps", "lac"):
        predicted = prediction_set(answer, 0.05, score=score)
        assert "a" in predicted.labels
        assert predicted.size >= 1


def test_aps_set_grows_with_the_threshold():
    answer = {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.3, "c": 0.1}}
    sizes = [prediction_set(answer, q, score="aps").size for q in (0.5, 0.8, 0.95, 1.0)]
    assert sizes == sorted(sizes)


def test_prediction_set_rejects_unknown_score():
    with pytest.raises(ValueError):
        prediction_set({"probabilities": {"a": 1.0}}, 0.5, score="nope")


def test_probabilities_are_normalised():
    answer = {"type": "choice", "choice": "a", "probabilities": {"a": 2.0, "b": 2.0}}
    normalised = probabilities(answer)
    assert pytest.approx(sum(normalised.values())) == 1.0
    assert top_probability(answer) == pytest.approx(0.5)


# ------------------------------------------------------ risk control: the guarantee


def test_certified_threshold_bound_is_within_alpha_by_construction():
    rng = random.Random(3)
    probs = [rng.random() for _ in range(300)]
    correct = [rng.random() < p for p in probs]
    cert = certified_threshold(probs, correct, alpha=0.05)
    assert cert.feasible
    assert cert.certification_bound <= 0.05 + 1e-12
    # Monotone non-increasing risk in lambda is what makes conformal risk control apply.
    curve = list(cert.risk_curve)
    assert all(curve[i] >= curve[i + 1] - 1e-12 for i in range(len(curve) - 1))


def test_certified_threshold_holds_on_fresh_exchangeable_samples():
    """Average per-query loss across many fresh test sets must stay at or below alpha."""
    alpha = 0.05
    trials = 200
    losses = []
    rng = random.Random(2026)
    for _ in range(trials):
        cal_probs = [min(1.0, abs(rng.gauss(0.7, 0.25))) for _ in range(400)]
        cal_correct = [rng.random() < p for p in cal_probs]
        cert = certified_threshold(cal_probs, cal_correct, alpha=alpha)
        assert cert.feasible
        test_probs = [min(1.0, abs(rng.gauss(0.7, 0.25))) for _ in range(400)]
        test_correct = [rng.random() < p for p in test_probs]
        report = selective_report(test_probs, test_correct, cert.threshold)
        losses.append(report.per_query_risk)
    mean_loss = sum(losses) / len(losses)
    assert mean_loss <= alpha * 1.25, f"mean per-query loss {mean_loss:.4f} exceeds alpha={alpha}"


def test_certified_threshold_is_infeasible_when_alpha_below_resolution():
    # With 10 calibration points, 1/(n+1) = 0.0909, so alpha = 0.01 cannot be certified.
    cert = certified_threshold([0.9] * 10, [True] * 10, alpha=0.01)
    assert not cert.feasible
    assert cert.threshold != cert.threshold  # NaN, and described as such
    assert "not" in cert.describe() or "no threshold" in cert.describe()


def test_selective_report_matches_hand_computation():
    probs = [0.9, 0.9, 0.9, 0.4, 0.4]
    correct = [True, True, False, True, False]
    report = selective_report(probs, correct, threshold=0.5)
    assert report.n == 5
    assert report.routed == 3
    assert report.routed_wrong == 1
    assert report.abstained == 2
    assert report.coverage == pytest.approx(0.6)
    assert report.selective_error == pytest.approx(1 / 3)
    assert report.per_query_risk == pytest.approx(0.2)


def test_risk_coverage_curve_is_monotone_in_coverage():
    probs = [i / 20 for i in range(20)]
    correct = [i % 3 == 0 for i in range(20)]
    curve = risk_coverage_curve(probs, correct)
    coverages = [row["coverage"] for row in curve]
    assert coverages == sorted(coverages, reverse=True)


def test_coverage_of_sets_reports_sizes_and_fallbacks():
    answers = [{"type": "choice", "choice": "a", "probabilities": {"a": 0.99, "b": 0.01}}] * 4
    rows = coverage_of_sets(answers, ["a"] * 4, [0.5, 1.0], score="aps")
    assert rows[-1]["coverage"] == 1.0
    assert rows[-1]["mean_set_size"] == 2.0
    assert rows[0]["top1_fallbacks"] == 4  # qhat below the top mass -> fallback counted


# --------------------------------------------------------------- interval helpers


def test_clopper_pearson_upper_matches_the_closed_form_for_zero_failures():
    # For k = 0 the exact one-sided bound is 1 - alpha^(1/n).
    for n, alpha in ((10, 0.05), (50, 0.01), (7, 0.2)):
        expected = 1 - alpha ** (1 / n)
        assert clopper_pearson_upper(0, n, alpha) == pytest.approx(expected, abs=1e-3)


def test_clopper_pearson_upper_exceeds_the_point_estimate_and_is_bounded():
    upper = clopper_pearson_upper(3, 30, 0.05)
    assert 0.1 < upper < 0.3
    assert clopper_pearson_upper(30, 30, 0.05) == 1.0


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(7, 20)
    assert low < 0.35 < high
    assert wilson_interval(0, 0) == (0.0, 1.0)
