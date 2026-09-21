"""Distribution-free guarantees on top of Jev's calibrated probabilities.

Jev returns probabilities, not guarantees. This module converts them into claims with
finite-sample coverage, using split conformal prediction and conformal risk control.

Three primitives
----------------
``conformal_quantile``
    The exchangeability-corrected threshold ``k``-th smallest calibration score with
    ``k = ceil((n+1)(1-alpha))``. Using this exact quantile (instead of the plain
    empirical one) is what buys a *finite-sample* — not just asymptotic — guarantee.

``prediction_set``
    Turns a ``choice`` answer into a **set** of labels that contains the truth with
    probability at least ``1 - alpha``. Two scores are supported:

    * ``"aps"``  adaptive prediction sets (Romano, Sesia, Candès & Sesia, 2020) —
      cumulative probability mass up to and including the true label.
    * ``"lac"``  least-ambiguous-set / margin score ``1 - p_true`` (Sadinle, Lei &
      Wasserman, 2019).

    An empty set is possible in the pure form when the test distribution is sharper
    than the calibration one; we fall back to the top-1 label (which can only
    *increase* coverage) and count how often that happened.

``certified_threshold``
    Conformal risk control (Angelopoulos, Bates, Fisch, Lei & Schuster, 2022) for a
    monotone loss. For a router the natural loss is

        L_{lambda}(x, y) = 1{auto-routed at lambda} * 1{wrong label}

    which is bounded by 1 and non-increasing in ``lambda``, so the theorem applies:

        E[L_{lambda_hat}(X, Y)] <= alpha   for a fresh exchangeable sample.

    Read that in production terms: **at most ``alpha`` of all incoming queries are
    auto-routed to the wrong handler**. It is a bound on the *per-query* rate of
    silent misroutes, which is the number that matters operationally. When no
    threshold on the grid attains ``alpha`` the function returns ``None`` and the
    caller must say so rather than pretend.

Nothing here is Jev-specific: any decision model that returns node probabilities can
be certified with these functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- helpers


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Finite-sample split-conformal threshold.

    Returns the ``k``-th smallest calibration score with ``k = ceil((n+1)(1-alpha))``,
    which guarantees ``P(score_new <= threshold) >= 1 - alpha`` for an exchangeable
    new sample. Returns ``+inf`` when ``k > n`` (the whole label space is required:
    honest, useless, and reported as such by the callers).
    """
    if not scores:
        raise ValueError("conformal_quantile needs a non-empty calibration set")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = len(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(sorted(scores)[k - 1])


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than Wald at the edges)."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), exact, stdlib only."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    if k == 0:
        # Exact, and avoids the cancellation the general branch would introduce for
        # small p where 1 - P(X >= 1) loses precision.
        return (1.0 - p) ** n
    # Sum the shorter tail to keep this fast and numerically sane.
    if k < n * p:
        total = 0.0
        log_p, log_q = math.log(p), math.log1p(-p)
        for i in range(k + 1):
            total += math.exp(
                math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
            )
        return min(1.0, total)
    total = 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    for i in range(k + 1, n + 1):
        total += math.exp(
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        )
    return max(0.0, 1.0 - total)


def clopper_pearson_upper(failures: int, trials: int, alpha: float = 0.05) -> float:
    """Exact (Clopper-Pearson) one-sided upper bound on a binomial rate.

    Used to report a *measured* selective error rate with an honest confidence
    statement next to the conformal guarantee, which is a different (predictive)
    object. Bisection on the exact binomial CDF; no SciPy needed.
    """
    if trials == 0:
        return 1.0
    if failures >= trials:
        return 1.0

    def tail(p: float) -> float:
        """P(X <= failures) - alpha: > 0 below the bound, < 0 above it."""
        return _binom_cdf(failures, trials, p) - alpha

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if tail(mid) > 0:
            lo = mid
        else:
            hi = mid
    return min(1.0, hi)


# ------------------------------------------------------------------- model answers


def probabilities(answer: dict[str, Any]) -> dict[str, float]:
    """Extract a normalised probability map from a Jev ``choice`` answer."""
    probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}
    if not probs:
        choice = answer.get("choice")
        if choice is None:
            raise ValueError("answer has neither probabilities nor choice")
        probs = {str(choice): 1.0}
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return {k: v / total for k, v in probs.items()}


def choice_of(answer: dict[str, Any]) -> str:
    choice = answer.get("choice")
    if choice is not None:
        return str(choice)
    return max(probabilities(answer).items(), key=lambda kv: kv[1])[0]


def top_probability(answer: dict[str, Any]) -> float:
    """``max_y P(y)`` — the score used by the certified routing threshold."""
    return max(probabilities(answer).values())


# ------------------------------------------------------------------- set scores


def aps_score(answer: dict[str, Any], true_label: str) -> float:
    """Adaptive prediction set score: cumulative mass up to and including the truth."""
    probs = probabilities(answer)
    items = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    cumulative = 0.0
    for label, prob in items:
        cumulative += prob
        if label == true_label:
            return cumulative
    return 1.0  # truth absent from the answer's support


def lac_score(answer: dict[str, Any], true_label: str) -> float:
    """Margin score ``1 - P(true)``."""
    return 1.0 - probabilities(answer).get(true_label, 0.0)


@dataclass
class SetPrediction:
    labels: tuple[str, ...]
    empty_before_fallback: bool = False

    @property
    def size(self) -> int:
        return len(self.labels)


def prediction_set(answer: dict[str, Any], qhat: float, score: str = "aps") -> SetPrediction:
    """Conformal prediction set for one ``choice`` answer at threshold ``qhat``."""
    probs = probabilities(answer)
    items = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    if score == "lac":
        cutoff = 1.0 - qhat
        labels = tuple(label for label, prob in items if prob >= cutoff)
    elif score == "aps":
        labels_list: list[str] = []
        cumulative = 0.0
        for label, prob in items:
            cumulative += prob
            if cumulative <= qhat:
                labels_list.append(label)
            else:
                break
        labels = tuple(labels_list)
    else:
        raise ValueError(f"unknown score '{score}' (expected 'aps' or 'lac')")

    was_empty = not labels
    if was_empty and items:
        labels = (items[0][0],)  # keep at least the argmax: can only raise coverage
    return SetPrediction(labels=labels, empty_before_fallback=was_empty)


# ------------------------------------------------------------------- risk control


@dataclass
class CertifiedThreshold:
    """Result of a conformal risk control fit on a calibration set."""

    alpha: float
    threshold: float
    lambda_grid: tuple[float, ...]
    risk_curve: tuple[float, ...]
    calibration_per_query_risk: float
    certification_bound: float
    n_calibration: int
    feasible: bool
    note: str = ""
    strictest_risk: float = float("nan")
    min_attainable_bound: float = float("nan")

    def describe(self) -> str:
        if not self.feasible:
            return (
                f"no threshold attains per-query risk <= {self.alpha:.3f}; the tightest "
                f"certificate these {self.n_calibration} calibration queries support is "
                f"{self.min_attainable_bound:.4f} ({self.note})"
            )
        return (
            f"lambda* = {self.threshold:.3f}: certified per-query loss "
            f"<= {self.certification_bound:.4f} (alpha = {self.alpha:.3f}, n = {self.n_calibration})"
        )


def certified_threshold(
    probabilities_max: Sequence[float],
    correct: Sequence[bool],
    alpha: float = 0.05,
    grid_size: int = 1001,
    loss_bound: float = 1.0,
) -> CertifiedThreshold:
    """Conformal risk control for the auto-route/abstain decision.

    ``probabilities_max[i]`` is ``max_y P(y|x_i)`` for calibration query ``i``;
    ``correct[i]`` says whether Jev's argmax label was the true one. The candidate
    thresholds are every observed confidence plus a dense grid, so the selected
    threshold is one the model actually produced.

    The loss is ``1{top-probability >= lambda} * 1{wrong}``, monotone non-increasing
    in ``lambda`` and bounded by 1, so conformal risk control applies and

        E[loss] <= (n/(n+1)) * R_hat(lambda_hat) + 1/(n+1) <= alpha.

    That is a bound on expected *silent misroutes per incoming query*, not on the
    error rate among routed queries (a different, non-exchangeable-looking quantity,
    reported separately by :func:`selective_report`).
    """
    if len(probabilities_max) != len(correct):
        raise ValueError("probabilities_max and correct must be the same length")
    n = len(probabilities_max)
    if n == 0:
        raise ValueError("certified_threshold needs calibration data")

    candidates = {0.0, 1.0}
    candidates.update(round(float(p), 4) for p in probabilities_max)
    step = 1.0 / (grid_size - 1)
    candidates.update(round(i * step, 4) for i in range(grid_size))
    grid = tuple(sorted(candidates))

    curve: list[float] = []
    for lam in grid:
        selected = 0
        wrong = 0
        for prob, ok in zip(probabilities_max, correct):
            if prob >= lam:
                selected += 1
                if not ok:
                    wrong += 1
        curve.append(wrong / n)

    slack = loss_bound / (n + 1)
    strictest_risk = curve[-1]
    min_attainable = (n / (n + 1)) * strictest_risk + slack
    chosen: float | None = None
    for lam, risk in zip(grid, curve):
        if (n / (n + 1)) * risk + slack <= alpha:
            chosen = lam
            break

    if chosen is None:
        return CertifiedThreshold(
            alpha=alpha,
            threshold=float("nan"),
            lambda_grid=grid,
            risk_curve=tuple(curve),
            calibration_per_query_risk=float("nan"),
            certification_bound=float("nan"),
            n_calibration=n,
            feasible=False,
            strictest_risk=strictest_risk,
            min_attainable_bound=min_attainable,
            note=(
                "two things set this floor: 1/(n+1) with "
                f"n = {n} is {slack:.4f}, and the model reports a top probability of 1.0 "
                f"on {sum(1 for p in probabilities_max if p >= 1.0)} calibration queries, of "
                "which the wrong ones keep the loss above zero no matter how strict the "
                "threshold gets -- collect more traffic and re-check the rounding"
            ),
        )

    index = grid.index(chosen)
    return CertifiedThreshold(
        alpha=alpha,
        threshold=float(chosen),
        lambda_grid=grid,
        risk_curve=tuple(curve),
        calibration_per_query_risk=curve[index],
        certification_bound=(n / (n + 1)) * curve[index] + slack,
        n_calibration=n,
        feasible=True,
        strictest_risk=strictest_risk,
        min_attainable_bound=min_attainable,
    )


# ------------------------------------------------------------------- reporting math


@dataclass
class SelectiveReport:
    """Measured behaviour of a threshold on held-out data."""

    n: int
    routed: int
    routed_correct: int
    routed_wrong: int
    abstained: int
    coverage: float
    selective_error: float
    selective_error_ci: tuple[float, float]
    selective_error_ucb: float
    per_query_risk: float
    per_query_risk_ci: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "routed": self.routed,
            "routed_correct": self.routed_correct,
            "routed_wrong": self.routed_wrong,
            "abstained": self.abstained,
            "coverage": round(self.coverage, 4),
            "selective_error": round(self.selective_error, 4),
            "selective_error_wilson95": [round(v, 4) for v in self.selective_error_ci],
            "selective_error_cp_ucb95": round(self.selective_error_ucb, 4),
            "per_query_risk": round(self.per_query_risk, 4),
            "per_query_risk_wilson95": [round(v, 4) for v in self.per_query_risk_ci],
        }


def selective_report(
    probabilities_max: Sequence[float],
    correct: Sequence[bool],
    threshold: float,
    alpha: float = 0.05,
) -> SelectiveReport:
    """Coverage and both flavours of risk for a fixed threshold on held-out data."""
    routed = routed_correct = routed_wrong = 0
    for prob, ok in zip(probabilities_max, correct):
        if prob >= threshold:
            routed += 1
            if ok:
                routed_correct += 1
            else:
                routed_wrong += 1
    n = len(probabilities_max)
    abstained = n - routed
    selective_error = (routed_wrong / routed) if routed else 0.0
    per_query_risk = routed_wrong / n if n else 0.0
    lo, hi = wilson_interval(routed_wrong, routed)
    return SelectiveReport(
        n=n,
        routed=routed,
        routed_correct=routed_correct,
        routed_wrong=routed_wrong,
        abstained=abstained,
        coverage=(routed / n) if n else 0.0,
        selective_error=selective_error,
        selective_error_ci=(lo, hi),
        selective_error_ucb=clopper_pearson_upper(routed_wrong, routed, alpha),
        per_query_risk=per_query_risk,
        per_query_risk_ci=wilson_interval(routed_wrong, n),
    )


def risk_coverage_curve(
    probabilities_max: Sequence[float],
    correct: Sequence[bool],
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float]]:
    """Risk-coverage trade-off (Geifman & El-Yaniv, 2017) at each threshold."""
    if thresholds is None:
        thresholds = [i / 100 for i in range(0, 101)]
    out: list[dict[str, float]] = []
    for lam in thresholds:
        report = selective_report(probabilities_max, correct, lam)
        out.append(
            {
                "threshold": round(float(lam), 4),
                "coverage": round(report.coverage, 4),
                "selective_error": round(report.selective_error, 6),
                "per_query_risk": round(report.per_query_risk, 6),
            }
        )
    return out


def coverage_of_sets(
    answers: Sequence[dict[str, Any]],
    truths: Sequence[str],
    thresholds: Sequence[float],
    score: str = "aps",
) -> list[dict[str, float]]:
    """Empirical coverage and mean set size for a family of thresholds (real numbers)."""
    out: list[dict[str, float]] = []
    for qhat in thresholds:
        hits = 0
        sizes = 0
        fallbacks = 0
        for answer, truth in zip(answers, truths):
            predicted = prediction_set(answer, qhat, score=score)
            hits += int(truth in predicted.labels)
            sizes += predicted.size
            fallbacks += int(predicted.empty_before_fallback)
        n = len(answers)
        lo, hi = wilson_interval(hits, n)
        out.append(
            {
                "qhat": round(float(qhat), 4),
                "coverage": round(hits / n, 4),
                "coverage_ci_lo": round(lo, 4),
                "coverage_ci_hi": round(hi, 4),
                "mean_set_size": round(sizes / n, 4),
                "top1_fallbacks": fallbacks,
            }
        )
    return out
