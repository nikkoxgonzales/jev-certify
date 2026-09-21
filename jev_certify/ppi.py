"""Prediction-powered inference: audit a live decision policy with few hand labels.

The problem this solves
-----------------------
You certified a router offline. Now it runs in production at 10,000 queries/day. How
do you *know* the certified error rate still holds, without hand-labelling 10,000
queries a day?

Prediction-powered inference (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic,
*Science* 382:669-674, 2023) answers it: label a small random sample by hand, let the
model predict the label on everything, and combine the two with a correction term.
The result is a confidence interval that is **provably valid** (under the usual
sampling assumptions) and much narrower than one built from hand labels alone.

PPI++ (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, NeurIPS 2023) adds a tuning
parameter ``lambda`` that is set to the (clipped) ratio of the covariance between the
model predictions and the labels to the variance of the model predictions. With
informative predictions it is strictly better than plain PPI, which is what this
module implements:

    theta_hat_lambda = mean_U(lambda * f) + mean_L(y - lambda * f)
    Var(theta_hat)   = lambda^2 * Var_U(f) / N + Var_L(y - lambda * f) / n

where ``f`` is the model's *numeric prediction of the labelled quantity* on the whole
pool (size ``N``) and ``(y, f)`` are the hand-labelled pairs (size ``n``).

For a router, the quantity worth auditing is

    Y = 1{this query was auto-routed AND the routed handler was correct}

i.e. the fraction of traffic that came out of the policy right, and

    f = 1{auto-routed} * max_y P(y)

which is exactly Jev's own estimate of that event. Also in this module: calibration
diagnostics (ECE, Brier, reliability bins) because PPI's efficiency depends on ``f``
being informative, and sample-size planning, because "how many labels do I need?" is
the first practical question a team asks.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

Z95 = 1.959963984540054


# --------------------------------------------------------------- calibration metrics


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of probabilistic predictions (Brier, 1950)."""
    if not probabilities:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(probabilities)


def reliability_bins(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> list[dict[str, float]]:
    """Reliability-diagram data: per-bin mean confidence, empirical frequency, count."""
    edges = [i / bins for i in range(bins + 1)]
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        members = [
            (p, y)
            for p, y in zip(probabilities, outcomes)
            if (lo <= p < hi) or (hi == 1.0 and p == 1.0)
        ]
        if not members:
            continue
        rows.append(
            {
                "bin": round((lo + hi) / 2, 4),
                "mean_confidence": round(sum(p for p, _ in members) / len(members), 4),
                "empirical_frequency": round(sum(y for _, y in members) / len(members), 4),
                "count": len(members),
            }
        )
    return rows


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> float:
    """Bin-weighted |confidence - accuracy| (Naeini et al., 2015)."""
    rows = reliability_bins(probabilities, outcomes, bins=bins)
    total = sum(row["count"] for row in rows)
    if not total:
        return float("nan")
    return sum(
        row["count"] / total * abs(row["mean_confidence"] - row["empirical_frequency"]) for row in rows
    )


# ----------------------------------------------------------------------- estimators


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _var(values: Sequence[float]) -> float:
    """Unbiased sample variance (0.0 for a single value)."""
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - 1)


@dataclass
class IntervalEstimate:
    """A point estimate with a two-sided confidence interval."""

    method: str
    estimate: float
    half_width: float
    n_labeled: int
    n_pool: int
    lambda_: float | None = None
    extra: dict[str, Any] | None = None

    @property
    def ci(self) -> tuple[float, float]:
        return (self.estimate - self.half_width, self.estimate + self.half_width)

    def as_dict(self) -> dict[str, Any]:
        low, high = self.ci
        out = {
            "method": self.method,
            "estimate": round(self.estimate, 5),
            "half_width": round(self.half_width, 5),
            "ci_low": round(low, 5),
            "ci_high": round(high, 5),
            "n_labeled": self.n_labeled,
            "n_pool": self.n_pool,
        }
        if self.lambda_ is not None:
            out["lambda"] = round(self.lambda_, 4)
        if self.extra:
            out.update(self.extra)
        return out


def classical_estimate(y_labeled: Sequence[float], n_pool: int, conf: float = 0.95) -> IntervalEstimate:
    """Hand labels only: the baseline every audit starts from."""
    z = Z95 if conf == 0.95 else _z_for(conf)
    n = len(y_labeled)
    mean = _mean(y_labeled)
    half = z * math.sqrt(_var(y_labeled) / n) if n > 1 else 1.0
    return IntervalEstimate("classical (labels only)", mean, half, n, n_pool)


def imputation_estimate(f_pool: Sequence[float], n_labeled: int, conf: float = 0.95) -> IntervalEstimate:
    """Model only: cheap, biased whenever the model is wrong, and it never knows."""
    z = Z95 if conf == 0.95 else _z_for(conf)
    n = len(f_pool)
    half = z * math.sqrt(_var(f_pool) / n) if n > 1 else 1.0
    return IntervalEstimate("imputation (model only)", _mean(f_pool), half, n_labeled, n)


def ppi_estimate(
    f_pool: Sequence[float],
    f_labeled: Sequence[float],
    y_labeled: Sequence[float],
    conf: float = 0.95,
    lam: float | None = None,
    rectified: bool = True,
) -> IntervalEstimate:
    """PPI++ (Angelopoulos et al., NeurIPS 2023) with an optional auto-tuned lambda."""
    z = Z95 if conf == 0.95 else _z_for(conf)
    n = len(y_labeled)
    N = len(f_pool)
    if n < 2:
        raise ValueError("PPI needs at least 2 labelled points for a variance estimate")

    if lam is None:
        var_f_labeled = _var(f_labeled)
        if var_f_labeled <= 1e-12:
            lam = 1.0
        else:
            cov = _mean([a * b for a, b in zip(y_labeled, f_labeled)]) - _mean(y_labeled) * _mean(f_labeled)
            cov *= n / (n - 1)
            lam = cov / var_f_labeled
            if rectified:
                lam = min(1.0, max(0.0, lam))

    residuals = [y - lam * f for y, f in zip(y_labeled, f_labeled)]
    variance = (lam * lam) * _var(f_pool) / N + _var(residuals) / n
    half = z * math.sqrt(max(variance, 0.0))
    estimate = lam * _mean(f_pool) + (_mean(y_labeled) - lam * _mean(f_labeled))
    return IntervalEstimate(
        "ppi++" if lam is not None else "ppi",
        estimate,
        half,
        n,
        N,
        lambda_=lam,
        extra={"rectified": rectified, "var_f_pool": round(_var(f_pool), 6), "var_residual": round(_var(residuals), 6)},
    )


def _z_for(conf: float) -> float:
    """Two-sided normal quantile without SciPy (bisection on erf)."""
    target = 0.5 * (1.0 + conf)
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------------- sample planning


def required_labels(
    f_pool: Sequence[float],
    f_labeled: Sequence[float],
    y_labeled: Sequence[float],
    target_half_width: float,
    conf: float = 0.95,
    lam: float | None = None,
) -> dict[str, float]:
    """How many hand labels each method needs for a target CI half-width.

    Inverts the variance formulas. ``inf`` means unreachable at any ``n``: the model's
    own variance over the pool (the ``lambda^2 Var(f)/N`` term) is already larger than
    the target, so auditing that target requires more traffic, not more labels.
    """
    z = Z95 if conf == 0.95 else _z_for(conf)
    N = len(f_pool)
    if lam is None:
        var_f_labeled = _var(f_labeled)
        if var_f_labeled <= 1e-12:
            lam = 1.0
        else:
            cov = _mean([a * b for a, b in zip(y_labeled, f_labeled)]) - _mean(y_labeled) * _mean(f_labeled)
            n = len(y_labeled)
            cov *= n / max(1, n - 1)
            lam = min(1.0, max(0.0, cov / var_f_labeled))

    residuals = [y - lam * f for y, f in zip(y_labeled, f_labeled)]
    var_classical = _var(y_labeled)
    var_resid = _var(residuals)
    pool_term = (lam * lam) * _var(f_pool) / N

    def solve(variance_acc: float) -> float:
        budget = (target_half_width / z) ** 2 - pool_term
        if budget <= 0:
            return math.inf
        return variance_acc / budget

    return {
        "lambda": lam,
        "classical_labels": round(solve(var_classical), 1),
        "ppi_labels": round(solve(var_resid), 1),
        "pool_term_contribution": round(pool_term, 8),
        "target_half_width": target_half_width,
    }


# ------------------------------------------------------------------ audit simulation


def audit_simulation(
    f_pool: Sequence[float],
    y_pool: Sequence[float],
    n_labeled: int,
    repeats: int = 500,
    conf: float = 0.95,
    seed: int = 7,
    ci_margin: float = 0.0,
) -> dict[str, Any]:
    """Monte-Carlo study of the three estimators on real pool data.

    Each repeat draws a fresh random labelling budget of ``n_labeled`` rows (which is
    what a human audit gives you), runs classical / imputation / PPI++ on it, and
    checks whether the interval covers the pool's true value. Reports, per method:
    empirical CI coverage, mean interval width, and the number of labelled rows each
    would have needed to reach the same width.

    This is a finite-population simulation, so "the truth" is the pool mean -- stated
    explicitly because it is exactly the kind of thing that should be stated.
    """
    rng = random.Random(seed)
    N = len(f_pool)
    truth = _mean(y_pool)
    margin = ci_margin
    stats: dict[str, dict[str, float]] = {
        name: {"covered": 0.0, "width": 0.0, "estimate_sum": 0.0, "runs": 0.0}
        for name in ("classical", "imputation", "ppi")
    }
    for _ in range(repeats):
        idx = sorted(rng.sample(range(N), n_labeled))
        f_lab = [f_pool[i] for i in idx]
        y_lab = [y_pool[i] for i in idx]
        f_unlab = [f_pool[i] for i in range(N) if i not in set(idx)]
        estimates = {
            "classical": classical_estimate(y_lab, N, conf=conf),
            "imputation": imputation_estimate(f_unlab or f_pool, n_labeled, conf=conf),
            "ppi": ppi_estimate(f_unlab or f_pool, f_lab, y_lab, conf=conf),
        }
        for name, est in estimates.items():
            low, high = est.ci
            stats[name]["covered"] += float(low - margin <= truth <= high + margin)
            stats[name]["width"] += 2 * est.half_width
            stats[name]["estimate_sum"] += est.estimate
            stats[name]["runs"] += 1

    out: dict[str, Any] = {
        "n_pool": N,
        "n_labeled": n_labeled,
        "repeats": repeats,
        "truth_pool_mean": round(truth, 5),
        "conf": conf,
        "methods": {},
    }
    for name, acc in stats.items():
        runs = acc["runs"] or 1
        out["methods"][name] = {
            "ci_coverage": round(acc["covered"] / runs, 4),
            "mean_width": round(acc["width"] / runs, 5),
            "mean_estimate": round(acc["estimate_sum"] / runs, 5),
            "bias": round(acc["estimate_sum"] / runs - truth, 5),
        }
    classical_width = out["methods"]["classical"]["mean_width"]
    ppi_width = out["methods"]["ppi"]["mean_width"]
    out["ppi_width_reduction_vs_classical"] = (
        round(1.0 - ppi_width / classical_width, 4) if classical_width else None
    )
    out["labels_ratio_classical_over_ppi"] = (
        round((classical_width / ppi_width) ** 2, 3) if ppi_width else None
    )
    return out
