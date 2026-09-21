"""Cost model: turn a certified risk level into a decision-theoretic operating point.

The certificate is half the story; the other half is what it costs. Given a certified
threshold from :mod:`jev_certify.conformal` this module computes, per 1,000 queries:

* how many queries Jev settles alone and at what measured token cost,
* how many escalate to the expensive fallback (an LLM or a human),
* the total cost, and the certified bound on silently misrouted traffic.

It also prices the two baselines a team would actually deploy without this machinery:

* **always-LLM** -- every query to the fallback, the expensive-but-safe default;
* **hand-picked threshold** -- the documented System One pattern ("start conservative
  and adjust as you observe results"): a fixed number like 0.9 chosen by taste, whose
  real per-query loss is only knowable after the fact.

Every price except Jev's is an explicit assumption, surfaced in the output rather than
buried, because a cost model with hidden constants is not a cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .conformal import CertifiedThreshold, selective_report

USD_PER_1K = 1000.0


@dataclass
class CostAssumptions:
    """Prices used for the parts of the pipeline Jev does not pay for."""

    jev_usd_per_1k_queries: float
    """Measured: sum of the provider's reported cost for 1,000 decisions."""

    fallback_usd_per_1k_queries: float = 4.20
    """Assumed price of settling an escalated query with a frontier LLM.

    Default is the listed input rate of a ~$1.05/M-token frontier model on ~4k tokens
    of context per query. Change it: the point of the output is the shape of the curve,
    not this constant.
    """

    human_review_usd_per_1k_queries: float = 0.0
    """Optional: cost of a human look at an escalated query (e.g. $1.50 per query
    => 1500.0). Left at zero so the default comparison is machine-only."""

    def as_dict(self) -> dict[str, float]:
        return {
            "jev_usd_per_1k_queries": round(self.jev_usd_per_1k_queries, 6),
            "fallback_usd_per_1k_queries": round(self.fallback_usd_per_1k_queries, 4),
            "human_review_usd_per_1k_queries": round(self.human_review_usd_per_1k_queries, 4),
        }


@dataclass
class OperatingPoint:
    """One deployable configuration and its measured outcome."""

    name: str
    alpha: float | None
    threshold: float
    certified: bool
    coverage: float
    selective_error: float
    selective_error_ucb: float
    per_query_loss: float
    certified_per_query_loss: float | None
    cost_per_1k_usd: float
    escalation_rate: float
    measured: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "threshold": round(self.threshold, 4),
            "certified": self.certified,
            "coverage": round(self.coverage, 4),
            "selective_error": round(self.selective_error, 4),
            "selective_error_ucb95": round(self.selective_error_ucb, 4),
            "per_query_loss": round(self.per_query_loss, 4),
            "certified_per_query_loss": (
                round(self.certified_per_query_loss, 4) if self.certified_per_query_loss is not None else None
            ),
            "escalation_rate": round(self.escalation_rate, 4),
            "cost_per_1k_usd": round(self.cost_per_1k_usd, 4),
            "measured": self.measured,
        }


def cost_per_1k(
    coverage: float,
    assumptions: CostAssumptions,
    escalation_extra_usd_per_1k: float | None = None,
) -> float:
    """Cost of 1,000 queries: one Jev decision each, plus escalations."""
    escalate = 1.0 - coverage
    fallback = (
        assumptions.fallback_usd_per_1k_queries
        if escalation_extra_usd_per_1k is None
        else escalation_extra_usd_per_1k
    )
    return assumptions.jev_usd_per_1k_queries + escalate * (fallback + assumptions.human_review_usd_per_1k_queries)


def operating_point_from_threshold(
    name: str,
    threshold: float,
    probabilities_max: Sequence[float],
    correct: Sequence[bool],
    assumptions: CostAssumptions,
    alpha: float | None = None,
    certified_per_query_loss: float | None = None,
    certified: bool = False,
) -> OperatingPoint:
    report = selective_report(probabilities_max, correct, threshold)
    return OperatingPoint(
        name=name,
        alpha=alpha,
        threshold=threshold,
        certified=certified,
        coverage=report.coverage,
        selective_error=report.selective_error,
        selective_error_ucb=report.selective_error_ucb,
        per_query_loss=report.per_query_risk,
        certified_per_query_loss=certified_per_query_loss,
        cost_per_1k_usd=cost_per_1k(report.coverage, assumptions),
        escalation_rate=1.0 - report.coverage,
        measured=report.as_dict(),
    )


def build_operating_points(
    certification: CertifiedThreshold,
    probabilities_max: Sequence[float],
    correct: Sequence[bool],
    assumptions: CostAssumptions,
    hand_picked: Sequence[float] = (0.5, 0.7, 0.9, 0.99),
) -> dict[str, Any]:
    """Compare the certified thresholds against hand-picked ones and always-LLM.

    ``probabilities_max``/``correct`` must be **held-out** data, never the calibration
    split: the certificate was computed there, so evaluating on it would flatter it.
    """
    points: list[OperatingPoint] = []

    if certification.feasible:
        points.append(
            operating_point_from_threshold(
                name=f"certified risk <= {certification.alpha:.0%}",
                threshold=certification.threshold,
                probabilities_max=probabilities_max,
                correct=correct,
                assumptions=assumptions,
                alpha=certification.alpha,
                certified_per_query_loss=certification.certification_bound,
                certified=True,
            )
        )
        # A saturated threshold for the trade-off table. It is *not* certified -- it is
        # what a team would reach by dialling the number up until the errors stop, with
        # no idea what the resulting bound is.
        points.append(
            operating_point_from_threshold(
                name="saturated threshold 0.999 (not certified)",
                threshold=0.999,
                probabilities_max=probabilities_max,
                correct=correct,
                assumptions=assumptions,
            )
        )
    for lam in hand_picked:
        points.append(
            operating_point_from_threshold(
                name=f"hand-picked threshold {lam:g}",
                threshold=lam,
                probabilities_max=probabilities_max,
                correct=correct,
                assumptions=assumptions,
            )
        )

    n = len(probabilities_max)
    always_llm_cost = assumptions.fallback_usd_per_1k_queries + assumptions.human_review_usd_per_1k_queries
    summary = {
        "assumptions": assumptions.as_dict(),
        "operating_points": [p.as_dict() for p in points],
        "always_llm": {
            "name": "always LLM (no router)",
            "coverage": 1.0,
            # Deliberately not filled in: the fallback model's accuracy was not measured
            # in this study, and copying Jev's error rate into this row would present an
            # assumption as a measurement.
            "selective_error": None,
            "cost_per_1k_usd": round(always_llm_cost, 4),
        },
        "note": (
            "per_query_loss is the expected share of incoming queries that are routed "
            "and answered wrongly; the certified column is the finite-sample bound from "
            "conformal risk control, the other columns are measurements. The always-LLM "
            "row's accuracy is NOT measured here -- only its cost is, from the stated "
            "assumption -- so compare it on price, not on quality."
        ),
    }
    summary["n_evaluation"] = n
    if certification.feasible:
        certified_point = points[0]
        summary["savings_vs_always_llm_pct"] = round(
            100.0 * (1.0 - certified_point.cost_per_1k_usd / always_llm_cost), 2
        ) if always_llm_cost else None
        summary["certificate_held"] = bool(
            certified_point.per_query_loss <= certification.alpha
        )
    return summary
