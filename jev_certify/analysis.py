"""Analysis: turn a journal of Jev decisions into certified, measured claims.

Nothing in this module calls the API. It reads the sampled pools (``plan-*.json``) and
the decision journal (``journal.jsonl``), rebuilds each request deterministically, and
computes:

* the **certified routing threshold** at several risk levels, via conformal risk
  control, plus the measured behaviour of that threshold on held-out data;
* **conformal prediction sets** (APS and margin scores) -- coverage and mean set size,
  which is the honest way to say "Jev is not sure, here are the labels it is choosing
  between";
* the **certified scope gate** -- a `noul` threshold that bounds the per-query rate of
  queries wrongly accepted as in-scope;
* a **prediction-powered audit** -- how narrow a confidence interval on the deployed
  accuracy you get from a small hand-labelled sample;
* a **drift section** -- the same procedure re-run on traffic the calibration never saw,
  because a guarantee that is never stress-tested is a claim, not a result.

The output is a single JSON-serialisable dict; :mod:`jev_certify.report` renders it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .cascade import CostAssumptions, build_operating_points
from .client import Decision, DecisionJournal
from .conformal import (
    aps_score,
    certified_threshold,
    choice_of,
    coverage_of_sets,
    lac_score,
    prediction_set,
    probabilities,
    risk_coverage_curve,
    selective_report,
    top_probability,
)
from .ppi import (
    audit_simulation,
    brier_score,
    expected_calibration_error,
    reliability_bins,
    required_labels,
)
from .tasks import Query, read_plan, requests_for

PRIMARY_ALPHAS = (0.01, 0.02, 0.05, 0.10)


# --------------------------------------------------------------------- data loading


@dataclass
class Scenario:
    """One deployment scenario: pools + the Jev decisions that answered them."""

    tag: str
    deployed_intents: list[str]
    calibration: list[Query]
    evaluation: list[Query]
    oos_evaluation: list[Query]
    shift_evaluation: list[Query]
    in_taxonomy_evaluation: list[Query]
    decisions: dict[str, Decision]

    def pool(self, name: str) -> list[Query]:
        return getattr(self, name)


def load_scenario(plan_path: str | Path, journal_path: str | Path, tag: str | None = None) -> Scenario:
    """Rebuild requests for a plan and attach the journalled Jev answer to each row."""
    plan = read_plan(plan_path)
    deployed = plan["deployed_intents"]
    journal = DecisionJournal(journal_path)
    records = journal.load()

    decisions: dict[str, Decision] = {}
    for key in (
        "calibration",
        "evaluation",
        "oos_evaluation",
        "shift_evaluation",
        "in_taxonomy_evaluation",
    ):
        rows: list[Query] = plan.get(key) or []
        if not rows:
            continue
        for (state, questions), row in zip(requests_for(rows, deployed), rows):
            from .client import decision_key

            lookup = decision_key("typesafe/jev-1.13", state, questions)
            record = records.get(lookup)
            if record is None:
                raise KeyError(
                    f"journal is missing a decision for {row.qid}; re-run experiments.collect "
                    "without --offline to fill the gaps (already-journalled rows cost nothing)"
                )
            decisions[row.qid] = Decision.from_record(record)

    return Scenario(
        tag=plan["tag"],
        deployed_intents=deployed,
        calibration=plan.get("calibration") or [],
        evaluation=plan.get("evaluation") or [],
        oos_evaluation=plan.get("oos_evaluation") or [],
        shift_evaluation=plan.get("shift_evaluation") or [],
        in_taxonomy_evaluation=plan.get("in_taxonomy_evaluation") or [],
        decisions=decisions,
    )


# -------------------------------------------------------------------- row extraction


def row_facts(row: Query, decision: Decision) -> dict[str, Any]:
    """Everything one decision tells us about one query, in plain Python types."""
    answer = decision.answers
    intent_answer = answer.get("intent", {})
    gate_answer = answer.get("in_scope", {})
    choice = choice_of(intent_answer) if intent_answer else None
    return {
        "qid": row.qid,
        "text": row.text,
        "gold_intent": row.gold_intent,
        "covered_by_taxonomy": row.covered_by_taxonomy,
        "choice": choice,
        "correct": bool(choice is not None and choice == row.gold_intent),
        "top_probability": top_probability(intent_answer) if intent_answer else 0.0,
        "confidence": float(intent_answer.get("confidence", 0.0) or 0.0),
        "probabilities": probabilities(intent_answer) if intent_answer else {},
        "in_scope_probability": float(gate_answer.get("noul", 0.0) or 0.0) if gate_answer else 0.0,
        "latency_s": decision.latency_s,
        "cost_usd": decision.cost_usd,
        "input_tokens": decision.input_tokens,
        "output_tokens": decision.output_tokens,
    }


def facts_for(rows: Sequence[Query], decisions: dict[str, Decision]) -> list[dict[str, Any]]:
    return [row_facts(row, decisions[row.qid]) for row in rows]


def _split(rows: list[dict[str, Any]]) -> tuple[list[float], list[bool]]:
    return [r["top_probability"] for r in rows], [r["correct"] for r in rows]


# ------------------------------------------------------------------ sub-analyses


def routing_certification(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    alphas: Sequence[float] = PRIMARY_ALPHAS,
    cost: CostAssumptions | None = None,
) -> dict[str, Any]:
    """Certified auto-route thresholds and their measured behaviour on held-out data."""
    cal_probs, cal_correct = _split(calibration)
    eval_probs, eval_correct = _split(evaluation)

    out: dict[str, Any] = {
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        # Accuracy is reported both ways on purpose: the calibration mix includes the
        # out-of-scope rows the gate needs, and those can never be routed "correctly",
        # so the mixed number understates the router and the in-scope number is the one
        # that describes the routing question.
        "calibration_accuracy_in_scope": (
            round(
                sum(1 for r in calibration if r["covered_by_taxonomy"] and r["correct"])
                / max(1, sum(1 for r in calibration if r["covered_by_taxonomy"])),
                4,
            )
        ),
        "calibration_accuracy_as_mixed": (
            round(sum(cal_correct) / len(cal_correct), 4) if cal_correct else None
        ),
        "top1_accuracy": round(sum(eval_correct) / len(eval_correct), 4) if eval_correct else None,
        "certificates": [],
        "curves": {},
    }

    for alpha in alphas:
        cert = certified_threshold(cal_probs, cal_correct, alpha=alpha)
        entry: dict[str, Any] = {
            "alpha": alpha,
            "feasible": cert.feasible,
            "threshold": round(cert.threshold, 4) if cert.feasible else None,
            "certified_per_query_loss": round(cert.certification_bound, 4) if cert.feasible else None,
            "calibration_per_query_loss": (
                round(cert.calibration_per_query_risk, 4) if cert.feasible else None
            ),
            "calibration_n": cert.n_calibration,
            "min_attainable_bound": round(cert.min_attainable_bound, 4),
            "note": cert.note,
        }
        if cert.feasible:
            report = selective_report(eval_probs, eval_correct, cert.threshold, alpha=alpha)
            entry["measured_on_holdout"] = report.as_dict()
            entry["certificate_held"] = bool(report.per_query_risk <= alpha)
            entry["coverage"] = round(report.coverage, 4)
        out["certificates"].append(entry)

    primary = next((c for c in out["certificates"] if c["alpha"] == 0.05 and c["feasible"]), None)
    if primary is not None and cost is not None:
        cert = certified_threshold(cal_probs, cal_correct, alpha=0.05)
        out["operating_points"] = build_operating_points(cert, eval_probs, eval_correct, cost)
        out["primary"] = primary
    out["curves"] = {
        "in_scope_only": risk_coverage_curve(eval_probs, eval_correct),
        "calibration": risk_coverage_curve(cal_probs, cal_correct),
    }
    return out


def prediction_set_analysis(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Conformal prediction sets: APS vs margin score, coverage and set size."""
    out: dict[str, Any] = {"alpha": alpha, "scores": {}}
    eval_truths = [r["gold_intent"] for r in evaluation]
    for name, scorer in (("aps", aps_score), ("lac", lac_score)):
        cal_scores = [
            scorer({"probabilities": r["probabilities"], "choice": r["choice"]}, r["gold_intent"])
            for r in calibration
        ]
        from .conformal import conformal_quantile

        qhat = conformal_quantile(cal_scores, alpha)
        thresholds = [qhat, qhat * 0.9, qhat * 1.1, 1.0, 0.5]
        curve = coverage_of_sets(
            [{"probabilities": r["probabilities"], "choice": r["choice"]} for r in evaluation],
            eval_truths,
            sorted({round(t, 6) for t in thresholds}),
            score=name,
        )
        target = next((c for c in curve if abs(c["qhat"] - round(qhat, 6)) < 1e-9), None)
        out["scores"][name] = {
            "qhat_calibration": None if qhat == float("inf") else round(qhat, 6),
            "infinite_qhat": qhat == float("inf"),
            "at_calibrated_qhat": target,
            "curve": curve,
        }
    return out


def probability_granularity(calibration: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> dict[str, Any]:
    """How coarse are the routing probabilities, and what does that cost?

    A per-query risk certificate cannot resolve below the model's own output
    granularity: if the model returns exactly 1.0 on queries it gets wrong, no threshold
    separates them, and the attainable bound floors out. Worth measuring explicitly
    because it is a property of the model, not of the statistics.
    """
    rows = calibration + evaluation
    values = [round(r["top_probability"], 6) for r in rows]
    distinct = sorted(set(values))
    multiples_of_001 = sum(1 for v in distinct if abs(v * 100 - round(v * 100)) < 1e-9)
    saturated = [r for r in rows if r["top_probability"] >= 0.9999]
    saturated_wrong = sum(1 for r in saturated if not r["correct"])

    # Histogram of the top probability, split into right and wrong answers: this is the
    # picture behind the resolution floor, so it is shipped rather than described.
    n_bins = 20
    counts = [0] * n_bins
    errors = [0] * n_bins
    for row in rows:
        index = min(n_bins - 1, int(row["top_probability"] * n_bins))
        counts[index] += 1
        if not row["correct"]:
            errors[index] += 1

    return {
        "answers": len(rows),
        "distinct_top_probabilities": len(distinct),
        "distinct_values_are_multiples_of_0.01": f"{multiples_of_001}/{len(distinct)}",
        "min_positive_top_probability": next((v for v in distinct if v > 0), None),
        "share_at_1.0": round(len(saturated) / len(rows), 4) if rows else None,
        "error_rate_at_1.0": (
            round(saturated_wrong / len(saturated), 4) if saturated else None
        ),
        "errors_at_1.0": saturated_wrong,
        "histogram_bin_width": round(1.0 / n_bins, 4),
        "histogram_counts": counts,
        "histogram_errors": errors,
        "note": (
            "Shares at 1.0 with errors behind them are the reason a strict threshold cannot "
            "drive the measured risk to zero, and therefore the reason a very small alpha is "
            "infeasible at a given sample size."
        ),
    }


def _matched_prevalence_evaluation(
    evaluation: list[dict[str, Any]], calibration_prevalence: float, seed: int = 5
) -> list[dict[str, Any]]:
    """Resample the evaluation mix so the uncovered share matches the calibration mix.

    The gate's per-query loss counts one failure per *incoming query*, so it scales with
    how much traffic the taxonomy does not cover. Calibrating on a 13% uncovered mix and
    deploying on a 43% uncovered mix is a different quantity, and separating the two is
    the point of this function.
    """
    covered = [r for r in evaluation if r["covered_by_taxonomy"]]
    uncovered = [r for r in evaluation if not r["covered_by_taxonomy"]]
    if not covered or not uncovered:
        return evaluation
    wanted = int(round(calibration_prevalence * len(covered) / max(1e-9, 1 - calibration_prevalence)))
    wanted = max(1, min(len(uncovered), wanted))
    rng = random.Random(seed)
    return covered + rng.sample(uncovered, wanted)


def gate_analysis(
    calibration: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    alphas: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
) -> dict[str, Any]:
    """The scope guardrail: a certified bound on wrongly accepting an out-of-scope query.

    The loss is ``1{accepted as in-scope} * 1{actually not covered}`` -- the share of
    incoming queries the gate silently hands to the router even though no deployed
    intent covers them. Conformal risk control bounds it at ``alpha``.
    """
    cal_probs = [r["in_scope_probability"] for r in calibration]
    cal_correct = [r["covered_by_taxonomy"] for r in calibration]
    eval_probs = [r["in_scope_probability"] for r in evaluation]
    eval_correct = [r["covered_by_taxonomy"] for r in evaluation]

    out: dict[str, Any] = {
        "calibration_n": len(calibration),
        "evaluation_n": len(evaluation),
        "certificates": [],
        "sample_answers": [
            {
                "text": r["text"][:90],
                "gold": r["gold_intent"] or "out-of-scope",
                "in_scope_probability": round(r["in_scope_probability"], 3),
            }
            for r in sorted(evaluation, key=lambda r: r["in_scope_probability"])[:6]
        ],
    }
    if len(set(cal_correct)) < 2:
        out["note"] = (
            "the calibration mix contains a single class, so no gate threshold can be fitted. "
            "A scope gate needs examples of traffic the deployed taxonomy does not cover -- "
            "this is a real deployment hazard, not a limitation of the method."
        )
        return out
    in_scope_eval = [r for r in evaluation if r["covered_by_taxonomy"]]
    oos_eval = [r for r in evaluation if not r["covered_by_taxonomy"]]
    calibration_prevalence = 1.0 - (sum(1 for c in cal_correct if c) / len(cal_correct))
    matched_mix = _matched_prevalence_evaluation(evaluation, calibration_prevalence)
    out["calibration_uncovered_prevalence"] = round(calibration_prevalence, 4)
    out["evaluation_uncovered_prevalence"] = round(
        len(oos_eval) / len(evaluation) if evaluation else 0.0, 4
    )
    for alpha in alphas:
        cert = certified_threshold(cal_probs, cal_correct, alpha=alpha)
        entry: dict[str, Any] = {
            "alpha": alpha,
            "feasible": cert.feasible,
            "threshold": round(cert.threshold, 4) if cert.feasible else None,
            "certified_false_accept_per_query": (
                round(cert.certification_bound, 4) if cert.feasible else None
            ),
            "min_attainable_bound": round(cert.min_attainable_bound, 4),
            "note": cert.note,
        }
        if cert.feasible:
            report = selective_report(eval_probs, eval_correct, cert.threshold, alpha=alpha)
            in_scope_pass = sum(1 for r in in_scope_eval if r["in_scope_probability"] >= cert.threshold)
            oos_accept = sum(1 for r in oos_eval if r["in_scope_probability"] >= cert.threshold)
            matched_report = selective_report(
                [r["in_scope_probability"] for r in matched_mix],
                [r["covered_by_taxonomy"] for r in matched_mix],
                cert.threshold,
                alpha=alpha,
            )
            entry.update(
                {
                    "measured": report.as_dict(),
                    "in_scope_pass_rate": round(in_scope_pass / len(in_scope_eval), 4) if in_scope_eval else None,
                    "out_of_scope_accept_rate": round(oos_accept / len(oos_eval), 4) if oos_eval else None,
                    "certificate_held": bool(report.per_query_risk <= alpha),
                    # The same threshold on a mix whose uncovered share matches the
                    # calibration mix: this is the isolated effect of prevalence.
                    "measured_on_prevalence_matched_mix": matched_report.as_dict(),
                    "certificate_held_at_matched_prevalence": bool(
                        matched_report.per_query_risk <= alpha
                    ),
                }
            )
        out["certificates"].append(entry)

    cps = [r["in_scope_probability"] for r in calibration] if calibration else []
    if cps:
        out["calibration_curve"] = risk_coverage_curve(eval_probs, eval_correct)
        out["noul_diagnostics"] = {
            "brier": round(brier_score(cps, [int(c) for c in cal_correct]), 4),
            "ece10": round(expected_calibration_error(cps, [int(c) for c in cal_correct], bins=10), 4),
            "reliability": reliability_bins(cps, [int(c) for c in cal_correct], bins=10),
        }
        # The routing probabilities carry the same diagnostic, because PPI's efficiency
        # depends on how informative they are.
        rps = [r["top_probability"] for r in calibration]
        rc = [int(r["correct"]) for r in calibration]
        out["routing_probability_diagnostics"] = {
            "brier": round(brier_score(rps, rc), 4),
            "ece10": round(expected_calibration_error(rps, rc, bins=10), 4),
            "reliability": reliability_bins(rps, rc, bins=10),
        }
    return out


def audit_analysis(
    pool: list[dict[str, Any]],
    threshold: float,
    label_budgets: Sequence[int] = (25, 50, 100, 200),
    repeats: int = 500,
) -> dict[str, Any]:
    """Prediction-powered audit of the deployed policy, across hand-labelling budgets.

    ``f`` is Jev's own estimate that a query is answered correctly by the certified
    policy (``1{routed} * max_y P(y)``) and ``y`` is the truth from the human labels.
    """
    out: dict[str, Any] = {
        "pool_n": len(pool),
        "threshold": round(threshold, 4),
        "routed_share": round(sum(1 for r in pool if r["top_probability"] >= threshold) / len(pool), 4),
        "true_routed_correct": round(
            sum(1 for r in pool if r["top_probability"] >= threshold and r["correct"]) / len(pool), 4
        ),
        "budgets": [],
    }
    f_pool = [r["top_probability"] if r["top_probability"] >= threshold else 0.0 for r in pool]
    y_pool = [
        float(1 if (r["top_probability"] >= threshold and r["correct"]) else 0) for r in pool
    ]
    for budget in label_budgets:
        if budget >= len(pool):
            continue
        sim = audit_simulation(f_pool, y_pool, budget, repeats=repeats)
        out["budgets"].append(sim)

    # How much traffic (not how many labels) is needed for a tighter audit. Practitioners
    # ask "how many labels do I need?"; the answer is often "none, you need more traffic".
    biggest = max((b["n_labeled"] for b in out["budgets"]), default=50)
    # A random stand-in labelled subset for variance estimation: picking the *most
    # confident* rows instead would understate the variance and oversell the method.
    label_rows = random.Random(11).sample(range(len(pool)), min(biggest, len(pool)))
    planning = {}
    for target in (0.05, 0.02, 0.01):
        planning[f"half_width_{target}"] = required_labels(
            f_pool,
            [f_pool[i] for i in label_rows],
            [y_pool[i] for i in label_rows],
            target_half_width=target,
        )
    out["planning"] = planning
    out["planning_note"] = (
        "labels needed to reach a given 95% interval half-width. Infinite means no amount of "
        "hand labelling gets there: the term lambda^2 Var(f)/N is set by how much traffic the "
        "policy has served, so the pool has to grow, not the labelling budget."
    )
    return out


def shift_analysis(
    scenario: Scenario,
    primary_threshold: float,
    primary_gate_threshold: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply an in-distribution certificate to traffic it was never calibrated on."""
    cal = facts_for(scenario.calibration, scenario.decisions)
    sections: dict[str, Any] = {}
    for name, rows in (
        ("evaluation_exchangeable", scenario.evaluation),
        ("in_taxonomy_exchangeable", scenario.in_taxonomy_evaluation),
        ("out_of_scope_traffic", scenario.oos_evaluation),
        ("unlisted_intents_traffic", scenario.shift_evaluation),
    ):
        if not rows:
            continue
        facts = facts_for(rows, scenario.decisions)
        probs, correct = _split(facts)
        report = selective_report(probs, correct, primary_threshold, alpha=alpha)
        covered = [r["covered_by_taxonomy"] for r in facts]
        gate_report = selective_report(
            [r["in_scope_probability"] for r in facts], covered, primary_gate_threshold, alpha=alpha
        )
        sections[name] = {
            "n": len(facts),
            "intent_accuracy": round(sum(correct) / len(correct), 4) if correct else None,
            "routing_above_threshold": report.as_dict(),
            "certificate_held": bool(report.per_query_risk <= alpha),
            "gate_false_accept_rate_per_query": round(gate_report.per_query_risk, 4),
            "gate_false_accept_rate_among_accepted": round(gate_report.selective_error, 4),
            "gate_certificate_held": bool(gate_report.per_query_risk <= alpha),
        }
    return {"alpha": alpha, "sections": sections}


def cost_analysis(facts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Measured Jev economics, straight from the provider's own usage records."""
    rows = [r for r in facts]
    n = len(rows) or 1
    latencies = sorted(r["latency_s"] for r in rows)

    def percentile(p: float) -> float:
        if not latencies:
            return float("nan")
        return latencies[min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))]

    cost_total = sum(r["cost_usd"] for r in rows)
    return {
        "decisions": len(rows),
        "cost_usd_total": round(cost_total, 6),
        "cost_usd_per_1k": round(cost_total / n * 1000.0, 4),
        "input_tokens_mean": round(sum(r["input_tokens"] for r in rows) / n, 1),
        "output_tokens_mean": round(sum(r["output_tokens"] for r in rows) / n, 1),
        "latency_p50_s": round(percentile(0.5), 3),
        "latency_p95_s": round(percentile(0.95), 3),
    }


# ------------------------------------------------------------------------ top level


def full_analysis(
    full: Scenario,
    restricted: Scenario | None = None,
    alpha: float = 0.05,
    cost_assumptions: CostAssumptions | None = None,
    audit_repeats: int = 500,
) -> dict[str, Any]:
    """Everything a reader needs to audit the certificate, in one JSON object."""
    cal = facts_for(full.calibration, full.decisions)
    ev = facts_for(full.evaluation, full.decisions)
    # The routeable pool for audits: queries whose gold label is in the deployed
    # taxonomy. Out-of-scope queries have no correct destination, so they belong to the
    # gate, not the accuracy audit.
    routeable = [r for r in ev + facts_for(full.in_taxonomy_evaluation, full.decisions) if r["covered_by_taxonomy"]]

    if cost_assumptions is None:
        measured = cost_analysis(cal + ev)
        cost_assumptions = CostAssumptions(jev_usd_per_1k_queries=measured["cost_usd_per_1k"])

    routing = routing_certification(cal, ev, cost=cost_assumptions)
    primary_threshold = float(routing["primary"]["threshold"]) if routing.get("primary") else 1.0

    # The gate is evaluated on the full mix of traffic it will actually see: queries the
    # taxonomy covers *and* the out-of-scope queries it exists to stop.
    oos_facts = facts_for(full.oos_evaluation, full.decisions)
    gate = gate_analysis(cal, ev + oos_facts)
    gate_cert = next((c for c in gate["certificates"] if c["alpha"] == alpha and c["feasible"]), None)
    gate_threshold = float(gate_cert["threshold"]) if gate_cert else 1.0

    result: dict[str, Any] = {
        "meta": {
            "model": "typesafe/jev-1.13 (OpenRouter Decisions API)",
            "dataset": "CLINC150 (Larson et al., EMNLP 2019) human test + out-of-scope splits",
            "alpha": alpha,
            "scenario": full.tag,
            "deployed_intents": len(full.deployed_intents),
            "journal_decisions": len(full.decisions),
        },
        "cost": cost_analysis(cal + ev + facts_for(full.oos_evaluation, full.decisions)),
        "routing": routing,
        "prediction_sets": prediction_set_analysis(cal, ev, alpha=alpha),
        "quantization": probability_granularity(cal, ev),
        "gate": gate,
        "audit": audit_analysis(routeable, primary_threshold, repeats=audit_repeats),
        "drift": shift_analysis(full, primary_threshold, gate_threshold, alpha=alpha),
        "examples": {
            "confident_correct": [
                {"text": r["text"], "gold": r["gold_intent"], "choice": r["choice"], "p": round(r["top_probability"], 3)}
                for r in sorted(ev, key=lambda r: -r["top_probability"])
                if r["correct"]
            ][:5],
            "confident_wrong": [
                {
                    "text": r["text"],
                    "gold": r["gold_intent"],
                    "choice": r["choice"],
                    "p": round(r["top_probability"], 3),
                    "runner_up": sorted(
                        [(k, v) for k, v in r["probabilities"].items() if k != r["choice"]],
                        key=lambda kv: -kv[1],
                    )[:1],
                }
                for r in sorted(ev, key=lambda r: -r["top_probability"])
                if not r["correct"]
            ][:5],
        },
    }

    if restricted is not None:
        r_cal = facts_for(restricted.calibration, restricted.decisions)
        r_tax = facts_for(restricted.in_taxonomy_evaluation, restricted.decisions)
        r_shift = facts_for(restricted.shift_evaluation, restricted.decisions)
        restricted_routing = routing_certification(r_cal, r_tax, alphas=(alpha,))
        restricted_threshold = (
            float(restricted_routing["certificates"][0]["threshold"])
            if restricted_routing["certificates"][0]["feasible"]
            else 1.0
        )
        restricted_gate = gate_analysis(r_cal, r_tax, alphas=(alpha,))
        r_gate_cert = next(
            (c for c in restricted_gate["certificates"] if c["alpha"] == alpha and c["feasible"]), None
        )
        result["restricted_deployment"] = {
            "deployed_intents": len(restricted.deployed_intents),
            "routing": restricted_routing,
            "gate": restricted_gate,
            "drift": shift_analysis(
                restricted,
                restricted_threshold,
                float(r_gate_cert["threshold"]) if r_gate_cert else 1.0,
                alpha=alpha,
            ),
            "note": (
                "Same model, same procedure, same threshold-fitting code -- only the "
                "traffic distribution changes. This is the section that shows what the "
                "certificate does and does not cover."
            ),
        }
    return result


def write_results(result: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return target
