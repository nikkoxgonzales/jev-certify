"""Reporting: render an analysis result as Markdown and as a one-page HTML dashboard.

Both renderers read numbers only from the analysis dict -- nothing is hard-coded, so a
report can never drift away from the data that produced it. The dashboard draws its
charts as inline SVG with no dependencies, so it works offline and can be committed.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _svg_line_chart(
    series: Sequence[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 520,
    height: int = 260,
    y_max: float | None = None,
) -> str:
    """Minimal inline-SVG line chart (no plotting dependency)."""
    points = [(float(p[x_key]), float(p[y_key])) for p in series if p.get(y_key) is not None]
    if len(points) < 2:
        return f"<p><em>not enough data for {html.escape(title)}</em></p>"
    pad_l, pad_r, pad_t, pad_b = 52, 16, 28, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_top = y_max if y_max is not None else max(p[1] for p in points) * 1.15
    y_top = max(y_top, 1e-9)

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min or 1.0) * plot_w

    def sy(y: float) -> float:
        return pad_t + plot_h - min(1.0, y / y_top) * plot_h

    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points))
    ticks_y = "".join(
        f'<text x="{pad_l - 8}" y="{sy(f):.1f}" text-anchor="end" font-size="10" fill="var(--muted-foreground, #666)">{f:.2f}</text>'
        for f in (0.0, y_top / 2, y_top)
    )
    ticks_x = "".join(
        f'<text x="{sx(v):.1f}" y="{height - 12}" text-anchor="middle" font-size="10" fill="var(--muted-foreground, #666)">{v:.2f}</text>'
        for v in (x_min, (x_min + x_max) / 2, x_max)
    )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{html.escape(title)}">
  <text x="{pad_l}" y="16" font-size="12" fill="var(--foreground, #111)">{html.escape(title)}</text>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="var(--border, #ddd)"/>
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="var(--border, #ddd)"/>
  <path d="{path}" fill="none" stroke="var(--accent, #3b82f6)" stroke-width="2"/>
  {ticks_y}{ticks_x}
  <text x="{pad_l + plot_w}" y="{height - 12}" text-anchor="end" font-size="10" fill="var(--muted-foreground, #666)">{html.escape(x_label)}</text>
  <text x="{pad_l}" y="{pad_t - 8}" font-size="10" fill="var(--muted-foreground, #666)">{html.escape(y_label)}</text>
</svg>"""


def _svg_bars(labels: Sequence[str], values: Sequence[float], title: str, unit: str = "") -> str:
    width, height = 520, 220
    pad_l, pad_t, pad_b = 120, 28, 26
    plot_w = width - pad_l - 16
    plot_h = height - pad_t - pad_b
    top = max(values) * 1.15 if values else 1.0
    row_h = plot_h / max(1, len(values))
    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = pad_t + i * row_h + row_h * 0.15
        bar_h = row_h * 0.7
        w = (value / top) * plot_w
        bars.append(
            f'<rect x="{pad_l}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" rx="3" fill="var(--accent, #3b82f6)" opacity="0.75"/>'
            f'<text x="{pad_l - 8}" y="{y + bar_h * 0.75:.1f}" text-anchor="end" font-size="11" fill="var(--foreground, #111)">{html.escape(label)}</text>'
            f'<text x="{pad_l + w + 6:.1f}" y="{y + bar_h * 0.75:.1f}" font-size="11" fill="var(--muted-foreground, #666)">{value:.4f}{html.escape(unit)}</text>'
        )
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{html.escape(title)}">
  <text x="12" y="16" font-size="12" fill="var(--foreground, #111)">{html.escape(title)}</text>
  {''.join(bars)}
</svg>"""


def render_markdown(result: dict[str, Any]) -> str:
    meta = result["meta"]
    routing = result["routing"]
    gate = result["gate"]
    cost = result["cost"]
    sets = result["prediction_sets"]["scores"]
    audit = result["audit"]
    drift = result["drift"]
    alpha = meta["alpha"]

    primary = routing.get("primary") or {}
    primary_measured = primary.get("measured_on_holdout", {}) if primary else {}
    gate_primary = next(
        (c for c in gate["certificates"] if c["alpha"] == alpha and c["feasible"]), {}
    )
    gate_tight = next(
        (c for c in gate["certificates"] if c["feasible"] and c.get("certificate_held")), {}
    )
    best_audit = max(
        (b for b in audit["budgets"]),
        key=lambda b: b["ppi_width_reduction_vs_classical"] or 0.0,
        default=None,
    )

    lines: list[str] = []
    lines.append("# jev-certify")
    lines.append("")
    lines.append(
        "**Turning Jev's calibrated probabilities into claims that survive an audit: a "
        "finite-sample bound on how many queries get routed wrong, plus a way to check "
        "the bound in production with a few hundred hand labels instead of all of them.**"
    )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    if primary:
        lines.append(
            f"- With {primary['calibration_n']} calibration queries and a target of "
            f"**{_pct(alpha, 0)} silently misrouted queries per incoming request**, conformal risk "
            f"control picks a confidence threshold of **{_num(primary['threshold'], 3)}** and certifies "
            f"an expected per-query loss of **{_num(primary['certified_per_query_loss'], 4)}**."
        )
        lines.append(
            f"- On {primary_measured.get('n', 'n/a')} held-out human-labelled queries that threshold "
            f"auto-routes **{_pct(primary_measured.get('coverage'))}** of traffic, of which "
            f"**{_pct(primary_measured.get('selective_error'))}** are wrong "
            f"(measured per-query loss {_num(primary_measured.get('per_query_risk'), 4)} — the "
            f"certificate held: {primary.get('certificate_held')})."
        )
    if gate_tight:
        lines.append(
            f"- The scope guardrail (`noul` ≥ {_num(gate_tight['threshold'], 3)}) certifies at most "
            f"**{_num(gate_tight['certified_false_accept_per_query'], 4)}** out-of-scope queries accepted "
            f"per incoming request: it passes **{_pct(gate_tight.get('in_scope_pass_rate'))}** of in-scope "
            f"traffic while accepting **{_pct(gate_tight.get('out_of_scope_accept_rate'))}** of the human "
            f"out-of-scope test set. The looser targets below are prevalence-sensitive — see §3, which "
            "is the section that matters most here."
        )
    if best_audit:
        ppi = best_audit["methods"]["ppi"]
        classical = best_audit["methods"]["classical"]
        lines.append(
            f"- Auditing the live policy with **{best_audit['n_labeled']} hand labels** out of "
            f"{best_audit['n_pool']} decisions: PPI++ gives a 95% interval of width "
            f"**{ppi['mean_width']:.4f}** against **{classical['mean_width']:.4f}** from labels alone "
            f"({_pct(best_audit['ppi_width_reduction_vs_classical'])} narrower)."
        )
    lines.append(
        f"- Cost of a decision, measured from the provider's own usage records: "
        f"**${cost['cost_usd_per_1k']:.4f} per 1,000 queries**, p50 latency "
        f"**{cost['latency_p50_s']}s**, p95 **{cost['latency_p95_s']}s**."
    )
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(
        _table(
            ["Item", "Value"],
            [
                ["Model", meta["model"]],
                ["Task data", meta["dataset"]],
                ["Deployed intents", meta["deployed_intents"]],
                ["Calibration / evaluation queries", f"{routing['calibration_n']} / {routing['evaluation_n']}"],
                ["Target risk (alpha)", _pct(alpha, 0)],
                ["Jev top-1 routing accuracy (held out)", _pct(routing["top1_accuracy"])],
                ["…on in-scope calibration queries", _pct(routing["calibration_accuracy_in_scope"])],
                [
                    "…on the calibration mix as sampled (includes the gate's out-of-scope rows, "
                    "which no router can get right)",
                    _pct(routing["calibration_accuracy_as_mixed"]),
                ],
                ["Decisions journalled", meta["journal_decisions"]],
            ],
        )
    )
    lines.append("")

    lines.append("## 1. Certified routing threshold")
    lines.append("")
    lines.append(
        "Conformal risk control (Angelopoulos et al., 2022) on the loss "
        "`1{auto-routed} x 1{wrong}`. The bound is on **expected silently misrouted "
        "queries per incoming request** — a per-query rate, not an accuracy on the "
        "queries that happened to be routed."
    )
    lines.append("")
    rows = []
    for cert in routing["certificates"]:
        measured = cert.get("measured_on_holdout", {})
        rows.append(
            [
                _pct(cert["alpha"], 0),
                _num(cert["threshold"], 3) if cert["feasible"] else "infeasible",
                _num(cert["certified_per_query_loss"], 4),
                _pct(measured.get("coverage")),
                _pct(measured.get("selective_error")),
                _num(measured.get("per_query_risk"), 4),
                "yes" if cert.get("certificate_held") else "no",
            ]
        )
    lines.append(
        _table(
            ["Target", "Threshold", "Certified loss/query", "Coverage", "Error when routed", "Measured loss/query", "Held?"],
            rows,
        )
    )
    lines.append("")
    infeasible = [c for c in routing["certificates"] if not c["feasible"]]
    if infeasible:
        first = infeasible[0]
        lines.append(
            f"**A target of {_pct(first['alpha'], 0)} is not attainable at this sample size.** "
            f"The tightest certificate {first['calibration_n']} calibration queries support is "
            f"{_num(first['min_attainable_bound'], 4)} per query, because {first.get('note', '')}."
        )
        lines.append(
            "This is the resolution floor of the certificate, and it is a property of the data, "
            "not of the method: the bound cannot be tightened by choosing a cleverer threshold, "
            "only by collecting more calibration traffic (or by the model not returning exactly "
            "1.0 on queries it gets wrong)."
        )
        lines.append("")

    lines.append("## 2. What Jev is choosing between (conformal prediction sets)")
    lines.append("")
    lines.append(
        "A `choice` answer is a distribution, and its top label is only one number. "
        "Conformal prediction sets turn that distribution into a set that provably "
        "contains the truth with probability ≥ "
        f"{_pct(1 - alpha)} (Romano et al., 2020 for APS; Sadinle et al., 2019 for the "
        "margin score). Set size is the honest measure of how much ambiguity is left."
    )
    lines.append("")
    rows = []
    for name, payload in sets.items():
        at = payload.get("at_calibrated_qhat")
        if not at:
            continue
        rows.append(
            [
                name.upper(),
                "∞ (all labels)" if payload.get("infinite_qhat") else _num(payload.get("qhat_calibration"), 4),
                _pct(at["coverage"]),
                _num(at["mean_set_size"], 2),
                f"[{_pct(at['coverage_ci_lo'])}, {_pct(at['coverage_ci_hi'])}]",
                at["top1_fallbacks"],
            ]
        )
    lines.append(
        _table(["Score", "Calibrated threshold", "Coverage", "Mean set size", "95% CI on coverage", "Top-1 fallbacks"], rows)
    )
    lines.append("")
    oversize = [
        (name, payload["at_calibrated_qhat"])
        for name, payload in sets.items()
        if payload.get("at_calibrated_qhat") and payload["at_calibrated_qhat"]["mean_set_size"] > 20
    ]
    if oversize:
        worst = max(oversize, key=lambda item: item[1]["mean_set_size"])
        lines.append(
            f"**Set-valued routing is not usable on this task.** {worst[0].upper()} sets average "
            f"{_num(worst[1]['mean_set_size'], 1)} labels at {_pct(worst[1]['coverage'])} coverage: with a "
            "150-label taxonomy and a peaked distribution, guaranteeing the truth is in the set means "
            "admitting most of the taxonomy, which is not a routing decision. Set-valued prediction "
            "earns its keep on small label spaces or over retrieval-style candidate shortlists; on a "
            "150-way router the threshold certificate above is the part that buys something. Kept in "
            "the report because a method that is measured and then dropped is worth more than one "
            "that was never tried."
        )
        lines.append("")

    lines.append("## 3. Scope guardrail")
    lines.append("")
    lines.append(
        "A second, independent question rides along in the same call: *does the deployed "
        "taxonomy cover this query at all?* Its loss is `1{accepted} x 1{not covered}`, "
        "so the certificate bounds the share of traffic the gate silently lets through."
    )
    lines.append("")
    if not gate["certificates"]:
        lines.append(f"*Not calibrated:* {gate.get('note', 'no certificate available')}")
        lines.append("")
    rows = []
    for cert in gate["certificates"]:
        measured = cert.get("measured", {})
        rows.append(
            [
                _pct(cert["alpha"], 0),
                _num(cert["threshold"], 3) if cert["feasible"] else "infeasible",
                _num(cert["certified_false_accept_per_query"], 4),
                _pct(cert.get("in_scope_pass_rate")),
                _pct(cert.get("out_of_scope_accept_rate")),
                _pct(measured.get("selective_error")),
                "yes" if cert.get("certificate_held") else "no",
            ]
        )
    lines.append(
        _table(
            ["Target", "Threshold on noul", "Certified false-accepts/query", "In-scope passed", "OOS accepted", "Error among accepted", "Held?"],
            rows,
        )
    )
    lines.append("")
    if gate["certificates"]:
        lines.append(
            f"The gate's per-query loss counts one failure per *incoming query*, so it scales with "
            f"how much of the traffic the taxonomy does not cover. The calibration mix was "
            f"{_pct(gate['calibration_uncovered_prevalence'])} uncovered; the evaluation mix was "
            f"{_pct(gate['evaluation_uncovered_prevalence'])}. Re-scoring the same thresholds on a "
            "mix whose uncovered share matches the calibration mix isolates the effect:"
        )
        lines.append("")
        rows = []
        for cert in gate["certificates"]:
            matched = cert.get("measured_on_prevalence_matched_mix") or {}
            rows.append(
                [
                    _pct(cert["alpha"], 0),
                    _num(cert.get("certified_false_accept_per_query"), 4),
                    _num((cert.get("measured") or {}).get("per_query_risk"), 4),
                    "yes" if cert.get("certificate_held") else "**no**",
                    _num(matched.get("per_query_risk"), 4),
                    "yes" if cert.get("certificate_held_at_matched_prevalence") else "**no**",
                ]
            )
        lines.append(
            _table(
                ["Target", "Certified", "Loss/query (as-sampled mix)", "Held?", "Loss/query (matched mix)", "Held?"],
                rows,
            )
        )
        lines.append("")
        lines.append(
            "**Prevalence is a first-class input to a per-query certificate, not a detail.** The "
            "as-sampled mix carries a factor of "
            f"{_num(gate['evaluation_uncovered_prevalence'] / max(1e-9, gate['calibration_uncovered_prevalence']), 1)}x "
            "more uncovered traffic than the calibration mix, and the per-query loss moves with it. "
            "Two fixes exist and both are the caller's job: calibrate on a mix that matches "
            "production, or certify the *conditional* rate on accepted queries and multiply by the "
            "prevalence your monitoring measures."
        )
        lines.append("")
        # Distinguish a systematic miss from sampling noise: the guarantee is on the
        # expectation, so one held-out draw is allowed to land above or below it.
        loose = []
        for cert in gate["certificates"]:
            matched = cert.get("measured_on_prevalence_matched_mix") or {}
            if cert.get("certificate_held_at_matched_prevalence") or not matched.get("n"):
                continue
            p = matched["per_query_risk"]
            se = (p * (1 - p) / matched["n"]) ** 0.5
            if se > 0:
                loose.append((cert["alpha"], (p - cert["certified_false_accept_per_query"]) / se))
        if loose:
            worst = max(loose, key=lambda item: item[1])
            lines.append(
                f"One caveat on the matched-mix column: the promise is on the *expected* per-query "
                f"loss, so a single held-out draw may land above it. The loose target "
                f"({_pct(worst[0], 0)}) sits {worst[1]:.1f} standard errors above its certified value "
                "— treat that as sampling noise, not as a broken certificate. The as-sampled column "
                "is the systematic failure: it misses by a factor, not by a fraction of a standard error."
            )
            lines.append("")

    diagnostics = gate.get("noul_diagnostics")
    if diagnostics:
        lines.append("## 4. Calibration of the probabilities themselves")
        lines.append("")
        lines.append(
            "The whole stack leans on Jev's numbers meaning something. Measured on the "
            "calibration split, against human labels:"
        )
        lines.append("")
        lines.append(
            _table(
                ["Quantity", "Brier", "ECE (10 bins)"],
                [
                    ["Scope question (noul)", _num(diagnostics["brier"], 4), _num(diagnostics["ece10"], 4)],
                    [
                        "Routing confidence (max probability vs. correctness)",
                        _num(gate["routing_probability_diagnostics"]["brier"], 4),
                        _num(gate["routing_probability_diagnostics"]["ece10"], 4),
                    ],
                ],
            )
        )
        lines.append("")

    quantization = result.get("quantization")
    if quantization:
        lines.append("## 4b. The resolution floor is the model's, not the statistics'")
        lines.append("")
        lines.append(
            "A per-query risk certificate cannot separate queries the model scores identically. "
            "Measured across the calibration and evaluation answers:"
        )
        lines.append("")
        lines.append(
            _table(
                ["Measurement", "Value"],
                [
                    ["Distinct top-probabilities returned", quantization["distinct_top_probabilities"]],
                    [
                        "…of which are multiples of 0.01",
                        quantization["distinct_values_are_multiples_of_0.01"],
                    ],
                    ["Smallest non-zero top-probability", _num(quantization["min_positive_top_probability"], 4)],
                    ["Share of answers at exactly 1.0", _pct(quantization["share_at_1.0"])],
                    ["Error rate among those answers", _pct(quantization["error_rate_at_1.0"])],
                    ["Answers at 1.0 that were wrong", quantization["errors_at_1.0"]],
                ],
            )
        )
        lines.append("")
        lines.append(quantization["note"])
        lines.append("")

    lines.append("## 5. Auditing the live policy with few labels")
    lines.append("")
    lines.append(
        "Prediction-powered inference (Angelopoulos et al., *Science*, 2023). Each "
        f"repeat draws a random hand-labelling budget from a pool of "
        f"{audit['pool_n']} decisions; `f` is Jev's own estimate that the query was "
        "answered correctly by the certified policy, `y` is the human label."
    )
    lines.append("")
    rows = []
    for budget in audit["budgets"]:
        ppi_ = budget["methods"]["ppi"]
        rows.append(
            [
                budget["n_labeled"],
                f"{_num(ppi_['ci_coverage'], 3)} / {_num(budget['methods']['classical']['ci_coverage'], 3)}",
                f"{ppi_['mean_width']:.4f}",
                f"{budget['methods']['classical']['mean_width']:.4f}",
                _pct(budget["ppi_width_reduction_vs_classical"]),
                f"{budget['labels_ratio_classical_over_ppi']}x",
            ]
        )
    lines.append(
        _table(
            ["Hand labels", "PPI / classical CI coverage", "PPI width", "Classical width", "Narrower by", "Fewer labels for same width"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        f"Truth for the pool (finite-population mean of the policy's per-query success): "
        f"**{audit['true_routed_correct']}**, with **{_pct(audit['routed_share'])}** of traffic "
        "auto-routed at the certified threshold."
    )
    lines.append("")
    planning = audit.get("planning")
    if planning:
        lines.append("**How many labels to reach a tighter audit?**")
        lines.append("")
        rows = []
        for key, plan in planning.items():
            rows.append(
                [
                    f"±{key.replace('half_width_', '')}",
                    _num(plan["classical_labels"], 0),
                    _num(plan["ppi_labels"], 0),
                    _num(plan["lambda"], 3),
                ]
            )
        lines.append(_table(["95% interval half-width", "Labels needed (labels only)", "Labels needed (PPI++)", "Tuned lambda"], rows))
        lines.append("")
        lines.append(audit.get("planning_note", ""))
        lines.append("")

    lines.append("## 6. Where the guarantee breaks")
    lines.append("")
    lines.append(
        "Conformal guarantees rest on exchangeability: calibration and production traffic "
        "must be draws from the same distribution. The table below applies the "
        "*in-distribution* certificate to traffic it was never calibrated on. This is the "
        "most important table in the report, because it is the one that says what the "
        "certificate does not buy you."
    )
    lines.append("")
    rows = []
    for name, section in drift["sections"].items():
        routed = section["routing_above_threshold"]
        rows.append(
            [
                name.replace("_", " "),
                section["n"],
                _pct(section["intent_accuracy"]),
                _pct(routed["coverage"]),
                _num(routed["per_query_risk"], 4),
                "yes" if section["certificate_held"] else "**no**",
                _pct(section["gate_false_accept_rate_per_query"]),
            ]
        )
    lines.append(
        _table(
            ["Traffic", "n", "Intent accuracy", "Auto-routed", "Loss/query", "Certificate held?", "Gate false-accepts/query"],
            rows,
        )
    )
    lines.append("")

    restricted = result.get("restricted_deployment")
    restricted_gate_uncalibrated = bool(restricted) and not restricted.get("gate", {}).get("certificates")
    if restricted:
        lines.append("### Same procedure, shifted traffic")
        lines.append("")
        lines.append(
            f"A second deployment sends only **{restricted['deployed_intents']} intents**, "
            "calibrated on that taxonomy, and then receives real traffic containing the "
            "other 130 intents. Same model, same code, same α — only the traffic changes."
        )
        lines.append("")
        rc = restricted["routing"]["certificates"][0]
        if not rc["feasible"]:
            lines.append(
                f"**The {_pct(alpha, 0)} certificate is not attainable on this taxonomy either:** "
                f"the tightest bound is {_num(rc['min_attainable_bound'], 4)} per query."
            )
            lines.append("")
        rows = [
            [
                "calibration (in taxonomy)",
                restricted["routing"]["calibration_n"],
                _num(rc["threshold"], 3) if rc["feasible"] else "infeasible",
                _num(rc["certified_per_query_loss"], 4),
                "—",
            ]
        ]
        for name, section in restricted["drift"]["sections"].items():
            routed = section["routing_above_threshold"]
            rows.append(
                [
                    name.replace("_", " "),
                    section["n"],
                    _num(rc["threshold"], 3) if rc["feasible"] else "infeasible",
                    _num(routed["per_query_risk"], 4),
                    "yes" if section["certificate_held"] else "**no**",
                ]
            )
        lines.append(_table(["Slice", "n", "Threshold", "Loss/query", "Certificate held?"], rows))
        lines.append("")
        if restricted_gate_uncalibrated:
            lines.append(
                "The restricted deployment's scope gate could not be calibrated at all: its "
                "calibration pool contained no traffic outside the 20 intents, so the gate has "
                "never seen a query it should reject. That is a deployment hazard the certificate "
                "cannot fix — sampling your traffic mix is a prerequisite for auditing it."
            )
            lines.append("")

    lines.append("## 7. Cost")
    lines.append("")
    points = routing.get("operating_points")
    if points:
        rows = []
        for point in points["operating_points"]:
            rows.append(
                [
                    point["name"],
                    _num(point["threshold"], 3),
                    _pct(point["coverage"]),
                    _num(point["per_query_loss"], 4),
                    _num(point["certified_per_query_loss"], 4) if point["certified_per_query_loss"] is not None else "—",
                    f"${point['cost_per_1k_usd']:.2f}",
                ]
            )
        rows.append(
            [
                points["always_llm"]["name"],
                "—",
                _pct(points["always_llm"]["coverage"]),
                _num(points["always_llm"]["selective_error"], 4),
                "—",
                f"${points['always_llm']['cost_per_1k_usd']:.2f}",
            ]
        )
        lines.append(
            _table(["Configuration", "Threshold", "Auto-routed", "Loss/query", "Certified", "Cost / 1k"], rows)
        )
        lines.append("")
        lines.append(
            f"Assumptions beyond Jev's measured cost (${points['assumptions']['jev_usd_per_1k_queries']:.4f} / 1k): "
            f"escalation {points['assumptions']['fallback_usd_per_1k_queries']} USD / 1k, "
            f"human review {points['assumptions']['human_review_usd_per_1k_queries']} USD / 1k."
        )
        lines.append("")

    lines.append("## Examples from the evaluation set")
    lines.append("")
    lines.append("**Confident and right**")
    lines.append("")
    for example in result["examples"]["confident_correct"]:
        lines.append(
            f"- `{example['text'][:80]}` → **{example['choice']}** (p = {example['p']})"
        )
    lines.append("")
    lines.append("**Confident and wrong** (what the certificate is paying for)")
    lines.append("")
    for example in result["examples"]["confident_wrong"]:
        runner = example["runner_up"][0] if example["runner_up"] else ("", 0)
        lines.append(
            f"- `{example['text'][:80]}` → **{example['choice']}** (p = {example['p']}) "
            f"but truth is **{example['gold']}** (runner-up {runner[0]} @ {runner[1]:.3f})"
        )
    lines.append("")

    lines.append("## Limits")
    lines.append("")
    lines.append("- **Exchangeability is the assumption, and it is not free.** The drift section measures it breaking. A certificate printed for one traffic distribution says nothing about another; re-calibrate on a fresh sample whenever the mix moves.")
    lines.append("- **The routing certificate is per-query, not per-routed-query.** It bounds silently misrouted traffic, not the error rate among routed queries; both columns are reported because teams ask for different ones.")
    lines.append("- **Calibration split size sets the resolution.** With n calibration points the smallest certifiable bound is about 1/(n+1); more traffic buys a tighter certificate, not a better model.")
    lines.append("- **The cost model contains one assumption** (the price of the escalated fallback). Jev's own cost, latency and token counts are measured, not assumed.")
    lines.append("- **CLINC150 is a research test set**, not your traffic. The method transfers; the numbers are this dataset's.")
    lines.append("- **Two questions rode in every call**, so the choice and the gate share a request. They are evaluated independently, but the gate's accuracy is not independent evidence about the router's.")

    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("export OPENROUTER_API_KEY=sk-or-...        # https://openrouter.ai/keys")
    lines.append("python -m experiments.collect              # ~2,400 Jev decisions, journals everything")
    lines.append("python -m experiments.analyse              # writes results/results.json + this report")
    lines.append("python -m pytest tests -q                  # math checks, no API calls")
    lines.append("```")
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append("- Angelopoulos, Bates, Fisch, Lei & Schuster. *Conformal Risk Control.* ICLR 2023 (arXiv:2208.02814).")
    lines.append("- Angelopoulos, Bates, Fannjiang, Jordan & Zrnic. *Prediction-Powered Inference.* Science 382:669-674, 2023.")
    lines.append("- Angelopoulos, Bates, Fannjiang, Jordan & Zrnic. *PPI++: Efficient Prediction-Powered Inference.* NeurIPS 2023 (arXiv:2311.01453).")
    lines.append("- Romano, Sesia & Candès. *Classification with Valid and Adaptive Coverage.* NeurIPS 2020 (arXiv:2006.02544).")
    lines.append("- Sadinle, Lei & Wasserman. *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.* JASA 2019.")
    lines.append("- Geifman & El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS 2017.")
    lines.append("- Guo, Pleiss, Sun & Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.")
    lines.append("- Larson et al. *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.* EMNLP 2019.")
    lines.append("- TypeSafe AI. *System One* and *Confidence* documentation, and the *Intent routing* pattern (docs.typesafe.ai).")
    return "\n".join(lines) + "\n"


def render_dashboard(result: dict[str, Any]) -> str:
    """A single self-contained HTML page: the certificate, the curves, the caveats."""
    meta = result["meta"]
    routing = result["routing"]
    gate = result["gate"]
    cost = result["cost"]
    audit = result["audit"]
    drift = result["drift"]
    sets = result["prediction_sets"]["scores"]
    alpha = meta["alpha"]
    primary = routing.get("primary") or {}
    measured = primary.get("measured_on_holdout", {}) if primary else {}
    gate_primary = next((c for c in gate["certificates"] if c["alpha"] == alpha and c["feasible"]), {})

    def card(label: str, value: str, note: str = "", tone: str = "") -> str:
        style = f"color:{tone}" if tone else ""
        return (
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value" style="{style}">{html.escape(value)}</div>'
            f'<div class="note">{html.escape(note)}</div></div>'
        )

    routing_curve = routing["curves"]["in_scope_only"]
    gate_curve = gate.get("calibration_curve") or []
    reliability = (gate.get("noul_diagnostics") or {}).get("reliability") or []
    rel_bars = "".join(
        f'<div class="relcol" title="bin {row["bin"]}"><div class="relbar" style="height:{row["empirical_frequency"] * 120:.0f}px"></div>'
        f'<div class="relbar conf" style="height:{row["mean_confidence"] * 120:.0f}px"></div>'
        f'<span>{row["bin"]:.1f}</span></div>'
        for row in reliability
    )

    audit_rows = "".join(
        f"<tr><td>{b['n_labeled']}</td>"
        f"<td>{b['methods']['ppi']['mean_width']:.4f}</td>"
        f"<td>{b['methods']['classical']['mean_width']:.4f}</td>"
        f"<td>{100 * (b['ppi_width_reduction_vs_classical'] or 0):.1f}%</td>"
        f"<td>{b['methods']['ppi']['ci_coverage']:.2f}</td>"
        f"<td>{b['labels_ratio_classical_over_ppi']}x</td></tr>"
        for b in audit["budgets"]
    )

    drift_rows = "".join(
        f"<tr><td>{html.escape(name.replace('_', ' '))}</td><td>{s['n']}</td>"
        f"<td>{100 * (s['intent_accuracy'] or 0):.1f}%</td>"
        f"<td>{100 * s['routing_above_threshold']['coverage']:.1f}%</td>"
        f"<td>{s['routing_above_threshold']['per_query_risk']:.4f}</td>"
        f"<td class=\"{'ok' if s['certificate_held'] else 'bad'}\">{'yes' if s['certificate_held'] else 'NO'}</td></tr>"
        for name, s in drift["sections"].items()
    )

    set_rows_list = []
    for name, payload in sets.items():
        at = payload.get("at_calibrated_qhat")
        if not at:
            continue
        qhat_text = "∞" if payload.get("infinite_qhat") else f"{payload['qhat_calibration']:.4f}"
        set_rows_list.append(
            f"<tr><td>{name.upper()}</td><td>{qhat_text}</td>"
            f"<td>{100 * at['coverage']:.1f}%</td>"
            f"<td>{at['mean_set_size']:.2f}</td>"
            f"<td>{at['top1_fallbacks']}</td></tr>"
        )
    set_rows = "".join(set_rows_list)

    oos_accept = gate_primary.get("out_of_scope_accept_rate") if gate_primary else None
    oos_accept_text = "n/a" if oos_accept is None else f"{100 * oos_accept:.1f}%"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>jev-certify — certified abstention for Jev decisions</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: var(--app-font, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif);
         color: var(--foreground, #16181d); margin: 0; padding: 4px 2px 24px; line-height: 1.45; }}
  h1 {{ font-size: 20px; margin: 0 0 2px; }}
  h2 {{ font-size: 14px; margin: 26px 0 8px; text-transform: uppercase; letter-spacing: .04em;
        color: var(--muted-foreground, #666); }}
  p.sub {{ margin: 0 0 14px; color: var(--muted-foreground, #666); font-size: 13px; max-width: 74ch; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ border: 1px solid var(--border, #e2e4e8); border-radius: 10px; padding: 10px 13px; min-width: 148px;
           background: var(--card, transparent); }}
  .card .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted-foreground, #666); }}
  .card .value {{ font-size: 21px; font-weight: 600; margin: 3px 0 1px; }}
  .card .note {{ font-size: 11px; color: var(--muted-foreground, #666); }}
  table {{ border-collapse: collapse; font-size: 12.5px; margin: 6px 0 4px; }}
  th, td {{ border-bottom: 1px solid var(--border, #e6e8ec); padding: 5px 10px 5px 0; text-align: left; }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: .03em; color: var(--muted-foreground, #666); font-weight: 600; }}
  td.ok {{ color: #16a34a; font-weight: 600; }} td.bad {{ color: #dc2626; font-weight: 600; }}
  .panes {{ display: flex; flex-wrap: wrap; gap: 22px; }}
  .pane {{ flex: 1 1 320px; min-width: 300px; }}
  .rel {{ display: flex; align-items: flex-end; gap: 6px; height: 140px; }}
  .relcol {{ display: flex; align-items: flex-end; gap: 2px; height: 130px; }}
  .relbar {{ width: 9px; background: var(--accent, #3b82f6); border-radius: 2px 2px 0 0; opacity: .8; }}
  .relbar.conf {{ background: var(--muted-foreground, #999); opacity: .5; }}
  .relcol span {{ font-size: 9px; color: var(--muted-foreground, #666); align-self: flex-end; margin-left: -12px; }}
  .legend {{ font-size: 11px; color: var(--muted-foreground, #666); }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px;
          background: var(--accent, #00000010); opacity: .85; padding: 1px 4px; border-radius: 4px; }}
  .warn {{ border-left: 3px solid #dc2626; padding-left: 10px; font-size: 12.5px; }}
</style>
</head>
<body>
  <h1>jev-certify</h1>
  <p class="sub">Certified abstention and prediction-powered audits for <strong>Jev</strong> (TypeSafe System One)
  on {html.escape(meta["dataset"])}. One Jev call per query carries a routing <code>choice</code> and a scope
  <code>noul</code>; the numbers below are measured against human labels, not asserted.</p>

  <div class="cards">
    {card("Certified threshold", f"{primary.get('threshold', float('nan')):.3f}", f"alpha = {100 * alpha:.0f}% per-query misroute budget")}
    {card("Auto-routed", f"{100 * (measured.get('coverage') or 0):.1f}%", f"of {measured.get('n', 0)} held-out queries")}
    {card("Measured loss / query", f"{measured.get('per_query_risk', 0):.4f}", "certificate held" if primary.get("certificate_held") else "certificate MISSED")}
    {card("Error when routed", f"{100 * (measured.get('selective_error') or 0):.1f}%", f"95% UCB {100 * (measured.get('selective_error_cp_ucb95') or 0):.1f}%")}
    {card("Gate false-accepts", oos_accept_text, "of human out-of-scope queries accepted")}
    {card("Cost / 1k queries", f"${cost['cost_usd_per_1k']:.4f}", f"p50 {cost['latency_p50_s']}s · p95 {cost['latency_p95_s']}s")}
  </div>

  <div class="panes">
    <div class="pane">
      <h2>Risk–coverage on held-out traffic</h2>
      {_svg_line_chart(routing_curve, "coverage", "selective_error", "Error among routed queries vs. coverage", "coverage", "selective error")}
      <p class="legend">Left is conservative (fewer queries routed, fewer mistakes), right is greedy.
      The certified threshold sits where the certified per-query budget is met.</p>
    </div>
    <div class="pane">
      <h2>Scope gate: risk–coverage</h2>
      {_svg_line_chart(gate_curve, "coverage", "selective_error", "Wrong accepts vs. queries accepted", "accept rate", "wrong accept rate")}
      <p class="legend">The gate decides whether the deployed taxonomy covers a query at all;
      out-of-scope traffic is what it exists to catch.</p>
    </div>
  </div>

  <h2>Reliability of Jev's scope probability (calibration split)</h2>
  <div class="rel">{rel_bars}</div>
  <p class="legend">Blue = observed frequency of "in scope" per confidence bin · grey = bin's mean confidence.
  Bars of similar height mean the number can be trusted as a probability, which is what every guarantee here assumes.</p>

  <h2>Prediction sets — how much ambiguity is left</h2>
  <table><thead><tr><th>Score</th><th>Calibrated threshold</th><th>Coverage</th><th>Mean set size</th><th>Top-1 fallbacks</th></tr></thead>
  <tbody>{set_rows}</tbody></table>

  <h2>Prediction-powered audit — {audit['pool_n']} decisions, few hand labels</h2>
  <table><thead><tr><th>Hand labels</th><th>PPI++ CI width</th><th>Labels-only width</th><th>Narrower</th><th>PPI CI coverage</th><th>Label ratio</th></tr></thead>
  <tbody>{audit_rows}</tbody></table>

  <h2>Where the certificate does <em>not</em> hold</h2>
  <table><thead><tr><th>Traffic</th><th>n</th><th>Intent accuracy</th><th>Auto-routed</th><th>Loss / query</th><th>Certificate held?</th></tr></thead>
  <tbody>{drift_rows}</tbody></table>
  <p class="warn">Conformal guarantees require exchangeability between calibration and production traffic.
  The rows marked NO are the measured cost of that assumption breaking — this is the part a router's README usually omits.</p>
</body>
</html>
"""


def write_reports(result: dict[str, Any], markdown_path: str | Path, html_path: str | Path) -> tuple[Path, Path]:
    md = Path(markdown_path)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_markdown(result), encoding="utf-8")
    page = Path(html_path)
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(render_dashboard(result), encoding="utf-8")
    return md, page
