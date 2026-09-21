"""Command line interface.

    jev-certify status                       # what is in the journal, what it cost
    jev-certify certify --alpha 0.05         # fit the certificate and print it
    jev-certify audit --labels 100           # prediction-powered audit at a label budget
    jev-certify report                       # write results/REPORT.md + report/dashboard.html

None of these subcommands call the API. Collection is a separate, explicit step
(``python -m experiments.collect``) so that no CLI invocation can quietly spend money.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import (
    audit_analysis,
    facts_for,
    full_analysis,
    gate_analysis,
    load_scenario,
    routing_certification,
)
from .cascade import CostAssumptions
from .client import DecisionJournal
from .report import write_reports
from .viz import write_charts

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results"


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    results = Path(args.results_dir)
    return (
        results / "journal.jsonl",
        results / f"plan-{args.tag}-full.json",
        results / f"plan-{args.tag}-restricted.json",
    )


def cmd_status(args: argparse.Namespace) -> int:
    journal_file, _, _ = _paths(args)
    if not journal_file.is_file():
        print(f"no journal at {journal_file}; run: python -m experiments.collect", file=sys.stderr)
        return 2
    journal = DecisionJournal(journal_file)
    records = journal.load()
    latencies = sorted(r.get("latency_s", 0.0) for r in records.values())
    costs = [float(r.get("usage", {}).get("cost", 0.0) or 0.0) for r in records.values()]
    tokens_in = sum(int(r.get("usage", {}).get("input_tokens", 0) or 0) for r in records.values())

    def pct(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))] if latencies else 0.0

    print(f"journal        : {journal_file}")
    print(f"decisions      : {len(records)}")
    print(f"input tokens   : {tokens_in:,}")
    print(f"total cost     : ${sum(costs):.6f}  (${sum(costs) / max(1, len(costs)) * 1000:.4f} per 1k)")
    print(f"latency p50/p95: {pct(0.5):.3f}s / {pct(0.95):.3f}s")
    print(f"model          : {next(iter(records.values()))['model'] if records else 'n/a'}")
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    journal_file, full_plan, _ = _paths(args)
    scenario = load_scenario(full_plan, journal_file)
    cal = facts_for(scenario.calibration, scenario.decisions)
    ev = facts_for(scenario.evaluation, scenario.decisions)
    routing = routing_certification(cal, ev, alphas=(args.alpha,))
    gate = gate_analysis(cal, ev, alphas=(args.alpha,))

    if args.json:
        print(json.dumps({"routing": routing, "gate": gate}, indent=2))
        return 0

    cert = routing["certificates"][0]
    print(f"Routing certificate (alpha = {args.alpha:.2%} per-query misroute budget)")
    if not cert["feasible"]:
        print("  infeasible: collect more calibration queries (1/(n+1) sets the floor)")
    else:
        measured = cert["measured_on_holdout"]
        print(f"  threshold                 : {cert['threshold']:.3f}")
        print(f"  certified loss / query    : {cert['certified_per_query_loss']:.4f}")
        print(f"  coverage on holdout       : {measured['coverage']:.2%} of {measured['n']} queries")
        print(f"  error when routed         : {measured['selective_error']:.2%} (95% UCB {measured['selective_error_cp_ucb95']:.2%})")
        print(f"  measured loss / query     : {measured['per_query_risk']:.4f}")
        print(f"  certificate held          : {cert['certificate_held']}")

    gate_cert = gate["certificates"][0]
    print(f"\nScope gate (alpha = {args.alpha:.2%} false-accept budget)")
    if not gate_cert["feasible"]:
        print("  infeasible")
    else:
        print(f"  threshold on noul         : {gate_cert['threshold']:.3f}")
        print(f"  certified false-accepts   : {gate_cert['certified_false_accept_per_query']:.4f} per query")
        print(f"  in-scope passed           : {gate_cert['in_scope_pass_rate']:.2%}")
        print(f"  out-of-scope accepted     : {gate_cert['out_of_scope_accept_rate']:.2%}")
        print(f"  certificate held          : {gate_cert['certificate_held']}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    journal_file, full_plan, _ = _paths(args)
    scenario = load_scenario(full_plan, journal_file)
    cal = facts_for(scenario.calibration, scenario.decisions)
    ev = facts_for(scenario.evaluation, scenario.decisions)
    routing = routing_certification(cal, ev, alphas=(args.alpha,))
    threshold = float(routing["primary"]["threshold"]) if routing.get("primary") else 1.0
    pool = [
        r
        for r in ev + facts_for(scenario.in_taxonomy_evaluation, scenario.decisions)
        if r["covered_by_taxonomy"]
    ]
    audit = audit_analysis(pool, threshold, label_budgets=(args.labels,), repeats=args.repeats)
    print(f"Auditing {audit['pool_n']} decisions at threshold {threshold:.3f}")
    print(f"  true share answered correctly: {audit['true_routed_correct']:.4f}")
    for entry in audit["budgets"]:
        for name in ("classical", "imputation", "ppi"):
            method = entry["methods"][name]
            print(
                f"  {name:11s} n={entry['n_labeled']:4d}  estimate={method['mean_estimate']:.4f}"
                f"  width={method['mean_width']:.4f}  covered={method['ci_coverage']:.3f}"
            )
        print(f"  -> PPI is {100 * (audit['budgets'][0]['ppi_width_reduction_vs_classical'] or 0):.1f}% narrower")
    if args.json:
        print(json.dumps(audit, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    journal_file, full_plan, restricted_plan = _paths(args)
    scenario = load_scenario(full_plan, journal_file)
    restricted = load_scenario(restricted_plan, journal_file) if restricted_plan.is_file() else None
    result = full_analysis(scenario, restricted, alpha=args.alpha)
    md, page = write_reports(
        result, Path(args.results_dir) / "REPORT.md", Path(args.report_dir) / "dashboard.html"
    )
    (Path(args.results_dir) / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    charts = write_charts(result, Path(args.charts_dir))
    print(f"wrote {md}\nwrote {page}")
    for chart in charts:
        print(f"wrote {chart}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-certify", description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--report-dir", default=str(ROOT / "report"))
    parser.add_argument("--charts-dir", default=str(ROOT / "docs"))
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--alpha", type=float, default=0.05)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="journal contents, cost and latency")
    status.set_defaults(func=cmd_status)

    certify = sub.add_parser("certify", help="fit and print the certified thresholds")
    certify.add_argument("--json", action="store_true")
    certify.set_defaults(func=cmd_certify)

    audit = sub.add_parser("audit", help="prediction-powered audit at a label budget")
    audit.add_argument("--labels", type=int, default=100)
    audit.add_argument("--repeats", type=int, default=300)
    audit.add_argument("--json", action="store_true")
    audit.set_defaults(func=cmd_audit)

    report = sub.add_parser("report", help="write REPORT.md and the HTML dashboard")
    report.set_defaults(func=cmd_report)

    pack = sub.add_parser(
        "pack-journal",
        help="write a slim journal (answers kept, request payloads dropped) for committing",
    )
    pack.set_defaults(func=cmd_pack_journal)
    return parser


def cmd_pack_journal(args: argparse.Namespace) -> int:
    """Write a slim journal: every recorded answer, none of the request payloads.

    The request bodies are the bulk of the file but they are not evidence -- they are
    reconstructed deterministically from the plan by ``jev_certify.tasks``, and
    ``load_scenario`` checks each reconstruction against the journal's content hash before
    using the stored answer. So dropping them shrinks the artifact that gets committed
    without weakening what it proves.
    """
    results = Path(args.results_dir)
    source = results / "journal.jsonl"
    target = results / "journal.slim.jsonl"
    if not source.is_file():
        print(f"no journal at {source}", file=sys.stderr)
        return 1
    keep = ("key", "model", "answers", "usage", "latency_s", "fetched_at")
    written = 0
    with source.open("r", encoding="utf-8") as reader, target.open("w", encoding="utf-8") as writer:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            writer.write(json.dumps({k: row[k] for k in keep if k in row}, separators=(",", ":")) + "\n")
            written += 1
    before, after = source.stat().st_size / 1e6, target.stat().st_size / 1e6
    print(f"wrote {target}")
    print(f"{written} records, {before:.1f} MB -> {after:.1f} MB")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
