"""Analysis entry point: journal + pools -> results.json + REPORT.md + dashboard.html.

    python -m experiments.analyse
    python -m experiments.analyse --alpha 0.02 --no-restricted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_certify.analysis import full_analysis, load_scenario, write_results  # noqa: E402
from jev_certify.report import write_reports  # noqa: E402
from jev_certify.viz import write_charts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--report-dir", default=str(ROOT / "report"))
    parser.add_argument("--charts-dir", default=str(ROOT / "docs"), help="where the README figures go")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-restricted", action="store_true")
    parser.add_argument("--repeats", type=int, default=500, help="Monte-Carlo repeats for the audit")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    journal = results_dir / "journal.jsonl"
    full_plan = results_dir / f"plan-{args.tag}-full.json"
    restricted_plan = results_dir / f"plan-{args.tag}-restricted.json"

    print(f"loading journal {journal} ({sum(1 for _ in journal.open(encoding='utf-8'))} decisions)")
    full = load_scenario(full_plan, journal)
    restricted = None
    if not args.no_restricted and restricted_plan.is_file():
        restricted = load_scenario(restricted_plan, journal)

    result = full_analysis(full, restricted, alpha=args.alpha, audit_repeats=args.repeats)

    out = write_results(result, results_dir / "results.json")
    md, page = write_reports(result, results_dir / "REPORT.md", Path(args.report_dir) / "dashboard.html")
    charts = write_charts(result, args.charts_dir)
    print(json.dumps({k: v for k, v in result["meta"].items()}, indent=2))
    print(f"wrote {out}")
    print(f"wrote {md}")
    print(f"wrote {page}")
    for chart in charts:
        print(f"wrote {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
