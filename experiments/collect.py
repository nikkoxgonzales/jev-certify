"""Data collection: spend real Jev calls and journal them.

This is the only script in the project that costs money. It runs two deployment
scenarios against CLINC150's human labels and writes every request/response pair to
``results/journal.jsonl`` plus the sampled pools to ``results/plan.json``.

    python -m experiments.collect                 # full run (~2,000 decisions)
    python -m experiments.collect --dry-run       # print request sizes only

Everything downstream (``experiments/analyse.py``) reads only those two files, so the
analysis can be changed and re-run for free -- which is also why the journal is
content-addressed: re-running this script after a crash costs nothing for the rows
already answered.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_certify.client import JevClient  # noqa: E402
from jev_certify.tasks import (  # noqa: E402
    build_scenario,
    load_clinc150,
    most_common_intents,
    requests_for,
    write_plan,
)

FULL_INTENTS_ARG = "full"


def plan_requests(plan: dict, key: str, deployed: list[str]) -> list[tuple[dict, dict]]:
    return requests_for(plan[key], deployed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="typesafe/jev-1.13")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retry-passes", type=int, default=3, help="re-attempt failed requests (journalled rows are free)")
    parser.add_argument("--journal", default=str(ROOT / "results" / "journal.jsonl"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--n-calibration", type=int, default=400)
    parser.add_argument("--n-evaluation", type=int, default=400)
    parser.add_argument("--n-oos-evaluation", type=int, default=300)
    parser.add_argument("--n-shift-evaluation", type=int, default=500)
    parser.add_argument("--n-lexicon", type=int, default=20, help="intents in the restricted deployment")
    parser.add_argument("--tag", default="v1", help="run tag, part of the journal filename key")
    parser.add_argument("--dry-run", action="store_true", help="print request sizes, spend nothing")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_clinc150()
    all_intents = most_common_intents(data, k=10_000)  # all 150, deterministic order
    full_intents = sorted({label for _, label in data["test"] if label != "oos"})
    restricted_intents = most_common_intents(data, k=args.n_lexicon)

    full_plan = build_scenario(
        data,
        deployed_intents=full_intents,
        n_calibration=args.n_calibration,
        n_evaluation=args.n_evaluation,
        n_oos_evaluation=args.n_oos_evaluation,
        seed=20260922,
        tag=f"{args.tag}-full",
    )
    restricted_plan = build_scenario(
        data,
        deployed_intents=restricted_intents,
        n_calibration=args.n_calibration,
        n_evaluation=0,
        n_oos_evaluation=0,
        n_oos_calibration=0,
        n_shift_evaluation=args.n_shift_evaluation,
        n_in_taxonomy_evaluation=args.n_evaluation,
        seed=20260922,
        tag=f"{args.tag}-restricted",
    )

    jobs: list[tuple[str, dict, dict]] = []
    for plan in (full_plan, restricted_plan):
        deployed = plan["deployed_intents"]
        for key in ("calibration", "evaluation", "oos_evaluation", "shift_evaluation", "in_taxonomy_evaluation"):
            rows = plan.get(key) or []
            for (state, questions), row in zip(requests_for(rows, deployed), rows):
                jobs.append((row.qid, state, questions))

    print(f"deployed intents: full={len(full_intents)} restricted={len(restricted_intents)}")
    print(f"planned decisions: {len(jobs)}")

    if args.dry_run:
        import hashlib

        state, questions = jobs[0][1], jobs[0][2]
        payload = json.dumps({"model": args.model, "state": state, "questions": questions})
        print(f"example payload bytes: {len(payload)}")
        print(f"example question ids: {list(questions)}")
        print(f"example criteria count: {len(questions['intent']['criteria'])}")
        print(f"rough input tokens/request: {len(payload) // 4}")
        print(f"rough input tokens total: {len(jobs) * len(payload) // 4}")
        print(f"rough cost at $0.042/M: ${len(jobs) * len(payload) / 4 * 0.042 / 1e6:.4f}")
        print(f"payload sha256[:12]: {hashlib.sha256(payload.encode()).hexdigest()[:12]}")
        write_plan(full_plan, results_dir / f"plan-{args.tag}-full.json")
        write_plan(restricted_plan, results_dir / f"plan-{args.tag}-restricted.json")
        return 0

    client = JevClient(model=args.model, journal=args.journal)
    preloaded = client.warm_from_journal()
    print(f"journal already holds {preloaded} decisions; new calls will pay only for misses")

    started = time.perf_counter()
    last = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.perf_counter()
        if now - last[0] > 5 or done == total:
            last[0] = now
            print(f"  {done}/{total} decisions  ({now - started:.0f}s)", flush=True)

    batches: list[tuple[str, str, list[tuple[dict, dict]]]] = []
    for name, plan in ((("full", full_plan)), ("restricted", restricted_plan)):
        deployed = plan["deployed_intents"]
        for key in ("calibration", "evaluation", "oos_evaluation", "shift_evaluation", "in_taxonomy_evaluation"):
            rows = plan.get(key) or []
            if rows:
                batches.append((name, key, requests_for(rows, deployed)))

    all_errors: list[str] = []
    for attempt in range(1, args.retry_passes + 1):
        errors_this_pass = 0
        for name, key, items in batches:
            print(f"[pass {attempt}][{name}/{key}] {len(items)} decisions", flush=True)
            _, errors = client.decide_many(items, workers=args.workers, progress=progress)
            if errors:
                errors_this_pass += len(errors)
                all_errors.extend(msg for _, msg in errors)
                print(f"  !! {len(errors)} request(s) failed, will retry next pass", flush=True)
                for _, msg in errors[:3]:
                    print(f"     {msg[:160]}", flush=True)
        if errors_this_pass == 0:
            break
        print(f"pass {attempt} finished with {errors_this_pass} failures", flush=True)

    write_plan(full_plan, results_dir / f"plan-{args.tag}-full.json")
    write_plan(restricted_plan, results_dir / f"plan-{args.tag}-restricted.json")
    stats = client.stats()
    (results_dir / "collection_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"journal: {args.journal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
