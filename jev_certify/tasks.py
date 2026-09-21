"""Tasks: real text-classification data with deterministic ground truth.

Why CLINC150 (Larson et al., EMNLP 2019, "An Evaluation Dataset for Intent
Classification and Out-of-Scope Prediction"):

* 150 intents, 4,500 held-out in-scope queries and **1,000 explicitly out-of-scope
  queries written by humans** -- the OOS half is what makes it a routing/abstention
  benchmark rather than a plain classifier benchmark.
* Ground truth is human, not synthesised by a model, so every number the certification
  reports is measured against something independent of Jev.
* The dataset's own headline result is about *knowing when you don't know*, which is
  exactly the question a certified decision layer has to answer.

The task built on it is a **certified semantic router**:

    state    = {"query": <user text>, "deployed_intents": [...]}
    question = intent    (choice over the deployed intents)
    question = in_scope  (noul: does the deployed taxonomy cover this query?)

Two scenarios ride on the same data:

``full``
    All 150 intents deployed. Calibration and evaluation are an exchangeable random
    split of the human test set -- the setting where the conformal guarantee holds.

``restricted``
    Only 20 intents deployed, but real traffic still contains the other 130 (plus the
    human OOS set). This is the deployment that actually happens: a small taxonomy goes
    live and the world keeps sending everything. It is the distribution shift that
    breaks the guarantee, measured rather than asserted.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "clinc150_full.json"


def humanize(intent: str) -> str:
    """``card_arrival`` -> ``card arrival``: the label text Jev sees in ``criteria``."""
    return intent.replace("_", " ").strip()


@dataclass
class Query:
    """One row of the router's input stream."""

    qid: str
    text: str
    gold_intent: str | None  # None for out-of-scope queries
    covered_by_taxonomy: bool
    scenario: str
    role: str  # "calibration" | "evaluation"

    @property
    def is_oos(self) -> bool:
        return self.gold_intent is None


def load_clinc150(path: str | Path | None = None) -> dict[str, list[list[str]]]:
    """Load the human train/val/test + oos splits of CLINC150."""
    file = Path(path) if path else DATA_FILE
    if not file.is_file():
        raise FileNotFoundError(
            f"{file} missing. Fetch it once with:\n"
            "  curl -sL -o data/clinc150_full.json "
            "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
        )
    with file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def intents_of(rows: Sequence[Sequence[str]]) -> list[str]:
    return sorted({label for _, label in rows if label != "oos"})


def build_scenario(
    data: dict[str, list[list[str]]],
    deployed_intents: Sequence[str],
    n_calibration: int = 400,
    n_evaluation: int = 400,
    n_oos_calibration: int = 60,
    n_oos_evaluation: int = 300,
    n_shift_evaluation: int = 500,
    n_in_taxonomy_evaluation: int = 150,
    seed: int = 20260922,
    tag: str = "full",
) -> dict[str, Any]:
    """Assemble the calibration/evaluation pools for one deployment scenario.

    Every pool is drawn with a fixed seed and is exactly reproducible. In-scope rows
    come from the human test split (calibration and evaluation are disjoint); OOS rows
    come from the human OOS test split.
    """
    rng = random.Random(seed)
    deployed = set(deployed_intents)

    in_scope_pool = [(text, label) for text, label in data["test"] if label in deployed]
    outside_pool = [(text, label) for text, label in data["test"] if label not in deployed]
    oos_pool = [(text, "oos") for text, _ in data["oos_test"]]

    rng.shuffle(in_scope_pool)
    rng.shuffle(outside_pool)
    rng.shuffle(oos_pool)

    def take(pool: list[tuple[str, str]], count: int) -> list[tuple[str, str]]:
        return pool[:count]

    calibration_rows = take(in_scope_pool, n_calibration)
    remainder = in_scope_pool[n_calibration:]
    evaluation_rows = take(remainder, n_evaluation)
    remainder = remainder[n_evaluation:]
    shift_rows = take(outside_pool, n_shift_evaluation)
    # A small in-taxonomy slice drawn from the same remainder, so a restricted
    # deployment can be compared against itself on home turf (guarantee holds) and on
    # shifted traffic (guarantee does not). Drawn last, so it never overlaps the
    # evaluation pool.
    in_taxonomy_rows = take(remainder, n_in_taxonomy_evaluation)
    oos_cal_rows = take(oos_pool, n_oos_calibration)
    oos_eval_rows = take(oos_pool[n_oos_calibration:], n_oos_evaluation)

    def row(text: str, label: str, role: str, index: int, kind: str) -> Query:
        covered = label in deployed
        return Query(
            qid=f"{tag}-{kind}-{index:04d}",
            text=text,
            gold_intent=None if label == "oos" else label,
            covered_by_taxonomy=covered,
            scenario=tag,
            role=role,
        )

    calibration = [row(t, l, "calibration", i, "cal") for i, (t, l) in enumerate(calibration_rows)]
    evaluation = [row(t, l, "evaluation", i, "eval") for i, (t, l) in enumerate(evaluation_rows)]
    # Out-of-scope rows join the calibration set too: a gate that has never seen a
    # "no" example cannot be calibrated for one.
    calibration += [row(t, l, "calibration", i, "oocal") for i, (t, l) in enumerate(oos_cal_rows)]
    oos_evaluation = [row(t, l, "evaluation", i, "ooseval") for i, (t, l) in enumerate(oos_eval_rows)]
    shift_evaluation = [row(t, l, "evaluation", i, "shift") for i, (t, l) in enumerate(shift_rows)]
    in_taxonomy_evaluation = [row(t, l, "evaluation", i, "intax") for i, (t, l) in enumerate(in_taxonomy_rows)]
    return {
        "tag": tag,
        "deployed_intents": list(deployed_intents),
        "calibration": calibration,
        "evaluation": evaluation,
        "oos_evaluation": oos_evaluation,
        "shift_evaluation": shift_evaluation,
        "in_taxonomy_evaluation": in_taxonomy_evaluation,
        "counts": {
            "deployed_intents": len(deployed_intents),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
            "oos_evaluation": len(oos_evaluation),
            "shift_evaluation": len(shift_evaluation),
        },
    }


def most_common_intents(data: dict[str, list[list[str]]], k: int = 20, split: str = "test") -> list[str]:
    counts: dict[str, int] = {}
    for _, label in data[split]:
        if label != "oos":
            counts[label] = counts.get(label, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [intent for intent, _ in ranked[:k]]


# ------------------------------------------------------------------- Jev requests


def routing_request(query: str, deployed_intents: Sequence[str], context: str | None = None) -> tuple[dict, dict]:
    """The one Jev call per query: a routing ``choice`` plus a scope ``noul``.

    Asking both together is the documented System One pattern (independent questions,
    evaluated in parallel, composed in code): the router needs the label, the guardrail
    needs to know whether the taxonomy covers the query at all.
    """
    state: dict[str, Any] = {"query": query, "deployed_intents": list(deployed_intents)}
    if context:
        state["channel"] = context
    questions: dict[str, Any] = {
        "intent": {
            "type": "choice",
            "instructions": {
                "question": "Which single deployed intent does `query` belong to?",
                "focus": "Judge the user's request, not what the answer would be.",
                "tie_break": (
                    "If several intents are close, pick the one that best matches the "
                    "action the user wants."
                ),
            },
            "criteria": {intent: humanize(intent) for intent in deployed_intents},
        },
        "in_scope": {
            "type": "noul",
            "instructions": {
                "question": "Does one of the intents in `deployed_intents` cover `query`?",
                "true": "The request asks for something one of the listed intents names",
                "false": (
                    "None of the listed intents covers the request, even if the request "
                    "is a normal thing a personal assistant could in principle do"
                ),
            },
        },
    }
    return state, questions


def requests_for(queries: Iterable[Query], deployed_intents: Sequence[str]) -> list[tuple[dict, dict]]:
    return [routing_request(row.text, deployed_intents) for row in queries]


def write_plan(plan: dict[str, Any], path: str | Path) -> None:
    """Persist the sampled pools so an analysis run can replay without re-sampling."""
    serialisable = {
        key: (
            [getattr(item, "__dict__", item) for item in value] if isinstance(value, list) else value
        )
        for key, value in plan.items()
    }
    Path(path).write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def read_plan(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in ("calibration", "evaluation", "oos_evaluation", "shift_evaluation", "in_taxonomy_evaluation"):
            out[key] = [Query(**row) for row in value]
        else:
            out[key] = value
    return out
