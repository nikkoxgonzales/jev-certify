"""Client/journal checks: content addressing, offline replay, and the real-data replay
that proves re-analysis never re-bills the API.

These tests never touch the network. ``JevClient(offline=True)`` refuses to make a
request and can only answer from the journal, so if a test passes offline the replay
path is genuinely working.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_certify.client import Decision, DecisionJournal, JevClient, JevError, decision_key
from jev_certify.tasks import read_plan, requests_for

ROOT = Path(__file__).resolve().parent.parent


def test_decision_key_is_content_addressed():
    state = {"query": "where is my card", "deployed_intents": ["card_arrival", "balance"]}
    questions = {"intent": {"type": "choice", "criteria": {"card_arrival": "card arrival"}}}
    first = decision_key("typesafe/jev-1.13", state, questions)
    assert first == decision_key("typesafe/jev-1.13", state, dict(questions))
    # Key order in the state must not matter: it is canonicalised before hashing.
    reordered = {"deployed_intents": state["deployed_intents"], "query": state["query"]}
    assert first == decision_key("typesafe/jev-1.13", reordered, questions)
    assert first != decision_key("typesafe/jev-1.13", {"query": "different"}, questions)
    assert first != decision_key("typesafe/jev-1.12", state, questions)


def test_journal_round_trip(tmp_path: Path):
    journal = DecisionJournal(tmp_path / "j.jsonl")
    decision = Decision(
        key="abc",
        model="typesafe/jev-1.13",
        state={"query": "hi"},
        questions={"intent": {"type": "choice"}},
        answers={"intent": {"type": "choice", "choice": "balance", "probabilities": {"balance": 1.0}}},
        usage={"input_tokens": 10, "output_tokens": 2, "cost": 0.0001},
        latency_s=0.5,
    )
    journal.append(decision)
    assert len(journal) == 1
    assert journal.total_cost_usd() == pytest.approx(0.0001)
    restored = Decision.from_record(journal.get("abc"))
    assert restored.answers["intent"]["choice"] == "balance"
    assert restored.served_from == "journal"
    assert restored.input_tokens == 10


def test_offline_client_serves_from_journal_and_refuses_otherwise(tmp_path: Path):
    journal_path = tmp_path / "j.jsonl"
    state = {"query": "transfer money"}
    questions = {"intent": {"type": "choice", "criteria": {"transfer": "transfer"}}}
    key = decision_key("typesafe/jev-1.13", state, questions)
    DecisionJournal(journal_path).append(
        Decision(
            key=key,
            model="typesafe/jev-1.13",
            state=state,
            questions=questions,
            answers={"intent": {"type": "choice", "choice": "transfer", "probabilities": {"transfer": 1.0}}},
            usage={"input_tokens": 5, "output_tokens": 1, "cost": 0.0},
        )
    )

    client = JevClient(offline=True, journal=journal_path)
    replay = client.decide(state, questions)
    assert replay.served_from == "journal"
    assert client.n_api_calls == 0

    with pytest.raises(JevError, match="offline mode"):
        client.decide({"query": "something new"}, questions)


def test_journal_tolerates_a_torn_tail_line(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    path.write_text('{"key": "a", "model": "m", "state": {}, "questions": {}, "answers": {}}\n{"key": "b", "mod', encoding="utf-8")
    journal = DecisionJournal(path)
    assert len(journal) == 1


def test_real_journal_replays_the_recorded_experiment_without_calling_the_api():
    """The central reproducibility claim, checked against the shipped data.

    Rebuilding every routing request from the saved pools must reproduce keys that are
    already in the journal; otherwise the report could not be regenerated for free.
    """
    results = ROOT / "results"
    plan_file = results / "plan-v1-full.json"
    journal_file = results / "journal.jsonl"
    if not plan_file.is_file() or not journal_file.is_file():
        pytest.skip("no recorded run in results/ yet -- run python -m experiments.collect")

    plan = read_plan(plan_file)
    journal = DecisionJournal(journal_file).load()
    rows = (plan.get("evaluation") or []) + (plan.get("calibration") or [])
    assert rows, "plan has no sampled rows"
    missing = 0
    for (state, questions) in requests_for(rows, plan["deployed_intents"]):
        if decision_key("typesafe/jev-1.13", state, questions) not in journal:
            missing += 1
    assert missing == 0, f"{missing} of {len(rows)} recorded requests are missing from the journal"


def test_offline_replay_of_a_recorded_scenario_yields_full_scenario():
    results = ROOT / "results"
    plan_file = results / "plan-v1-full.json"
    journal_file = results / "journal.jsonl"
    if not plan_file.is_file() or not journal_file.is_file():
        pytest.skip("no recorded run in results/ yet")

    from jev_certify.analysis import load_scenario

    scenario = load_scenario(plan_file, journal_file)
    assert len(scenario.calibration) > 0
    assert len(scenario.evaluation) > 0
    # Every answer carries the two questions the router asked.
    sample = next(iter(scenario.decisions.values()))
    assert set(sample.answers) == {"intent", "in_scope"}
    assert json.dumps(sample.to_record())  # serialisable, so the journal stays portable
