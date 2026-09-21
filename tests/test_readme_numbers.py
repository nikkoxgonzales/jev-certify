"""Every headline number in the README must match results/results.json.

A report that can drift away from the data that produced it is worse than no report, and
prose is the easiest place to let a number rot. Each test below pins one published claim to
the recorded result. They skip when the run is absent or was taken at a different alpha, so
the suite stays honest rather than failing for the wrong reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "results.json"


@pytest.fixture(scope="module")
def results() -> dict:
    if not RESULTS.is_file():
        pytest.skip("no recorded run -- python -m experiments.collect && python -m experiments.analyse")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_primary_certificate_matches_the_readme(results: dict, readme: str):
    if results["meta"]["alpha"] != 0.05:
        pytest.skip("README pins the alpha=0.05 run")
    primary = results["routing"]["primary"]
    measured = primary["measured_on_holdout"]
    assert primary["threshold"] == 0.831
    assert primary["certified_per_query_loss"] == 0.0499
    assert measured["coverage"] == 0.8475
    assert measured["selective_error"] == 0.0265
    assert primary["certificate_held"] is True
    for token in ("0.831", "84.75%", "2.65%"):
        assert token in readme, f"README no longer states {token}"


def test_resolution_floor_matches_the_readme(results: dict, readme: str):
    alpha_one = next(c for c in results["routing"]["certificates"] if c["alpha"] == 0.01)
    assert alpha_one["feasible"] is False
    assert alpha_one["min_attainable_bound"] == 0.0195
    quantization = results["quantization"]
    assert quantization["share_at_1.0"] == 0.564
    assert quantization["errors_at_1.0"] == 9
    assert quantization["distinct_top_probabilities"] == 94
    assert quantization["distinct_values_are_multiples_of_0.01"] == "60/94"
    for token in ("56.4%", "1.95%", "94 distinct"):
        assert token in readme, f"README no longer states {token}"


def test_gate_numbers_match_the_readme(results: dict, readme: str):
    gate_one = next(c for c in results["gate"]["certificates"] if c["alpha"] == 0.01)
    assert gate_one["threshold"] == 0.671
    assert gate_one["certificate_held"] is True
    assert gate_one["in_scope_pass_rate"] == 0.82
    assert gate_one["out_of_scope_accept_rate"] == 0.0167
    # The prevalence failure must stay visible: 5% target, not held, materially over.
    gate_five = next(c for c in results["gate"]["certificates"] if c["alpha"] == 0.05)
    assert gate_five["certificate_held"] is False
    assert gate_five["measured"]["per_query_risk"] > gate_five["certified_false_accept_per_query"] * 3
    assert "prevalence" in readme.lower()


def test_audit_numbers_match_the_readme(results: dict, readme: str):
    smallest = min(results["audit"]["budgets"], key=lambda b: b["n_labeled"])
    assert smallest["n_labeled"] == 25
    assert smallest["methods"]["ppi"]["mean_width"] == 0.1017
    assert round(smallest["methods"]["classical"]["mean_width"], 4) == 0.2696
    assert smallest["labels_ratio_classical_over_ppi"] == 7.025
    planning = results["audit"]["planning"]
    assert round(planning["half_width_0.05"]["ppi_labels"]) == 40
    assert round(planning["half_width_0.05"]["classical_labels"]) == 254
    assert planning["half_width_0.02"]["ppi_labels"] == float("inf")
    for token in ("0.1017", "0.2696", "254", "±0.02"):
        assert token in readme, f"README no longer states {token}"


def test_prediction_sets_stay_reported_as_a_negative_result(results: dict, readme: str):
    scores = results["prediction_sets"]["scores"]
    assert scores["aps"]["at_calibrated_qhat"]["mean_set_size"] == 148.5375
    assert scores["lac"]["at_calibrated_qhat"]["mean_set_size"] == 150.0
    assert "148.5" in readme and "useless" in readme.lower()


def test_drift_and_shift_findings_match_the_readme(results: dict, readme: str):
    drift = results["drift"]["sections"]
    assert drift["evaluation_exchangeable"]["certificate_held"] is True
    oos = drift["out_of_scope_traffic"]
    assert oos["certificate_held"] is False
    assert oos["routing_above_threshold"]["per_query_risk"] == 0.23
    restricted = results["restricted_deployment"]
    assert restricted["routing"]["certificates"][0]["threshold"] == 0.0
    assert restricted["drift"]["sections"]["unlisted_intents_traffic"]["routing_above_threshold"][
        "per_query_risk"
    ] == 1.0
    assert not restricted["gate"]["certificates"], "the restricted gate must stay reported as uncalibratable"
    assert "1.0000" in readme


def test_cost_claims_match_the_readme(results: dict, readme: str):
    assert results["cost"]["cost_usd_per_1k"] == 0.1455
    assert results["cost"]["latency_p50_s"] == 0.37
    assert results["cost"]["latency_p95_s"] == 0.744
    points = results["routing"]["operating_points"]
    assert points["always_llm"]["selective_error"] is None, "an unmeasured baseline must stay unmeasured"
    assert 81 <= points["savings_vs_always_llm_pct"] <= 82
    # The cost table's whole argument is that the certificate beats the hand-picked
    # alternatives, so pin the rows it compares against.
    by_name = {p["name"]: p for p in points["operating_points"]}
    certified = by_name["certified risk <= 5%"]
    hand_099 = by_name["hand-picked threshold 0.99"]
    saturated = next(p for p in points["operating_points"] if p["name"].startswith("saturated"))
    assert certified["coverage"] == 0.8475 and round(certified["cost_per_1k_usd"], 2) == 0.79
    assert hand_099["coverage"] == 0.7050 and round(hand_099["cost_per_1k_usd"], 2) == 1.38
    assert saturated["certified"] is False and round(saturated["cost_per_1k_usd"], 2) == 1.71
    assert certified["coverage"] > hand_099["coverage"], "the README claim that certified beats hand-picked 0.99"
    for token in ("$0.1455", "0.37 s", "not measured", "$1.38", "$1.71"):
        assert token in readme, f"README no longer states {token}"
