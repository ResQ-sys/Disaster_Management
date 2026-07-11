"""
Scenario / validation tests for the HP Disaster Relief Agent.

These are deterministic checks (no LLM calls) covering the core logic:
urgency scoring, volunteer need↔resource matching, coordination-message
drafting, free-text need extraction, the RAG relevance gate, and the
wildfire/GLOF spatial assessments. Run with:  pytest -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resq_project.workflow import compute_urgency_score
from resq_project import coordination as coord


# ── Urgency scoring ────────────────────────────────────────────────────
def test_urgency_high_scenario():
    s = compute_urgency_score({
        "imd_alert_level": "RED", "district_risk": {"risk_tier": "CRITICAL"},
        "needs": ["Medical", "Rescue"], "glof_alert": {"level": "WATCH"},
        "wildfire_risk": {"level": "HIGH"}, "escalation_needed": True,
    })
    assert s["score"] >= 75 and s["level"] == "CRITICAL"
    assert s["breakdown"]["imd_alert"] == 35


def test_urgency_low_scenario():
    s = compute_urgency_score({
        "imd_alert_level": "GREEN", "district_risk": {"risk_tier": "LOW"},
        "needs": ["Food"], "glof_alert": {}, "wildfire_risk": {"level": "MINIMAL"},
        "escalation_needed": False,
    })
    assert s["score"] < 30 and s["level"] == "LOW"


def test_urgency_monotonic():
    """More severe inputs must not lower the score."""
    base = {"imd_alert_level": "YELLOW", "district_risk": {"risk_tier": "MEDIUM"},
            "needs": ["Food"], "glof_alert": {}, "wildfire_risk": {}, "escalation_needed": False}
    worse = {**base, "imd_alert_level": "RED"}
    assert compute_urgency_score(worse)["score"] >= compute_urgency_score(base)["score"]


# ── Volunteer matching ─────────────────────────────────────────────────
def test_category_is_hard_filter():
    need = {"category": "Medical", "location": "Kullu", "quantity": 10}
    res = {"category": "Food", "location": "Kullu", "quantity": 100}
    assert coord.score_match(need, res)["eligible"] is False


def test_matching_prefers_same_location_and_coverage():
    need = {"category": "Water", "location": "Kangra school", "quantity": 100}
    near = {"category": "Water", "location": "Kangra", "quantity": 250,
            "availability": "Available", "contact_status": "Verified"}
    far = {"category": "Water", "location": "Shimla", "quantity": 20,
           "availability": "Pending", "contact_status": "Pending"}
    assert coord.score_match(need, near)["score"] > coord.score_match(need, far)["score"]


def test_richer_sample_data_includes_both_matched_and_escalated_cases():
    matches = coord.match_needs_to_resources()
    assert len(matches) >= 5
    assert any(m["best_match"] is not None for m in matches)
    assert any(m["best_match"] is None for m in matches)


def test_worklist_sorted_by_urgency():
    matches = coord.match_needs_to_resources()
    ranks = [coord.URGENCY_RANK.get(str(m["need"].get("urgency", "")).upper(), 0) for m in matches]
    assert ranks == sorted(ranks, reverse=True)


# ── Coordination message ───────────────────────────────────────────────
def test_coordination_message_contains_key_fields():
    matches = coord.match_needs_to_resources()
    msg = coord.draft_coordination_message(matches[0])
    assert "COORDINATION" in msg
    assert "approval" in msg.lower()          # human-in-the-loop reminder present
    assert matches[0]["need"]["request_id"] in msg


def test_unmatched_message_escalates():
    m = {"need": {"request_id": "X1", "category": "Fuel", "location": "Nowhere",
                  "quantity": 5, "urgency": "High", "reported_by": "t", "notes": ""},
         "best_match": None}
    msg = coord.draft_coordination_message(m)
    assert "NO matching resource" in msg and "scalate" in msg


# ── Free-text / tweet extraction ───────────────────────────────────────
def test_extract_need_from_text():
    n = coord.extract_need_from_text(
        "URGENT: 30 people trapped, need rescue at Manali ward 3")
    assert n["category"] == "Rescue"
    assert n["urgency"] == "Critical"
    assert n["quantity"] == 30


# ── Streaming state merge ──────────────────────────────────────────────
def test_stream_merge_accumulates_reducer_channels():
    """stream_mode='updates' deltas carry only NEW node_log entries; the
    merge must append them, not overwrite the accumulated trace."""
    from resq_project.workflow import merge_stream_delta
    state = {"node_log": ["✓ intake_agent"], "error_log": [], "x": 1}
    merge_stream_delta(state, {"node_log": ["✓ glof_monitor_agent"], "x": 2})
    merge_stream_delta(state, {"node_log": ["✓ resource_finder_agent"], "error_log": ["boom"]})
    assert state["node_log"] == [
        "✓ intake_agent", "✓ glof_monitor_agent", "✓ resource_finder_agent"]
    assert state["error_log"] == ["boom"]
    assert state["x"] == 2   # ordinary fields still overwritten


def test_provider_counts_exclude_monitoring_assets():
    from resq_project import charts
    state = {
        "hospitals": [{"name": "A"}],
        "shelters": [{"name": "B"}, {"name": "C"}],
        "cwc_stations": [{"station": "S1"}],
        "glacial_lakes": [{"lake_id": "L1"}, {"lake_id": "L2"}],
    }
    assert charts.provider_type_counts(state) == {"Hospitals": 1, "Shelters": 2}
    assert charts.monitoring_asset_counts(state) == {"CWC Stations": 1, "Glacial Lakes": 2}


def test_pdf_inline_markup_renders_bold():
    from resq_project.pdf_report import _inline_markup

    rendered = _inline_markup("Alert **critical** now")
    assert "<b>critical</b>" in rendered


# ── Inventory-aware allocation ─────────────────────────────────────────
def _water_provider(qty=100):
    return {"resource_id": "R1", "provider_name": "Jal Unit", "category": "Water",
            "location": "Kangra", "quantity": qty,
            "availability": "Available", "contact_status": "Verified"}


def test_allocation_prevents_double_promising():
    needs = [
        {"request_id": "A1", "category": "Water", "location": "Kangra", "quantity": 80, "urgency": "Critical"},
        {"request_id": "A2", "category": "Water", "location": "Kangra", "quantity": 50, "urgency": "High"},
    ]
    out = coord.match_needs_to_resources(needs, [_water_provider(100)], allocate=True)
    by_id = {m["need"]["request_id"]: m for m in out}
    assert by_id["A1"]["best_match"]["committed_units"] == 80
    # second need only gets what is actually left, not the full listed stock
    assert by_id["A2"]["best_match"]["committed_units"] == 20
    total = sum(m["best_match"]["committed_units"] for m in out)
    assert total <= 100


def test_allocation_urgency_order_and_exhaustion():
    needs = [
        {"request_id": "B2", "category": "Water", "location": "Kangra", "quantity": 10, "urgency": "Low"},
        {"request_id": "B1", "category": "Water", "location": "Kangra", "quantity": 100, "urgency": "Critical"},
    ]
    out = coord.match_needs_to_resources(needs, [_water_provider(100)], allocate=True)
    by_id = {m["need"]["request_id"]: m for m in out}
    # Critical need is served first even though it was listed second …
    assert by_id["B1"]["best_match"]["committed_units"] == 100
    # … and the exhausted provider is no longer offered to the Low need
    assert by_id["B1"]["status"] == "MATCHED"
    assert by_id["B2"]["best_match"] is None and by_id["B2"]["status"] == "UNMATCHED"


def test_allocation_does_not_mutate_inputs():
    resources = [_water_provider(100)]
    needs = [{"request_id": "C1", "category": "Water", "location": "Kangra",
              "quantity": 60, "urgency": "High"}]
    coord.match_needs_to_resources(needs, resources, allocate=True)
    assert resources[0]["quantity"] == 100


# ── Dispatch ledger ────────────────────────────────────────────────────
def test_dispatch_ledger_decrements_stock(monkeypatch, tmp_path):
    monkeypatch.setattr(coord, "DISPATCH_LEDGER", tmp_path / "ledger.jsonl")
    coord.log_dispatch({"resource_id": "R001", "units": 15, "request_id": "N001"})
    coord.log_dispatch({"resource_id": "R001", "units": 5, "request_id": "N009"})
    assert coord.dispatched_totals()["R001"] == 20

    r001 = next(r for r in coord.load_resources() if r["resource_id"] == "R001")
    assert r001["quantity_original"] == 40
    assert r001["dispatched_units"] == 20
    assert r001["quantity"] == 20   # remaining = listed − dispatched


def test_dispatch_ledger_never_goes_negative(monkeypatch, tmp_path):
    monkeypatch.setattr(coord, "DISPATCH_LEDGER", tmp_path / "ledger.jsonl")
    coord.log_dispatch({"resource_id": "R001", "units": 999, "request_id": "X"})
    r001 = next(r for r in coord.load_resources() if r["resource_id"] == "R001")
    assert r001["quantity"] == 0


# ── Tweet feed triage + extraction evaluation ──────────────────────────
def test_tweet_feed_triage():
    from resq_project import tweet_triage as tt
    tweets = tt.load_tweet_feed()
    assert len(tweets) >= 40
    needs = tt.triage_tweets(tweets)
    assert len(needs) == len(tweets)
    assert all(n["request_id"].startswith("TW-") for n in needs)
    assert all(n["category"] in coord.CATEGORIES for n in needs)


def test_rule_extraction_baseline_accuracy():
    """Pin the measured rule-based baseline (docs/extraction_eval.md)."""
    from resq_project import tweet_triage as tt
    res = tt.evaluate_extraction(coord.extract_need_from_text)
    assert res["n"] >= 40
    assert res["accuracy"]["category"] >= 0.70
    assert res["accuracy"]["urgency"] >= 0.90
    assert res["accuracy"]["quantity"] >= 0.90
    assert res["overall"] >= 0.85


# ── Operations map gazetteer ───────────────────────────────────────────
def test_opsmap_gazetteer_offline():
    from resq_project.opsmap import locate
    lat, lon, source = locate("Kullu bus stand")
    assert source.startswith("gazetteer") and 31 < lat < 33 and 76 < lon < 79
    # longest-name match wins ("paonta sahib", not a shorter partial)
    assert locate("relief camp near Paonta Sahib")[2] == "gazetteer · Paonta Sahib"


# ── RAG relevance gate (retrieval only, no LLM) ────────────────────────
@pytest.mark.parametrize("q", [
    "Which hospitals are in Kullu?",
    "GLOF risk in Lahul and Spiti",
    "nearest CWC river monitoring station",
])
def test_rag_accepts_in_scope(q):
    from resq_project.chatbot import retrieve, CONFIDENT_DISTANCE
    hits = retrieve(q)
    assert hits and hits[0]["distance"] <= CONFIDENT_DISTANCE


@pytest.mark.parametrize("q", [
    "What is the capital of France?",
    "Who won the cricket match yesterday?",
    "How do I bake a chocolate cake?",
])
def test_rag_rejects_out_of_scope(q):
    from resq_project.chatbot import retrieve, CONFIDENT_DISTANCE
    hits = retrieve(q)
    top = hits[0]["distance"] if hits else 99
    assert top > CONFIDENT_DISTANCE   # gate would refuse before the LLM


# ── Wildfire proneness ─────────────────────────────────────────────────
def test_wildfire_prone_vs_remote():
    from resq_project.tools import assess_wildfire_risk
    shimla = assess_wildfire_risk(31.104, 77.173)
    remote = assess_wildfire_risk(32.226, 78.071)
    assert shimla["prone"] is True and shimla["level"] in ("HIGH", "MODERATE")
    assert remote["prone"] is False and remote["count_10km"] == 0


# ── GLOF glacial-lake query ────────────────────────────────────────────
def test_glacial_lake_query_flags_increase():
    from resq_project.tools import query_glacial_lakes
    lakes = query_glacial_lakes("LAHUL AND SPITI", 32.5, 77.5, n_results=5)
    assert lakes, "expected glacial lakes for Lahul & Spiti"
    assert any(lk.get("status") == "increase" for lk in lakes)
