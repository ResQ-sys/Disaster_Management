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


def test_all_sample_needs_get_matched():
    matches = coord.match_needs_to_resources()
    assert len(matches) >= 5
    # every sample need should find at least a candidate (not UNMATCHED)
    assert all(m["best_match"] is not None for m in matches)


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
