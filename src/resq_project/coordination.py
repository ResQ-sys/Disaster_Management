"""
Volunteer need ↔ resource matching + human-in-the-loop coordination.

This is the module that fulfils the core project framing: match reported
*needs* (needs.csv) against available *resources* (resources.csv), score each
match by category / location / quantity / urgency / availability, draft a
coordination message for each match, and keep humans in control via an
explicit approve → send gate that is logged for audit.

All matching logic here is deterministic and explainable (no LLM), so the
routing decisions are transparent and reproducible.
"""

import json
import csv
import re
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from resq_project.config import (
    APPROVALS_LOG,
    DISPATCH_LEDGER,
    NEEDS_CSV,
    RESOURCES_CSV,
)

# ── Scoring weights ────────────────────────────────────────────────────
URGENCY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "": 0}
CATEGORIES = ["Medical", "Shelter", "Food", "Water", "Transport", "Rescue"]


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════
def load_needs() -> list[dict]:
    if not Path(NEEDS_CSV).exists():
        return []
    with open(NEEDS_CSV, newline="", encoding="utf-8-sig") as f:
        return [{k: (v if v is not None else "") for k, v in row.items()} for row in csv.DictReader(f)]


def load_resources(apply_ledger: bool = True) -> list[dict]:
    """Load the resource pool; by default apply the dispatch ledger so
    `quantity` reflects what is actually still available (never below 0)."""
    if not Path(RESOURCES_CSV).exists():
        return []
    with open(RESOURCES_CSV, newline="", encoding="utf-8-sig") as f:
        resources = [{k: (v if v is not None else "") for k, v in row.items()}
                     for row in csv.DictReader(f)]

    dispatched = dispatched_totals() if apply_ledger else {}
    for res in resources:
        try:
            original = int(float(res.get("quantity", 0) or 0))
        except (TypeError, ValueError):
            original = 0
        used = int(dispatched.get(str(res.get("resource_id", "")), 0))
        res["quantity_original"] = original
        res["dispatched_units"] = used
        res["quantity"] = max(0, original - used)
    return resources


# ══════════════════════════════════════════════════════════════════════
# Dispatch ledger — approved sends consume provider stock
# ══════════════════════════════════════════════════════════════════════
def log_dispatch(record: dict) -> None:
    """Record an approved dispatch (resource_id, provider, units, request_id)
    so the provider's remaining stock reflects what has been committed."""
    DISPATCH_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(DISPATCH_LEDGER, "a") as f:
        f.write(json.dumps(record) + "\n")


def dispatched_totals() -> dict:
    """Total units committed per resource_id across the ledger."""
    if not DISPATCH_LEDGER.exists():
        return {}
    totals: dict = {}
    with open(DISPATCH_LEDGER) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("resource_id", ""))
            try:
                units = int(float(rec.get("units", 0) or 0))
            except (TypeError, ValueError):
                units = 0
            if rid:
                totals[rid] = totals.get(rid, 0) + units
    return totals


def read_dispatches(limit: int = 50) -> list[dict]:
    if not DISPATCH_LEDGER.exists():
        return []
    with open(DISPATCH_LEDGER) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[-limit:]


# ══════════════════════════════════════════════════════════════════════
# Matching engine
# ══════════════════════════════════════════════════════════════════════
def _location_tokens(text: str) -> set:
    return {t for t in re.split(r"[\s,]+", str(text).lower()) if len(t) > 2}


def _location_score(need_loc: str, res_loc: str) -> float:
    """0-1 overlap of location tokens (town/ward names)."""
    a, b = _location_tokens(need_loc), _location_tokens(res_loc)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_match(need: dict, resource: dict) -> dict:
    """Score a single need↔resource pair (0-100) with a transparent breakdown."""
    # Category is a hard filter — different categories cannot match.
    if str(need.get("category", "")).lower() != str(resource.get("category", "")).lower():
        return {"score": 0, "eligible": False}

    breakdown = {}

    # Category match (hard requirement satisfied)
    breakdown["category"] = 40

    # Location proximity (token overlap)
    loc = _location_score(need.get("location", ""), resource.get("location", ""))
    breakdown["location"] = round(25 * loc, 1)

    # Quantity coverage
    try:
        need_qty = float(need.get("quantity", 0) or 0)
        res_qty = float(resource.get("quantity", 0) or 0)
        coverage = min(res_qty / need_qty, 1.0) if need_qty > 0 else 1.0
    except (TypeError, ValueError):
        coverage = 0.0
    breakdown["quantity"] = round(20 * coverage, 1)

    # Availability / verification
    avail = str(resource.get("availability", "")).lower() == "available"
    verified = str(resource.get("contact_status", "")).lower() == "verified"
    breakdown["availability"] = (8 if avail else 0) + (7 if verified else 0)

    score = round(sum(breakdown.values()), 1)
    gap = 0
    try:
        gap = max(0, int(float(need.get("quantity", 0)) - float(resource.get("quantity", 0))))
    except (TypeError, ValueError):
        pass

    return {
        "score": score,
        "eligible": True,
        "coverage_pct": round(coverage * 100),
        "quantity_gap": gap,
        "breakdown": breakdown,
    }


def match_needs_to_resources(needs=None, resources=None, allocate: bool = False) -> list[dict]:
    """For every need, rank eligible resources and attach the best match.

    With allocate=True the matcher becomes inventory-aware: needs are
    processed in urgency order and each best match *reserves* its units from
    a working copy of the pool, so one provider's stock is never promised to
    two needs at once. Each best match then carries `committed_units`.
    """
    needs = needs if needs is not None else load_needs()
    resources = resources if resources is not None else load_resources()

    # Working copies so allocation never mutates the caller's resource dicts.
    pool = [dict(res) for res in resources]

    if allocate:
        needs = sorted(needs, key=lambda n: -URGENCY_RANK.get(
            str(n.get("urgency", "")).upper(), 0))

    results = []
    for need in needs:
        try:
            need_qty = int(float(need.get("quantity", 0) or 0))
        except (TypeError, ValueError):
            need_qty = 0

        scored = []
        for res in pool:
            try:
                remaining = int(float(res.get("quantity", 0) or 0))
            except (TypeError, ValueError):
                remaining = 0
            # Exhausted providers can't serve a quantified need anymore.
            if allocate and need_qty > 0 and remaining <= 0:
                continue
            s = score_match(need, res)
            if s.get("eligible"):
                # Snapshot: the draft shows availability as of *this* match,
                # even after later needs decrement the working pool.
                scored.append({"resource": dict(res), "_pool_res": res, **s})
        scored.sort(key=lambda x: (x["score"], URGENCY_RANK.get(
            str(x["resource"].get("urgency_capacity", "")).upper(), 0)), reverse=True)

        best = scored[0] if scored else None
        if best and allocate:
            pool_res = best["_pool_res"]
            available = int(float(pool_res.get("quantity", 0) or 0))
            committed = min(need_qty, available) if need_qty > 0 else 0
            best["committed_units"] = committed
            pool_res["quantity"] = available - committed
        for entry in scored:
            entry.pop("_pool_res", None)

        results.append({
            "need": need,
            "best_match": best,
            "alternatives": scored[1:3],
            "status": "MATCHED" if best and best["score"] >= 55 else
                      ("PARTIAL" if best else "UNMATCHED"),
        })
    # Sort the worklist by need urgency, then unmatched first within a level
    results.sort(key=lambda r: (
        -URGENCY_RANK.get(str(r["need"].get("urgency", "")).upper(), 0),
        0 if r["status"] != "MATCHED" else 1,
    ))
    return results


# ══════════════════════════════════════════════════════════════════════
# Free-text / tweet → need extraction (rule-based starter)
# ══════════════════════════════════════════════════════════════════════
_CATEGORY_KEYWORDS = {
    "Medical": ["medical", "medicine", "injured", "injury", "first aid", "doctor", "ambulance", "hospital", "trauma"],
    "Shelter": ["shelter", "displaced", "homeless", "tent", "roof", "stranded", "stay"],
    "Food": ["food", "meal", "ration", "hungry", "eat", "kits"],
    "Water": ["water", "drinking", "thirsty", "tanker"],
    "Transport": ["transport", "vehicle", "evacuate", "evacuation", "rescue boat", "pickup"],
    "Rescue": ["rescue", "trapped", "stuck", "search", "sos", "help us"],
}
_URGENCY_KEYWORDS = {
    "Critical": ["urgent", "critical", "immediately", "dying", "emergency", "sos", "asap"],
    "High": ["soon", "quickly", "high", "serious"],
}


def extract_need_from_text(text: str) -> dict:
    """Rule-based extraction of a structured need from a free-text message/tweet.

    Starter NLP: keyword matching for category + urgency, first quantity number,
    and a naive location guess. Clearly heuristic — meant to seed the worklist.
    """
    t = (text or "").lower()

    category = ""
    best_hits = 0
    for cat, kws in _CATEGORY_KEYWORDS.items():
        hits = sum(1 for k in kws if k in t)
        if hits > best_hits:
            best_hits, category = hits, cat

    urgency = "Medium"
    for lvl, kws in _URGENCY_KEYWORDS.items():
        if any(k in t for k in kws):
            urgency = lvl
            break

    qty_match = re.search(r"\b(\d{1,4})\b", t)
    quantity = int(qty_match.group(1)) if qty_match else ""

    # naive location: capitalised word(s) from the original text
    loc_match = re.search(r"(?:in|at|near)\s+([A-Z][\w]+(?:\s+[A-Z]?[\w]+){0,2})", text or "")
    location = loc_match.group(1).strip() if loc_match else ""

    return {
        "request_id": "MSG-" + datetime.now().strftime("%H%M%S"),
        "reported_by": "Extracted from message",
        "location": location,
        "category": category or "Rescue",
        "quantity": quantity,
        "urgency": urgency,
        "contact_status": "Pending",
        "notes": (text or "").strip()[:160],
        "extraction": "rule-based (heuristic)",
    }


# ══════════════════════════════════════════════════════════════════════
# Coordination message drafting (human-in-the-loop)
# ══════════════════════════════════════════════════════════════════════
def draft_coordination_message(match: dict, coordinator_email: str = "") -> str:
    """Produce a ready-to-send coordination message for a matched need↔resource.

    This is a DRAFT for human approval — it is never sent automatically.
    """
    need = match["need"]
    best = match.get("best_match")
    coordinator_email = (coordinator_email or "").strip() or "demo.coordinator@example.com"

    header = (
        f"DISASTER RELIEF COORDINATION — {need.get('category', 'N/A').upper()} "
        f"[{str(need.get('urgency', '')).upper()} priority]"
    )

    if not best:
        return (
            f"{header}\n\n"
            f"TO: {coordinator_email}\n"
            f"SUBJECT: Approval required to escalate need #{need.get('request_id')}\n\n"
            f"CC: District Control Room\n\n"
            f"Need #{need.get('request_id')}: {need.get('quantity')} unit(s) of "
            f"{need.get('category')} at {need.get('location')} "
            f"(reported by {need.get('reported_by')}).\n"
            f"Notes: {need.get('notes')}\n\n"
            f"⚠ NO matching resource found in the available pool. "
            f"Escalate to district control room (HPSDMA 1077 / NDMA 1078) for external support.\n\n"
            f"— Action for human coordinator: verify need, approve escalation, then escalate."
        )

    res = best["resource"]
    coverage = best.get("coverage_pct", 0)
    gap = best.get("quantity_gap", 0)
    gap_line = (f"\n⚠ Partial coverage: shortfall of {gap} unit(s); consider a second provider."
                if gap > 0 else "")

    return (
        f"{header}\n\n"
        f"TO: {coordinator_email}\n"
        f"SUBJECT: Approve outreach to {res.get('provider_name')} for need #{need.get('request_id')}\n\n"
        f"Matched provider: {res.get('provider_name')} ({res.get('location')})\n"
        f"FROM: HP Disaster Relief Coordination\n\n"
        f"Request #{need.get('request_id')}: {need.get('quantity')} unit(s) of "
        f"{need.get('category')} needed at {need.get('location')} "
        f"(reported by {need.get('reported_by')}).\n"
        f"Notes: {need.get('notes')}\n\n"
        f"You are listed with {res.get('quantity')} unit(s) available "
        f"({res.get('availability')}, contact {res.get('contact_status')}). "
        f"Estimated coverage: {coverage}%.{gap_line}\n\n"
        f"On approval, send outreach requesting dispatch of "
        f"{min(int(need.get('quantity') or 0), int(res.get('quantity') or 0))} "
        f"unit(s) and expected arrival time.\n\n"
        f"— Match confidence: {best.get('score')}/100. "
        f"Requires human coordinator approval before sending."
    )


# ══════════════════════════════════════════════════════════════════════
# Human-in-the-loop approval logging (audit trail)
# ══════════════════════════════════════════════════════════════════════
def log_approval(record: dict) -> None:
    """Append a human decision (approve/edit/send/reject) to the audit log."""
    APPROVALS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(APPROVALS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_approvals(limit: int = 50) -> list[dict]:
    if not APPROVALS_LOG.exists():
        return []
    with open(APPROVALS_LOG) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[-limit:]


def send_coordinator_email(recipient: str, subject: str, body: str,
                           sender_email: str, smtp_password: str) -> dict:
    """Send a real email to the coordinator using hardcoded Gmail SMTP."""
    if not recipient:
        return {"sent": False, "error": "Coordinator email is empty."}
    if not sender_email:
        return {"sent": False, "error": "Sender email is empty."}
    if not smtp_password:
        return {"sent": False, "error": "SMTP password is empty."}

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender_email, smtp_password)
            server.send_message(msg)
        return {"sent": True, "error": ""}
    except Exception as e:
        return {"sent": False, "error": str(e)}
