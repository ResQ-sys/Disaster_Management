"""
Disaster-tweet batch triage + extraction evaluation.

Puts the project's reference dataset (Kaggle Disaster Tweets) to real use:
a feed of disaster tweets/field messages is converted into structured relief
needs that seed the coordination worklist, and the two available extractors
are evaluated against a hand-labelled sample:

  1. rule-based  — coordination.extract_need_from_text (keyword heuristics)
  2. LLM         — llama3.2:1b with a strict JSON contract, falling back to
                   the rule-based extractor if the model is unreachable or
                   returns unparseable output

data/disaster_tweets_sample.csv is a hand-labelled starter sample styled on
the Kaggle Disaster Tweets format (synthetic starter data, as permitted by
the project brief). Labels: category, urgency, quantity, location.
"""

import csv
import json
from pathlib import Path

from resq_project.config import TWEETS_CSV
from resq_project.coordination import CATEGORIES, extract_need_from_text

_URGENCIES = ["Critical", "High", "Medium", "Low"]

_LLM_SYSTEM = f"""You extract structured relief needs from disaster tweets.
Respond ONLY with valid JSON, no prose, in this exact shape:
{{"category": "...", "urgency": "...", "quantity": ..., "location": "..."}}

Rules:
- category: one of {CATEGORIES} — what is being asked for.
- urgency: one of {_URGENCIES} — how time-critical the wording is.
- quantity: integer number of units/people needed, or null if no unit count
  is stated (durations like "2 days" are NOT quantities).
- location: the place name mentioned, or "" if none.

Examples:
"URGENT: 30 people trapped near Aut, need rescue" ->
{{"category": "Rescue", "urgency": "Critical", "quantity": 30, "location": "Aut"}}
"Bus needed to move 45 students from Theog school" ->
{{"category": "Transport", "urgency": "Medium", "quantity": 45, "location": "Theog"}}
"No clean water in Sundernagar colony after flood" ->
{{"category": "Water", "urgency": "Medium", "quantity": null, "location": "Sundernagar"}}"""


# ══════════════════════════════════════════════════════════════════════
# Tweet feed loading + batch triage
# ══════════════════════════════════════════════════════════════════════
def load_tweet_feed(path=None) -> list[dict]:
    """Load the labelled tweet sample (tweet_id, text, label_* columns)."""
    path = Path(path or TWEETS_CSV)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(f)]


def triage_tweets(tweets=None) -> list[dict]:
    """Convert a tweet feed into structured needs for the coordination worklist.

    Uses the deterministic rule-based extractor so batch triage works offline
    and reproducibly; the LLM extractor is available for comparison/eval.
    """
    tweets = tweets if tweets is not None else load_tweet_feed()
    needs = []
    for tw in tweets:
        need = extract_need_from_text(tw.get("text", ""))
        need["request_id"] = f"TW-{tw.get('tweet_id', len(needs) + 1)}"
        need["reported_by"] = "Tweet feed"
        needs.append(need)
    return needs


# ══════════════════════════════════════════════════════════════════════
# LLM extractor (strict JSON contract + rule-based fallback)
# ══════════════════════════════════════════════════════════════════════
def extract_need_with_llm(text: str, get_llm) -> dict:
    """Extract a structured need with the local LLM; fall back to rules.

    `get_llm` is a callable returning a chat model (injected, same pattern as
    chatbot.answer, to avoid a hard dependency on the workflow module).
    """
    fallback = extract_need_from_text(text)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        resp = llm.invoke([SystemMessage(content=_LLM_SYSTEM),
                           HumanMessage(content=f"Tweet: {text}\nJSON:")])
        raw = resp.content.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)

        category = str(data.get("category", "")).title()
        urgency = str(data.get("urgency", "")).title()
        quantity = data.get("quantity")
        quantity = int(quantity) if quantity not in (None, "", "null") else ""

        return {
            **fallback,
            "category": category if category in CATEGORIES else fallback["category"],
            "urgency": urgency if urgency in _URGENCIES else fallback["urgency"],
            "quantity": quantity,
            "location": str(data.get("location", "") or "").strip() or fallback["location"],
            "extraction": "llm (json contract)",
        }
    except Exception:
        return {**fallback, "extraction": "rule-based (LLM fallback)"}


def extract_need_hybrid(text: str, get_llm) -> dict:
    """Evidence-based split of labour between the two extractors.

    The measured comparison (docs/extraction_eval.md) shows the LLM wins on
    the semantic fields (category, location) while the rules win on the
    literal fields (urgency keywords, stated quantities — the 1B LLM
    hallucinates numbers). The hybrid takes each field from its winner.
    """
    rule = extract_need_from_text(text)
    llm = extract_need_with_llm(text, get_llm)
    return {
        **rule,
        "category": llm["category"],
        "location": llm["location"] or rule["location"],
        "extraction": "hybrid (LLM category/location + rule urgency/quantity)",
    }


# ══════════════════════════════════════════════════════════════════════
# Evaluation against the hand labels
# ══════════════════════════════════════════════════════════════════════
def _norm_tokens(text: str) -> set:
    return {t for t in str(text).lower().replace(",", " ").split() if t}


def _location_correct(extracted: str, label: str) -> bool:
    """Lenient match: the labelled town appears among the extracted tokens
    (extractors often capture 'Kullu bus stand' for label 'Kullu')."""
    if not label:
        return not str(extracted).strip()
    return bool(_norm_tokens(label) & _norm_tokens(extracted))


def _quantity_correct(extracted, label: str) -> bool:
    if str(label).strip() == "":
        return str(extracted).strip() == ""
    try:
        return int(float(extracted)) == int(float(label))
    except (TypeError, ValueError):
        return False


def evaluate_extraction(extractor, tweets=None) -> dict:
    """Score an extractor (callable text -> need dict) against the labels.

    Returns per-field accuracy plus a per-tweet row breakdown so mistakes
    are inspectable, not just counted.
    """
    tweets = tweets if tweets is not None else load_tweet_feed()
    fields = ["category", "urgency", "quantity", "location"]
    correct = dict.fromkeys(fields, 0)
    rows = []

    for tw in tweets:
        need = extractor(tw["text"])
        row = {"tweet_id": tw.get("tweet_id"), "text": tw.get("text")}
        checks = {
            "category": str(need.get("category", "")).lower() == str(tw.get("label_category", "")).lower(),
            "urgency": str(need.get("urgency", "")).lower() == str(tw.get("label_urgency", "")).lower(),
            "quantity": _quantity_correct(need.get("quantity", ""), tw.get("label_quantity", "")),
            "location": _location_correct(need.get("location", ""), tw.get("label_location", "")),
        }
        for f in fields:
            correct[f] += checks[f]
            row[f] = "✓" if checks[f] else f"✗ got {need.get(f, '')!r}, want {tw.get('label_' + f, '')!r}"
        rows.append(row)

    n = len(tweets) or 1
    return {
        "n": len(tweets),
        "accuracy": {f: round(correct[f] / n, 3) for f in fields},
        "overall": round(sum(correct.values()) / (n * len(fields)), 3),
        "rows": rows,
    }
