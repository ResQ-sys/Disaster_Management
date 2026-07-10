"""
Grounded RAG chatbot for the HP Disaster Relief Agent.

Answers questions STRICTLY from the ingested Himachal Pradesh disaster data
(ChromaDB collections). Two layers guard against hallucination / out-of-scope:

  1. Relevance gate — retrieve across all collections; if nothing clears the
     cosine-distance threshold, refuse immediately WITHOUT calling the LLM.
  2. Grounded prompt — the LLM is instructed to answer ONLY from the retrieved
     context and to say "I don't know" if the context is insufficient.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from resq_project.config import (
    COLLECTION_HOSPITALS, COLLECTION_SHELTERS, COLLECTION_SCHOOLS,
    COLLECTION_CWC, COLLECTION_KNOWLEDGE, COLLECTION_GLACIAL,
    HP_EMERGENCY_NUMBERS, RELIEF_RATES, RELIEF_RATES_SOURCE,
)
from resq_project.tools import (
    _get_collection, _haversine, geocode_place, query_hospitals, query_shelters,
)
from resq_project.i18n import t

# Collections searched, with a friendly label for citations
_SEARCH_COLLECTIONS = {
    COLLECTION_KNOWLEDGE: "Disaster Knowledge (HIMCOSTE / IMD / NDMA)",
    COLLECTION_HOSPITALS: "NHP Hospitals",
    COLLECTION_SHELTERS:  "DAY-NULM Shelters",
    COLLECTION_SCHOOLS:   "Govt Schools (shelter proxy)",
    COLLECTION_CWC:       "CWC River Stations",
    COLLECTION_GLACIAL:   "CWC Glacial Lakes (GLOF)",
}

# Cosine distance (0 = identical, 2 = opposite). Keep only reasonably close
# matches; require at least one confident match to treat the query as in-scope.
KEEP_MAX_DISTANCE     = 1.00   # discard supporting chunks farther than this
CONFIDENT_DISTANCE    = 0.70   # need >=1 hit at least this close to be in-scope
                               # (calibrated: in-scope ≤0.59, out-of-scope ≥0.77)
TOP_K_PER_COLLECTION  = 4
MAX_CONTEXT_CHUNKS    = 10

OUT_OF_SCOPE_MSG = (
    "I don't know — that's outside the scope of my Himachal Pradesh disaster "
    "data sources (hospitals, shelters, CWC river stations, glacial-lake/GLOF "
    "monitoring, and HP disaster guidance)."
)

SYSTEM_PROMPT_TEMPLATE = """You are the Himachal Pradesh Disaster Relief assistant.

Answer the user's QUESTION using ONLY the facts in the CONTEXT below. The
CONTEXT is retrieved from official HP disaster data sources.

STRICT RULES:
- Use ONLY information present in the CONTEXT. Never use outside or prior knowledge.
- Do NOT invent or guess names, numbers, phone contacts, coordinates, or statistics.
- If the CONTEXT does not clearly contain the answer, reply EXACTLY (in English):
  "I don't know based on my Himachal Pradesh disaster data sources."
- Otherwise, be concise, factual, and specific, and write your answer in {language}.
- Quote figures/names exactly as in the CONTEXT.
- Do not answer questions unrelated to Himachal Pradesh disaster response.
"""


def retrieve(question: str):
    """Query every collection; return kept chunks sorted by distance."""
    hits = []
    for coll_name, label in _SEARCH_COLLECTIONS.items():
        try:
            col = _get_collection(coll_name)
            n = min(TOP_K_PER_COLLECTION, col.count())
            if n == 0:
                continue
            res = col.query(
                query_texts=[question],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            continue

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            if dist is None or dist > KEEP_MAX_DISTANCE:
                continue
            src = (meta or {}).get("source", label)
            hits.append({"doc": doc, "distance": dist, "label": label, "source": src})

    hits.sort(key=lambda h: h["distance"])
    return hits[:MAX_CONTEXT_CHUNKS]


def answer(question: str, get_llm, language: str = "English") -> dict:
    """Return a grounded answer dict: {answer, grounded, sources, hits}.

    `get_llm` is a callable returning a chat model (injected to avoid a hard
    dependency on the workflow module). `language` controls the language of
    a *successful* answer only — the refusal message stays a fixed English
    string so the `refused` detection below keeps working regardless of
    the user's language preference.
    """
    q = (question or "").strip()
    if not q:
        return {"answer": "Please ask a question.", "grounded": False, "sources": []}

    hits = retrieve(q)

    # Relevance gate — refuse before the LLM if nothing is clearly relevant.
    if not hits or hits[0]["distance"] > CONFIDENT_DISTANCE:
        return {"answer": OUT_OF_SCOPE_MSG, "grounded": False, "sources": [], "hits": hits}

    context = "\n\n".join(
        f"[{i+1}] (source: {h['source']})\n{h['doc']}" for i, h in enumerate(hits)
    )
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {q}\n\nAnswer:"

    try:
        llm = get_llm()
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=language or "English")
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        text = resp.content.strip()
    except Exception as e:
        return {"answer": f"(assistant error: {e})", "grounded": False, "sources": []}

    # Unique source labels, ordered by best match
    seen, sources = set(), []
    for h in hits:
        if h["label"] not in seen:
            seen.add(h["label"])
            sources.append(h["label"])

    refused = "i don't know" in text.lower()
    return {
        "answer": text,
        "grounded": not refused,
        "sources": [] if refused else sources,
        "hits": hits,
    }


# ══════════════════════════════════════════════════════════════════════
# PROPERTY DAMAGE ASSISTANT
#
# Opens automatically after a disaster alert/incident is reported. This is
# a fixed, button-driven decision tree (no free-text/LLM parsing needed —
# every branch is a known choice), so it stays deterministic and auditable.
# Purely informational: it never files, submits, or writes to any backend;
# every compensation figure carries a verification disclaimer.
# ══════════════════════════════════════════════════════════════════════

# District-centre fallback for geocoding hospitals/shelters that don't carry
# their own coordinates. Mirrors workflow.py's DISTRICT_CENTERS — kept as a
# small local copy so this module doesn't pull in the LangGraph workflow
# just for one constant (same duplication pattern app.py already uses).
_PD_DISTRICT_CENTERS = {
    "KANGRA": (32.10, 76.27), "MANDI": (31.71, 76.93),
    "SHIMLA": (31.10, 77.17), "KULLU": (31.95, 77.11),
    "SOLAN": (30.91, 77.10), "SIRMOUR": (30.56, 77.66),
    "BILASPUR": (31.34, 76.76), "HAMIRPUR": (31.68, 76.52),
    "CHAMBA": (32.55, 76.12), "UNA": (31.47, 76.27),
    "KINNAUR": (31.59, 78.44), "LAHUL AND SPITI": (32.55, 77.60),
}

CLAIM_PROCEDURE_STEPS = [
    "Report the damage to your area **Patwari / Revenue department (DDMA)** — "
    "Revenue staff carry out the official damage assessment in HP.",
    "Take **dated photos and video** of the damage from multiple angles as evidence.",
    "Keep ready: **Aadhaar, bank account details, ration card, and ownership/tenancy proof.**",
    "Submit the relief application at the **Tehsil / SDM (Revenue) office.**",
    "A **Patwari verifies the damage** on-site and files a report.",
    "After verification, relief is **credited to your bank account** under SDRF norms.",
]
CLAIM_REFERENCE = (
    "**Where to go:** Patwari / Tehsil (Revenue) office  \n"
    "**Helplines:** 1077 & 1070  \n"
    "**HP SDMA:** hpsdma.nic.in"
)


def _nearest_by_distance(candidates: list[dict], district: str,
                          user_lat: float, user_lon: float, limit: int = 5) -> list[dict]:
    """Resolve each candidate's approximate coordinates (geocode by name +
    district, falling back to the district centre) and sort by haversine
    distance from the user — the same resolution ladder route_planning_agent
    uses in workflow.py, so distances stay consistent with the rest of the app.

    Each result carries its resolved "latitude"/"longitude" (in addition to
    "distance_km"/"approx") so callers can plot it on a map, not just show
    the distance as text.
    """
    fallback = _PD_DISTRICT_CENTERS.get(district.upper(), (user_lat, user_lon))
    resolved = []
    for c in candidates:
        name = c.get("name") or "facility"
        geo = geocode_place(f"{name}, {district.title()}, Himachal Pradesh, India")
        lat, lon = geo if geo else fallback
        resolved.append({
            **c,
            "latitude": lat,
            "longitude": lon,
            "distance_km": round(_haversine(user_lat, user_lon, lat, lon), 1),
            "approx": geo is None,
        })
    resolved.sort(key=lambda x: x["distance_km"])
    return resolved[:limit]


def nearest_hospitals(district: str, user_lat: float, user_lon: float, limit: int = 5) -> list[dict]:
    """Real hospitals from the NHP directory (ChromaDB), sorted by distance."""
    candidates = query_hospitals(district, need_type="emergency", n_results=max(limit, 6))
    return _nearest_by_distance(candidates, district, user_lat, user_lon, limit)


def nearest_shelters(district: str, user_lat: float, user_lon: float, limit: int = 5) -> list[dict]:
    """Real shelters (DAY-NULM + govt schools), sorted by distance."""
    candidates = query_shelters(district, n_results=max(limit, 6))
    return _nearest_by_distance(candidates, district, user_lat, user_lon, limit)


def _tel_pill(label: str, number: str) -> str:
    return (
        f"<a href='tel:{number}' style='text-decoration:none'>"
        f"<span style='background:#dc2626;color:white;padding:6px 14px;"
        f"border-radius:20px;font-size:13px;font-weight:600;margin-right:8px;"
        f"display:inline-block;margin-bottom:6px'>📞 {label}: {number}</span></a>"
    )


def _seed_alert_blocks() -> list[dict]:
    return [
        {"type": "assistant_text", "content": t(
            "🚨 **Disaster alert acknowledged.** Let's start with immediate safety, "
            "then work through the property damage.\n\n"
            "**Is anyone injured or trapped right now?**"
        )},
    ]


def _seed_idle_blocks() -> list[dict]:
    return [
        {"type": "assistant_text", "content": t(
            "👋 Ask me anything about HP hospitals, shelters, river/GLOF monitoring, "
            "or disaster guidance. Need help with **property damage**? Tap the "
            "button below."
        )},
    ]


def render_hp_assistant(agent_state: dict | None, get_llm, language: str = "English") -> None:
    """Render the single, unified HP Disaster Assistant chat.

    One transcript, one input surface, two jobs: grounded free-text Q&A over
    ingested HP disaster data, and a guided, button-driven property-damage
    flow that opens automatically once an incident/alert (`agent_state`) is
    reported. `get_llm` is injected (same DI pattern as `answer()`) so this
    module doesn't need a hard import of the LangGraph workflow. `language`
    controls both the free-text Q&A answers and the guided flow's on-screen
    text (via i18n.t) — each transcript entry is translated at the moment
    it's added, so past messages don't retroactively change language.
    """
    if st.session_state.get("hp_chat") is None:
        if agent_state:
            st.session_state.hp_chat = {"blocks": _seed_alert_blocks(), "mode": "ask_injury"}
        else:
            st.session_state.hp_chat = {"blocks": _seed_idle_blocks(), "mode": "chat"}
    hp = st.session_state.hp_chat

    for block in hp["blocks"]:
        _render_hp_block(block)

    district = (agent_state or {}).get("district", "")
    user_lat = (agent_state or {}).get("latitude")
    user_lon = (agent_state or {}).get("longitude")
    mode = hp["mode"]

    if mode == "ask_injury":
        c1, c2 = st.columns(2)
        if c1.button(t("🚑 Yes — injured / trapped"), use_container_width=True, key="hp_injury_yes"):
            hp["blocks"].append({"type": "user_text", "content": t("Yes — injured / trapped")})
            hp["blocks"].append({"type": "emergency_numbers"})
            with st.spinner(t("📍 Finding nearest hospitals...")):
                hospitals = nearest_hospitals(district, user_lat, user_lon)
            hp["blocks"].append({"type": "hospitals", "items": hospitals})
            hp["blocks"].append({"type": "assistant_text", "content": t(
                "**Call the numbers above now** and head to the nearest hospital, or "
                "wait for help if it's not safe to move. Let's also check the "
                "property damage.\n\n**How badly is the property damaged?**"
            )})
            hp["mode"] = "ask_severity"
            st.rerun()
        if c2.button(t("No injuries reported"), use_container_width=True, key="hp_injury_no"):
            hp["blocks"].append({"type": "user_text", "content": t("No injuries reported")})
            hp["blocks"].append({"type": "assistant_text", "content": t(
                "Good to hear. Let's check the property damage.\n\n"
                "**How badly is the property damaged?**"
            )})
            hp["mode"] = "ask_severity"
            st.rerun()

    elif mode == "ask_severity":
        c1, c2, c3, c4 = st.columns(4)
        if c1.button(t("🏚️ Fully damaged"), use_container_width=True, key="hp_sev_full"):
            hp["blocks"].append({"type": "user_text", "content": t("Fully damaged")})
            with st.spinner(t("📍 Finding nearest shelters...")):
                shelters = nearest_shelters(district, user_lat, user_lon)
            hp["blocks"].append({"type": "shelters", "items": shelters})
            hp["blocks"].append({"type": "compensation", "severity": "FULLY_DAMAGED"})
            hp["blocks"].append({"type": "claim_procedure"})
            hp["mode"] = "chat"
            st.rerun()
        if c2.button(t("🧱 Partially damaged"), use_container_width=True, key="hp_sev_partial"):
            hp["blocks"].append({"type": "user_text", "content": t("Partially damaged")})
            hp["blocks"].append({"type": "assistant_text",
                                 "content": t("**Is it safe for you to stay in the house tonight?**")})
            hp["mode"] = "ask_safe_tonight"
            st.rerun()
        if c3.button(t("🔧 Minor damage"), use_container_width=True, key="hp_sev_minor"):
            hp["blocks"].append({"type": "user_text", "content": t("Minor damage")})
            hp["blocks"].append({"type": "compensation", "severity": "MINOR"})
            hp["blocks"].append({"type": "claim_procedure"})
            hp["mode"] = "chat"
            st.rerun()
        if c4.button(t("✅ Not damaged"), use_container_width=True, key="hp_sev_none"):
            hp["blocks"].append({"type": "user_text", "content": t("Not damaged")})
            hp["blocks"].append({"type": "assistant_text", "content": t(
                "That's a relief — no compensation or claim process is needed. Stay "
                "alert to further disaster advisories and re-check the property once "
                "the situation clears."
            )})
            hp["mode"] = "chat"
            st.rerun()

    elif mode == "ask_safe_tonight":
        c1, c2 = st.columns(2)
        if c1.button(t("Yes — safe to stay"), use_container_width=True, key="hp_safe_yes"):
            hp["blocks"].append({"type": "user_text", "content": t("Yes — safe to stay tonight")})
            hp["blocks"].append({"type": "compensation", "severity": "PARTIALLY_DAMAGED"})
            hp["blocks"].append({"type": "claim_procedure"})
            hp["mode"] = "chat"
            st.rerun()
        if c2.button(t("No — not safe"), use_container_width=True, key="hp_safe_no"):
            hp["blocks"].append({"type": "user_text", "content": t("No — not safe to stay tonight")})
            with st.spinner(t("📍 Finding nearest shelters...")):
                shelters = nearest_shelters(district, user_lat, user_lon)
            hp["blocks"].append({"type": "shelters", "items": shelters})
            hp["blocks"].append({"type": "compensation", "severity": "PARTIALLY_DAMAGED"})
            hp["blocks"].append({"type": "claim_procedure"})
            hp["mode"] = "chat"
            st.rerun()

    else:
        # No guided question pending — free-text Q&A, with a manual way to
        # (re)start the property-damage flow from the same input surface.
        with st.form("hp_chat_form", clear_on_submit=True):
            user_q = st.text_input(
                t("Your question"),
                placeholder=t("e.g. Which hospitals are in Kullu? Or ask about property damage help."),
                label_visibility="collapsed",
            )
            col_send, col_pd, col_clear = st.columns([2, 2, 1])
            send = col_send.form_submit_button(t("Ask ➤"), use_container_width=True)
            start_pd = col_pd.form_submit_button(t("🏚️ Property damage help"), use_container_width=True)
            clear = col_clear.form_submit_button(t("🗑️ Clear"), use_container_width=True)

        if clear:
            st.session_state.hp_chat = None
            st.rerun()

        if start_pd:
            hp["blocks"].append({"type": "user_text", "content": t("I need help with property damage")})
            hp["blocks"].append({"type": "assistant_text",
                                 "content": t("Sure — **is anyone injured or trapped right now?**")})
            hp["mode"] = "ask_injury"
            st.rerun()

        if send and user_q.strip():
            hp["blocks"].append({"type": "user_text", "content": user_q})
            with st.spinner(t("🔎 Searching HP disaster data...")):
                result = answer(user_q, get_llm, language)
            hp["blocks"].append({
                "type": "assistant_text",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })
            st.rerun()


def _render_hp_block(block: dict) -> None:
    kind = block["type"]

    if kind == "assistant_text":
        with st.chat_message("assistant"):
            st.markdown(block["content"])
            if block.get("sources"):
                st.caption("📚 Sources: " + ", ".join(block["sources"]))

    elif kind == "user_text":
        with st.chat_message("user"):
            st.markdown(block["content"])

    elif kind == "emergency_numbers":
        with st.chat_message("assistant"):
            st.markdown(t("**📞 Emergency numbers — tap to call:**"))
            st.markdown(
                "".join(_tel_pill(label, number) for label, number in HP_EMERGENCY_NUMBERS.items()),
                unsafe_allow_html=True,
            )

    elif kind == "hospitals":
        with st.chat_message("assistant"):
            items = block["items"]
            if not items:
                st.warning(t("No hospitals found in the local directory for this district — "
                             "call 1077 (district control room) for assistance."))
            else:
                st.markdown(t("**🏥 Nearest hospitals ({n}, sorted by distance):**", n=len(items)))
                for h in items:
                    approx = t(" (approx. location)") if h.get("approx") else ""
                    st.markdown(
                        f"- **{h.get('name', 'N/A')}** — ≈{h.get('distance_km', 'N/A')} km{approx} · "
                        f"📞 {h.get('contact', 'N/A')}"
                    )

    elif kind == "shelters":
        with st.chat_message("assistant"):
            items = block["items"]
            if not items:
                st.warning(t("No shelters found in the local directory for this district — "
                             "call 1077 (district control room) for the nearest relief camp."))
            else:
                st.markdown(t("**🏠 Nearest shelters ({n}, sorted by distance):**", n=len(items)))
                for s in items:
                    approx = t(" (approx. location)") if s.get("approx") else ""
                    cap = t(" · Capacity: {v}", v=s['capacity']) if s.get("capacity") else ""
                    st.markdown(f"- **{s.get('name', 'N/A')}** — ≈{s.get('distance_km', 'N/A')} km{approx}{cap}")

    elif kind == "compensation":
        rate = RELIEF_RATES[block["severity"]]
        with st.chat_message("assistant"):
            st.markdown(f"**💰 Approximate relief — {t(rate['label'])}: {rate['display']}**")
            st.caption(t("Source: {v}", v=t(RELIEF_RATES_SOURCE)))
            st.warning(
                t("⚠️ This amount is **approximate** and is finalized only after **official "
                  "verification** by the Patwari / Revenue department. Do not treat this as a "
                  "guaranteed payout.")
            )

    elif kind == "claim_procedure":
        with st.chat_message("assistant"):
            st.markdown(t("**📋 Claim procedure (informational):**"))
            for i, step_text in enumerate(CLAIM_PROCEDURE_STEPS, 1):
                st.markdown(f"{i}. {t(step_text)}")
            st.info(t(CLAIM_REFERENCE))
