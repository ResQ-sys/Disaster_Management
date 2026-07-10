"""
Grounded RAG chatbot for the HP Disaster Relief Agent.

Answers questions STRICTLY from the ingested Himachal Pradesh disaster data
(ChromaDB collections). Two layers guard against hallucination / out-of-scope:

  1. Relevance gate — retrieve across all collections; if nothing clears the
     cosine-distance threshold, refuse immediately WITHOUT calling the LLM.
  2. Grounded prompt — the LLM is instructed to answer ONLY from the retrieved
     context and to say "I don't know" if the context is insufficient.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from resq_project.config import (
    COLLECTION_HOSPITALS, COLLECTION_SHELTERS, COLLECTION_SCHOOLS,
    COLLECTION_CWC, COLLECTION_KNOWLEDGE, COLLECTION_GLACIAL,
)
from resq_project.tools import (
    _get_collection, query_cwc_stations, query_glacial_lakes,
    query_hospitals, query_knowledge,
)

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

SYSTEM_PROMPT = """You are the Himachal Pradesh Disaster Relief assistant.

Answer the user's QUESTION using ONLY the facts in the CONTEXT below. The
CONTEXT is retrieved from official HP disaster data sources.

STRICT RULES:
- Use ONLY information present in the CONTEXT. Never use outside or prior knowledge.
- Do NOT invent or guess names, numbers, phone contacts, coordinates, or statistics.
- If the CONTEXT does not clearly contain the answer, reply EXACTLY:
  "I don't know based on my Himachal Pradesh disaster data sources."
- Be concise, factual, and specific. Quote figures/names exactly as in the CONTEXT.
- Do not answer questions unrelated to Himachal Pradesh disaster response.
"""


def _tokens(text: str) -> set[str]:
    toks = set()
    for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(tok) <= 2:
            continue
        if tok.endswith("s") and len(tok) > 4:
            tok = tok[:-1]
        toks.add(tok)
    return toks


def _lexical_distance(question: str, doc: str) -> float:
    q = _tokens(question)
    d = _tokens(doc)
    if not q or not d:
        return 1.0
    overlap = len(q & d)
    if overlap == 0:
        return 1.0
    return max(0.0, 1 - (overlap / len(q)))


def _fallback_retrieve(question: str):
    q_lower = (question or "").lower()
    candidates = []

    if any(word in q_lower for word in ["hospital", "medical", "doctor"]):
        district = "KULLU" if "kullu" in q_lower else ""
        for row in query_hospitals(district or "KULLU", n_results=TOP_K_PER_COLLECTION):
            doc = (
                f"Hospital: {row.get('name')} | District: {row.get('district')} | "
                f"Type: {row.get('type')} | Contact: {row.get('contact')}"
            )
            candidates.append({
                "doc": doc,
                "distance": _lexical_distance(question, doc),
                "label": _SEARCH_COLLECTIONS[COLLECTION_HOSPITALS],
                "source": "NHP Hospitals (fallback)",
            })

    if any(word in q_lower for word in ["glof", "glacial", "lake", "spiti", "lahul"]):
        district = "LAHUL AND SPITI" if ("spiti" in q_lower or "lahul" in q_lower) else ""
        for row in query_glacial_lakes(district, 32.5, 77.5, n_results=TOP_K_PER_COLLECTION):
            doc = (
                f"Glacial lake {row.get('lake_id')} in {row.get('district')} | "
                f"Status: {row.get('status')} | River: {row.get('river')} | "
                f"Area change: {row.get('area_pct_change')}%"
            )
            candidates.append({
                "doc": doc,
                "distance": _lexical_distance(question, doc),
                "label": _SEARCH_COLLECTIONS[COLLECTION_GLACIAL],
                "source": "CWC Glacial Lakes (fallback)",
            })

    if any(word in q_lower for word in ["cwc", "river", "station", "monitoring"]):
        for row in query_cwc_stations("KULLU", river="Beas", n_results=TOP_K_PER_COLLECTION):
            doc = (
                f"CWC station {row.get('station')} | District: {row.get('district')} | "
                f"River: {row.get('river')} | Type: {row.get('site_type')}"
            )
            candidates.append({
                "doc": doc,
                "distance": _lexical_distance(question, doc),
                "label": _SEARCH_COLLECTIONS[COLLECTION_CWC],
                "source": "CWC River Stations (fallback)",
            })

    for doc in query_knowledge(question, n_results=TOP_K_PER_COLLECTION):
        candidates.append({
            "doc": doc,
            "distance": _lexical_distance(question, doc),
            "label": _SEARCH_COLLECTIONS[COLLECTION_KNOWLEDGE],
            "source": "Disaster Knowledge (fallback)",
        })

    kept = [c for c in candidates if c["distance"] <= KEEP_MAX_DISTANCE]
    kept.sort(key=lambda h: h["distance"])
    return kept[:MAX_CONTEXT_CHUNKS]


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
    hits = hits[:MAX_CONTEXT_CHUNKS]
    return hits or _fallback_retrieve(question)


def answer(question: str, get_llm) -> dict:
    """Return a grounded answer dict: {answer, grounded, sources, hits}.

    `get_llm` is a callable returning a chat model (injected to avoid a hard
    dependency on the workflow module).
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
        resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
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
