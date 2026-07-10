"""
Grounded RAG chatbot for the HP Disaster Relief Agent.

Answers questions STRICTLY from the ingested Himachal Pradesh disaster data
(ChromaDB collections). Two layers guard against hallucination / out-of-scope:

  1. Relevance gate — retrieve across all collections; if nothing clears the
     cosine-distance threshold, refuse immediately WITHOUT calling the LLM.
  2. Grounded prompt — the LLM is instructed to answer ONLY from the retrieved
     context and to say "I don't know" if the context is insufficient.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from resq_project.config import (
    COLLECTION_HOSPITALS, COLLECTION_SHELTERS, COLLECTION_SCHOOLS,
    COLLECTION_CWC, COLLECTION_KNOWLEDGE, COLLECTION_GLACIAL,
)
from resq_project.tools import _get_collection

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
