"""
Self-provisioning for the ChromaDB RAG vector store.

The retrieval collections (hospitals, schools, shelters, CWC stations, glacial
lakes, disaster knowledge) are built by ``scripts/ingest.py``. Rather than make
that a manual setup step, this module lets the app/workflow build the store on
first use: if any required collection is missing or empty, it runs the full
ingestion exactly once. When the store is already complete every check is a
cheap no-op, so it is safe to call on every startup.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    import chromadb
except Exception:  # pragma: no cover - chromadb optional at import time
    chromadb = None

from resq_project.config import (
    CHROMA_DIR,
    COLLECTION_HOSPITALS, COLLECTION_SHELTERS, COLLECTION_SCHOOLS,
    COLLECTION_CWC, COLLECTION_KNOWLEDGE, COLLECTION_GLACIAL,
)

# Collections the app relies on for grounded retrieval.
REQUIRED_COLLECTIONS = (
    COLLECTION_HOSPITALS, COLLECTION_SCHOOLS, COLLECTION_SHELTERS,
    COLLECTION_CWC, COLLECTION_GLACIAL, COLLECTION_KNOWLEDGE,
)


def missing_collections() -> list[str]:
    """Required collections that are absent or empty (so retrieval would fall
    back to stub data). Returns everything if ChromaDB can't be opened."""
    if chromadb is None:
        return list(REQUIRED_COLLECTIONS)
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        counts = {}
        for col in client.list_collections():
            try:
                counts[col.name] = client.get_collection(col.name).count()
            except Exception:
                counts[col.name] = 0
    except Exception:
        return list(REQUIRED_COLLECTIONS)
    return [name for name in REQUIRED_COLLECTIONS if counts.get(name, 0) <= 0]


def vector_store_ready() -> bool:
    """True when every required collection exists and is non-empty."""
    return not missing_collections()


def _load_ingest_module():
    """Import ``scripts/ingest.py`` by path (it lives outside the package)."""
    ingest_path = Path(__file__).resolve().parents[2] / "scripts" / "ingest.py"
    spec = importlib.util.spec_from_file_location("resq_ingest", ingest_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ingestion script at {ingest_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_vector_store(force: bool = False) -> list[str]:
    """Build any missing/empty RAG collections, running ingestion if needed.

    Returns the list of collections that were (re)built — empty when the store
    was already complete, making this cheap to call on every app start.
    Set ``force=True`` to rebuild everything from source.
    """
    missing = list(REQUIRED_COLLECTIONS) if force else missing_collections()
    if not missing:
        return []
    ingest = _load_ingest_module()
    ingest.run_ingestion(overwrite=force)
    return missing
