"""
RAG Ingestion Pipeline
Processes all HP disaster data sources → ChromaDB collections
Uses: sentence-transformers (all-MiniLM-L6-v2) + ChromaDB

Sources ingested:
  1. himachal_hospitals_289.csv        → hp_hospitals
  2. SCHOOL_GOVT_MARCH_2021.pdf        → hp_schools  (govt only, shelter proxy)
  3. Shelter Info.pdf + curated values → hp_shelters
  4. TableViewStationForecastData.xlsx → hp_cwc_stations
  5. Landslide PDF + NDMA text         → hp_disaster_knowledge
"""

import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from resq_project.config import (
    CHROMA_DIR, HOSPITAL_CSV, SCHOOL_PDF,
    CWC_EXCEL, LANDSLIDE_PDF, GLACIAL_LAKES_CSV, EMBEDDING_MODEL,
    COLLECTION_HOSPITALS, COLLECTION_SHELTERS, COLLECTION_SCHOOLS,
    COLLECTION_CWC, COLLECTION_KNOWLEDGE, COLLECTION_GLACIAL,
    DISTRICT_RISK, BLOCKED_CORRIDORS
)


# ── ChromaDB client (persistent) ───────────────────────────────────────
def get_chroma_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


# ── Sentence-Transformers Embedding Function ───────────────────────────
def get_embedding_fn():
    """Returns ChromaDB-compatible sentence-transformers embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_or_create_collection(client, name: str, overwrite: bool = False):
    ef = get_embedding_fn()
    if overwrite:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )


# ══════════════════════════════════════════════════════════════════════
# 1. HOSPITALS
# ══════════════════════════════════════════════════════════════════════
def ingest_hospitals(client, overwrite=False):
    logger.info("Ingesting hospitals...")
    col = get_or_create_collection(client, COLLECTION_HOSPITALS, overwrite)

    if col.count() > 0 and not overwrite:
        logger.info(f"  → Skipping: {col.count()} hospitals already loaded")
        return

    df = pd.read_csv(HOSPITAL_CSV)
    df = df.fillna("")

    documents, metadatas, ids = [], [], []

    for _, row in df.iterrows():
        # Construct searchable text document
        doc = (
            f"Hospital: {row['Hospital Name']} | "
            f"District: {row['District']} | "
            f"Type: {row['Hospital Type']} | "
            f"Contact: {row['Hospital Contact']} | "
            f"Specialities: {row['Specialities']}"
        )
        meta = {
            "hospital_id":   str(row['Hospital Id']),
            "name":          str(row['Hospital Name']),
            "district":      str(row['District']).upper(),
            "type":          str(row['Hospital Type']),
            "contact":       str(row['Hospital Contact']),
            "specialities":  str(row['Specialities']),
            "source":        "NHP_HP_Hospital_Directory",
        }
        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"hospital_{row['Hospital Id']}")

    # Batch upsert
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        col.upsert(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    logger.success(f"  ✓ {col.count()} hospitals ingested")


# ══════════════════════════════════════════════════════════════════════
# 2. GOVERNMENT SCHOOLS (shelter proxies)
# ══════════════════════════════════════════════════════════════════════
def ingest_schools(client, overwrite=False):
    logger.info("Ingesting government schools (shelter proxies)...")
    col = get_or_create_collection(client, COLLECTION_SCHOOLS, overwrite)

    if col.count() > 0 and not overwrite:
        logger.info(f"  → Skipping: {col.count()} schools already loaded")
        return

    reader = PdfReader(str(SCHOOL_PDF))
    documents, metadatas, ids = [], [], []

    school_id = 0
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = text.split('\n')

        current_district = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract district from lines
            district_match = re.search(
                r'(Bilaspur|Chamba|Hamirpur|Kangra|Kinnaur|Kullu|Mandi|Shimla|Sirmour|Solan|Una|Lahul)',
                line, re.IGNORECASE
            )
            if district_match:
                current_district = district_match.group(1).upper()

            # Only capture government schools (used as emergency shelters)
            if 'GOVT' in line.upper() and ('SR SEC' in line.upper() or 'HIGH SCHOOL' in line.upper() or 'MIDDLE' in line.upper()):
                # Extract school name from line
                school_name_match = re.search(r'GOVT[\w\s&.()/-]+?(?=\s*-\s*\w+\s+\d{10}|\s*$)', line, re.IGNORECASE)
                school_name = school_name_match.group(0).strip() if school_name_match else line[:80]

                # Extract contact
                contact_match = re.search(r'\d{10}', line)
                contact = contact_match.group(0) if contact_match else "N/A"

                # Extract place/village name
                place_match = re.search(r'\d{4}\s*-\s*([\w\s]+?)\s*-\s*GOVT', line, re.IGNORECASE)
                place = place_match.group(1).strip() if place_match else ""

                if current_district and school_name:
                    doc = (
                        f"Government School (Emergency Shelter): {school_name} | "
                        f"Place: {place} | "
                        f"District: {current_district} | "
                        f"Contact: {contact} | "
                        f"Type: Government School - Emergency Shelter Proxy"
                    )
                    meta = {
                        "name":     school_name[:200],
                        "district": current_district,
                        "place":    place,
                        "contact":  contact,
                        "type":     "GOVT_SCHOOL_SHELTER",
                        "source":   "HP_Education_Dept_March_2021",
                    }
                    documents.append(doc)
                    metadatas.append(meta)
                    ids.append(f"school_{school_id:05d}")
                    school_id += 1

    # Batch upsert
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        col.upsert(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    logger.success(f"  ✓ {col.count()} government schools ingested as shelter proxies")


# ══════════════════════════════════════════════════════════════════════
# 3. NULM SHELTERS (from context document)
# ══════════════════════════════════════════════════════════════════════
def ingest_shelters(client, overwrite=False):
    logger.info("Ingesting DAY-NULM shelters...")
    col = get_or_create_collection(client, COLLECTION_SHELTERS, overwrite)

    if col.count() > 0 and not overwrite:
        logger.info(f"  → Skipping: {col.count()} shelters already loaded")
        return

    # Hardcoded from the DAY-NULM document (context provided)
    # Cities with actual NULM shelter capacity > 0
    nulm_shelters = [
        {"city": "Baddi",        "district": "SOLAN",   "nulm_capacity": 40,  "total_shelters": 1},
        {"city": "Bilaspur",     "district": "BILASPUR","nulm_capacity": 25,  "total_shelters": 1},
        {"city": "Chamba",       "district": "CHAMBA",  "nulm_capacity": 16,  "total_shelters": 1},
        {"city": "Dharamsala",   "district": "KANGRA",  "nulm_capacity": 50,  "total_shelters": 1},
        {"city": "Ghumarwin",    "district": "BILASPUR","nulm_capacity": 25,  "total_shelters": 1},
        {"city": "Hamirpur",     "district": "HAMIRPUR","nulm_capacity": 30,  "total_shelters": 1},
        {"city": "Jawalamukhi",  "district": "KANGRA",  "nulm_capacity": 12,  "total_shelters": 1},
        {"city": "Kullu",        "district": "KULLU",   "nulm_capacity": 25,  "total_shelters": 1},
        {"city": "Manali",       "district": "KULLU",   "nulm_capacity": 28,  "total_shelters": 1},
        {"city": "Mehatpur",     "district": "UNA",     "nulm_capacity": 25,  "total_shelters": 1},
        {"city": "Nahan",        "district": "SIRMOUR", "nulm_capacity": 15,  "total_shelters": 1},
        {"city": "Naina Devi",   "district": "BILASPUR","nulm_capacity": 25,  "total_shelters": 1},
        {"city": "Paonta Sahib", "district": "SIRMOUR", "nulm_capacity": 40,  "total_shelters": 1},
        {"city": "Parwanoo",     "district": "SOLAN",   "nulm_capacity": 38,  "total_shelters": 1},
        {"city": "Rohru",        "district": "SHIMLA",  "nulm_capacity": 20,  "total_shelters": 1},
        {"city": "Shimla Urban", "district": "SHIMLA",  "nulm_capacity": 180, "total_shelters": 2},
        {"city": "Solan",        "district": "SOLAN",   "nulm_capacity": 34,  "total_shelters": 1},
        {"city": "Talai",        "district": "HAMIRPUR","nulm_capacity": 15,  "total_shelters": 1},
        {"city": "Una",          "district": "UNA",     "nulm_capacity": 20,  "total_shelters": 1},
        # Non-NULM with capacity
        {"city": "Dehra",        "district": "KANGRA",  "nulm_capacity": 0,   "non_nulm_capacity": 32},
        {"city": "Nagrota Bagwan","district":"KANGRA",  "nulm_capacity": 0,   "non_nulm_capacity": 48},
        {"city": "Nalagarh",     "district": "SOLAN",   "nulm_capacity": 0,   "non_nulm_capacity": 25},
        {"city": "Kangra",       "district": "KANGRA",  "nulm_capacity": 0,   "non_nulm_capacity": 12},
    ]

    documents, metadatas, ids = [], [], []
    for i, s in enumerate(nulm_shelters):
        capacity = s.get("nulm_capacity", 0) + s.get("non_nulm_capacity", 0)
        doc = (
            f"Emergency Shelter: {s['city']} | "
            f"District: {s['district']} | "
            f"Capacity: {capacity} persons | "
            f"NULM Shelter | Source: DAY-NULM HP"
        )
        meta = {
            "city":          s["city"],
            "district":      s["district"],
            "capacity":      str(capacity),
            "nulm_capacity": str(s.get("nulm_capacity", 0)),
            "type":          "NULM_SHELTER",
            "source":        "DAY_NULM_HP",
        }
        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"shelter_{i:03d}")

    col.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.success(f"  ✓ {col.count()} NULM shelters ingested")


# ══════════════════════════════════════════════════════════════════════
# 4. CWC RIVER MONITORING STATIONS
# ══════════════════════════════════════════════════════════════════════
def ingest_cwc_stations(client, overwrite=False):
    logger.info("Ingesting CWC river monitoring stations...")
    col = get_or_create_collection(client, COLLECTION_CWC, overwrite)

    if col.count() > 0 and not overwrite:
        logger.info(f"  → Skipping: {col.count()} stations already loaded")
        return

    df = pd.read_excel(str(CWC_EXCEL))
    hp_df = df[df['State name'] == 'Himachal Pradesh'].copy()
    hp_df = hp_df.fillna("")

    documents, metadatas, ids = [], [], []

    for idx, row in hp_df.iterrows():
        doc = (
            f"CWC River Monitoring Station: {row['Station Name']} | "
            f"District: {row['District / Town']} | "
            f"River: {row['River Name']} | "
            f"Basin: {row['Basin Name']} | "
            f"Station Type: {row['Type Of Site']} | "
            f"Division: {row['Division Name']} | "
            f"Coordinates: {row['Latitude']:.4f}N, {row['longitude']:.4f}E"
        )
        meta = {
            "name":       str(row['Station Name']),
            "district":   str(row['District / Town']).upper(),
            "river":      str(row['River Name']),
            "basin":      str(row['Basin Name']),
            "site_type":  str(row['Type Of Site']),
            "division":   str(row['Division Name']),
            "latitude":   str(row['Latitude']),
            "longitude":  str(row['longitude']),
            "source":     "CWC_National_Flood_Forecast_Registry",
        }
        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"cwc_{idx}")

    col.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.success(f"  ✓ {col.count()} CWC stations ingested")


# ══════════════════════════════════════════════════════════════════════
# 4b. GLACIAL LAKES — GLOF monitoring (CWC Sept 2025)
# ══════════════════════════════════════════════════════════════════════
def ingest_glacial_lakes(client, overwrite=False):
    logger.info("Ingesting glacial lakes (GLOF monitoring)...")
    col = get_or_create_collection(client, COLLECTION_GLACIAL, overwrite)

    if col.count() > 0 and not overwrite:
        logger.info(f"  → Skipping: {col.count()} glacial lakes already loaded")
        return

    if not GLACIAL_LAKES_CSV.exists():
        logger.warning(f"  → {GLACIAL_LAKES_CSV} not found — run scripts/build_glacial_lakes_csv.py first")
        return

    df = pd.read_csv(GLACIAL_LAKES_CSV).fillna("")
    documents, metadatas, ids = [], [], []

    for i, row in df.iterrows():
        trend = str(row["status"]).upper()
        pct = row["area_pct_change"]
        doc = (
            f"Glacial Lake (GLOF monitoring): {row['lake_id']} | "
            f"District: {row['district']} | Basin: {row['basin']} | "
            f"River: {row['river'] or 'N/A'} | "
            f"Water-spread area trend: {trend} ({pct}% vs base year) | "
            f"Coordinates: {row['latitude']}N, {row['longitude']}E | "
            f"Monitored: {row['monitored_period']} | "
            f"NOTE: Based on previous-year monthly satellite monitoring, not real-time. "
            f"Expanding lakes indicate elevated GLOF (Glacial Lake Outburst Flood) risk. "
            f"Source: {row['source']}"
        )
        meta = {
            "lake_id":          str(row["lake_id"]),
            "district":         str(row["district"]).upper(),
            "basin":            str(row["basin"]),
            "river":            str(row["river"]),
            "latitude":         str(row["latitude"]),
            "longitude":        str(row["longitude"]),
            "area_pct_change":  str(pct),
            "status":           str(row["status"]),
            "monitored_period": str(row["monitored_period"]),
            "type":             "GLACIAL_LAKE_GLOF",
            "source":           str(row["source"]),
        }
        documents.append(doc)
        metadatas.append(meta)
        ids.append(f"glof_{row['lake_id']}")

    col.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.success(f"  ✓ {col.count()} glacial lakes ingested (GLOF monitoring)")


# ══════════════════════════════════════════════════════════════════════
# 5. DISASTER KNOWLEDGE BASE (Landslide PDF + Risk data)
# ══════════════════════════════════════════════════════════════════════
def ingest_knowledge_base(client, overwrite=False):
    logger.info("Ingesting disaster knowledge base...")
    col = get_or_create_collection(client, COLLECTION_KNOWLEDGE, overwrite)

    if col.count() > 0 and not overwrite:
        logger.info(f"  → Skipping: {col.count()} knowledge chunks already loaded")
        return

    documents, metadatas, ids = [], [], []
    chunk_id = 0

    # ── 5a. Landslide PDF chunks ────────────────────────────────────
    if LANDSLIDE_PDF.exists():
        reader = PdfReader(str(LANDSLIDE_PDF))
        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "") + "\n"

        # Sentence-aware chunking (500 chars, 100 overlap)
        chunks = _chunk_text(full_text, chunk_size=500, overlap=100)
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            documents.append(chunk)
            metadatas.append({
                "source": "HIMCOSTE_Landslide_Inventory_2023",
                "type":   "landslide_report",
            })
            ids.append(f"know_{chunk_id:05d}")
            chunk_id += 1
        logger.info(f"  → {chunk_id} chunks from landslide PDF")

    # ── 5b. District risk tier knowledge ───────────────────────────
    for district, info in DISTRICT_RISK.items():
        doc = (
            f"District Risk Profile: {district} | "
            f"Landslides in 2023 monsoon: {info['landslides_2023']} | "
            f"Risk Tier: {info['tier']} | "
            f"Key Rivers: {', '.join(info['key_rivers'])} | "
            f"Source: HIMCOSTE Post-Monsoon Landslide Inventory 2023"
        )
        documents.append(doc)
        metadatas.append({
            "source":   "HIMCOSTE_District_Risk_Tiers",
            "type":     "risk_profile",
            "district": district,
        })
        ids.append(f"know_{chunk_id:05d}")
        chunk_id += 1

    # ── 5c. Blocked NH corridors ────────────────────────────────────
    for road in BLOCKED_CORRIDORS:
        doc = (
            f"Road Risk: {road['road']} at {road['segment']} | "
            f"District: {road['district']} | "
            f"Risk Type: {road['risk']} | "
            f"High-risk Season: {road['season']} | "
            f"WARNING: This route segment is prone to closure during monsoon. "
            f"Verify status before recommending. Source: HIMCOSTE 2023 Report"
        )
        documents.append(doc)
        metadatas.append({
            "source":   "HIMCOSTE_Blocked_Corridors",
            "type":     "road_risk",
            "district": road['district'],
            "road":     road['road'],
        })
        ids.append(f"know_{chunk_id:05d}")
        chunk_id += 1

    # ── 5d. HP Emergency Contacts knowledge ────────────────────────
    emergency_contacts = {
        "NDMA National": "1078",
        "HP Police": "100",
        "HP Ambulance": "108",
        "Fire HP": "101",
        "HPSDMA Control Room Shimla": "0177-2620131",
        "NDRF HP": "0172-2749165",
        "IGMC Shimla": "0177-2658585",
        "Dr RPGMC Tanda Kangra": "01892-267114",
        "Zonal Hospital Mandi": "01905-235243",
        "Zonal Hospital Kullu": "01902-222343",
    }
    for org, contact in emergency_contacts.items():
        doc = (
            f"Emergency Contact: {org} | "
            f"Number: {contact} | "
            f"Use in disaster escalation and resource coordination"
        )
        documents.append(doc)
        metadatas.append({"source": "HP_Emergency_Contacts", "type": "emergency_contact"})
        ids.append(f"know_{chunk_id:05d}")
        chunk_id += 1

    # ── 5e. IMD Alert guidance ──────────────────────────────────────
    imd_guidance = [
        ("RED Alert HP", "Extremely heavy rain >204.5mm/day. Evacuate flood plains and landslide zones. Activate NDRF. All river crossings dangerous."),
        ("ORANGE Alert HP", "Very heavy rain 115-204mm/day. Alert all district control rooms. Avoid river banks and hill slopes. Monitor CWC stations."),
        ("YELLOW Alert HP", "Heavy rain 64-115mm/day. Be updated. Restrict movement in Kangra, Mandi, Solan, Sirmour high-risk zones."),
        ("Monsoon HP", "HP monsoon June-September. Average 1334mm rainfall in 2023. Districts Kangra, Mandi most affected historically."),
    ]
    for title, text in imd_guidance:
        doc = f"IMD Weather Guidance: {title} | {text}"
        documents.append(doc)
        metadatas.append({"source": "IMD_HP_Guidelines", "type": "weather_guidance"})
        ids.append(f"know_{chunk_id:05d}")
        chunk_id += 1

    # Batch upsert
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        col.upsert(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    logger.success(f"  ✓ {col.count()} knowledge chunks ingested")


# ── Helper: text chunker ───────────────────────────────────────────────
def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Sentence-aware chunking with overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_sents, overlap_len = [], 0
            for s in reversed(current):
                if overlap_len + len(s) < overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_sents
            current_len = overlap_len
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Master ingestion entry point ───────────────────────────────────────
def run_ingestion(overwrite: bool = False):
    """Run full data pipeline. Set overwrite=True to rebuild from scratch."""
    logger.info("=" * 60)
    logger.info("HP Disaster Agent — Data Ingestion Pipeline")
    logger.info("=" * 60)

    client = get_chroma_client()

    ingest_hospitals(client, overwrite)
    ingest_schools(client, overwrite)
    ingest_shelters(client, overwrite)
    ingest_cwc_stations(client, overwrite)
    ingest_glacial_lakes(client, overwrite)
    ingest_knowledge_base(client, overwrite)

    # Summary
    logger.info("\n── Collection Summary ──")
    for name in [COLLECTION_HOSPITALS, COLLECTION_SCHOOLS,
                 COLLECTION_SHELTERS, COLLECTION_CWC, COLLECTION_GLACIAL,
                 COLLECTION_KNOWLEDGE]:
        try:
            col = client.get_collection(name, embedding_function=get_embedding_fn())
            logger.info(f"  {name}: {col.count()} documents")
        except Exception:
            logger.warning(f"  {name}: NOT FOUND")

    logger.success("\n✓ Ingestion complete. ChromaDB ready at: " + str(CHROMA_DIR))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Rebuild all collections")
    args = parser.parse_args()
    run_ingestion(overwrite=args.overwrite)
