"""
HP Disaster Relief Resource Matching Agent
Configuration & Constants
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[2]
PRIMARY_DATA_DIR = BASE_DIR / "data"
FALLBACK_DATA_DIR = BASE_DIR / "source_data"
CHROMA_DIR     = BASE_DIR / "chroma_db"


def resolve_data_file(filename: str, prefer_primary: bool = False) -> Path:
    """Resolve a data file from data/ or source_data/ depending on where it exists."""
    primary_path = PRIMARY_DATA_DIR / filename
    fallback_path = FALLBACK_DATA_DIR / filename

    if prefer_primary and primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    return primary_path


# Source data files
HOSPITAL_CSV        = resolve_data_file("himachal_hospitals_289.csv")
SCHOOL_PDF          = resolve_data_file("SCHOOL_GOVT_MARCH_2021.pdf")
SHELTER_PDF         = resolve_data_file("Shelter Info.pdf")
SHELTER_CSV         = resolve_data_file("hp_shelters.csv", prefer_primary=True)   # optional converted file
CWC_EXCEL           = resolve_data_file("TableViewStationForecastData.xlsx")
LANDSLIDE_PDF       = resolve_data_file("Landslide Inventory Mapping (Post Monsoon for Himachal Pradesh) -2023.pdf")
BLOCKED_ROADS_CSV   = resolve_data_file("hp_blocked_corridors.csv", prefer_primary=True)
EMERGENCY_CONTACTS  = resolve_data_file("hp_emergency_contacts.json", prefer_primary=True)
GLACIAL_LAKES_CSV   = resolve_data_file("hp_glacial_lakes.csv")   # CWC GLOF monitoring (Sep 2025)
WILDFIRE_CSV        = resolve_data_file("Past_Data_For_Wildfire_detection_HP.csv")  # VIIRS fire hotspots
NEEDS_CSV           = resolve_data_file("needs.csv", prefer_primary=True)       # volunteer/relief needs
RESOURCES_CSV       = resolve_data_file("resources.csv", prefer_primary=True)   # volunteer/relief resources
TWEETS_CSV          = resolve_data_file("disaster_tweets_sample.csv", prefer_primary=True)  # labelled tweet feed (Kaggle Disaster Tweets style)
APPROVALS_LOG       = BASE_DIR / "logs" / "approvals.jsonl"                     # human-in-loop audit log
DISPATCH_LEDGER     = BASE_DIR / "logs" / "dispatch_ledger.jsonl"               # approved dispatches → inventory decrements

# ── Runtime Configuration ──────────────────────────────────────────────
LLM_PROVIDER         = os.getenv("LLM_PROVIDER", "ollama")   # ollama | openai | anthropic | grok | gemini
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OPENAI_MODEL         = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL      = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
XAI_BASE_URL         = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
GROK_MODEL           = os.getenv("GROK_MODEL", "grok-4.5")
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
ORS_API_KEY          = os.getenv("ORS_API_KEY", "")          # openrouteservice.org
AGENT_COORDINATOR_EMAIL = os.getenv("AGENT_COORDINATOR_EMAIL", "demo.coordinator@example.com")

# ── Embedding Model ────────────────────────────────────────────────────
# Using sentence-transformers as required
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim

# ── ChromaDB Collections ───────────────────────────────────────────────
COLLECTION_HOSPITALS  = "hp_hospitals"
COLLECTION_SHELTERS   = "hp_shelters"
COLLECTION_SCHOOLS    = "hp_schools"
COLLECTION_CWC        = "hp_cwc_stations"
COLLECTION_KNOWLEDGE  = "hp_disaster_knowledge"   # landslide PDF + NDMA guidelines
COLLECTION_GLACIAL    = "hp_glacial_lakes"        # CWC GLOF monitoring (Sep 2025)

# ── LangGraph LLM ─────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.1

# ── HP Districts & Risk Tiers (from HIMCOSTE 2023 Landslide Inventory) ─
DISTRICT_RISK = {
    "KANGRA":         {"landslides_2023": 4027, "tier": "CRITICAL",  "key_rivers": ["Beas", "Banganga"]},
    "MANDI":          {"landslides_2023": 2169, "tier": "CRITICAL",  "key_rivers": ["Beas", "Uhl", "Suketi"]},
    "SOLAN":          {"landslides_2023": 1930, "tier": "HIGH",      "key_rivers": ["Ashwini", "Ghamber"]},
    "HAMIRPUR":       {"landslides_2023": 1666, "tier": "HIGH",      "key_rivers": ["Beas", "Swan"]},
    "SIRMOUR":        {"landslides_2023": 1395, "tier": "HIGH",      "key_rivers": ["Giri", "Yamuna", "Tons"]},
    "KULLU":          {"landslides_2023": 729,  "tier": "MEDIUM",    "key_rivers": ["Beas", "Parvati", "Sainj"]},
    "SHIMLA":         {"landslides_2023": 483,  "tier": "MEDIUM",    "key_rivers": ["Satluj", "Pabbar"]},
    "CHAMBA":         {"landslides_2023": 534,  "tier": "MEDIUM",    "key_rivers": ["Ravi", "Chenab"]},
    "BILASPUR":       {"landslides_2023": 442,  "tier": "MEDIUM",    "key_rivers": ["Satluj", "Swan"]},
    "UNA":            {"landslides_2023": 98,   "tier": "LOW",       "key_rivers": ["Swan", "Beas"]},
    "KINNAUR":        {"landslides_2023": 65,   "tier": "LOW",       "key_rivers": ["Satluj", "Baspa", "Spiti"]},
    "LAHUL AND SPITI":{"landslides_2023": 31,   "tier": "LOW",       "key_rivers": ["Chandra", "Bhaga", "Spiti"]},
}

# ── Blocked NH Corridors (from HIMCOSTE 2023 report) ──────────────────
BLOCKED_CORRIDORS = [
    {"road": "NH-154", "segment": "Kotrupi",           "district": "MANDI",   "risk": "chronic_annual",  "season": "July-Sept"},
    {"road": "NH-5",   "segment": "Nigulsari",         "district": "KINNAUR", "risk": "subsidence",      "season": "July-Aug"},
    {"road": "NH-154A","segment": "Khara Mukh-Chamba", "district": "CHAMBA",  "risk": "rockslide",       "season": "July-Aug"},
    {"road": "NH-3",   "segment": "Sissu",             "district": "LAHUL AND SPITI","risk": "slope_failure","season": "July-Sept"},
    {"road": "NH-5",   "segment": "Kiarighat-Solan",   "district": "SOLAN",   "risk": "road_widening",   "season": "June-Sept"},
    {"road": "NH-3",   "segment": "Rohtang Pass",      "district": "KULLU",   "risk": "snowfall_closure","season": "Oct-May"},
    {"road": "NH-21",  "segment": "Kullu-Manali Beas corridor", "district": "KULLU", "risk": "flash_flood","season": "July-Aug"},
]

# ── Disaster Types & Response Protocols ───────────────────────────────
DISASTER_TYPES = ["Flash Flood", "Landslide", "Cloudburst", "GLOF", "Wildfire", "Avalanche", "Drought", "Road Blockage"]

# ── IMD Alert Levels ───────────────────────────────────────────────────
IMD_ALERTS = {
    "RED":    "Extremely heavy rain (>204.5mm/day). Take Action. Evacuate low-lying areas.",
    "ORANGE": "Very heavy rain (115.6-204.4mm/day). Be Alert. Avoid river zones.",
    "YELLOW": "Heavy rain (64.5-115.5mm/day). Be Updated. Monitor conditions.",
    "GREEN":  "Normal conditions. No immediate risk.",
}

# ── Property Damage Assistant ──────────────────────────────────────────
# HP emergency helplines (verified). Kept as a config constant, not hardcoded
# inline, so the numbers can be updated in one place.
HP_EMERGENCY_NUMBERS = {
    "National Emergency":    "112",
    "District Control Room": "1077",
    "State Control Room":    "1070",
    "Ambulance":             "108",
}

# Approximate relief under the HP special relief package (announced July
# 2025). These rates are revised after major disasters — update here (or
# point RELIEF_RATES_SOURCE at a live/remote feed) rather than hardcoding
# the amounts anywhere else in the app.
RELIEF_RATES = {
    "FULLY_DAMAGED":     {"label": "Fully damaged house",     "amount": 700000, "display": "₹7,00,000"},
    "PARTIALLY_DAMAGED": {"label": "Partially damaged house", "amount": 100000, "display": "₹1,00,000"},
    "MINOR":             {"label": "Minor damage",            "amount": None,   "display": "Assessed on-site"},
}
RELIEF_RATES_SOURCE = "HP Special Relief Package, July 2025 — revised after major disasters; amounts are approximate until officially verified."

# ── Open-Meteo ─────────────────────────────────────────────────────────
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# ── OpenRouteService ───────────────────────────────────────────────────
ORS_BASE_URL = "https://api.openrouteservice.org/v2"

# ── NDMA SACHET — CAP disaster alert feed (Himachal Pradesh) ──────────
NDMA_HP_RSS_URL = "https://sachet.ndma.gov.in/cap_public_website/rss/rss_himachal.xml"
