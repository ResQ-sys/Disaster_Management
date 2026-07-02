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
DATA_DIR       = PRIMARY_DATA_DIR if PRIMARY_DATA_DIR.exists() else FALLBACK_DATA_DIR
CHROMA_DIR     = BASE_DIR / "chroma_db"

# Source data files
HOSPITAL_CSV        = DATA_DIR / "himachal_hospitals_289.csv"
SCHOOL_PDF          = DATA_DIR / "SCHOOL_GOVT_MARCH_2021.pdf"
SHELTER_PDF         = DATA_DIR / "Shelter Info.pdf"
SHELTER_CSV         = PRIMARY_DATA_DIR / "hp_shelters.csv"   # optional converted file
CWC_EXCEL           = DATA_DIR / "TableViewStationForecastData.xlsx"
LANDSLIDE_PDF       = DATA_DIR / "Landslide Inventory Mapping (Post Monsoon for Himachal Pradesh) -2023.pdf"
BLOCKED_ROADS_CSV   = DATA_DIR / "hp_blocked_corridors.csv"
EMERGENCY_CONTACTS  = DATA_DIR / "hp_emergency_contacts.json"

# ── API Keys (set in .env) ─────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENWEATHER_API_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
ORS_API_KEY          = os.getenv("ORS_API_KEY", "")          # openrouteservice.org

# ── Embedding Model ────────────────────────────────────────────────────
# Using sentence-transformers as required
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim

# ── ChromaDB Collections ───────────────────────────────────────────────
COLLECTION_HOSPITALS  = "hp_hospitals"
COLLECTION_SHELTERS   = "hp_shelters"
COLLECTION_SCHOOLS    = "hp_schools"
COLLECTION_CWC        = "hp_cwc_stations"
COLLECTION_KNOWLEDGE  = "hp_disaster_knowledge"   # landslide PDF + NDMA guidelines

# ── LangGraph LLM ─────────────────────────────────────────────────────
LLM_MODEL       = "gpt-4o-mini"    # cost-efficient; swap to gpt-4o for demo
LLM_TEMPERATURE = 0.1              # low temp = consistent, factual outputs

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
DISASTER_TYPES = ["Flash Flood", "Landslide", "Cloudburst", "GLOF", "Avalanche", "Drought", "Road Blockage"]

# ── IMD Alert Levels ───────────────────────────────────────────────────
IMD_ALERTS = {
    "RED":    "Extremely heavy rain (>204.5mm/day). Take Action. Evacuate low-lying areas.",
    "ORANGE": "Very heavy rain (115.6-204.4mm/day). Be Alert. Avoid river zones.",
    "YELLOW": "Heavy rain (64.5-115.5mm/day). Be Updated. Monitor conditions.",
    "GREEN":  "Normal conditions. No immediate risk.",
}

# ── OpenWeatherMap ─────────────────────────────────────────────────────
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5"

# ── OpenRouteService ───────────────────────────────────────────────────
ORS_BASE_URL = "https://api.openrouteservice.org/v2"
