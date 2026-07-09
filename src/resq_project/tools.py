"""
LangGraph Tool Functions
All external API calls and ChromaDB queries wrapped as callable tools.

Tools:
  get_weather()           → Open-Meteo current + forecast
  query_hospitals()       → ChromaDB nearest hospitals
  query_shelters()        → ChromaDB shelters + schools
  query_cwc_stations()    → CWC river monitoring nearest to location
  query_knowledge()       → RAG over landslide PDF + risk knowledge
  get_route()             → OpenRouteService driving directions
  check_road_risk()       → Blocked corridor risk check
  get_district_risk()     → HIMCOSTE district risk tier
"""

import math
import requests
from typing import Optional
from loguru import logger
import chromadb
from chromadb.utils import embedding_functions

from resq_project.config import (
    CHROMA_DIR, EMBEDDING_MODEL, ORS_API_KEY,
    COLLECTION_HOSPITALS, COLLECTION_SHELTERS, COLLECTION_SCHOOLS,
    COLLECTION_CWC, COLLECTION_KNOWLEDGE, COLLECTION_GLACIAL,
    OPEN_METEO_BASE_URL, ORS_BASE_URL,
    BLOCKED_CORRIDORS, DISTRICT_RISK, WILDFIRE_CSV
)


# ── Shared ChromaDB client (singleton) ────────────────────────────────
_chroma_client = None
_embedding_fn  = None

def _get_client():
    global _chroma_client, _embedding_fn
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _embedding_fn  = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _chroma_client, _embedding_fn

def _get_collection(name: str):
    client, ef = _get_client()
    return client.get_collection(name, embedding_function=ef)


# ══════════════════════════════════════════════════════════════════════
# TOOL 1 — WEATHER
# ══════════════════════════════════════════════════════════════════════
def get_weather(lat: float, lon: float, district: str = "") -> dict:
    """
    Fetch current weather + 24h forecast from Open-Meteo.
    Returns rainfall, temp, wind, and IMD-style alert level.
    Does not require any API key.
    """
    try:
        resp = requests.get(
            OPEN_METEO_BASE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,weather_code",
                "hourly": "precipitation",
                "forecast_days": 2,
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()

        current = payload.get("current", {})
        hourly = payload.get("hourly", {})
        precipitation = hourly.get("precipitation", [])
        rainfall_1h = precipitation[0] if precipitation else 0.0
        forecast_rain_24h = sum(float(value or 0.0) for value in precipitation[:24])
        temp = current.get("temperature_2m", 0.0)
        wind_speed = current.get("wind_speed_10m", 0.0)
        description = _describe_weather_code(current.get("weather_code"))

        # Derive IMD-style alert
        alert_level = _derive_imd_alert(forecast_rain_24h)

        return {
            "district":          district or f"{lat:.2f}N {lon:.2f}E",
            "temperature_c":     round(temp, 1),
            "wind_speed_kmh":    round(wind_speed * 3.6, 1),
            "current_rain_mmph": round(rainfall_1h, 2),
            "forecast_rain_24h": round(forecast_rain_24h, 1),
            "description":       description,
            "alert_level":       alert_level["level"],
            "alert_message":     alert_level["message"],
            "source":            "Open-Meteo",
        }

    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return _mock_weather(district)


def _derive_imd_alert(rain_24h: float) -> dict:
    """Map 24h rainfall to IMD alert level."""
    if rain_24h >= 204.5:
        return {"level": "RED",    "message": "Extremely heavy rain. Evacuate flood plains and landslide zones immediately."}
    elif rain_24h >= 115.6:
        return {"level": "ORANGE", "message": "Very heavy rain. Alert district control rooms. Avoid river banks and hill slopes."}
    elif rain_24h >= 64.5:
        return {"level": "YELLOW", "message": "Heavy rain. Be updated. Restrict movement in high-risk zones (Kangra, Mandi, Solan)."}
    else:
        return {"level": "GREEN",  "message": "Normal conditions. Monitor for changes."}


def _mock_weather(district: str) -> dict:
    """Mock weather fallback when weather API is unavailable."""
    return {
        "district":          district or "Unknown",
        "temperature_c":     22.5,
        "wind_speed_kmh":    18.0,
        "current_rain_mmph": 12.5,
        "forecast_rain_24h": 85.0,
        "description":       "Heavy rain (MOCK DATA)",
        "alert_level":       "YELLOW",
        "alert_message":     "Heavy rain. Mock data returned because live weather lookup failed.",
        "source":            "MOCK",
    }


def _describe_weather_code(code: Optional[int]) -> str:
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Severe thunderstorm with hail",
    }
    return weather_codes.get(code, "Weather conditions unavailable")


# ══════════════════════════════════════════════════════════════════════
# TOOL 2 — HOSPITALS (ChromaDB)
# ══════════════════════════════════════════════════════════════════════
def query_hospitals(district: str, need_type: str = "emergency", n_results: int = 5) -> list[dict]:
    """
    Query ChromaDB for hospitals in the given district.
    need_type: 'emergency', 'surgery', 'trauma', 'general'
    Returns list of hospital dicts.
    """
    try:
        col = _get_collection(COLLECTION_HOSPITALS)
        query_text = f"hospital {need_type} {district} Himachal Pradesh"

        results = col.query(
            query_texts=[query_text],
            n_results=min(n_results, col.count()),
            where={"district": {"$eq": district.upper()}}
        )

        hospitals = []
        if results["metadatas"] and results["metadatas"][0]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                hospitals.append({
                    "name":       meta.get("name"),
                    "district":   meta.get("district"),
                    "type":       meta.get("type"),
                    "contact":    meta.get("contact"),
                    "specialities": meta.get("specialities"),
                    "relevance":  round(1 - dist, 3),
                })

        # Fallback: if district filter returns 0, query without filter
        if not hospitals:
            results = col.query(
                query_texts=[query_text],
                n_results=min(n_results, col.count())
            )
            if results["metadatas"] and results["metadatas"][0]:
                for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                    hospitals.append({
                        "name":       meta.get("name"),
                        "district":   meta.get("district"),
                        "type":       meta.get("type"),
                        "contact":    meta.get("contact"),
                        "specialities": meta.get("specialities"),
                        "relevance":  round(1 - dist, 3),
                    })

        return hospitals

    except Exception as e:
        logger.error(f"Hospital query error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
# TOOL 3 — SHELTERS (NULM + Schools)
# ══════════════════════════════════════════════════════════════════════
def query_shelters(district: str, n_results: int = 5) -> list[dict]:
    """
    Query ChromaDB for shelters in district.
    Queries both NULM shelters and government schools (shelter proxies).
    """
    results = []

    # NULM shelters
    try:
        col = _get_collection(COLLECTION_SHELTERS)
        qr = col.query(
            query_texts=[f"emergency shelter {district} Himachal Pradesh"],
            n_results=min(3, col.count()),
            where={"district": {"$eq": district.upper()}}
        )
        if qr["metadatas"] and qr["metadatas"][0]:
            for meta in qr["metadatas"][0]:
                results.append({
                    "name":     f"NULM Shelter - {meta.get('city')}",
                    "district": meta.get("district"),
                    "capacity": meta.get("capacity"),
                    "type":     "NULM_SHELTER",
                    "contact":  "Contact District Collector for access",
                    "source":   "DAY-NULM HP",
                })
    except Exception as e:
        logger.error(f"Shelter query error: {e}")

    # Government schools as backup shelters
    try:
        col = _get_collection(COLLECTION_SCHOOLS)
        qr = col.query(
            query_texts=[f"government school shelter {district}"],
            n_results=min(n_results - len(results), col.count()),
            where={"district": {"$eq": district.upper()}}
        )
        if qr["metadatas"] and qr["metadatas"][0]:
            for meta in qr["metadatas"][0]:
                results.append({
                    "name":     meta.get("name"),
                    "district": meta.get("district"),
                    "capacity": "",
                    "type":     "GOVT_SCHOOL_SHELTER",
                    "contact":  "",
                    "source":   "HP Education Dept",
                })
    except Exception as e:
        logger.error(f"School shelter query error: {e}")

    return results[:n_results]


# ══════════════════════════════════════════════════════════════════════
# TOOL 4 — CWC STATIONS (river monitoring)
# ══════════════════════════════════════════════════════════════════════
def query_cwc_stations(district: str, river: str = "", n_results: int = 3) -> list[dict]:
    """
    Query CWC monitoring stations for a given district/river.
    Returns station details for real-time CWC forecast lookup.
    """
    try:
        col = _get_collection(COLLECTION_CWC)
        query_text = f"CWC river monitoring station {district} {river} flood level"

        # Try district-filtered query first
        try:
            qr = col.query(
                query_texts=[query_text],
                n_results=min(n_results, col.count()),
                where={"district": {"$eq": district.upper()}}
            )
        except Exception:
            qr = col.query(query_texts=[query_text], n_results=min(n_results, col.count()))

        stations = []
        if qr["metadatas"] and qr["metadatas"][0]:
            for meta in qr["metadatas"][0]:
                stations.append({
                    "station":    meta.get("name"),
                    "district":   meta.get("district"),
                    "river":      meta.get("river"),
                    "site_type":  meta.get("site_type"),
                    "latitude":   float(meta.get("latitude", 0)),
                    "longitude":  float(meta.get("longitude", 0)),
                    "cwc_url":    f"https://ffs.india.gov.in (search: {meta.get('name')})",
                    "source":     "CWC National Flood Forecast Registry",
                })
        return stations

    except Exception as e:
        logger.error(f"CWC query error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
# TOOL 5 — KNOWLEDGE BASE RAG
# ══════════════════════════════════════════════════════════════════════
def query_knowledge(query: str, n_results: int = 4) -> list[str]:
    """
    RAG query over the disaster knowledge base.
    Returns relevant text chunks from HIMCOSTE report, risk profiles, etc.
    """
    try:
        col = _get_collection(COLLECTION_KNOWLEDGE)
        qr = col.query(
            query_texts=[query],
            n_results=min(n_results, col.count())
        )
        chunks = []
        if qr["documents"] and qr["documents"][0]:
            for doc, meta in zip(qr["documents"][0], qr["metadatas"][0]):
                source = meta.get("source", "unknown")
                chunks.append(f"[{source}] {doc}")
        return chunks

    except Exception as e:
        logger.error(f"Knowledge query error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
# TOOL 6 — ROUTE PLANNING (OpenRouteService)
# ══════════════════════════════════════════════════════════════════════
def get_route(
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    profile: str = "driving-car"
) -> dict:
    """
    Get driving route from origin to destination using OpenRouteService.
    Profiles: 'driving-car', 'foot-walking'

    ORS is the sole routing source: if no API key is configured or the API
    call fails, this returns an error route (no distance/time) rather than a
    straight-line estimate.
    """
    if not ORS_API_KEY:
        logger.warning("ORS API key not set — cannot compute route.")
        return {
            "distance_km":  None,
            "duration_min": None,
            "turn_by_turn": [],
            "source":       "unavailable",
            "error":        "ORS_API_KEY not set — add it to .env to enable routing.",
        }

    try:
        url = f"{ORS_BASE_URL}/directions/{profile}"
        resp = requests.post(url,
            json={"coordinates": [[origin_lon, origin_lat], [dest_lon, dest_lat]]},
            headers={
                "Authorization": ORS_API_KEY,
                "Content-Type": "application/json"
            },
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        summary = data["routes"][0]["summary"]
        steps   = data["routes"][0]["segments"][0]["steps"]

        return {
            "distance_km":     round(summary["distance"] / 1000, 2),
            "duration_min":    round(summary["duration"] / 60, 1),
            "turn_by_turn":    [s["instruction"] for s in steps[:5]],
            "source":          "OpenRouteService",
        }

    except Exception as e:
        logger.error(f"ORS routing error: {e}")
        return {
            "distance_km":  None,
            "duration_min": None,
            "turn_by_turn": [],
            "source":       "error",
            "error":        f"OpenRouteService request failed: {e}",
        }


def straight_line_route(origin_lat, origin_lon, dest_lat, dest_lon) -> dict:
    """Approximate as-the-crow-flies route, used only as a fallback when ORS
    routing is unavailable. Distance is the straight-line (haversine) distance;
    time assumes ~30 km/h on HP mountain roads. Clearly labelled as approximate.
    """
    dist_km = _haversine(origin_lat, origin_lon, dest_lat, dest_lon)
    return {
        "distance_km":  round(dist_km, 1),
        "duration_min": round(dist_km / 30 * 60),   # ~30 km/h mountain roads
        "turn_by_turn": [],
        "source":       "straight_line_approx",
    }


# Simple in-process cache so we don't re-hit Nominatim for the same place.
_GEOCODE_CACHE: dict[str, Optional[tuple]] = {}


def geocode_place(query: str) -> Optional[tuple]:
    """Geocode a free-text place to (lat, lon) via Nominatim (OpenStreetMap).

    Returns None if the query cannot be resolved or geocoding is unavailable.
    Nominatim's usage policy asks for a descriptive user-agent and low volume,
    both of which we satisfy (single lookup per request, cached).
    """
    if not query or not query.strip():
        return None

    key = query.strip().lower()
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]

    result = None
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="resq_disaster_agent")
        loc = geolocator.geocode(query, timeout=10)
        if loc:
            result = (round(loc.latitude, 4), round(loc.longitude, 4))
    except Exception as e:
        logger.warning(f"geocode_place failed for {query!r}: {e}")

    _GEOCODE_CACHE[key] = result
    return result


# ══════════════════════════════════════════════════════════════════════
# TOOL 7 — ROAD RISK CHECK
# ══════════════════════════════════════════════════════════════════════
def check_road_risk(district: str, month: int = None) -> list[dict]:
    """
    Check if any known blocked NH corridors are in the given district.
    Source: HIMCOSTE 2023 Landslide Inventory.
    """
    import datetime
    if month is None:
        month = datetime.datetime.now().month

    risky = []
    monsoon_months = {6, 7, 8, 9}

    for corridor in BLOCKED_CORRIDORS:
        if corridor["district"].upper() == district.upper():
            is_active_season = month in monsoon_months
            risky.append({
                "road":       corridor["road"],
                "segment":    corridor["segment"],
                "risk_type":  corridor["risk"],
                "season":     corridor["season"],
                "currently_risky": is_active_season,
                "warning":    f"⚠ {corridor['road']} at {corridor['segment']} is prone to blockage. "
                              f"Verify status before travel. Source: HIMCOSTE 2023.",
            })
    return risky


# ══════════════════════════════════════════════════════════════════════
# TOOL 8 — DISTRICT RISK TIER
# ══════════════════════════════════════════════════════════════════════
def get_district_risk(district: str) -> dict:
    """
    Return district risk tier from HIMCOSTE 2023 landslide inventory.
    """
    district_upper = district.upper()
    info = DISTRICT_RISK.get(district_upper, {})
    if info:
        return {
            "district":        district_upper,
            "risk_tier":       info["tier"],
            "landslides_2023": info["landslides_2023"],
            "key_rivers":      info["key_rivers"],
            "source":          "HIMCOSTE_Landslide_Inventory_2023",
        }
    return {
        "district":  district_upper,
        "risk_tier": "UNKNOWN",
        "message":   "District not found in risk database",
    }


# ══════════════════════════════════════════════════════════════════════
# TOOL 9 — NEAREST RESOURCE (spatial)
# ══════════════════════════════════════════════════════════════════════
def find_nearest_cwc_station(user_lat: float, user_lon: float) -> dict:
    """
    Find closest CWC river monitoring station to user's coordinates.
    Uses geopy-style haversine distance.
    """
    try:
        col = _get_collection(COLLECTION_CWC)
        all_data = col.get(include=["metadatas"])
        
        nearest, min_dist = None, float("inf")
        for meta in all_data["metadatas"]:
            try:
                slat = float(meta.get("latitude", 0))
                slon = float(meta.get("longitude", 0))
                dist = _haversine(user_lat, user_lon, slat, slon)
                if dist < min_dist:
                    min_dist = dist
                    nearest = {**meta, "distance_km": round(dist, 2)}
            except Exception:
                continue

        return nearest or {"error": "No CWC stations found"}

    except Exception as e:
        logger.error(f"Nearest CWC station error: {e}")
        return {"error": str(e)}


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ══════════════════════════════════════════════════════════════════════
# TOOL 10 — GLACIAL LAKES (GLOF monitoring, CWC Sept 2025)
# ══════════════════════════════════════════════════════════════════════
def query_glacial_lakes(district: str = "", user_lat: float = None,
                        user_lon: float = None, n_results: int = 5) -> list[dict]:
    """
    Return HP glacial lakes relevant to a GLOF assessment.

    Prioritises lakes in the user's district, then by proximity to the user's
    coordinates. Each lake carries its water-spread-area trend (increase =
    elevated GLOF risk). Data is from CWC's previous-year monthly satellite
    monitoring (September 2025), NOT real-time.
    """
    try:
        col = _get_collection(COLLECTION_GLACIAL)
        all_data = col.get(include=["metadatas"])
        lakes = []
        for meta in all_data["metadatas"]:
            try:
                llat = float(meta.get("latitude", 0))
                llon = float(meta.get("longitude", 0))
            except (TypeError, ValueError):
                llat = llon = 0.0
            dist = None
            if user_lat is not None and user_lon is not None and llat and llon:
                dist = round(_haversine(user_lat, user_lon, llat, llon), 1)
            lakes.append({
                "lake_id":         meta.get("lake_id"),
                "district":        meta.get("district"),
                "basin":           meta.get("basin"),
                "river":           meta.get("river"),
                "latitude":        llat,
                "longitude":       llon,
                "status":          meta.get("status"),
                "area_pct_change": meta.get("area_pct_change"),
                "monitored_period": meta.get("monitored_period"),
                "distance_km":     dist,
                "source":          meta.get("source"),
            })

        d = (district or "").upper()

        def sort_key(lk):
            same_district = 0 if lk["district"] == d else 1
            increasing = 0 if lk["status"] == "increase" else 1
            proximity = lk["distance_km"] if lk["distance_km"] is not None else 9e9
            return (same_district, increasing, proximity)

        lakes.sort(key=sort_key)
        return lakes[:n_results]

    except Exception as e:
        logger.error(f"Glacial lake query error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
# TOOL 11 — WILDFIRE PRONENESS (historical VIIRS fire hotspots)
# ══════════════════════════════════════════════════════════════════════
_wildfire_pts = None   # cached (lat_array, lon_array)


def _load_wildfire_points():
    """Load & cache the historical fire-hotspot coordinates as numpy arrays."""
    global _wildfire_pts
    if _wildfire_pts is None:
        import pandas as pd
        df = pd.read_csv(WILDFIRE_CSV, usecols=["Lat", "Lon"]).dropna()
        _wildfire_pts = (df["Lat"].to_numpy(dtype=float),
                         df["Lon"].to_numpy(dtype=float))
    return _wildfire_pts


def assess_wildfire_risk(user_lat: float, user_lon: float) -> dict:
    """
    Flag whether a location (lat/lon) is prone to wildfires, based on the
    density of historical satellite fire detections (VIIRS) nearby.

    Levels by count of past fire hotspots within 10 km:
        HIGH     >= 100   |  MODERATE 25-99  |  LOW 1-24  |  MINIMAL 0
    'prone' is True for MODERATE and HIGH.

    NOTE: Based on past-year historical satellite fire-detection points,
    not a live fire feed.
    """
    try:
        import numpy as np
        lat, lon = _load_wildfire_points()

        R = 6371.0
        dlat = np.radians(lat - user_lat)
        dlon = np.radians(lon - user_lon)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(user_lat)) * np.cos(np.radians(lat)) * np.sin(dlon / 2) ** 2)
        dist = R * 2 * np.arcsin(np.sqrt(a))

        count_5 = int((dist <= 5).sum())
        count_10 = int((dist <= 10).sum())
        nearest = round(float(dist.min()), 1) if dist.size else None

        if count_10 >= 100:
            level = "HIGH"
        elif count_10 >= 25:
            level = "MODERATE"
        elif count_10 >= 1:
            level = "LOW"
        else:
            level = "MINIMAL"

        prone = level in ("HIGH", "MODERATE")
        msg = {
            "HIGH":     "High wildfire proneness — dense history of fire detections nearby.",
            "MODERATE": "Moderate wildfire proneness — notable fire history in the area.",
            "LOW":      "Low wildfire proneness — limited fire history nearby.",
            "MINIMAL":  "Minimal wildfire proneness — no recorded fires within 10 km.",
        }[level]

        return {
            "level":       level,
            "prone":       prone,
            "count_5km":   count_5,
            "count_10km":  count_10,
            "nearest_km":  nearest,
            "message":     msg,
            "disclaimer":  "Based on past-year historical satellite fire detections (VIIRS), not a live fire feed.",
            "source":      "VIIRS Active Fire / Hotspot history (HP)",
        }

    except Exception as e:
        logger.error(f"Wildfire assessment error: {e}")
        return {"level": "UNKNOWN", "prone": False, "error": str(e)}
