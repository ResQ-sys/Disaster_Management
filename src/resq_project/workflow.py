"""
HP Disaster Relief Agent — LangGraph Workflow

WHY LANGGRAPH OVER CREWAI:
  1. Disaster type creates GENUINE conditional branches:
       Flash Flood  → CWC station check + river risk
       Landslide    → HIMCOSTE risk tier + blocked roads
       GLOF/Avalanche → different escalation path
  2. State (location, disaster type, found resources, route)
     must PERSIST and ACCUMULATE across all 5 nodes
  3. Conditional edges: if no resource found → skip route → go straight to escalation
  4. Human-in-the-loop possible at the escalation node
  LangGraph's explicit state machine gives full control over all of this.
  CrewAI would hide these branches behind sequential task chaining.

Graph Nodes (Agents):
  intake_agent          → collects user inputs, enriches with weather + district risk
  resource_finder_agent → queries ChromaDB for hospitals, shelters, CWC stations
  matching_agent        → scores and ranks resources by type/urgency/distance
  route_planning_agent  → calls ORS for route + checks NH road risk
  escalation_agent      → generates final report, flags if no resource found
"""

from typing import TypedDict, Annotated
import operator
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from resq_project.config import LLM_MODEL, LLM_TEMPERATURE, OLLAMA_BASE_URL
from resq_project.tools import (
    get_weather, query_hospitals, query_shelters,
    query_cwc_stations, query_knowledge, get_route,
    check_road_risk, get_district_risk, find_nearest_cwc_station,
    geocode_place, straight_line_route, query_glacial_lakes,
    assess_wildfire_risk
)
import re


# ══════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# TypedDict = every node reads/writes from the same shared state object
# ══════════════════════════════════════════════════════════════════════
class DisasterState(TypedDict):
    # ── User Inputs ────────────────────────────────────────────────
    user_name:       str
    district:        str
    location_desc:   str          # "Near Manali bus stand, Kullu"
    latitude:        float
    longitude:       float
    disaster_type:   str          # "Flash Flood" | "Landslide" | "Cloudburst" etc.
    needs:           list[str]    # ["Medical", "Shelter", "Food", "Rescue"]
    language:        str          # preferred language for AI-generated report text

    # ── Enriched Context (Intake Agent fills these) ────────────────
    weather:         dict
    district_risk:   dict
    nearest_cwc:     dict
    imd_alert_level: str
    wildfire_risk:   dict         # historical VIIRS fire-hotspot proneness

    # ── GLOF Monitoring (GLOF Monitor Agent fills these) ───────────
    glacial_lakes:   list[dict]
    glof_alert:      dict

    # ── Found Resources (Resource Finder fills these) ──────────────
    hospitals:       list[dict]
    shelters:        list[dict]
    cwc_stations:    list[dict]
    knowledge_chunks: list[str]

    # ── Matched Resources (Matching Agent fills these) ─────────────
    matched_resources: list[dict]
    priority_resource: dict
    match_reasoning:   str

    # ── Route (Route Planning Agent fills these) ───────────────────
    route:           dict
    routes:          list
    road_risks:      list[dict]
    route_warning:   str

    # ── Final Output (Escalation Agent fills these) ────────────────
    urgency:         dict         # explainable urgency score
    final_report:    str
    escalation_needed: bool
    escalation_reason: str
    emergency_contacts: list[str]

    # ── Control flow ───────────────────────────────────────────────
    error_log:       Annotated[list[str], operator.add]
    node_log:        Annotated[list[str], operator.add]


# ══════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════
def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url=OLLAMA_BASE_URL,
    )


# ══════════════════════════════════════════════════════════════════════
# URGENCY SCORING — deterministic & explainable (0-100)
# ══════════════════════════════════════════════════════════════════════
_ALERT_URGENCY = {"RED": 35, "ORANGE": 25, "YELLOW": 15, "GREEN": 5}
_TIER_URGENCY  = {"CRITICAL": 25, "HIGH": 18, "MEDIUM": 10, "LOW": 5, "UNKNOWN": 8}
_NEED_URGENCY  = {"Rescue": 10, "Medical": 10, "Evacuation": 9,
                  "Shelter": 6, "Water": 6, "Food": 4}


def compute_urgency_score(state: "DisasterState") -> dict:
    """Transparent urgency score from situation signals. Returns score, level
    and a per-factor breakdown so the routing decision is explainable."""
    b = {}
    b["imd_alert"] = _ALERT_URGENCY.get(str(state.get("imd_alert_level", "GREEN")).upper(), 5)
    b["district_tier"] = _TIER_URGENCY.get(
        str(state.get("district_risk", {}).get("risk_tier", "UNKNOWN")).upper(), 8)

    # Needs — take the two most severe reported needs (capped)
    need_pts = sorted((_NEED_URGENCY.get(n, 3) for n in state.get("needs", [])), reverse=True)
    b["needs"] = sum(need_pts[:2])

    glof = state.get("glof_alert", {}) or {}
    b["glof"] = 12 if glof.get("level") == "WATCH" else (6 if glof.get("level") == "ADVISORY" else 0)

    wf = state.get("wildfire_risk", {}) or {}
    b["wildfire"] = {"HIGH": 10, "MODERATE": 6, "LOW": 2}.get(wf.get("level"), 0)

    b["escalation"] = 10 if state.get("escalation_needed") else 0

    score = min(100, round(sum(b.values())))
    level = ("CRITICAL" if score >= 75 else "HIGH" if score >= 50 else
             "MODERATE" if score >= 30 else "LOW")
    return {"score": score, "level": level, "breakdown": b}


# ══════════════════════════════════════════════════════════════════════
# NODE 1 — INTAKE AGENT
# Role: Understand the situation; enrich with weather + district risk
# ══════════════════════════════════════════════════════════════════════
def intake_agent(state: DisasterState) -> DisasterState:
    # Fetch real-time context
    weather      = get_weather(state["latitude"], state["longitude"], state["district"])
    district_risk = get_district_risk(state["district"])
    nearest_cwc  = find_nearest_cwc_station(state["latitude"], state["longitude"])
    wildfire_risk = assess_wildfire_risk(state["latitude"], state["longitude"])
    risk_knowledge = query_knowledge(
        f"disaster risk {state['district']} {state['disaster_type']} Himachal Pradesh"
    )

    return {
        **state,
        "weather":         weather,
        "district_risk":   district_risk,
        "nearest_cwc":     nearest_cwc,
        "wildfire_risk":   wildfire_risk,
        "imd_alert_level": weather.get("alert_level", "GREEN"),
        "knowledge_chunks": risk_knowledge,
        "node_log":        [f"✓ intake_agent: {weather['alert_level']} alert, {district_risk.get('risk_tier')} risk district"
                            + f" | wildfire {wildfire_risk.get('level', 'N/A')}"],
    }


# ══════════════════════════════════════════════════════════════════════
# NODE 1b — GLOF MONITOR AGENT
# Role: Flag expanding glacial lakes near the user (GLOF early-warning)
# Data: CWC monthly satellite monitoring (Sep 2025) — previous-year, not live
# ══════════════════════════════════════════════════════════════════════
GLOF_SENSITIVE = ("GLOF", "Flash Flood", "Cloudburst")


def glof_monitor_agent(state: DisasterState) -> DisasterState:
    district = state["district"]
    disaster = state["disaster_type"]
    lakes = query_glacial_lakes(district, state["latitude"], state["longitude"], n_results=5)

    increasing = [lk for lk in lakes if lk.get("status") == "increase"]
    alert = {}
    if increasing:
        nearest = min(
            increasing,
            key=lambda lk: lk["distance_km"] if lk["distance_km"] is not None else 9e9,
        )
        in_district = [lk for lk in increasing if lk.get("district") == district.upper()]
        # Heighten level for water-driven hazards where GLOF is directly relevant
        level = "WATCH" if disaster in GLOF_SENSITIVE else "ADVISORY"
        dist_txt = f"~{nearest['distance_km']} km away" if nearest["distance_km"] is not None else "in the region"
        river_txt = nearest.get("river") or nearest.get("basin") or "Himalayan"
        alert = {
            "level":             level,
            "count_increasing":  len(increasing),
            "count_in_district": len(in_district),
            "nearest":           nearest,
            "monitored_period":  nearest.get("monitored_period", "September 2025"),
            "message": (
                f"{len(increasing)} glacial lake(s) near {district.title()} showed an "
                f"increase in water-spread area in the latest CWC monthly monitoring. "
                f"Nearest expanding lake {nearest['lake_id']} ({river_txt} basin, {dist_txt}, "
                f"+{nearest.get('area_pct_change')}%). This indicates elevated GLOF risk "
                f"downstream — monitor CWC advisories and avoid river banks."
            ),
            "disclaimer": (
                "Based on previous-year monthly satellite monitoring "
                f"({nearest.get('monitored_period', 'Sep 2025')}), not real-time water levels."
            ),
        }

    return {
        **state,
        "glacial_lakes": lakes,
        "glof_alert":    alert,
        "node_log": [
            f"✓ glof_monitor_agent: {len(lakes)} lakes considered, "
            f"{len(increasing)} increasing"
            + (f" → {alert['level']} alert" if alert else " → no alert")
        ],
    }


# ══════════════════════════════════════════════════════════════════════
# NODE 2 — RESOURCE FINDER AGENT
# Role: Query ChromaDB for all relevant resources
# Different resources fetched based on disaster type
# ══════════════════════════════════════════════════════════════════════
def resource_finder_agent(state: DisasterState) -> DisasterState:
    district      = state["district"]
    needs         = state["needs"]
    disaster_type = state["disaster_type"]

    hospitals, shelters, cwc_stations = [], [], []

    # Medical need or any injury-risk disaster → fetch hospitals
    if "Medical" in needs or disaster_type in ["Flash Flood", "Landslide", "Cloudburst", "GLOF", "Avalanche", "Wildfire"]:
        hospitals = query_hospitals(district, need_type="emergency", n_results=5)

    # Shelter need → fetch NULM shelters + schools
    if "Shelter" in needs or disaster_type in ["Flash Flood", "Cloudburst", "Landslide", "Wildfire"]:
        shelters = query_shelters(district, n_results=5)

    # Flood/GLOF → CWC station data critical
    if disaster_type in ["Flash Flood", "GLOF", "Cloudburst"]:
        river = state.get("nearest_cwc", {}).get("river", "")
        cwc_stations = query_cwc_stations(district, river=river, n_results=3)

    # Additional contextual knowledge
    know_query = f"{disaster_type} response resources {district} Himachal Pradesh relief"
    extra_knowledge = query_knowledge(know_query, n_results=3)

    # Merge with existing knowledge chunks
    all_knowledge = list(set(state.get("knowledge_chunks", []) + extra_knowledge))

    return {
        **state,
        "hospitals":      hospitals,
        "shelters":       shelters,
        "cwc_stations":   cwc_stations,
        "knowledge_chunks": all_knowledge,
        "node_log":       [f"✓ resource_finder_agent: found {len(hospitals)} hospitals, "
                           f"{len(shelters)} shelters, {len(cwc_stations)} CWC stations"],
    }


# ══════════════════════════════════════════════════════════════════════
# NODE 3 — MATCHING & PRIORITIZATION AGENT
# Role: Score and rank resources; pick top priority
# ══════════════════════════════════════════════════════════════════════
def matching_agent(state: DisasterState) -> DisasterState:
    llm = get_llm()

    all_resources = []
    for h in state.get("hospitals", []):
        h["resource_type"] = "HOSPITAL"
        all_resources.append(h)
    for s in state.get("shelters", []):
        s["resource_type"] = "SHELTER"
        all_resources.append(s)

    if not all_resources:
        return {
            **state,
            "matched_resources": [],
            "priority_resource": {},
            "match_reasoning":   "No resources found in district. Escalation required.",
            "escalation_needed": True,
            "escalation_reason": f"No resources found in {state['district']} for needs: {state['needs']}",
            "node_log": ["⚠ matching_agent: NO resources found — escalation flag set"],
        }

    system = """You are the Matching and Prioritization Agent for HP Disaster Relief.
    Given the situation and available resources, select the top priority resource.
    Respond ONLY in valid JSON:
    {
        "priority_resource": {"name": "...", "type": "...", "contact": "...", "reason": "..."},
        "ranked_resources": [{"name": "...", "type": "...", "contact": "...", "rank_reason": "..."}],
        "reasoning": "2-sentence explanation of prioritization"
    }"""

    prompt = f"""
Situation:
- District: {state['district']} | Disaster: {state['disaster_type']}
- Needs: {', '.join(state['needs'])}
- IMD Alert: {state.get('imd_alert_level', 'UNKNOWN')}
- District Risk Tier: {state.get('district_risk', {}).get('risk_tier', 'UNKNOWN')}

Available Resources:
{json.dumps(all_resources, indent=2)}

Context from knowledge base:
{chr(10).join(state.get('knowledge_chunks', [])[:3])}

Select and rank resources. Return valid JSON only.
"""
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        result   = json.loads(response.content.strip().strip("```json").strip("```"))
        priority = result.get("priority_resource", all_resources[0])
        ranked   = result.get("ranked_resources", all_resources)
        reasoning = result.get("reasoning", "")
    except Exception as e:
        priority  = all_resources[0] if all_resources else {}
        ranked    = all_resources
        reasoning = f"Fallback ranking used (LLM parse error: {e})"

    return {
        **state,
        "matched_resources": ranked,
        "priority_resource": priority,
        "match_reasoning":   reasoning,
        "node_log": [f"✓ matching_agent: priority → {priority.get('name', 'N/A')} ({priority.get('resource_type', 'N/A')})"],
    }


# ══════════════════════════════════════════════════════════════════════
# NODE 4 — ROUTE PLANNING AGENT
# Role: Generate route + check NH road risk from HIMCOSTE data
# ══════════════════════════════════════════════════════════════════════
DISTRICT_CENTERS = {
    "KANGRA":          (32.10, 76.27), "MANDI":   (31.71, 76.93),
    "SHIMLA":          (31.10, 77.17), "KULLU":   (31.95, 77.11),
    "SOLAN":           (30.91, 77.10), "SIRMOUR": (30.56, 77.66),
    "BILASPUR":        (31.34, 76.76), "HAMIRPUR":(31.68, 76.52),
    "CHAMBA":          (32.55, 76.12), "UNA":     (31.47, 76.27),
    "KINNAUR":         (31.59, 78.44), "LAHUL AND SPITI": (32.55, 77.60),
}


# Generic words in facility names that aren't place names — stripped so we can
# geocode the *town* embedded in the name (e.g. "Civil Hospital Dalhousie").
_GENERIC_NAME_TOKENS = {
    "hospital", "civil", "referral", "community", "health", "centre", "center",
    "chc", "phc", "gmc", "pt", "district", "regional", "ayurvedic", "and", "the",
    "nulm", "shelter", "govt", "government", "general", "sub", "subdivisional",
    "dispensary", "clinic", "medical", "college", "of", "hp", "himachal", "pradesh",
}


def _extract_place(name: str) -> str:
    """Pull the likely town/place token(s) out of a facility name."""
    toks = re.split(r"[\s,\-]+", (name or "").strip())
    keep = [t for t in toks if t and t.lower().strip(".") not in _GENERIC_NAME_TOKENS and not t.isdigit()]
    return " ".join(keep[-2:]) if keep else ""


def _resolve_dest(resource, name, district, center):
    """Best-effort (lat, lon, source) for a resource — approximate is fine.

    Ladder: resource coords → geocode full name → geocode the town in the
    name → district center. Always returns a usable point.
    """
    try:
        r_lat = float(resource.get("latitude", 0) or 0)
        r_lon = float(resource.get("longitude", 0) or 0)
        if r_lat and r_lon:
            return r_lat, r_lon, "resource coordinates"
    except (TypeError, ValueError):
        pass

    geo = geocode_place(f"{name}, {district.title()}, Himachal Pradesh, India")
    if geo:
        return geo[0], geo[1], "geocoded resource"

    place = _extract_place(name)
    if place:
        geo = geocode_place(f"{place}, {district.title()}, Himachal Pradesh, India")
        if geo:
            return geo[0], geo[1], f"approx · {place}"

    return center[0], center[1], "district center"


def _route_to_resource(resource, user_lat, user_lon, district, center):
    """Compute an (approximate) route from the user's location to a resource.

    Resolves a best-effort destination, then uses ORS road routing when
    available and falls back to a labelled straight-line estimate otherwise,
    so a distance/time always shows. Returns a route dict enriched with the
    resource name/type and destination coordinates.
    """
    if not resource:
        return {}

    name = resource.get("name", "resource")
    d_lat, d_lon, dest_source = _resolve_dest(resource, name, district, center)

    same_point = abs(d_lat - user_lat) < 1e-4 and abs(d_lon - user_lon) < 1e-4
    if same_point:
        route = {
            "distance_km":  0.0,
            "duration_min": 0.0,
            "turn_by_turn": [f"{name} is within the immediate vicinity of the reported location."],
            "source":       "same_locality",
        }
    else:
        route = get_route(user_lat, user_lon, d_lat, d_lon)
        # ORS unavailable/failed → approximate straight-line so a number shows.
        if route.get("distance_km") is None:
            route = straight_line_route(user_lat, user_lon, d_lat, d_lon)

    route["name"]          = name
    route["resource_type"] = resource.get("resource_type", resource.get("type", "RESOURCE"))
    route["dest_source"]   = dest_source
    route["dest_lat"]      = d_lat
    route["dest_lon"]      = d_lon
    return route


def route_planning_agent(state: DisasterState) -> DisasterState:
    priority    = state.get("priority_resource", {})
    district    = state["district"]
    disaster    = state["disaster_type"]

    # Check NH corridor risk for this district
    road_risks = check_road_risk(district)

    # If no priority resource, skip routing
    if not priority:
        return {
            **state,
            "road_risks":   road_risks,
            "route":        {},
            "routes":       [],
            "route_warning": "No priority resource found — cannot compute route.",
            "node_log": ["⚠ route_planning_agent: skipped — no priority resource"],
        }

    # Build route warning from blocked corridors
    active_risks = [r for r in road_risks if r.get("currently_risky")]
    route_warning = ""
    if active_risks:
        route_warning = (
            "⚠ WARNING: The following routes are prone to closure in current season: "
            + " | ".join(r["warning"] for r in active_risks)
        )
    elif disaster in ["Flash Flood", "Cloudburst", "Landslide", "Wildfire"]:
        route_warning = "⚠ Check local road status before travel — active disaster conditions."

    user_lat = state["latitude"]
    user_lon = state["longitude"]
    center = DISTRICT_CENTERS.get(district.upper(), (user_lat + 0.1, user_lon + 0.1))

    # Priority route (used by the final report + summary metrics).
    route = _route_to_resource(priority, user_lat, user_lon, district, center)

    # Per-type routes: distance/time to the top hospital AND the top shelter,
    # so the user sees both — not only the single priority pick.
    routes = []
    top_hospital = next(iter(state.get("hospitals", [])), None)
    top_shelter  = next(iter(state.get("shelters", [])), None)
    if top_hospital:
        routes.append(_route_to_resource(top_hospital, user_lat, user_lon, district, center))
    if top_shelter:
        routes.append(_route_to_resource(top_shelter, user_lat, user_lon, district, center))

    return {
        **state,
        "road_risks":    road_risks,
        "route":         route,
        "routes":        routes,
        "route_warning": route_warning,
        "node_log": [
            f"✓ route_planning_agent: priority {route.get('distance_km', 'N/A')} km / "
            f"{route.get('duration_min', 'N/A')} min | {len(routes)} typed routes | "
            f"{len(active_risks)} active road risks"
        ],
    }


# ══════════════════════════════════════════════════════════════════════
# Report label translation (Hindi) — deterministic, LLM-free.
#
# The report's facts (numbers, alert codes, urgency scores, place names)
# are built directly here in Python and only ever have their fixed labels
# translated via this dict. A small local model (llama3.2:1b) asked to
# translate/regenerate the WHOLE factual report was observed to hallucinate
# — e.g. reporting a GREEN weather alert as RED. Numbers and codes must
# never pass through the LLM in a different language than they were
# computed in. Only the short "Recommendation" paragraph below is
# LLM-generated (and kept low-stakes: plain-language advice, no figures).
# ══════════════════════════════════════════════════════════════════════
_REPORT_LABELS_HI = {
    "Disaster Response Report": "आपदा प्रतिक्रिया रिपोर्ट",
    "Situation": "स्थिति",
    "Person": "व्यक्ति",
    "Location": "स्थान",
    "district": "जिला",
    "Disaster": "आपदा",
    "Needs": "आवश्यकताएं",
    "Urgency Score": "तात्कालिकता स्कोर",
    "Weather": "मौसम",
    "Forecast rain (24h)": "वर्षा पूर्वानुमान (24 घंटे)",
    "Alert": "चेतावनी",
    "District Risk (HIMCOSTE 2023)": "जिला जोखिम (HIMCOSTE 2023)",
    "Risk Tier": "जोखिम स्तर",
    "Landslides in 2023": "2023 में भूस्खलन",
    "Priority Resource": "प्राथमिकता संसाधन",
    "Name": "नाम",
    "Type": "प्रकार",
    "Contact": "संपर्क",
    "None found — escalation required": "कोई नहीं मिला — आगे बढ़ाना आवश्यक",
    "Route to Priority Resource": "प्राथमिकता संसाधन तक मार्ग",
    "Distance": "दूरी",
    "Duration": "अवधि",
    "Road warning": "सड़क चेतावनी",
    "None": "कोई नहीं",
    "CWC Monitoring": "CWC निगरानी",
    "Nearest Station": "निकटतम स्टेशन",
    "Live data": "लाइव डेटा",
    "GLOF Monitoring (glacial lake outburst risk)": "GLOF निगरानी (हिमानी झील प्रकोप बाढ़ जोखिम)",
    "No expanding glacial lakes flagged near this location.":
        "इस स्थान के पास कोई बढ़ती हिमानी झील चिह्नित नहीं।",
    "Wildfire Proneness": "जंगल की आग की संभावना",
    "Other Resources Found": "अन्य मिले संसाधन",
    "Escalation Needed": "आगे बढ़ाना आवश्यक",
    "Yes": "हाँ", "No": "नहीं",
    "Reason": "कारण",
    "Recommendation": "सिफारिश",
    "NOTE: This is AI-assisted information. Always verify with HPSDMA and "
    "local authorities. Call 1078 (NDMA) for official support.":
        "नोट: यह AI-सहायता प्राप्त जानकारी है। हमेशा HPSDMA और स्थानीय अधिकारियों से "
        "सत्यापित करें। आधिकारिक सहायता हेतु 1078 (NDMA) पर कॉल करें।",
    "NDMA Helpline": "NDMA हेल्पलाइन",
    "HP Police": "HP पुलिस",
    "HP Ambulance (108)": "HP एम्बुलेंस (108)",
    "Fire": "फायर",
    "Control Room": "नियंत्रण कक्ष",
    "Nearest Hospital": "निकटतम अस्पताल",
}

_CODE_LABELS_HI = {
    "CRITICAL": "अति गंभीर", "HIGH": "उच्च", "MEDIUM": "मध्यम", "MODERATE": "मध्यम",
    "LOW": "निम्न", "MINIMAL": "न्यूनतम", "UNKNOWN": "अज्ञात", "N/A": "उपलब्ध नहीं",
    "RED": "लाल", "ORANGE": "नारंगी", "YELLOW": "पीला", "GREEN": "हरा",
}


def _rl(text: str, language: str) -> str:
    """Translate a fixed report label (never a data value) to Hindi."""
    return _REPORT_LABELS_HI.get(text, text) if language == "Hindi" else text


def _cl(code, language: str):
    """Translate a short enum code (risk tier, alert level, ...) to Hindi."""
    return _CODE_LABELS_HI.get(str(code).upper(), code) if language == "Hindi" else code


# ══════════════════════════════════════════════════════════════════════
# NODE 5 — ESCALATION & REPORT AGENT
# Role: Generate final structured report; escalate if no resource found
# ══════════════════════════════════════════════════════════════════════
def escalation_agent(state: DisasterState) -> DisasterState:
    llm = get_llm()
    language = state.get("language") or "English"

    # HP emergency contacts (always included)
    emergency_contacts = [
        f"{_rl('NDMA Helpline', language)}: 1078",
        f"{_rl('HP Police', language)}: 100",
        f"{_rl('HP Ambulance (108)', language)}: 108",
        f"{_rl('Fire', language)}: 101",
        f"HPSDMA {state['district']} {_rl('Control Room', language)}: 0177-2620131",
        f"NDRF HP: 0172-2749165",
    ]

    # Add top hospital contact
    if state.get("hospitals"):
        h = state["hospitals"][0]
        emergency_contacts.append(f"{_rl('Nearest Hospital', language)} ({h.get('name')}): {h.get('contact')}")

    escalation_needed = state.get("escalation_needed", False)
    if not state.get("priority_resource"):
        escalation_needed = True

    # Explainable urgency score (uses final escalation flag)
    urgency = compute_urgency_score({**state, "escalation_needed": escalation_needed})

    # ── Deterministic facts block — built in Python, never touched by the
    # LLM, so numbers/codes/names can never be mistranslated or hallucinated.
    weather = state.get("weather", {}) or {}
    district_risk = state.get("district_risk", {}) or {}
    priority = state.get("priority_resource", {}) or {}
    route = state.get("route", {}) or {}
    nearest_cwc = state.get("nearest_cwc", {}) or {}
    glof_alert = state.get("glof_alert", {}) or {}
    wildfire = state.get("wildfire_risk", {}) or {}
    other_resources = state.get("matched_resources", [])[1:4]

    R = lambda k: _rl(k, language)
    lines = [
        f"# {R('Disaster Response Report')}",
        f"## {R('Situation')}",
        f"- **{R('Person')}:** {state.get('user_name', 'Unknown')}",
        f"- **{R('Location')}:** {state.get('location_desc')} ({state['district']} {R('district')})",
        f"- **{R('Disaster')}:** {state['disaster_type']}",
        f"- **{R('Needs')}:** {', '.join(state.get('needs', []))}",
        f"- **{R('Urgency Score')}:** {urgency['score']}/100 ({_cl(urgency['level'], language)})",
        "",
        f"## {R('Weather')}",
        f"- {weather.get('description', 'N/A')}",
        f"- **{R('Forecast rain (24h)')}:** {weather.get('forecast_rain_24h', 'N/A')} mm",
        f"- **{R('Alert')}:** {_cl(state.get('imd_alert_level', 'N/A'), language)}",
        "",
        f"## {R('District Risk (HIMCOSTE 2023)')}",
        f"- **{R('Risk Tier')}:** {_cl(district_risk.get('risk_tier', 'N/A'), language)}",
        f"- **{R('Landslides in 2023')}:** {district_risk.get('landslides_2023', 'N/A')}",
        "",
        f"## {R('Priority Resource')}",
    ]
    if priority:
        lines += [
            f"- **{R('Name')}:** {priority.get('name', 'N/A')}",
            f"- **{R('Type')}:** {priority.get('resource_type', priority.get('type', 'N/A'))}",
            f"- **{R('Contact')}:** {priority.get('contact', 'N/A')}",
        ]
    else:
        lines.append(f"- {R('None found — escalation required')}")
    lines += [
        "",
        f"## {R('Route to Priority Resource')}",
        f"- **{R('Distance')}:** {route.get('distance_km', 'N/A')} km",
        f"- **{R('Duration')}:** {route.get('duration_min', 'N/A')} min",
        f"- **{R('Road warning')}:** {state.get('route_warning') or R('None')}",
        "",
        f"## {R('CWC Monitoring')}",
        f"- **{R('Nearest Station')}:** {nearest_cwc.get('name', 'N/A')} ({nearest_cwc.get('river', 'N/A')})",
        f"- **{R('Live data')}:** https://ffs.india-water.gov.in",
        "",
        f"## {R('GLOF Monitoring (glacial lake outburst risk)')}",
    ]
    if glof_alert:
        lines.append(f"- {glof_alert.get('message', '')} ({glof_alert.get('disclaimer', '')})")
    else:
        lines.append(f"- {R('No expanding glacial lakes flagged near this location.')}")
    lines += [
        "",
        f"## {R('Wildfire Proneness')}",
        f"- **{_cl(wildfire.get('level', 'N/A'), language)}** — {wildfire.get('message', 'N/A')} "
        f"({wildfire.get('disclaimer', '')})",
    ]
    if other_resources:
        lines += ["", f"## {R('Other Resources Found')}"]
        for r in other_resources:
            lines.append(f"- {r.get('name', 'N/A')} ({r.get('resource_type', r.get('type', 'N/A'))})")
    lines += [
        "",
        f"## {R('Escalation Needed')}",
        f"- **{R('Yes') if escalation_needed else R('No')}**"
        + (f" — {R('Reason')}: {state.get('escalation_reason', '')}" if escalation_needed else ""),
    ]
    facts_report = "\n".join(lines)

    # ── Short, low-stakes LLM narrative — plain-language advice only, no
    # figures to get wrong. Kept deliberately short: a small local model
    # (llama3.2:1b) reliably follows a "write in Hindi" instruction only
    # on short, focused prompts, not on long structured-data generation.
    situation_summary = (
        f"Disaster: {state['disaster_type']} in {state['district']} district. "
        f"Urgency: {urgency['level']}. Needs: {', '.join(state.get('needs', []))}. "
        + (f"Nearest resource: {priority.get('name')} ({route.get('distance_km', '?')} km away). "
           if priority else "No local resource found — escalate to district authorities. ")
        + ("GLOF risk elevated nearby. " if glof_alert else "")
        + ("Wildfire-prone area. " if wildfire.get("prone") else "")
        + (f"Road risk: {state['route_warning']}. " if state.get("route_warning") else "")
    )
    if language == "Hindi":
        narrative_system = (
            "IMPORTANT: Write your ENTIRE response in Hindi (Devanagari script only). "
            "Do not use English words.\n"
            "You are a disaster relief assistant. Given a short situation summary, write "
            "a brief 3-4 sentence actionable recommendation for the person. Do not restate "
            "exact numbers; just give clear, plain-language guidance on what to do next."
        )
    else:
        narrative_system = (
            "You are a disaster relief assistant. Given a short situation summary, write "
            "a brief 3-4 sentence actionable recommendation for the person. Do not restate "
            "exact numbers; just give clear, plain-language guidance on what to do next."
        )
    try:
        narrative_resp = llm.invoke([
            SystemMessage(content=narrative_system),
            HumanMessage(content=situation_summary),
        ])
        narrative = narrative_resp.content.strip()
    except Exception:
        narrative = situation_summary  # fallback: still informative, just not prose

    note = _rl(
        "NOTE: This is AI-assisted information. Always verify with HPSDMA and "
        "local authorities. Call 1078 (NDMA) for official support.",
        language,
    )
    final_report = f"{facts_report}\n\n## {R('Recommendation')}\n\n{narrative}\n\n**{note}**"

    return {
        **state,
        "urgency":           urgency,
        "final_report":      final_report,
        "escalation_needed": escalation_needed,
        "emergency_contacts": emergency_contacts,
        "node_log": [
            f"✓ escalation_agent: report generated | "
            f"escalation={'YES ⚠' if escalation_needed else 'No'}"
        ],
    }


# ══════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGES
# These define the branching logic — core reason to use LangGraph
# ══════════════════════════════════════════════════════════════════════
def should_skip_to_escalation(state: DisasterState) -> str:
    """
    After matching: if no resources found, skip route planning,
    go straight to escalation.
    """
    if state.get("escalation_needed") or not state.get("priority_resource"):
        return "skip_to_escalation"
    return "continue_to_route"


def disaster_type_router(state: DisasterState) -> str:
    """
    After intake: route based on disaster type.
    Flood/GLOF → resource finder with CWC emphasis
    Landslide   → resource finder with road-risk emphasis
    Others      → standard resource finder
    All paths currently merge to same resource_finder node,
    but this edge lets you split in future for disaster-specific agents.
    """
    return "resource_finder"   # All types currently use same node


# ══════════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════════
def build_graph():
    graph = StateGraph(DisasterState)

    # Add nodes
    graph.add_node("intake_agent",         intake_agent)
    graph.add_node("glof_monitor_agent",   glof_monitor_agent)
    graph.add_node("resource_finder_agent", resource_finder_agent)
    graph.add_node("matching_agent",        matching_agent)
    graph.add_node("route_planning_agent",  route_planning_agent)
    graph.add_node("escalation_agent",      escalation_agent)

    # Entry point
    graph.set_entry_point("intake_agent")

    # Edges: intake → (conditional by disaster type) → GLOF monitor → resource_finder
    graph.add_conditional_edges(
        "intake_agent",
        disaster_type_router,
        {"resource_finder": "glof_monitor_agent"}
    )

    # GLOF monitor → resource_finder
    graph.add_edge("glof_monitor_agent", "resource_finder_agent")

    # resource_finder → matching
    graph.add_edge("resource_finder_agent", "matching_agent")

    # matching → (conditional: resources found?) → route OR escalation
    graph.add_conditional_edges(
        "matching_agent",
        should_skip_to_escalation,
        {
            "continue_to_route":    "route_planning_agent",
            "skip_to_escalation":   "escalation_agent",
        }
    )

    # route → escalation
    graph.add_edge("route_planning_agent", "escalation_agent")

    # escalation → END
    graph.add_edge("escalation_agent", END)

    return graph.compile()


# ── Convenience runner ─────────────────────────────────────────────────
def run_agent(
    user_name:     str,
    district:      str,
    location_desc: str,
    latitude:      float,
    longitude:     float,
    disaster_type: str,
    needs:         list[str],
    language:      str = "English",
) -> DisasterState:
    """Run the full LangGraph pipeline and return final state."""
    app = build_graph()

    initial_state: DisasterState = {
        "user_name":      user_name,
        "district":       district,
        "location_desc":  location_desc,
        "latitude":       latitude,
        "longitude":      longitude,
        "disaster_type":  disaster_type,
        "needs":          needs,
        "language":       language,
        # All other fields start empty
        "weather": {}, "district_risk": {}, "nearest_cwc": {}, "imd_alert_level": "",
        "wildfire_risk": {}, "glacial_lakes": [], "glof_alert": {},
        "hospitals": [], "shelters": [], "cwc_stations": [], "knowledge_chunks": [],
        "matched_resources": [], "priority_resource": {}, "match_reasoning": "",
        "route": {}, "routes": [], "road_risks": [], "route_warning": "",
        "urgency": {}, "final_report": "", "escalation_needed": False,
        "escalation_reason": "", "emergency_contacts": [],
        "error_log": [], "node_log": [],
    }

    final_state = app.invoke(initial_state)
    return final_state
