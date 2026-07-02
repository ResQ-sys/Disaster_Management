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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from resq_project.config import LLM_MODEL, LLM_TEMPERATURE
from resq_project.tools import (
    get_weather, query_hospitals, query_shelters,
    query_cwc_stations, query_knowledge, get_route,
    check_road_risk, get_district_risk, find_nearest_cwc_station
)


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

    # ── Enriched Context (Intake Agent fills these) ────────────────
    weather:         dict
    district_risk:   dict
    nearest_cwc:     dict
    imd_alert_level: str

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
    road_risks:      list[dict]
    route_warning:   str

    # ── Final Output (Escalation Agent fills these) ────────────────
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
    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


# ══════════════════════════════════════════════════════════════════════
# NODE 1 — INTAKE AGENT
# Role: Understand the situation; enrich with weather + district risk
# ══════════════════════════════════════════════════════════════════════
def intake_agent(state: DisasterState) -> DisasterState:
    llm = get_llm()

    # Fetch real-time context
    weather      = get_weather(state["latitude"], state["longitude"], state["district"])
    district_risk = get_district_risk(state["district"])
    nearest_cwc  = find_nearest_cwc_station(state["latitude"], state["longitude"])
    risk_knowledge = query_knowledge(
        f"disaster risk {state['district']} {state['disaster_type']} Himachal Pradesh"
    )

    # LLM: assess situation severity
    system = """You are the Intake Agent for HP Disaster Relief. 
    Assess the situation and provide a concise severity summary (2-3 sentences).
    Be factual, calm, and action-oriented. Never fabricate data."""

    prompt = f"""
Situation Report:
- District: {state['district']} | Location: {state['location_desc']}
- Disaster Type: {state['disaster_type']}
- Needs Reported: {', '.join(state['needs'])}
- Weather: {weather['description']} | Rainfall 24h forecast: {weather['forecast_rain_24h']}mm | Alert: {weather['alert_level']}
- District Risk Tier (HIMCOSTE 2023): {district_risk.get('risk_tier')} | Landslides 2023: {district_risk.get('landslides_2023', 'N/A')}
- Nearest CWC Station: {nearest_cwc.get('name', 'N/A')} on {nearest_cwc.get('river', 'N/A')} ({nearest_cwc.get('distance_km', 'N/A')} km)
- Relevant Context: {' | '.join(risk_knowledge[:2])}

Provide a 2-sentence situation severity assessment.
"""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        **state,
        "weather":         weather,
        "district_risk":   district_risk,
        "nearest_cwc":     nearest_cwc,
        "imd_alert_level": weather.get("alert_level", "GREEN"),
        "knowledge_chunks": risk_knowledge,
        "node_log":        [f"✓ intake_agent: {weather['alert_level']} alert, {district_risk.get('risk_tier')} risk district"],
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
    if "Medical" in needs or disaster_type in ["Flash Flood", "Landslide", "Cloudburst", "GLOF", "Avalanche"]:
        hospitals = query_hospitals(district, need_type="emergency", n_results=5)

    # Shelter need → fetch NULM shelters + schools
    if "Shelter" in needs or disaster_type in ["Flash Flood", "Cloudburst", "Landslide"]:
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

    resource_count = len(hospitals) + len(shelters) + len(cwc_stations)

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
    elif disaster in ["Flash Flood", "Cloudburst", "Landslide"]:
        route_warning = "⚠ Check local road status before travel — active disaster conditions."

    # Attempt routing if we have coordinates
    route = {}
    priority_name = priority.get("name", "resource")

    # For hospitals we can do distance estimate using user coordinates
    # (In production: geocode hospital address to get lat/lon)
    user_lat = state["latitude"]
    user_lon = state["longitude"]

    # Rough hospital coordinate estimation (center of district)
    # In production: use Google Geocoding API on hospital address
    district_centers = {
        "KANGRA":          (32.10, 76.27), "MANDI":   (31.71, 76.93),
        "SHIMLA":          (31.10, 77.17), "KULLU":   (31.95, 77.11),
        "SOLAN":           (30.91, 77.10), "SIRMOUR": (30.56, 77.66),
        "BILASPUR":        (31.34, 76.76), "HAMIRPUR":(31.68, 76.52),
        "CHAMBA":          (32.55, 76.12), "UNA":     (31.47, 76.27),
        "KINNAUR":         (31.59, 78.44), "LAHUL AND SPITI": (32.55, 77.60),
    }
    dest_center = district_centers.get(district.upper(), (user_lat + 0.1, user_lon + 0.1))
    route = get_route(user_lat, user_lon, dest_center[0], dest_center[1])

    return {
        **state,
        "road_risks":    road_risks,
        "route":         route,
        "route_warning": route_warning,
        "node_log": [
            f"✓ route_planning_agent: {route.get('distance_km', 'N/A')} km, "
            f"{route.get('duration_min', 'N/A')} min | "
            f"{len(active_risks)} active road risks"
        ],
    }


# ══════════════════════════════════════════════════════════════════════
# NODE 5 — ESCALATION & REPORT AGENT
# Role: Generate final structured report; escalate if no resource found
# ══════════════════════════════════════════════════════════════════════
def escalation_agent(state: DisasterState) -> DisasterState:
    llm = get_llm()

    # HP emergency contacts (always included)
    emergency_contacts = [
        "NDMA Helpline: 1078",
        "HP Police: 100",
        "HP Ambulance (108): 108",
        "Fire: 101",
        f"HPSDMA {state['district']} Control Room: 0177-2620131",
        "NDRF HP: 0172-2749165",
    ]

    # Add top hospital contact
    if state.get("hospitals"):
        h = state["hospitals"][0]
        emergency_contacts.append(f"Nearest Hospital ({h.get('name')}): {h.get('contact')}")

    escalation_needed = state.get("escalation_needed", False)
    if not state.get("priority_resource"):
        escalation_needed = True

    system = """You are the Escalation and Report Agent for HP Disaster Relief.
    Generate a clear, actionable disaster response report.
    Format: markdown with sections.
    Be concise, factual, and action-oriented.
    Always end with: 'NOTE: This is AI-assisted information. Always verify with local authorities.'"""

    prompt = f"""
Generate a disaster response report:

SITUATION:
- Person: {state.get('user_name', 'Unknown')}
- Location: {state.get('location_desc')} ({state['district']} district)
- Disaster: {state['disaster_type']}
- Needs: {', '.join(state.get('needs', []))}

WEATHER (OpenWeatherMap):
- {state.get('weather', {}).get('description', 'N/A')}
- Forecast rain 24h: {state.get('weather', {}).get('forecast_rain_24h', 'N/A')} mm
- Alert: {state.get('imd_alert_level', 'N/A')}

DISTRICT RISK (HIMCOSTE 2023):
- Risk Tier: {state.get('district_risk', {}).get('risk_tier', 'N/A')}
- Landslides in 2023: {state.get('district_risk', {}).get('landslides_2023', 'N/A')}

PRIORITY RESOURCE:
{json.dumps(state.get('priority_resource', {}), indent=2)}

ROUTE:
- Distance: {state.get('route', {}).get('distance_km', 'N/A')} km
- Duration: {state.get('route', {}).get('duration_min', 'N/A')} min
- Warning: {state.get('route_warning', 'None')}

CWC MONITORING:
- Nearest Station: {state.get('nearest_cwc', {}).get('name', 'N/A')} on {state.get('nearest_cwc', {}).get('river', 'N/A')}
- Live data: https://ffs.india.gov.in

ALL RESOURCES FOUND:
{json.dumps(state.get('matched_resources', [])[:3], indent=2)}

ESCALATION NEEDED: {escalation_needed}
{f"REASON: {state.get('escalation_reason', '')}" if escalation_needed else ""}

Generate the complete response report in markdown.
"""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    return {
        **state,
        "final_report":      response.content,
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
    graph.add_node("resource_finder_agent", resource_finder_agent)
    graph.add_node("matching_agent",        matching_agent)
    graph.add_node("route_planning_agent",  route_planning_agent)
    graph.add_node("escalation_agent",      escalation_agent)

    # Entry point
    graph.set_entry_point("intake_agent")

    # Edges: intake → (conditional by disaster type) → resource_finder
    graph.add_conditional_edges(
        "intake_agent",
        disaster_type_router,
        {"resource_finder": "resource_finder_agent"}
    )

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
    needs:         list[str]
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
        # All other fields start empty
        "weather": {}, "district_risk": {}, "nearest_cwc": {}, "imd_alert_level": "",
        "hospitals": [], "shelters": [], "cwc_stations": [], "knowledge_chunks": [],
        "matched_resources": [], "priority_resource": {}, "match_reasoning": "",
        "route": {}, "road_risks": [], "route_warning": "",
        "final_report": "", "escalation_needed": False,
        "escalation_reason": "", "emergency_contacts": [],
        "error_log": [], "node_log": [],
    }

    final_state = app.invoke(initial_state)
    return final_state
