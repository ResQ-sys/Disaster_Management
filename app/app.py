"""
HP Disaster Relief Resource Matching Agent
Streamlit UI
"""

import streamlit as st
import json
import folium
from streamlit_folium import st_folium

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from resq_project.config import DISASTER_TYPES, DISTRICT_RISK
from resq_project.workflow import run_agent

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HP Disaster Relief Agent",
    page_icon="🏔️",
    layout="wide"
)

# ── Alert color map ────────────────────────────────────────────────────
ALERT_COLORS = {
    "RED":    ("#dc2626", "🔴"),
    "ORANGE": ("#ea580c", "🟠"),
    "YELLOW": ("#ca8a04", "🟡"),
    "GREEN":  ("#16a34a", "🟢"),
}

RISK_COLORS = {
    "CRITICAL": "#dc2626",
    "HIGH":     "#ea580c",
    "MEDIUM":   "#ca8a04",
    "LOW":      "#16a34a",
    "UNKNOWN":  "#6b7280",
}

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d6a4f 100%);
            padding: 24px 32px; border-radius: 12px; margin-bottom: 24px;'>
  <h1 style='color:white; margin:0; font-size:28px;'>
    🏔️ HP Disaster Relief Resource Matching Agent
  </h1>
  <p style='color:#a8d5e2; margin:8px 0 0 0; font-size:14px;'>
    Himachal Pradesh • Multi-hazard: Flood | Landslide | GLOF | Avalanche | Cloudburst
  </p>
  <p style='color:#94a3b8; margin:4px 0 0 0; font-size:12px;'>
    Powered by: LangGraph • Ollama llama3.2:1b • ChromaDB • Open-Meteo •
    Sources: HIMCOSTE 2023 | CWC | NHP Hospitals | DAY-NULM | HP Education Dept
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📋 Situation Details")
    st.markdown("---")

    user_name = st.text_input("Your Name", placeholder="e.g. Ramesh Kumar")

    district = st.selectbox(
        "District *",
        options=sorted(DISTRICT_RISK.keys()),
        help="Select the affected district in Himachal Pradesh"
    )

    location_desc = st.text_input(
        "Location Description",
        placeholder="e.g. Near Kullu bus stand, Beas riverbank"
    )

    st.markdown("**Approximate Coordinates** (or use map)")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=31.95, step=0.01, format="%.4f")
    with col2:
        longitude = st.number_input("Longitude", value=77.11, step=0.01, format="%.4f")

    disaster_type = st.selectbox("Disaster Type *", DISASTER_TYPES)

    needs = st.multiselect(
        "Immediate Needs *",
        ["Medical", "Shelter", "Food", "Rescue", "Evacuation", "Water"],
        default=["Medical", "Shelter"]
    )

    st.markdown("---")
    run_btn = st.button("🚨 Find Relief Resources", type="primary", use_container_width=True)
    st.markdown("---")

    # Quick reference
    risk_info = DISTRICT_RISK.get(district, {})
    st.markdown(f"**{district} Risk Profile**")
    tier = risk_info.get("tier", "UNKNOWN")
    color = RISK_COLORS.get(tier, "#6b7280")
    st.markdown(
        f"<span style='background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px'>{tier}</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"Landslides (2023): **{risk_info.get('landslides_2023', 'N/A')}**")
    st.markdown(f"Key Rivers: {', '.join(risk_info.get('key_rivers', []))}")

    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("""
    <small>
    🏥 NHP Hospitals (289 HP facilities)<br>
    🏫 HP Edu Dept Schools (shelter proxy)<br>
    🏠 DAY-NULM Shelters (54 cities)<br>
    🌊 CWC Stations (52 HP stations)<br>
    ⛰️ HIMCOSTE Landslide Inventory 2023<br>
    🌧️ Open-Meteo (no API key)<br>
    🗺️ OpenRouteService (routing)
    </small>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN PANEL — MAP + RESULTS
# ══════════════════════════════════════════════════════════════════════
if not run_btn:
    # Default view: HP map
    st.markdown("### 🗺️ Himachal Pradesh — Disaster Risk Map")

    m = folium.Map(location=[31.8, 77.2], zoom_start=7, tiles="CartoDB positron")

    # Plot district risk tiers
    district_coords = {
        "KANGRA": (32.10, 76.27), "MANDI": (31.71, 76.93),
        "SHIMLA": (31.10, 77.17), "KULLU": (31.95, 77.11),
        "SOLAN": (30.91, 77.10), "SIRMOUR": (30.56, 77.66),
        "BILASPUR": (31.34, 76.76), "HAMIRPUR": (31.68, 76.52),
        "CHAMBA": (32.55, 76.12), "UNA": (31.47, 76.27),
        "KINNAUR": (31.59, 78.44), "LAHUL AND SPITI": (32.55, 77.60),
    }
    tier_colors = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "beige", "LOW": "green"}

    for dist, coords in district_coords.items():
        info = DISTRICT_RISK.get(dist, {})
        color = tier_colors.get(info.get("tier", "LOW"), "gray")
        folium.CircleMarker(
            location=coords, radius=12,
            color=color, fill=True, fill_color=color, fill_opacity=0.6,
            popup=f"<b>{dist}</b><br>Risk: {info.get('tier')}<br>Landslides 2023: {info.get('landslides_2023')}"
        ).add_to(m)
        folium.Marker(
            location=coords,
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px;font-weight:bold;color:#1e293b">{dist.split()[0]}</div>'
            )
        ).add_to(m)

    st_folium(m, width=900, height=500)
    st.info("ℹ️ Fill in the sidebar and click **Find Relief Resources** to activate the agent.")

else:
    # ── Validate inputs ──────────────────────────────────────────────
    if not district or not disaster_type or not needs:
        st.error("⚠️ Please fill in District, Disaster Type, and at least one Need.")
        st.stop()

    # ── Run the agent ────────────────────────────────────────────────
    with st.spinner("🔄 Running disaster response pipeline... (5 LangGraph nodes)"):
        try:
            state = run_agent(
                user_name=user_name or "Anonymous",
                district=district,
                location_desc=location_desc or f"{district}, HP",
                latitude=latitude,
                longitude=longitude,
                disaster_type=disaster_type,
                needs=needs
            )
            success = True
        except Exception as e:
            st.error(f"Agent error: {e}")
            success = False

    if not success:
        st.stop()

    # ══════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ══════════════════════════════════════════════════════════════════

    # ── Top metrics row ──────────────────────────────────────────────
    weather      = state.get("weather", {})
    district_risk = state.get("district_risk", {})
    alert        = state.get("imd_alert_level", "GREEN")
    alert_color, alert_icon = ALERT_COLORS.get(alert, ("#6b7280", "⚪"))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🌧️ IMD Alert",    f"{alert_icon} {alert}")
    m2.metric("⛰️ Risk Tier",    district_risk.get("risk_tier", "N/A"))
    m3.metric("🌡️ Temp",        f"{weather.get('temperature_c', 'N/A')}°C")
    m4.metric("💧 Rain (24h)",   f"{weather.get('forecast_rain_24h', 'N/A')} mm")
    m5.metric("🏥 Resources Found", len(state.get("matched_resources", [])))

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Response Report", "🏥 Resources", "🗺️ Route & Map", "🌊 CWC Stations", "⚙️ Agent Log"
    ])

    # ── TAB 1: Final Report ──────────────────────────────────────────
    with tab1:
        if state.get("escalation_needed"):
            st.error("🚨 ESCALATION REQUIRED — Insufficient local resources found")

        report = state.get("final_report", "No report generated.")
        st.markdown(report)

        st.markdown("---")
        st.markdown("**📞 Emergency Contacts**")
        for contact in state.get("emergency_contacts", []):
            st.markdown(f"• {contact}")

        st.warning("⚠️ NOTE: This is AI-assisted disaster information. Always verify with HPSDMA and local authorities. Call 1078 (NDMA) for official support.")

    # ── TAB 2: Resources ─────────────────────────────────────────────
    with tab2:
        priority = state.get("priority_resource", {})
        if priority:
            st.success(f"✅ **Priority Resource: {priority.get('name', 'N/A')}**")
            col1, col2 = st.columns(2)
            col1.markdown(f"**Type:** {priority.get('resource_type', priority.get('type', 'N/A'))}")
            col1.markdown(f"**Contact:** {priority.get('contact', 'N/A')}")
            col2.markdown(f"**District:** {priority.get('district', district)}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🏥 Hospitals ({len(state.get('hospitals', []))} found)**")
            for h in state.get("hospitals", [])[:5]:
                with st.expander(h.get("name", "N/A")):
                    st.write(f"**Type:** {h.get('type')}")
                    st.write(f"**Contact:** {h.get('contact')}")
                    st.write(f"**Specialities:** {h.get('specialities', 'N/A')[:80]}")

        with col2:
            st.markdown(f"**🏠 Shelters ({len(state.get('shelters', []))} found)**")
            for s in state.get("shelters", [])[:5]:
                with st.expander(s.get("name", "N/A")):
                    st.write(f"**Type:** {s.get('type')}")
                    st.write(f"**Capacity:** {s.get('capacity', 'N/A')}")
                    st.write(f"**Contact:** {s.get('contact', 'N/A')}")

        st.markdown("---")
        st.markdown(f"**Matching Reasoning:** {state.get('match_reasoning', 'N/A')}")

    # ── TAB 3: Route & Map ───────────────────────────────────────────
    with tab3:
        route = state.get("route", {})
        road_risks = state.get("road_risks", [])

        col1, col2 = st.columns(2)
        col1.metric("📍 Distance", f"{route.get('distance_km', 'N/A')} km")
        col2.metric("⏱️ Est. Time", f"{route.get('duration_min', 'N/A')} min")

        if state.get("route_warning"):
            st.warning(state["route_warning"])

        if route.get("turn_by_turn"):
            st.markdown("**Route Steps:**")
            for step in route["turn_by_turn"]:
                st.markdown(f"→ {step}")

        # Road risk table
        if road_risks:
            st.markdown("---")
            st.markdown(f"**⛰️ Road Risk Assessment — {district}**")
            for risk in road_risks:
                status_color = "🔴" if risk.get("currently_risky") else "🟡"
                st.markdown(
                    f"{status_color} **{risk['road']}** at {risk['segment']} — "
                    f"{risk['risk_type']} | Season: {risk['season']}"
                )

        # Map
        m = folium.Map(location=[latitude, longitude], zoom_start=11, tiles="CartoDB positron")
        folium.Marker([latitude, longitude],
                      popup="Your Location",
                      icon=folium.Icon(color="red", icon="exclamation-sign")).add_to(m)

        # Nearest CWC station
        cwc = state.get("nearest_cwc", {})
        if cwc.get("latitude") and cwc.get("longitude"):
            try:
                folium.Marker(
                    [float(cwc["latitude"]), float(cwc["longitude"])],
                    popup=f"CWC: {cwc.get('name')} on {cwc.get('river')}",
                    icon=folium.Icon(color="blue", icon="tint")
                ).add_to(m)
            except Exception:
                pass

        st_folium(m, width=850, height=400)

    # ── TAB 4: CWC Stations ──────────────────────────────────────────
    with tab4:
        st.markdown("### 🌊 CWC River Monitoring Stations")
        st.markdown(
            "These are official Central Water Commission stations. "
            "Check **[ffs.india.gov.in](https://ffs.india.gov.in)** for live water level data."
        )

        nearest = state.get("nearest_cwc", {})
        if nearest:
            st.info(
                f"📡 **Nearest CWC Station:** {nearest.get('name')} | "
                f"River: {nearest.get('river')} | "
                f"Distance: {nearest.get('distance_km')} km"
            )

        for s in state.get("cwc_stations", []):
            with st.expander(f"{s.get('station')} — {s.get('river')}"):
                st.write(f"**District:** {s.get('district')}")
                st.write(f"**Site Type:** {s.get('site_type')}")
                st.write(f"**Coordinates:** {s.get('latitude')}, {s.get('longitude')}")
                st.write(f"**Live Data:** {s.get('cwc_url')}")

    # ── TAB 5: Agent Log ─────────────────────────────────────────────
    with tab5:
        st.markdown("### ⚙️ LangGraph Agent Execution Log")
        for log in state.get("node_log", []):
            icon = "⚠️" if "⚠" in log else "✅"
            st.markdown(f"{icon} `{log}`")

        if state.get("error_log"):
            st.markdown("**Errors:**")
            for err in state["error_log"]:
                st.error(err)

        st.markdown("---")
        st.markdown("**LangGraph Pipeline:**")
        st.markdown("""
        ```
        START
          ↓
        [intake_agent]          ← weather + district risk + nearest CWC
          ↓ (conditional: disaster type route)
        [resource_finder_agent] ← hospitals + shelters + CWC stations from ChromaDB
          ↓
        [matching_agent]        ← LLM ranks & prioritizes resources
          ↓ (conditional: resources found?)
          ├── YES → [route_planning_agent] ← ORS route + NH road risk check
          │           ↓
          └── NO  → [escalation_agent] ← final report + emergency contacts
                      ↓
                     END
        ```
        """)


# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:11px'>
IIT Mandi AAI Himshikhar 2026 Capstone Project |
HP Disaster Relief Resource Matching Agent |
Data: HIMCOSTE 2023 • CWC • NHP • DAY-NULM • HP Education Dept • Open-Meteo
</div>
""", unsafe_allow_html=True)
