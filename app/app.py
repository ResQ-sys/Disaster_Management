"""
HP Disaster Relief Resource Matching Agent
Streamlit UI
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from resq_project.config import AGENT_COORDINATOR_EMAIL, DISASTER_TYPES, DISTRICT_RISK
from resq_project.workflow import run_agent, get_llm
from resq_project.tools import assess_wildfire_risk
from resq_project import chatbot
from resq_project import coordination as coord


@st.cache_data(show_spinner=False)
def wildfire_flag(lat, lon):
    """Cached wildfire-proneness lookup for the given coordinates."""
    return assess_wildfire_risk(lat, lon)


def rag_answer(question):
    """Grounded RAG answer over the ingested HP disaster data."""
    return chatbot.answer(question, get_llm)

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HP Disaster Relief Agent",
    page_icon="🏔️",
    layout="wide"
)

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None
if "agent_error" not in st.session_state:
    st.session_state.agent_error = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_coordinator_email" not in st.session_state:
    st.session_state.agent_coordinator_email = AGENT_COORDINATOR_EMAIL
if "sender_email" not in st.session_state:
    st.session_state.sender_email = ""
if "smtp_password" not in st.session_state:
    st.session_state.smtp_password = ""

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

# ── District center coordinates (fallback when geocoding is unavailable) ─
DISTRICT_COORDS = {
    "KANGRA": (32.10, 76.27), "MANDI": (31.71, 76.93),
    "SHIMLA": (31.10, 77.17), "KULLU": (31.95, 77.11),
    "SOLAN": (30.91, 77.10), "SIRMOUR": (30.56, 77.66),
    "BILASPUR": (31.34, 76.76), "HAMIRPUR": (31.68, 76.52),
    "CHAMBA": (32.55, 76.12), "UNA": (31.47, 76.27),
    "KINNAUR": (31.59, 78.44), "LAHUL AND SPITI": (32.55, 77.60),
}


@st.cache_data(show_spinner=False)
def geocode_location(district: str, location_desc: str):
    """Approximate (lat, lon, source) for the location the user described.

    Tries Nominatim (OpenStreetMap) using the location description + district;
    falls back to the district center coordinates if geocoding fails or the
    query returns nothing.
    """
    fallback = DISTRICT_COORDS.get(district, (31.80, 77.20))

    if location_desc and location_desc.strip():
        query = ", ".join(filter(None, [
            location_desc.strip(),
            district.title(),
            "Himachal Pradesh, India",
        ]))
        try:
            geolocator = Nominatim(user_agent="resq_disaster_agent")
            loc = geolocator.geocode(query, timeout=10)
            if loc:
                return round(loc.latitude, 4), round(loc.longitude, 4), "geocoded (OSM)"
        except Exception:
            pass

    return round(fallback[0], 4), round(fallback[1], 4), "district center"

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
    Himachal Pradesh • Multi-hazard: Flood | Landslide | GLOF | Wildfire | Avalanche | Cloudburst
  </p>
  <p style='color:#94a3b8; margin:4px 0 0 0; font-size:12px;'>
    Powered by: LangGraph • Ollama llama3.2:1b • ChromaDB • Open-Meteo •
    Sources: HIMCOSTE 2023 | CWC | NHP Hospitals | DAY-NULM | HP Education Dept
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# RAG CHATBOT — grounded Q&A over ingested HP disaster data
# ══════════════════════════════════════════════════════════════════════
with st.expander("💬 Ask the HP Disaster Assistant (grounded RAG chatbot)", expanded=False):
    st.caption(
        "Answers **only** from ingested HP disaster data — hospitals, shelters, "
        "CWC river stations, GLOF glacial-lake monitoring, and HP disaster guidance. "
        "Out-of-scope questions get an honest *\"I don't know\"* (no hallucination)."
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("📚 Sources: " + ", ".join(msg["sources"]))

    with st.form("rag_chat_form", clear_on_submit=True):
        user_q = st.text_input(
            "Your question",
            placeholder="e.g. Which hospitals are in Kullu? Which GLOF lakes are increasing?",
            label_visibility="collapsed",
        )
        col_send, col_clear = st.columns([1, 1])
        send = col_send.form_submit_button("Ask ➤", use_container_width=True)
        clear = col_clear.form_submit_button("🗑️ Clear chat", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if send and user_q.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.spinner("🔎 Searching HP disaster data..."):
            result = rag_answer(user_q)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
# VOLUNTEER NEED ↔ RESOURCE MATCHING + HUMAN-IN-THE-LOOP COORDINATION
# ══════════════════════════════════════════════════════════════════════
if "approved_msgs" not in st.session_state:
    st.session_state.approved_msgs = {}   # request_id -> "approved"/"sent"/"rejected"

with st.expander("🤝 Volunteer Need–Resource Matching & Coordination (human-in-the-loop)", expanded=False):
    st.caption(
        "Matches reported **needs** against available **volunteer, NGO, and relief resources** "
        "(deterministic scoring by category, location, quantity, availability). "
        "Every coordination message is emailed to the **agent coordinator email** after approval — "
        "nothing is dispatched automatically."
    )
    st.caption(f"Coordinator email for demo approvals: `{st.session_state.agent_coordinator_email}`")

    # Optional: extract a new need from a free-text message / tweet
    with st.form("extract_need_form", clear_on_submit=True):
        msg_text = st.text_input(
            "Add a need from a field message / tweet (optional, rule-based extraction)",
            placeholder="e.g. URGENT: 30 people trapped, need rescue and medical at Manali ward 3",
        )
        extracted = st.form_submit_button("➕ Extract & add need")
    if "extra_needs" not in st.session_state:
        st.session_state.extra_needs = []
    if extracted and msg_text.strip():
        st.session_state.extra_needs.append(coord.extract_need_from_text(msg_text))
        st.rerun()

    needs = coord.load_needs() + st.session_state.extra_needs
    resources = coord.load_resources()
    matches = coord.match_needs_to_resources(needs, resources)

    st.markdown(f"**Worklist — {len(matches)} needs** "
                f"({sum(1 for m in matches if m['status']=='MATCHED')} matched, "
                f"{sum(1 for m in matches if m['status']!='MATCHED')} need attention), "
                f"sorted by urgency.")

    STATUS_ICON = {"MATCHED": "🟢", "PARTIAL": "🟡", "UNMATCHED": "🔴"}
    for m in matches:
        need = m["need"]
        best = m["best_match"]
        rid = str(need.get("request_id"))
        decided = st.session_state.approved_msgs.get(rid)
        title = (f"{STATUS_ICON.get(m['status'],'⚪')} {rid} · {need.get('category')} · "
                 f"{str(need.get('urgency','')).upper()} · {need.get('location')}"
                 + (f"  →  {best['resource']['provider_name']} ({best['score']}/100)" if best else "  →  no match")
                 + (f"   ✅ {decided.upper()}" if decided else ""))
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if best:
                bcol = st.columns(4)
                bcol[0].metric("Match score", f"{best['score']}/100")
                bcol[1].metric("Coverage", f"{best.get('coverage_pct',0)}%")
                bcol[2].metric("Gap", best.get("quantity_gap", 0))
                bcol[3].metric("Provider verified",
                               "Yes" if str(best['resource'].get('contact_status','')).lower()=="verified" else "No")

            draft = coord.draft_coordination_message(
                m, coordinator_email=st.session_state.agent_coordinator_email)
            edited = st.text_area("Coordination message (editable)", value=draft,
                                  height=200, key=f"msg_{rid}")

            a1, a2, a3 = st.columns(3)
            if a1.button("✅ Approve & mark sent", key=f"send_{rid}", use_container_width=True):
                subject = (
                    f"Disaster coordination approval needed: {need.get('category')} "
                    f"request {rid}"
                )
                result = coord.send_coordinator_email(
                    st.session_state.agent_coordinator_email,
                    subject,
                    edited,
                    st.session_state.sender_email,
                    st.session_state.smtp_password,
                )
                if result["sent"]:
                    coord.log_approval({"request_id": rid, "action": "approved_sent",
                                        "provider": (best or {}).get("resource", {}).get("provider_name", "N/A"),
                                        "category": need.get("category"),
                                        "coordinator_email": st.session_state.agent_coordinator_email,
                                        "message": edited,
                                        "edited": edited != draft})
                    st.session_state.approved_msgs[rid] = "sent"
                    st.rerun()
                st.error(f"Email send failed: {result['error']}")
            if a2.button("📝 Log edit (no send)", key=f"edit_{rid}", use_container_width=True):
                coord.log_approval({"request_id": rid, "action": "edited",
                                    "category": need.get("category"),
                                    "coordinator_email": st.session_state.agent_coordinator_email,
                                    "message": edited})
                st.session_state.approved_msgs[rid] = "approved"
                st.rerun()
            if a3.button("🚫 Reject / escalate", key=f"rej_{rid}", use_container_width=True):
                coord.log_approval({"request_id": rid, "action": "rejected",
                                    "category": need.get("category"),
                                    "coordinator_email": st.session_state.agent_coordinator_email,
                                    "reason": "human rejected"})
                st.session_state.approved_msgs[rid] = "rejected"
                st.rerun()

    # Audit trail
    approvals = coord.read_approvals(limit=20)
    if approvals:
        st.markdown("---")
        st.markdown(f"**🧾 Human-in-the-loop audit log** (last {len(approvals)})")
        for a in reversed(approvals):
            st.caption(f"{a.get('timestamp','')[:19]} · {a.get('request_id')} · "
                       f"**{a.get('action')}** · {a.get('category','')} "
                       f"{'· ✏️ edited' if a.get('edited') else ''}")

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📋 Situation Details")
    st.markdown("---")

    user_name = st.text_input("Your Name", placeholder="e.g. Ramesh Kumar")
    st.session_state.agent_coordinator_email = st.text_input(
        "Agent coordinator email",
        value=st.session_state.agent_coordinator_email,
        help="Volunteer/NGO coordination drafts are addressed to this email for demo approval.")
    st.session_state.sender_email = st.text_input(
        "User email",
        value=st.session_state.sender_email,
        help="Gmail address used to send the coordinator email.")
    st.session_state.smtp_password = st.text_input(
        "SMTP password",
        value=st.session_state.smtp_password,
        type="password",
        help="Use the Gmail app password for the sender email.")

    district = st.selectbox(
        "District *",
        options=sorted(DISTRICT_RISK.keys()),
        help="Select the affected district in Himachal Pradesh"
    )

    location_desc = st.text_input(
        "Location Description",
        placeholder="e.g. Near Kullu bus stand, Beas riverbank"
    )

    # Approximate coordinates are derived automatically from the district +
    # location description rather than entered by hand.
    latitude, longitude, geo_source = geocode_location(district, location_desc)
    st.markdown("**Approximate Coordinates** (auto-derived)")
    st.markdown(
        f"<span style='background:#1e3a5f;color:white;padding:4px 10px;"
        f"border-radius:6px;font-size:13px'>📍 {latitude:.4f}, {longitude:.4f}</span> "
        f"<small style='color:#94a3b8'>· {geo_source}</small>",
        unsafe_allow_html=True
    )

    # ── Wildfire proneness flag (based purely on lat/lon) ─────────────
    wf = wildfire_flag(latitude, longitude)
    WF_COLORS = {"HIGH": "#dc2626", "MODERATE": "#ea580c",
                 "LOW": "#ca8a04", "MINIMAL": "#16a34a", "UNKNOWN": "#6b7280"}
    wf_color = WF_COLORS.get(wf.get("level", "UNKNOWN"), "#6b7280")
    wf_icon = "🔥" if wf.get("prone") else "🟢"
    st.markdown("**🔥 Wildfire Proneness**")
    st.markdown(
        f"<span style='background:{wf_color};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:12px'>{wf_icon} {wf.get('level', 'N/A')}"
        f"{' · PRONE' if wf.get('prone') else ''}</span> "
        f"<small style='color:#94a3b8'>{wf.get('count_10km', 0)} past fires ≤10km</small>",
        unsafe_allow_html=True
    )

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
    🤝 Volunteer & NGO resource pool (demo coordination registry)<br>
    🌊 CWC Stations (52 HP stations)<br>
    ⛰️ HIMCOSTE Landslide Inventory 2023<br>
    🌧️ Open-Meteo (no API key)<br>
    🗺️ OpenRouteService (routing)
    </small>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN PANEL — MAP + RESULTS
# ══════════════════════════════════════════════════════════════════════
if run_btn:
    if not district or not disaster_type or not needs:
        st.session_state.agent_error = "⚠️ Please fill in District, Disaster Type, and at least one Need."
        st.session_state.agent_state = None
    else:
        with st.spinner("🔄 Running disaster response pipeline... (6 LangGraph nodes)"):
            try:
                st.session_state.agent_state = run_agent(
                    user_name=user_name or "Anonymous",
                    district=district,
                    location_desc=location_desc or f"{district}, HP",
                    latitude=latitude,
                    longitude=longitude,
                    disaster_type=disaster_type,
                    needs=needs
                )
                st.session_state.agent_error = None
            except Exception as e:
                st.session_state.agent_error = f"Agent error: {e}"
                st.session_state.agent_state = None

if st.session_state.agent_error:
    st.error(st.session_state.agent_error)

state = st.session_state.agent_state

if state is None:
    # Default view: HP map
    st.markdown("### 🗺️ Himachal Pradesh — Disaster Risk Map")

    m = folium.Map(location=[31.8, 77.2], zoom_start=7, tiles="CartoDB positron")

    # Plot district risk tiers
    tier_colors = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "beige", "LOW": "green"}

    for dist, coords in DISTRICT_COORDS.items():
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
    # ══════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ══════════════════════════════════════════════════════════════════

    # ── Top metrics row ──────────────────────────────────────────────
    weather      = state.get("weather", {})
    district_risk = state.get("district_risk", {})
    alert        = state.get("imd_alert_level", "GREEN")
    alert_color, alert_icon = ALERT_COLORS.get(alert, ("#6b7280", "⚪"))

    glof_alert = state.get("glof_alert", {}) or {}

    urgency = state.get("urgency", {}) or {}

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("🚨 Urgency", f"{urgency.get('score', 'N/A')}/100",
              urgency.get("level", ""))
    m2.metric("🌧️ IMD Alert",    f"{alert_icon} {alert}")
    m3.metric("⛰️ Risk Tier",    district_risk.get("risk_tier", "N/A"))
    m4.metric("🌡️ Temp",        f"{weather.get('temperature_c', 'N/A')}°C")
    m5.metric("💧 Rain (24h)",   f"{weather.get('forecast_rain_24h', 'N/A')} mm")
    m6.metric("🏥 Resources", len(state.get("matched_resources", [])))
    m7.metric("🧊 GLOF", f"{glof_alert.get('level', 'None')}"
              + (f" ({glof_alert.get('count_increasing')})" if glof_alert else ""))

    # ── Urgency banner (explainable breakdown) ───────────────────────
    if urgency:
        u_lvl = urgency.get("level")
        u_banner = (st.error if u_lvl in ("CRITICAL",) else
                    st.warning if u_lvl in ("HIGH",) else st.info)
        bd = urgency.get("breakdown", {})
        bd_txt = " · ".join(f"{k}:{v}" for k, v in bd.items() if v)
        u_banner(f"🚨 **Urgency {urgency.get('score')}/100 ({u_lvl})** — {bd_txt}")

    # ── GLOF alert banner ────────────────────────────────────────────
    if glof_alert:
        banner = st.error if glof_alert.get("level") == "WATCH" else st.warning
        banner(f"🧊 **GLOF {glof_alert.get('level')}** — {glof_alert.get('message')}")
        st.caption(f"⏳ {glof_alert.get('disclaimer')}")

    # ── Wildfire proneness banner ────────────────────────────────────
    wf = state.get("wildfire_risk", {}) or {}
    if wf.get("prone"):
        st.warning(f"🔥 **Wildfire proneness: {wf.get('level')}** — {wf.get('message')} "
                   f"({wf.get('count_10km', 0)} past fire detections within 10 km)")
        st.caption(f"⏳ {wf.get('disclaimer')}")
    elif wf:
        st.info(f"🟢 **Wildfire proneness: {wf.get('level')}** — {wf.get('message')}")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Response Report", "🏥 Resources", "🗺️ Route & Map",
        "🌊 CWC Stations", "🧊 GLOF Watch", "⚙️ Agent Log"
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

        # ── Human-in-the-loop coordination message for this request ──
        st.markdown("---")
        st.markdown("**🤝 Coordination Message — human approval required**")
        st.caption("Draft message for the coordinator email. Review/edit, then approve to send it by email. "
                   "Nothing is sent automatically before approval.")
        priority = state.get("priority_resource", {})
        pseudo_match = {
            "need": {
                "request_id": f"AGENT-{state.get('district','')[:3].upper()}",
                "reported_by": state.get("user_name", "Anonymous"),
                "location": state.get("location_desc", ""),
                "category": ", ".join(state.get("needs", [])) or "Assistance",
                "quantity": "", "urgency": urgency.get("level", ""),
                "notes": f"{state.get('disaster_type','')} — urgency {urgency.get('score','?')}/100",
            },
            "best_match": {
                "resource": {
                    "provider_name": priority.get("name", "N/A"),
                    "location": priority.get("district", district),
                    "quantity": priority.get("capacity", ""),
                    "availability": "Listed", "contact_status": "Unverified",
                },
                "score": 0, "coverage_pct": 0, "quantity_gap": 0,
            } if priority else None,
        }
        agent_rid = pseudo_match["need"]["request_id"]
        decided = st.session_state.get("approved_msgs", {}).get(agent_rid)
        draft = coord.draft_coordination_message(
            pseudo_match, coordinator_email=st.session_state.agent_coordinator_email)
        edited = st.text_area("Coordination message (editable)", value=draft, height=200,
                              key=f"agentmsg_{agent_rid}")
        ac1, ac2 = st.columns(2)
        if ac1.button("✅ Approve & mark sent", key=f"agentsend_{agent_rid}", use_container_width=True):
            subject = (
                f"Disaster coordination approval needed: {state.get('disaster_type','Incident')} "
                f"{agent_rid}"
            )
            result = coord.send_coordinator_email(
                st.session_state.agent_coordinator_email,
                subject,
                edited,
                st.session_state.sender_email,
                st.session_state.smtp_password,
            )
            if result["sent"]:
                coord.log_approval({"request_id": agent_rid, "action": "approved_sent",
                                    "provider": priority.get("name", "N/A"),
                                    "category": ", ".join(state.get("needs", [])),
                                    "coordinator_email": st.session_state.agent_coordinator_email,
                                    "message": edited, "edited": edited != draft})
                st.session_state.setdefault("approved_msgs", {})[agent_rid] = "sent"
                st.rerun()
            st.error(f"Email send failed: {result['error']}")
        if ac2.button("🚫 Reject / escalate", key=f"agentrej_{agent_rid}", use_container_width=True):
            coord.log_approval({"request_id": agent_rid, "action": "rejected",
                                "coordinator_email": st.session_state.agent_coordinator_email,
                                "reason": "human rejected"})
            st.session_state.setdefault("approved_msgs", {})[agent_rid] = "rejected"
            st.rerun()
        if decided:
            st.success(f"Coordinator decision logged: **{decided.upper()}**")

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
                    if s.get("type") != "GOVT_SCHOOL_SHELTER":
                        st.write(f"**Capacity:** {s.get('capacity', 'N/A')}")
                        st.write(f"**Contact:** {s.get('contact', 'N/A')}")
                    else:
                        st.caption("School shelter details are activated through district administration.")

        st.markdown("---")
        st.markdown(f"**Matching Reasoning:** {state.get('match_reasoning', 'N/A')}")

    # ── TAB 3: Route & Map ───────────────────────────────────────────
    with tab3:
        route = state.get("route", {})
        routes = state.get("routes", [])
        road_risks = state.get("road_risks", [])

        TYPE_META = {
            "HOSPITAL": ("🏥", "Hospital"),
            "SHELTER":  ("🏠", "Shelter"),
        }

        st.markdown("**🚑 Distance & Estimated Time to Each Resource**")
        if routes:
            cols = st.columns(len(routes))
            for col, r in zip(cols, routes):
                icon, label = TYPE_META.get(r.get("resource_type", ""), ("📍", "Resource"))
                dist_val = r.get("distance_km")
                dur_val  = r.get("duration_min")
                approx   = r.get("source") in ("straight_line_approx", "same_locality")
                prefix   = "≈ " if approx else ""
                with col:
                    st.markdown(f"**{icon} {label}** — {r.get('name', 'N/A')}")
                    c1, c2 = st.columns(2)
                    c1.metric("📍 Distance", f"{prefix}{dist_val} km" if dist_val is not None else "N/A")
                    c2.metric("⏱️ Time", f"{prefix}{dur_val} min" if dur_val is not None else "N/A")
                    if r.get("source") == "straight_line_approx":
                        st.caption("≈ straight-line estimate (add ORS_API_KEY for road distance)")
                    elif r.get("dest_source", "").startswith("approx"):
                        st.caption(f"approx location · {r['dest_source']}")
                    if r.get("error"):
                        st.caption(f"⚠️ {r['error']}")
        else:
            st.info("No hospital/shelter routes available.")

        if route.get("error"):
            st.error(f"🛣️ Routing unavailable — {route['error']}")

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
        st.caption(f"📍 Approximate location: {latitude:.4f}, {longitude:.4f} ({geo_source})")
        m = folium.Map(location=[latitude, longitude], zoom_start=11, tiles="CartoDB positron")
        folium.Marker([latitude, longitude],
                      popup="Your Location",
                      icon=folium.Icon(color="red", icon="exclamation-sign")).add_to(m)

        # One marker + connecting line per typed route (hospital / shelter)
        MARKER_STYLE = {
            "HOSPITAL": ("green", "plus-sign", "#16a34a"),
            "SHELTER":  ("blue", "home", "#2563eb"),
        }
        for r in routes:
            if not (r.get("dest_lat") and r.get("dest_lon")):
                continue
            dest_ll = [r["dest_lat"], r["dest_lon"]]
            m_color, m_icon, line_color = MARKER_STYLE.get(
                r.get("resource_type", ""), ("gray", "info-sign", "#6b7280"))
            dist_val = r.get("distance_km")
            folium.Marker(
                dest_ll,
                popup=f"{r.get('resource_type', 'Resource')}: {r.get('name', 'N/A')}"
                      + (f" ({dist_val} km)" if dist_val is not None else ""),
                icon=folium.Icon(color=m_color, icon=m_icon)
            ).add_to(m)
            if dest_ll != [latitude, longitude]:
                folium.PolyLine([[latitude, longitude], dest_ll],
                                color=line_color, weight=3, opacity=0.7).add_to(m)

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
            "Check **[ffs.india-water.gov.in](https://ffs.india-water.gov.in)** for live water level data."
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

    # ── TAB 5: GLOF Watch ────────────────────────────────────────────
    with tab5:
        st.markdown("### 🧊 Glacial Lake Outburst Flood (GLOF) Watch")
        st.info(
            "⏳ **Note:** This data is based on **previous-year monthly satellite "
            "monitoring** by the Central Water Commission (September 2025) — it reflects "
            "**water-spread-area trends, not real-time water levels**. Expanding lakes "
            "indicate *elevated* GLOF risk; always verify with live CWC advisories."
        )

        if glof_alert:
            lvl = glof_alert.get("level")
            (st.error if lvl == "WATCH" else st.warning)(
                f"🧊 **GLOF {lvl}** — {glof_alert.get('count_increasing')} expanding "
                f"glacial lake(s) near {district.title()} "
                f"({glof_alert.get('count_in_district', 0)} in-district)."
            )
            nr = glof_alert.get("nearest", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Nearest expanding lake", nr.get("lake_id", "N/A"))
            c2.metric("Distance", f"{nr.get('distance_km', 'N/A')} km")
            c3.metric("Area change", f"+{nr.get('area_pct_change', 'N/A')}%")
        else:
            st.success("✅ No expanding glacial lakes flagged near this location in the latest monitoring.")

        lakes = state.get("glacial_lakes", [])
        if lakes:
            st.markdown("**Monitored glacial lakes (nearest first):**")
            # Map of lakes
            glof_map = folium.Map(location=[latitude, longitude], zoom_start=8, tiles="CartoDB positron")
            folium.Marker([latitude, longitude], popup="Your Location",
                          icon=folium.Icon(color="red", icon="exclamation-sign")).add_to(glof_map)
            for lk in lakes:
                rising = lk.get("status") == "increase"
                color = "red" if rising else ("blue" if lk.get("status") == "decrease" else "gray")
                try:
                    glof_map.add_child(folium.CircleMarker(
                        location=[float(lk["latitude"]), float(lk["longitude"])],
                        radius=7, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                        popup=f"{lk['lake_id']} · {lk.get('river') or lk.get('basin')} · "
                              f"{lk.get('status')} ({lk.get('area_pct_change')}%)"
                    ))
                except (TypeError, ValueError):
                    pass
            st_folium(glof_map, width=850, height=380)

            for lk in lakes:
                icon = "🔴" if lk.get("status") == "increase" else ("🔵" if lk.get("status") == "decrease" else "⚪")
                with st.expander(
                    f"{icon} {lk.get('lake_id')} — {lk.get('district', '').title()} "
                    f"({lk.get('status')}, {lk.get('area_pct_change')}%)"
                ):
                    st.write(f"**Basin / River:** {lk.get('basin')} / {lk.get('river') or 'N/A'}")
                    st.write(f"**Coordinates:** {lk.get('latitude')}, {lk.get('longitude')}")
                    st.write(f"**Distance from you:** {lk.get('distance_km', 'N/A')} km")
                    st.write(f"**Monitored period:** {lk.get('monitored_period')}")
                    st.write(f"**Source:** {lk.get('source')}")
        else:
            st.caption("No glacial-lake monitoring records available for this area.")

    # ── TAB 6: Agent Log ─────────────────────────────────────────────
    with tab6:
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
        [glof_monitor_agent]    ← expanding glacial lakes (CWC GLOF monitoring)
          ↓
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
Data: HIMCOSTE 2023 • CWC (incl. GLOF Glacial Lake Monitoring Sep 2025) • VIIRS Wildfire History • NHP • DAY-NULM • HP Education Dept • Open-Meteo
</div>
""", unsafe_allow_html=True)
