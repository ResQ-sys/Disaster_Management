"""
HP Disaster Relief Resource Matching Agent
Streamlit UI
"""

import base64

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
from resq_project.chatbot import nearest_hospitals, nearest_shelters
from resq_project import coordination as coord
from resq_project import ndma_alerts
from resq_project.i18n import t, district_label, disaster_type_label, need_label, code_label

# Disaster-type icon shown as a corner badge on the district map
DISASTER_ICONS = {
    "Flash Flood": "🌊", "Landslide": "🏔️", "Cloudburst": "🌧️",
    "GLOF": "🧊", "Wildfire": "🔥", "Avalanche": "❄️",
    "Drought": "☀️", "Road Blockage": "🚧",
}


def render_stat_cards(cards: list[tuple[str, str, str, str]]) -> None:
    """Render metric cards in a responsive CSS grid so labels/values wrap
    onto a new line instead of being clipped — st.metric truncates in
    narrow columns, which was cutting off longer (and Hindi) text.

    Each card is (label, value, sub, accent_color) — the accent color reuses
    the app's existing semantic palette (ALERT_COLORS/RISK_COLORS/etc.) as a
    left border, so the cards read as part of the same visual language as
    the rest of the app rather than generic boxes.
    """
    cards_html = "".join(
        f"""
        <div style='background:white; border-left:4px solid {color}; border-radius:8px;
                    padding:14px 12px; box-shadow:0 1px 3px rgba(0,0,0,0.08);'>
          <div style='font-size:12px; color:#64748b; margin-bottom:6px; word-break:break-word;'>{label}</div>
          <div style='font-size:19px; font-weight:700; color:#1e293b; word-break:break-word;'>{value}</div>
          {f"<div style='font-size:12px; color:#64748b; margin-top:2px; word-break:break-word;'>{sub}</div>" if sub else ""}
        </div>"""
        for label, value, sub, color in cards
    )
    st.markdown(
        f"<div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(140px,1fr)); "
        f"gap:12px; margin-bottom:16px;'>{cards_html}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_district_facilities(district: str, user_lat: float, user_lon: float):
    """Hospitals + shelters for the district, geocoded and sorted. Cached so
    other widget interactions elsewhere on the page don't re-trigger the
    (slow, per-facility) geocoding."""
    hospitals = nearest_hospitals(district, user_lat, user_lon, limit=15)
    shelters = nearest_shelters(district, user_lat, user_lon, limit=15)
    return hospitals, shelters


def render_district_map(district: str, user_lat: float, user_lon: float, disaster_type: str):
    """Disaster Risk Map — shown only after a report is generated. Zoomed
    into the reported district, plotting that district's hospitals and
    shelters, with the reported disaster type badged in the corner."""
    st.markdown(t("### 🗺️ Himachal Pradesh — Disaster Risk Map"))

    with st.spinner(t("📍 Loading hospitals and shelters for {d}...", d=district_label(district))):
        hospitals, shelters = get_district_facilities(district, user_lat, user_lon)

    m = folium.Map(location=[user_lat, user_lon], zoom_start=11, tiles="CartoDB positron")

    folium.Marker(
        [user_lat, user_lon], popup=t("Your Location"),
        icon=folium.Icon(color="red", icon="exclamation-sign"),
    ).add_to(m)

    for h in hospitals:
        if h.get("latitude") and h.get("longitude"):
            folium.Marker(
                [h["latitude"], h["longitude"]],
                popup=f"🏥 {h.get('name', 'N/A')} — {h.get('distance_km', '?')} km",
                icon=folium.Icon(color="green", icon="plus-sign"),
            ).add_to(m)

    for s in shelters:
        if s.get("latitude") and s.get("longitude"):
            folium.Marker(
                [s["latitude"], s["longitude"]],
                popup=f"🏠 {s.get('name', 'N/A')} — {s.get('distance_km', '?')} km",
                icon=folium.Icon(color="blue", icon="home"),
            ).add_to(m)

    # Disaster-type badge, injected into the map's own HTML root (not a
    # sibling Streamlit element) so it reliably overlays the map's corner
    # instead of risking cross-iframe stacking issues.
    icon = DISASTER_ICONS.get(disaster_type, "⚠️")
    badge_html = f"""
    <div style="position:absolute; top:10px; right:10px; z-index:9999;
                background:white; border-radius:8px; padding:6px 12px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3);
                display:flex; align-items:center; gap:8px;">
      <span style="font-size:26px; line-height:1;">{icon}</span>
      <span style="font-size:13px; font-weight:600; color:#1e293b;">{disaster_type_label(disaster_type)}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(badge_html))

    st_folium(m, width=900, height=500, key="district_risk_map")
    st.caption(t(
        "🏥 {nh} hospitals · 🏠 {ns} shelters shown for {d}",
        nh=len(hospitals), ns=len(shelters), d=district_label(district),
    ))


@st.cache_data(show_spinner=False)
def wildfire_flag(lat, lon):
    """Cached wildfire-proneness lookup for the given coordinates."""
    return assess_wildfire_risk(lat, lon)


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
if "hp_chat" not in st.session_state:
    st.session_state.hp_chat = None
if "agent_coordinator_email" not in st.session_state:
    st.session_state.agent_coordinator_email = AGENT_COORDINATOR_EMAIL
if "sender_email" not in st.session_state:
    st.session_state.sender_email = ""
if "smtp_password" not in st.session_state:
    st.session_state.smtp_password = ""
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "language_pref" not in st.session_state:
    st.session_state.language_pref = "English"

# ══════════════════════════════════════════════════════════════════════
# LANDING PAGE — collects name / email / SMTP password / language once,
# before the rest of the app renders
# ══════════════════════════════════════════════════════════════════════
LANGUAGE_OPTIONS = ["English", "Hindi"]


def render_landing_page():
    # The landing page form itself always renders in whatever language was
    # last selected in this session (default English), so switching to
    # Hindi and reopening "Edit login details" shows the form in Hindi too.
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d6a4f 100%);
                padding: 36px 40px; border-radius: 16px; margin: 40px auto 24px auto;
                max-width: 560px; text-align:center;'>
      <h1 style='color:white; margin:0; font-size:26px;'>{t("🏔️ HP Disaster Relief Agent")}</h1>
      <p style='color:#a8d5e2; margin:10px 0 0 0; font-size:14px;'>
        {t("Tell us who you are before we start — this personalizes AI-generated reports "
           "and chat answers, and lets you send coordination emails directly from the app.")}
      </p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("landing_form"):
            name = st.text_input(t("Your name *"), placeholder="e.g. Ramesh Kumar")
            language = st.selectbox(
                t("Language preference"), LANGUAGE_OPTIONS,
                index=LANGUAGE_OPTIONS.index(st.session_state.language_pref),
                help=t("This language will be used across the whole app — reports, chat "
                       "answers, and on-screen labels/buttons.")
            )

            st.markdown("---")
            st.caption(
                t("📧 **Email & SMTP password are optional** — only required if you want the "
                  "app to send coordination emails **directly to relief authorities** on your "
                  "behalf. You can leave these blank and still use the rest of the app.")
            )
            email = st.text_input(t("Your email"), placeholder="you@example.com")
            smtp_password = st.text_input(
                t("SMTP password"), type="password",
                help=t("Gmail app password for the email above — only used if you actually "
                       "send a coordination email."))

            submitted = st.form_submit_button(t("🚀 Continue to app"), use_container_width=True)

        if submitted:
            if not name.strip():
                st.error(t("Please enter your name to continue."))
            else:
                st.session_state.user_name = name.strip()
                st.session_state.sender_email = email.strip()
                st.session_state.smtp_password = smtp_password
                st.session_state.language_pref = language
                st.session_state.logged_in = True
                st.rerun()


if not st.session_state.logged_in:
    render_landing_page()
    st.stop()

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
    "MODERATE": "#ca8a04",
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
# HEADER — a clean flat-illustration mountain skyline (self-contained SVG,
# baked-in gradient sky) instead of a plain CSS gradient block.
# ══════════════════════════════════════════════════════════════════════
_SKYLINE_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 300' preserveAspectRatio='xMidYMax slice'>"
    "<defs><linearGradient id='sky' x1='0' y1='0' x2='0' y2='1'>"
    "<stop offset='0%' stop-color='#1e3a5f'/><stop offset='100%' stop-color='#2d6a4f'/>"
    "</linearGradient></defs>"
    "<rect width='1200' height='300' fill='url(#sky)'/>"
    "<polygon points='0,300 0,200 220,100 440,195 660,120 880,210 1080,130 1200,190 1200,300' "
    "fill='#ffffff' fill-opacity='0.06'/>"
    "<polygon points='0,300 0,250 260,150 520,245 780,165 1040,250 1200,210 1200,300' "
    "fill='#ffffff' fill-opacity='0.10'/>"
    "</svg>"
)
_SKYLINE_URI = "data:image/svg+xml;base64," + base64.b64encode(_SKYLINE_SVG.encode()).decode()

st.markdown(f"""
<div style="background-image:url('{_SKYLINE_URI}'); background-size:cover;
            background-position:center; padding:26px 32px; border-radius:12px;
            margin-bottom:24px;">
  <h1 style='color:white; margin:0; font-size:28px;'>
    {t("🏔️ HP Disaster Relief Resource Matching Agent")}
  </h1>
  <p style='color:#a8d5e2; margin:8px 0 0 0; font-size:14px;'>
    {t("Himachal Pradesh • Multi-hazard: Flood | Landslide | GLOF | Wildfire | Avalanche | Cloudburst")}
  </p>
  <p style='color:#94a3b8; margin:4px 0 0 0; font-size:12px;'>
    {t("Powered by: LangGraph • Ollama llama3.2:1b • ChromaDB • Open-Meteo • "
       "Sources: HIMCOSTE 2023 | CWC | NHP Hospitals | DAY-NULM | HP Education Dept")}
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# NDMA SACHET HEADLINE ALERTS — live CAP feed, last 7 days, English headline
# read directly from each alert's official English <cap:info> block
# ══════════════════════════════════════════════════════════════════════
with st.spinner("📰 Fetching latest NDMA alerts..."):
    _ndma_alerts = ndma_alerts.get_alerts(7)
ndma_alerts.render_headline_ticker(_ndma_alerts)

# ══════════════════════════════════════════════════════════════════════
# RAG CHATBOT — grounded Q&A over ingested HP disaster data
# ══════════════════════════════════════════════════════════════════════
with st.expander(
    t("💬 HP Disaster Assistant"),
    expanded=bool(st.session_state.agent_state),
):
    st.caption(
        t("One chatbot for two jobs: grounded Q&A over ingested HP disaster data "
          "(hospitals, shelters, CWC stations, GLOF monitoring, disaster guidance), and "
          "guided, **informational-only** help with property damage — it never files, "
          "submits, or stores anything.")
    )
    chatbot.render_hp_assistant(st.session_state.agent_state, get_llm, st.session_state.language_pref)

# ══════════════════════════════════════════════════════════════════════
# VOLUNTEER NEED ↔ RESOURCE MATCHING + HUMAN-IN-THE-LOOP COORDINATION
# ══════════════════════════════════════════════════════════════════════
if "approved_msgs" not in st.session_state:
    st.session_state.approved_msgs = {}   # request_id -> "approved"/"sent"/"rejected"

with st.expander(t("🤝 Volunteer Need–Resource Matching & Coordination (human-in-the-loop)"), expanded=False):
    st.caption(
        t("Matches reported **needs** against available **volunteer, NGO, and relief resources** "
          "(deterministic scoring by category, location, quantity, availability). "
          "Every coordination message is emailed to the **agent coordinator email** after approval — "
          "nothing is dispatched automatically.")
    )
    st.caption(t("Coordinator email for demo approvals: `{email}`", email=st.session_state.agent_coordinator_email))

    # Optional: extract a new need from a free-text message / tweet
    with st.form("extract_need_form", clear_on_submit=True):
        msg_text = st.text_input(
            t("Add a need from a field message / tweet (optional, rule-based extraction)"),
            placeholder=t("e.g. URGENT: 30 people trapped, need rescue and medical at Manali ward 3"),
        )
        extracted = st.form_submit_button(t("➕ Extract & add need"))
    if "extra_needs" not in st.session_state:
        st.session_state.extra_needs = []
    if extracted and msg_text.strip():
        st.session_state.extra_needs.append(coord.extract_need_from_text(msg_text))
        st.rerun()

    needs = coord.load_needs() + st.session_state.extra_needs
    resources = coord.load_resources()
    matches = coord.match_needs_to_resources(needs, resources)

    st.markdown(t(
        "**Worklist — {n} needs** ({matched} matched, {unmatched} need attention), sorted by urgency.",
        n=len(matches),
        matched=sum(1 for m in matches if m['status'] == 'MATCHED'),
        unmatched=sum(1 for m in matches if m['status'] != 'MATCHED'),
    ))

    STATUS_ICON = {"MATCHED": "🟢", "PARTIAL": "🟡", "UNMATCHED": "🔴"}
    for m in matches:
        need = m["need"]
        best = m["best_match"]
        rid = str(need.get("request_id"))
        decided = st.session_state.approved_msgs.get(rid)
        title = (f"{STATUS_ICON.get(m['status'],'⚪')} {rid} · {need_label(need.get('category'))} · "
                 f"{code_label(str(need.get('urgency','')).upper())} · {need.get('location')}"
                 + (f"  →  {best['resource']['provider_name']} ({best['score']}/100)" if best else f"  →  {t('no match')}")
                 + (f"   ✅ {code_label(decided.upper())}" if decided else ""))
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if best:
                bcol = st.columns(4)
                bcol[0].metric(t("Match score"), f"{best['score']}/100")
                bcol[1].metric(t("Coverage"), f"{best.get('coverage_pct',0)}%")
                bcol[2].metric(t("Gap"), best.get("quantity_gap", 0))
                bcol[3].metric(t("Provider verified"),
                               t("Yes") if str(best['resource'].get('contact_status','')).lower()=="verified" else t("No"))

            draft = coord.draft_coordination_message(
                m, coordinator_email=st.session_state.agent_coordinator_email)
            edited = st.text_area(t("Coordination message (editable)"), value=draft,
                                  height=200, key=f"msg_{rid}")

            a1, a2, a3 = st.columns(3)
            if a1.button(t("✅ Approve & mark sent"), key=f"send_{rid}", use_container_width=True):
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
                st.error(t("Email send failed: {error}", error=result['error']))
            if a2.button(t("📝 Log edit (no send)"), key=f"edit_{rid}", use_container_width=True):
                coord.log_approval({"request_id": rid, "action": "edited",
                                    "category": need.get("category"),
                                    "coordinator_email": st.session_state.agent_coordinator_email,
                                    "message": edited})
                st.session_state.approved_msgs[rid] = "approved"
                st.rerun()
            if a3.button(t("🚫 Reject / escalate"), key=f"rej_{rid}", use_container_width=True):
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
        st.markdown(t("**🧾 Human-in-the-loop audit log** (last {n})", n=len(approvals)))
        for a in reversed(approvals):
            st.caption(f"{a.get('timestamp','')[:19]} · {a.get('request_id')} · "
                       f"**{code_label(str(a.get('action','')).upper())}** · {need_label(a.get('category',''))} "
                       f"{t('· ✏️ edited') if a.get('edited') else ''}")

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### 👤 {t('Your details (set at login)')}")
    st.text_input(t("Your Name"), value=st.session_state.user_name, disabled=True, key="sb_name")
    st.text_input(t("Your email"), value=st.session_state.sender_email, disabled=True, key="sb_email")
    st.text_input(t("SMTP password"), value=st.session_state.smtp_password, type="password",
                  disabled=True, key="sb_smtp")
    st.text_input(t("Language preference"), value=st.session_state.language_pref, disabled=True, key="sb_lang")
    if st.button(t("✏️ Edit login details"), use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
    user_name = st.session_state.user_name

    st.markdown("---")
    st.markdown(f"### {t('📋 Situation Details')}")
    st.markdown("---")

    st.session_state.agent_coordinator_email = st.text_input(
        t("Agent coordinator email"),
        value=st.session_state.agent_coordinator_email,
        help=t("Volunteer/NGO coordination drafts are addressed to this email for demo approval."))

    district = st.selectbox(
        t("District *"),
        options=sorted(DISTRICT_RISK.keys()),
        format_func=district_label,
        help=t("Select the affected district in Himachal Pradesh")
    )

    location_desc = st.text_input(
        t("Location Description"),
        placeholder=t("e.g. Near Kullu bus stand, Beas riverbank")
    )

    # Approximate coordinates are derived automatically from the district +
    # location description rather than entered by hand.
    latitude, longitude, geo_source = geocode_location(district, location_desc)
    st.markdown(t("**Approximate Coordinates** (auto-derived)"))
    st.markdown(
        f"<span style='background:#1e3a5f;color:white;padding:4px 10px;"
        f"border-radius:6px;font-size:13px'>📍 {latitude:.4f}, {longitude:.4f}</span> "
        f"<small style='color:#94a3b8'>· {t(geo_source)}</small>",
        unsafe_allow_html=True
    )

    # ── Wildfire proneness flag (based purely on lat/lon) ─────────────
    wf = wildfire_flag(latitude, longitude)
    WF_COLORS = {"HIGH": "#dc2626", "MODERATE": "#ea580c",
                 "LOW": "#ca8a04", "MINIMAL": "#16a34a", "UNKNOWN": "#6b7280"}
    wf_color = WF_COLORS.get(wf.get("level", "UNKNOWN"), "#6b7280")
    wf_icon = "🔥" if wf.get("prone") else "🟢"
    st.markdown(t("**🔥 Wildfire Proneness**"))
    st.markdown(
        f"<span style='background:{wf_color};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:12px'>{wf_icon} {code_label(wf.get('level', 'N/A'))}"
        f"{t(' · PRONE') if wf.get('prone') else ''}</span> "
        f"<small style='color:#94a3b8'>{t('{n} past fires ≤10km', n=wf.get('count_10km', 0))}</small>",
        unsafe_allow_html=True
    )

    disaster_type = st.selectbox(t("Disaster Type *"), DISASTER_TYPES, format_func=disaster_type_label)

    needs = st.multiselect(
        t("Immediate Needs *"),
        ["Medical", "Shelter", "Food", "Rescue", "Evacuation", "Water"],
        default=["Medical", "Shelter"],
        format_func=need_label,
    )

    st.markdown("---")
    run_btn = st.button(t("🚨 Find Relief Resources"), type="primary", use_container_width=True)
    st.markdown("---")

    # Quick reference
    risk_info = DISTRICT_RISK.get(district, {})
    st.markdown(t("{d} Risk Profile", d=district_label(district)))
    tier = risk_info.get("tier", "UNKNOWN")
    color = RISK_COLORS.get(tier, "#6b7280")
    st.markdown(
        f"<span style='background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px'>{code_label(tier)}</span>",
        unsafe_allow_html=True
    )
    st.markdown(t("Landslides (2023): **{n}**", n=risk_info.get('landslides_2023', 'N/A')))
    st.markdown(t("Key Rivers: {rivers}", rivers=', '.join(risk_info.get('key_rivers', []))))

    st.markdown("---")
    st.markdown(t("**Data Sources**"))
    st.markdown(t("""
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
    """), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN PANEL — MAP + RESULTS
# ══════════════════════════════════════════════════════════════════════
if run_btn:
    if not district or not disaster_type or not needs:
        st.session_state.agent_error = t("⚠️ Please fill in District, Disaster Type, and at least one Need.")
        st.session_state.agent_state = None
    else:
        with st.spinner(t("🔄 Running disaster response pipeline... (6 LangGraph nodes)")):
            try:
                st.session_state.agent_state = run_agent(
                    user_name=user_name or "Anonymous",
                    district=district,
                    location_desc=location_desc or f"{district}, HP",
                    latitude=latitude,
                    longitude=longitude,
                    disaster_type=disaster_type,
                    needs=needs,
                    language=st.session_state.language_pref,
                )
                st.session_state.agent_error = None
                # Fresh incident → fresh assistant conversation (re-seeds the alert prompt).
                st.session_state.hp_chat = None
            except Exception as e:
                st.session_state.agent_error = t("Agent error: {error}", error=e)
                st.session_state.agent_state = None

if st.session_state.agent_error:
    st.error(st.session_state.agent_error)

state = st.session_state.agent_state

if state is None:
    # Default view: no map yet — the Disaster Risk Map only renders once a
    # report has been generated, focused on the reported district.
    st.info(t("ℹ️ Fill in the sidebar and click **Find Relief Resources** to activate the agent."))

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

    urgency_color = RISK_COLORS.get(str(urgency.get("level", "")).upper(), "#6b7280")
    alert_row_color = ALERT_COLORS.get(alert, ("#6b7280", "⚪"))[0]
    risk_color = RISK_COLORS.get(str(district_risk.get("risk_tier", "")).upper(), "#6b7280")
    glof_color = "#dc2626" if glof_alert.get("level") == "WATCH" else (
        "#ea580c" if glof_alert else "#16a34a")

    render_stat_cards([
        (t("🚨 Urgency"), f"{urgency.get('score', 'N/A')}/100", code_label(urgency.get("level", "")), urgency_color),
        (t("🌧️ IMD Alert"), f"{alert_icon} {code_label(alert)}", "", alert_row_color),
        (t("⛰️ Risk Tier"), code_label(district_risk.get("risk_tier", "N/A")), "", risk_color),
        (t("🌡️ Temp"), f"{weather.get('temperature_c', 'N/A')}°C", "", "#3b82f6"),
        (t("💧 Rain (24h)"), f"{weather.get('forecast_rain_24h', 'N/A')} mm", "", "#3b82f6"),
        (t("🏥 Resources"), str(len(state.get("matched_resources", []))), "", "#16a34a"),
        (t("🧊 GLOF"), code_label(glof_alert.get('level', 'None'))
         + (f" ({glof_alert.get('count_increasing')})" if glof_alert else ""), "", glof_color),
    ])

    # ── Urgency banner (explainable breakdown) ───────────────────────
    if urgency:
        u_lvl = urgency.get("level")
        u_banner = (st.error if u_lvl in ("CRITICAL",) else
                    st.warning if u_lvl in ("HIGH",) else st.info)
        bd = urgency.get("breakdown", {})
        bd_txt = " · ".join(f"{k}:{v}" for k, v in bd.items() if v)
        u_banner(t("🚨 **Urgency {score}/100 ({level})** — {breakdown}",
                   score=urgency.get('score'), level=code_label(u_lvl), breakdown=bd_txt))

    # ── GLOF alert banner ────────────────────────────────────────────
    if glof_alert:
        banner = st.error if glof_alert.get("level") == "WATCH" else st.warning
        banner(t("🧊 **GLOF {level}** — {message}",
                 level=code_label(glof_alert.get('level')), message=glof_alert.get('message')))
        st.caption(f"⏳ {glof_alert.get('disclaimer')}")

    # ── Wildfire proneness banner ────────────────────────────────────
    wf = state.get("wildfire_risk", {}) or {}
    if wf.get("prone"):
        st.warning(t("🔥 **Wildfire proneness: {level}** — {message} "
                     "({n} past fire detections within 10 km)",
                     level=code_label(wf.get('level')), message=wf.get('message'),
                     n=wf.get('count_10km', 0)))
        st.caption(f"⏳ {wf.get('disclaimer')}")
    elif wf:
        st.info(t("🟢 **Wildfire proneness: {level}** — {message}",
                  level=code_label(wf.get('level')), message=wf.get('message')))

    st.markdown("---")

    # ── Disaster Risk Map — district-focused, hospitals + shelters + the
    # reported disaster type, generated only now that the report exists ──
    render_district_map(district, latitude, longitude, disaster_type)

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        t("📋 Response Report"), t("🏥 Resources"), t("🗺️ Route & Map"),
        t("🌊 CWC Stations"), t("🧊 GLOF Watch"), t("⚙️ Agent Log")
    ])

    # ── TAB 1: Final Report ──────────────────────────────────────────
    with tab1:
        if state.get("escalation_needed"):
            st.error(t("🚨 ESCALATION REQUIRED — Insufficient local resources found"))

        report = state.get("final_report") or t("No report generated.")
        st.markdown(report)

        st.markdown("---")
        st.markdown(t("**📞 Emergency Contacts**"))
        for contact in state.get("emergency_contacts", []):
            st.markdown(f"• {contact}")

        # ── Human-in-the-loop coordination message for this request ──
        st.markdown("---")
        st.markdown(t("**🤝 Coordination Message — human approval required**"))
        st.caption(t("Draft message for the coordinator email. Review/edit, then approve to send it by email. "
                     "Nothing is sent automatically before approval."))
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
        edited = st.text_area(t("Coordination message (editable)"), value=draft, height=200,
                              key=f"agentmsg_{agent_rid}")
        ac1, ac2 = st.columns(2)
        if ac1.button(t("✅ Approve & mark sent"), key=f"agentsend_{agent_rid}", use_container_width=True):
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
            st.error(t("Email send failed: {error}", error=result['error']))
        if ac2.button(t("🚫 Reject / escalate"), key=f"agentrej_{agent_rid}", use_container_width=True):
            coord.log_approval({"request_id": agent_rid, "action": "rejected",
                                "coordinator_email": st.session_state.agent_coordinator_email,
                                "reason": "human rejected"})
            st.session_state.setdefault("approved_msgs", {})[agent_rid] = "rejected"
            st.rerun()
        if decided:
            st.success(t("Coordinator decision logged: **{decision}**", decision=decided.upper()))

    # ── TAB 2: Resources ─────────────────────────────────────────────
    with tab2:
        priority = state.get("priority_resource", {})
        if priority:
            st.success(t("✅ **Priority Resource: {name}**", name=priority.get('name', 'N/A')))
            col1, col2 = st.columns(2)
            col1.markdown(t("**Type:** {v}", v=priority.get('resource_type', priority.get('type', 'N/A'))))
            col1.markdown(t("**Contact:** {v}", v=priority.get('contact', 'N/A')))
            col2.markdown(t("**District:** {v}", v=priority.get('district', district)))

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(t("**🏥 Hospitals ({n} found)**", n=len(state.get('hospitals', []))))
            for h in state.get("hospitals", [])[:5]:
                with st.expander(h.get("name", "N/A")):
                    st.write(t("**Type:** {v}", v=h.get('type')))
                    st.write(t("**Contact:** {v}", v=h.get('contact')))
                    st.write(t("**Specialities:** {v}", v=h.get('specialities', 'N/A')[:80]))

        with col2:
            st.markdown(t("**🏠 Shelters ({n} found)**", n=len(state.get('shelters', []))))
            for s in state.get("shelters", [])[:5]:
                with st.expander(s.get("name", "N/A")):
                    st.write(t("**Type:** {v}", v=s.get('type')))
                    if s.get("type") != "GOVT_SCHOOL_SHELTER":
                        st.write(t("**Capacity:** {v}", v=s.get('capacity', 'N/A')))
                        st.write(t("**Contact:** {v}", v=s.get('contact', 'N/A')))
                    else:
                        st.caption(t("School shelter details are activated through district administration."))

        st.markdown("---")
        st.markdown(t("**Matching Reasoning:** {v}", v=state.get('match_reasoning', 'N/A')))

    # ── TAB 3: Route & Map ───────────────────────────────────────────
    with tab3:
        route = state.get("route", {})
        routes = state.get("routes", [])
        road_risks = state.get("road_risks", [])

        TYPE_META = {
            "HOSPITAL": ("🏥", t("Hospital")),
            "SHELTER":  ("🏠", t("Shelter")),
        }

        st.markdown(t("**🚑 Distance & Estimated Time to Each Resource**"))
        if routes:
            cols = st.columns(len(routes))
            for col, r in zip(cols, routes):
                icon, label = TYPE_META.get(r.get("resource_type", ""), ("📍", t("Resource")))
                dist_val = r.get("distance_km")
                dur_val  = r.get("duration_min")
                approx   = r.get("source") in ("straight_line_approx", "same_locality")
                prefix   = "≈ " if approx else ""
                with col:
                    st.markdown(f"**{icon} {label}** — {r.get('name', 'N/A')}")
                    c1, c2 = st.columns(2)
                    c1.metric(t("📍 Distance"), f"{prefix}{dist_val} km" if dist_val is not None else "N/A")
                    c2.metric(t("⏱️ Time"), f"{prefix}{dur_val} min" if dur_val is not None else "N/A")
                    if r.get("source") == "straight_line_approx":
                        st.caption(t("≈ straight-line estimate (add ORS_API_KEY for road distance)"))
                    elif r.get("dest_source", "").startswith("approx"):
                        st.caption(t("approx location · {v}", v=r['dest_source']))
                    if r.get("error"):
                        st.caption(f"⚠️ {r['error']}")
        else:
            st.info(t("No hospital/shelter routes available."))

        if route.get("error"):
            st.error(t("🛣️ Routing unavailable — {v}", v=route['error']))

        if state.get("route_warning"):
            st.warning(state["route_warning"])

        if route.get("turn_by_turn"):
            st.markdown(t("**Route Steps:**"))
            for step in route["turn_by_turn"]:
                st.markdown(f"→ {step}")

        # Road risk table
        if road_risks:
            st.markdown("---")
            st.markdown(t("**⛰️ Road Risk Assessment — {d}**", d=district_label(district)))
            for risk in road_risks:
                status_color = "🔴" if risk.get("currently_risky") else "🟡"
                st.markdown(
                    f"{status_color} **{risk['road']}** {t('at {v}', v=risk['segment'])} — "
                    f"{risk['risk_type']} | {t('Season: {v}', v=risk['season'])}"
                )

        # Map
        st.caption(t("📍 Approximate location: {lat}, {lon} ({src})",
                     lat=f"{latitude:.4f}", lon=f"{longitude:.4f}", src=t(geo_source)))
        m = folium.Map(location=[latitude, longitude], zoom_start=11, tiles="CartoDB positron")
        folium.Marker([latitude, longitude],
                      popup=t("Your Location"),
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
        st.markdown(t("### 🌊 CWC River Monitoring Stations"))
        st.markdown(
            t("These are official Central Water Commission stations. Check "
              "**[ffs.india-water.gov.in](https://ffs.india-water.gov.in)** for live "
              "water level data.")
        )

        nearest = state.get("nearest_cwc", {})
        if nearest:
            st.info(
                t("📡 **Nearest CWC Station:** {name} | River: {river} | Distance: {dist} km",
                  name=nearest.get('name'), river=nearest.get('river'), dist=nearest.get('distance_km'))
            )

        for s in state.get("cwc_stations", []):
            with st.expander(f"{s.get('station')} — {s.get('river')}"):
                st.write(t("**District:** {v}", v=s.get('district')))
                st.write(t("**Site Type:** {v}", v=s.get('site_type')))
                st.write(t("**Coordinates:** {v}", v=f"{s.get('latitude')}, {s.get('longitude')}"))
                st.write(t("**Live Data:** {v}", v=s.get('cwc_url')))

    # ── TAB 5: GLOF Watch ────────────────────────────────────────────
    with tab5:
        st.markdown(t("### 🧊 Glacial Lake Outburst Flood (GLOF) Watch"))
        st.info(
            t("⏳ **Note:** This data is based on **previous-year monthly satellite "
              "monitoring** by the Central Water Commission (September 2025) — it reflects "
              "**water-spread-area trends, not real-time water levels**. Expanding lakes "
              "indicate *elevated* GLOF risk; always verify with live CWC advisories.")
        )

        if glof_alert:
            lvl = glof_alert.get("level")
            (st.error if lvl == "WATCH" else st.warning)(
                f"🧊 **GLOF {code_label(lvl)}** — {glof_alert.get('count_increasing')} expanding "
                f"glacial lake(s) near {district_label(district)} "
                f"({glof_alert.get('count_in_district', 0)} in-district)."
            )
            nr = glof_alert.get("nearest", {})
            c1, c2, c3 = st.columns(3)
            c1.metric(t("Nearest expanding lake"), nr.get("lake_id", "N/A"))
            c2.metric(t("Distance"), f"{nr.get('distance_km', 'N/A')} km")
            c3.metric(t("Area change"), f"+{nr.get('area_pct_change', 'N/A')}%")
        else:
            st.success(t("✅ No expanding glacial lakes flagged near this location in the latest monitoring."))

        lakes = state.get("glacial_lakes", [])
        if lakes:
            st.markdown(t("**Monitored glacial lakes (nearest first):**"))
            # Map of lakes
            glof_map = folium.Map(location=[latitude, longitude], zoom_start=8, tiles="CartoDB positron")
            folium.Marker([latitude, longitude], popup=t("Your Location"),
                          icon=folium.Icon(color="red", icon="exclamation-sign")).add_to(glof_map)
            for lk in lakes:
                rising = lk.get("status") == "increase"
                color = "red" if rising else ("blue" if lk.get("status") == "decrease" else "gray")
                try:
                    glof_map.add_child(folium.CircleMarker(
                        location=[float(lk["latitude"]), float(lk["longitude"])],
                        radius=7, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                        popup=f"{lk['lake_id']} · {lk.get('river') or lk.get('basin')} · "
                              f"{code_label(lk.get('status'))} ({lk.get('area_pct_change')}%)"
                    ))
                except (TypeError, ValueError):
                    pass
            st_folium(glof_map, width=850, height=380)

            for lk in lakes:
                icon = "🔴" if lk.get("status") == "increase" else ("🔵" if lk.get("status") == "decrease" else "⚪")
                with st.expander(
                    f"{icon} {lk.get('lake_id')} — {district_label(lk.get('district', ''))} "
                    f"({code_label(lk.get('status'))}, {lk.get('area_pct_change')}%)"
                ):
                    st.write(t("**Basin / River:** {v}", v=f"{lk.get('basin')} / {lk.get('river') or 'N/A'}"))
                    st.write(t("**Coordinates:** {v}", v=f"{lk.get('latitude')}, {lk.get('longitude')}"))
                    st.write(t("**Distance from you:** {v} km", v=lk.get('distance_km', 'N/A')))
                    st.write(t("**Monitored period:** {v}", v=lk.get('monitored_period')))
                    st.write(t("**Source:** {v}", v=lk.get('source')))
        else:
            st.caption(t("No glacial-lake monitoring records available for this area."))

    # ── TAB 6: Agent Log ─────────────────────────────────────────────
    with tab6:
        st.markdown(t("### ⚙️ LangGraph Agent Execution Log"))
        for log in state.get("node_log", []):
            icon = "⚠️" if "⚠" in log else "✅"
            st.markdown(f"{icon} `{log}`")

        if state.get("error_log"):
            st.markdown(t("**Errors:**"))
            for err in state["error_log"]:
                st.error(err)

        st.markdown("---")
        st.markdown(t("**LangGraph Pipeline:**"))
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
st.markdown(f"""
<div style='text-align:center; color:#94a3b8; font-size:11px'>
{t('''
IIT Mandi AAI Himshikhar 2026 Capstone Project |
HP Disaster Relief Resource Matching Agent |
Data: HIMCOSTE 2023 • CWC (incl. GLOF Glacial Lake Monitoring Sep 2025) • VIIRS Wildfire History • NHP • DAY-NULM • HP Education Dept • Open-Meteo
''')}
</div>
""", unsafe_allow_html=True)
