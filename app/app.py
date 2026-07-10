"""
HP Disaster Relief Resource Matching Agent
Streamlit UI
"""

import html
import time

import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from resq_project.config import AGENT_COORDINATOR_EMAIL, DISASTER_TYPES, DISTRICT_RISK
from resq_project.workflow import run_agent, get_llm, stream_agent
from resq_project.tools import assess_wildfire_risk
from resq_project import chatbot
from resq_project import coordination as coord
from resq_project import charts
from resq_project.pdf_report import build_pdf
from resq_project.llm_client import (
    active_model_label,
    active_grok_model,
    active_ollama_model,
    active_provider,
    list_ollama_models,
    ollama_reachable,
    provider_ready,
    set_anthropic_model,
    set_grok_model,
    set_ollama_model,
    set_openai_model,
    set_provider,
)
from resq_project import opsmap
from resq_project import tweet_triage

STAGE_LABELS = {
    "intake_agent": "Intake agent · incident parsing and weather intake",
    "glof_monitor_agent": "GLOF monitor agent · expanding glacial lakes",
    "resource_finder_agent": "Resource finder agent · hospitals, shelters, CWC stations",
    "matching_agent": "Matching agent · rank and prioritize resources",
    "route_planning_agent": "Route planning agent · route and road risk",
    "escalation_agent": "Escalation/report agent · final response report",
    "completed": "Complete",
    "error": "Error",
}
STAGE_ORDER = [
    "intake_agent",
    "glof_monitor_agent",
    "resource_finder_agent",
    "matching_agent",
    "route_planning_agent",
    "escalation_agent",
]


@st.cache_data(show_spinner=False)
def wildfire_flag(lat, lon):
    """Cached wildfire-proneness lookup for the given coordinates."""
    return assess_wildfire_risk(lat, lon)


@st.cache_data(show_spinner=False)
def locate_place(text: str):
    """Cached town-level locator for need/resource locations (ops map)."""
    return opsmap.locate(text)


def rag_answer(question):
    """Grounded RAG answer over the ingested HP disaster data."""
    return chatbot.answer(question, get_llm)


def stage_label(node_name: str) -> str:
    return STAGE_LABELS.get(node_name, node_name.replace("_", " ").title())


def stage_node_html(node_name: str) -> str:
    return f"<span class='node-trace'>{html.escape(node_name)}</span>"


def stage_completion_html(prefix: str, node_name: str, label: str, elapsed_s: float, step_num: int | None = None) -> str:
    return f"<span class='node-trace'>{html.escape(prefix)} {html.escape(node_name)}</span>"


def render_count_fallback(title: str, counts: dict, x_label: str = "Value") -> None:
    st.markdown(f"**{title}**")
    if not counts:
        st.info("No data available.")
        return
    chart_rows = [{"Label": key, x_label: value} for key, value in counts.items()]
    st.dataframe(chart_rows, width="stretch", hide_index=True)

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HP Disaster Relief Agent",
    page_icon="🏔️",
    layout="wide"
)

st.markdown("""
<style>
  html, body, [class*="css"], .stMarkdown, .stMarkdown p { font-size: 1.14rem; }
  .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
  .hero {
    background:
      radial-gradient(circle at top right, rgba(245,166,35,.18), transparent 32%),
      linear-gradient(135deg, #102a43 0%, #1a4b5f 52%, #168a7a 100%);
    padding: 1.5rem 1.8rem;
    border-radius: 20px;
    color: white;
    box-shadow: 0 12px 30px rgba(16,42,67,.22);
    margin-bottom: 1.1rem;
  }
  .hero h1 { color: #fff; font-size: 2.75rem; margin: 0 0 .25rem 0; font-weight: 850; letter-spacing: -.03em; }
  .hero p { color: #d7e7ee; margin: 0; font-size: 1.14rem; }
  .hero .meta { color: #b8d4da; font-size: 1.08rem; margin-top: .45rem; }
  .hero-grid {
    display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.65rem; margin-top: 1rem;
  }
  .hero-chip {
    background: rgba(255,255,255,.1);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 14px;
    padding: .7rem .8rem;
  }
  .hero-chip strong { display:block; color:#fff; font-size:1.08rem; margin-bottom:.12rem; }
  .hero-chip span { color:#cde3e7; font-size:.98rem; }
  [data-testid="stMetricValue"] { font-size: 2.4rem; font-weight: 800; }
  [data-testid="stMetricLabel"] { font-size: 1.12rem; opacity: .88; }
  .stTabs [data-baseweb="tab"] { font-size: 1.14rem; padding: .7rem 1.05rem; }
  .stTabs [aria-selected="true"] { font-weight: 700; }
  .pill {
    display:inline-block; padding:.26rem .6rem; border-radius:999px; font-size:.82rem; font-weight:700;
    background:#e7f2ef; color:#125e56; border:1px solid #cbe5df;
  }
  .node-trace { color:#168a34; font-weight:800; }
  .stage-panel {
    border:1px solid #cbe5df; background:#f4fbf7; border-radius:14px; padding:.9rem 1rem; margin:.2rem 0 .4rem 0;
  }
  .stage-panel p { margin:.15rem 0; }
</style>
""", unsafe_allow_html=True)

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
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "agent_stage" not in st.session_state:
    st.session_state.agent_stage = ""
if "agent_stage_history" not in st.session_state:
    st.session_state.agent_stage_history = []

_provider_labels = {
    "Ollama (local)": "ollama",
    "OpenAI (API)": "openai",
    "Anthropic (API)": "anthropic",
    "Grok (xAI API)": "grok",
}
if st.session_state.get("llm_provider_choice") in _provider_labels:
    set_provider(_provider_labels[st.session_state["llm_provider_choice"]])
if st.session_state.get("ollama_model_choice"):
    set_ollama_model(st.session_state["ollama_model_choice"])
if st.session_state.get("openai_model_choice"):
    set_openai_model(st.session_state["openai_model_choice"])
if st.session_state.get("anthropic_model_choice"):
    set_anthropic_model(st.session_state["anthropic_model_choice"])
if st.session_state.get("grok_model_choice"):
    set_grok_model(st.session_state["grok_model_choice"])

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
st.markdown(f"""
<div class="hero">
  <h1>ResQ · Disaster Relief Resource Matching Agent</h1>
  <p>Himachal Pradesh multi-hazard coordination: one incident report in, one ranked and explainable action plan out.</p>
  <div class="meta">LangGraph · ChromaDB / fallback retrieval · Open-Meteo · OpenRouteService · Human approval gate · Active model: {active_model_label()}</div>
  <div class="hero-grid">
    <div class="hero-chip"><strong>Decision support</strong><span>Never auto-dispatches outbound action</span></div>
    <div class="hero-chip"><strong>Explainable urgency</strong><span>0–100 score with factor breakdown</span></div>
    <div class="hero-chip"><strong>Grounded retrieval</strong><span>Hospitals, shelters, CWC, GLOF and wildfire context</span></div>
    <div class="hero-chip"><strong>Coordination-ready</strong><span>Approve-before-send draft emails with audit log</span></div>
  </div>
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
        send = col_send.form_submit_button("Ask ➤", width="stretch")
        clear = col_clear.form_submit_button("🗑️ Clear chat", width="stretch")

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

    # Batch triage: pull the disaster-tweet feed into the worklist
    tcol1, tcol2 = st.columns([1, 2])
    if tcol1.button("📨 Load tweet feed (batch triage)", width="stretch"):
        known = {n.get("request_id") for n in st.session_state.extra_needs}
        st.session_state.extra_needs.extend(
            n for n in tweet_triage.triage_tweets() if n["request_id"] not in known)
        st.rerun()
    tcol2.caption(
        "Converts `data/disaster_tweets_sample.csv` (Kaggle Disaster Tweets style) into "
        "structured needs via the extraction pipeline — see `docs/extraction_eval.md` for "
        "measured accuracy."
    )

    needs = coord.load_needs() + st.session_state.extra_needs
    resources = coord.load_resources()          # quantities already net of dispatch ledger
    matches = coord.match_needs_to_resources(needs, resources, allocate=True)

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
                bcol = st.columns(5)
                bcol[0].metric("Match score", f"{best['score']}/100")
                bcol[1].metric("Coverage", f"{best.get('coverage_pct',0)}%")
                bcol[2].metric("Gap", best.get("quantity_gap", 0))
                bcol[3].metric("Reserved units", best.get("committed_units", 0),
                               help="Units held for this need from the provider's remaining stock "
                                    "(inventory-aware allocation — no double-promising).")
                bcol[4].metric("Provider verified",
                               "Yes" if str(best['resource'].get('contact_status','')).lower()=="verified" else "No")

            draft = coord.draft_coordination_message(
                m, coordinator_email=st.session_state.agent_coordinator_email)
            edited = st.text_area("Coordination message (editable)", value=draft,
                                  height=200, key=f"msg_{rid}")

            a1, a2, a3 = st.columns(3)
            if a1.button("✅ Approve & mark sent", key=f"send_{rid}", width="stretch"):
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
                    # Consume the provider's stock so later matches see it
                    units = (best or {}).get("committed_units", 0)
                    if best and units:
                        coord.log_dispatch({
                            "request_id": rid,
                            "resource_id": best["resource"].get("resource_id", ""),
                            "provider": best["resource"].get("provider_name", ""),
                            "category": need.get("category"),
                            "units": units,
                        })
                    st.session_state.approved_msgs[rid] = "sent"
                    st.rerun()
                st.error(f"Email send failed: {result['error']}")
            if a2.button("📝 Log edit (no send)", key=f"edit_{rid}", width="stretch"):
                coord.log_approval({"request_id": rid, "action": "edited",
                                    "category": need.get("category"),
                                    "coordinator_email": st.session_state.agent_coordinator_email,
                                    "message": edited})
                st.session_state.approved_msgs[rid] = "approved"
                st.rerun()
            if a3.button("🚫 Reject / escalate", key=f"rej_{rid}", width="stretch"):
                coord.log_approval({"request_id": rid, "action": "rejected",
                                    "category": need.get("category"),
                                    "coordinator_email": st.session_state.agent_coordinator_email,
                                    "reason": "human rejected"})
                st.session_state.approved_msgs[rid] = "rejected"
                st.rerun()

    # Provider inventory (net of dispatch ledger)
    st.markdown("---")
    st.markdown("**📦 Provider inventory** — listed stock minus approved dispatches")
    st.dataframe(
        [{
            "Provider": r.get("provider_name"), "Category": r.get("category"),
            "Location": r.get("location"), "Listed": r.get("quantity_original", 0),
            "Dispatched": r.get("dispatched_units", 0), "Remaining": r.get("quantity", 0),
        } for r in resources],
        width="stretch", hide_index=True,
    )

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
# OPERATIONS MAP — every need, provider, and allocation on one picture
# ══════════════════════════════════════════════════════════════════════
with st.expander("🗺️ Operations Map — needs, providers & allocations", expanded=False):
    st.caption(
        "Control-room picture: every open **need** (colored by match status), every "
        "**provider** (blue), and dashed lines showing the current best allocation. "
        "Town-level accuracy from an offline gazetteer (OSM geocoding as fallback)."
    )
    STATUS_MAP_COLOR = {"MATCHED": "green", "PARTIAL": "orange", "UNMATCHED": "red"}

    ops_map = folium.Map(location=[31.7, 77.0], zoom_start=8, tiles="CartoDB positron")

    provider_pts = {}
    for r in resources:
        loc = locate_place(str(r.get("location", "")))
        if not loc:
            continue
        provider_pts[str(r.get("resource_id", ""))] = (loc[0], loc[1])
        folium.Marker(
            [loc[0], loc[1]],
            popup=(f"<b>{r.get('provider_name')}</b><br>{r.get('category')} · "
                   f"remaining {r.get('quantity', 0)} / {r.get('quantity_original', 0)}"),
            icon=folium.Icon(color="blue", icon="briefcase"),
        ).add_to(ops_map)

    plotted_needs = 0
    for m in matches:
        need = m["need"]
        loc = locate_place(str(need.get("location", "")))
        if not loc:
            continue
        plotted_needs += 1
        color = STATUS_MAP_COLOR.get(m["status"], "gray")
        folium.CircleMarker(
            location=[loc[0], loc[1]], radius=8,
            color=color, fill=True, fill_color=color, fill_opacity=0.8,
            popup=(f"<b>{need.get('request_id')}</b> · {need.get('category')} × "
                   f"{need.get('quantity') or '?'}<br>{need.get('urgency')} · {m['status']}"),
        ).add_to(ops_map)
        best = m.get("best_match")
        if best:
            ppt = provider_pts.get(str(best["resource"].get("resource_id", "")))
            if ppt:
                folium.PolyLine(
                    [[loc[0], loc[1]], list(ppt)],
                    color=color, weight=2, opacity=0.65, dash_array="6",
                ).add_to(ops_map)

    st_folium(ops_map, width=900, height=450)
    st.caption(
        f"{plotted_needs} needs and {len(provider_pts)} providers plotted · "
        f"🟢 matched · 🟠 partial · 🔴 unmatched · dashed line = proposed allocation"
    )

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Model Control")
    provider_options = _provider_labels
    provider_labels = list(provider_options.keys())
    current_provider = active_provider()
    current_index = next((i for i, label in enumerate(provider_labels) if provider_options[label] == current_provider), 0)
    provider_label = st.selectbox("LLM Provider", provider_labels, index=current_index, key="llm_provider_choice")
    selected_provider = provider_options[provider_label]
    set_provider(selected_provider)

    if selected_provider == "ollama":
        installed_models = list_ollama_models()
        model_options = installed_models or [active_ollama_model()]
        model_index = model_options.index(active_ollama_model()) if active_ollama_model() in model_options else 0
        ollama_model = st.selectbox("Ollama model", model_options, index=model_index, key="ollama_model_choice")
        set_ollama_model(ollama_model)
        if provider_ready("ollama"):
            st.caption(f"Ollama reachable · model `{ollama_model}`")
        else:
            st.warning("Ollama is not reachable right now. The workflow will fail until the local server is available.")
    elif selected_provider == "openai":
        openai_model = st.text_input("OpenAI model", value="gpt-4o-mini", key="openai_model_choice")
        set_openai_model(openai_model)
        st.caption("Requires `OPENAI_API_KEY` in the environment.")
        if not provider_ready("openai"):
            st.warning("OpenAI provider selected but `OPENAI_API_KEY` is not set.")
    elif selected_provider == "anthropic":
        anthropic_model = st.text_input("Anthropic model", value="claude-3-5-sonnet-latest", key="anthropic_model_choice")
        set_anthropic_model(anthropic_model)
        st.caption("Requires `ANTHROPIC_API_KEY` in the environment.")
        if not provider_ready("anthropic"):
            st.warning("Anthropic provider selected but `ANTHROPIC_API_KEY` is not set.")
    else:
        grok_model = st.text_input("Grok model", value=active_grok_model(), key="grok_model_choice")
        set_grok_model(grok_model)
        st.caption("Requires `XAI_API_KEY` in the environment. Uses xAI's OpenAI-compatible API without reasoning options enabled.")
        if not provider_ready("grok"):
            st.warning("Grok provider selected but `XAI_API_KEY` is not set.")

    st.markdown("---")
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
    run_btn = st.button("🚨 Find Relief Resources", type="primary", width="stretch")
    st.caption(f"Active provider: `{active_provider()}` · Active model: `{active_model_label()}`")
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
        ordered_nodes = list(STAGE_ORDER)
        progress = st.progress(0.0, text="Starting LangGraph pipeline…")
        stage_box = st.empty()
        try:
            final_state = None
            stage_history = []
            run_started = time.perf_counter()
            for node_name, partial_state in stream_agent(
                user_name=user_name or "Anonymous",
                district=district,
                location_desc=location_desc or f"{district}, HP",
                latitude=latitude,
                longitude=longitude,
                disaster_type=disaster_type,
                needs=needs,
            ):
                final_state = partial_state
                st.session_state.agent_stage = node_name
                step_num = ordered_nodes.index(node_name) + 1 if node_name in ordered_nodes else 1
                label = stage_label(node_name)
                elapsed = time.perf_counter() - run_started
                stage_history.append({
                    "node": node_name,
                    "label": label,
                    "elapsed_s": round(elapsed, 2),
                    "step_num": step_num,
                })
                st.session_state.agent_stage_history = stage_history
                stage_box.markdown(
                    "<div class='stage-panel'>"
                    + "".join(
                        f"<p>{stage_completion_html('✓', item['node'], item['label'], item['elapsed_s'], item['step_num'])}</p>"
                        for item in stage_history
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
                progress.progress(step_num / len(ordered_nodes), text=f"Stage {step_num}/{len(ordered_nodes)} · {label}")

            if final_state is None:
                raise RuntimeError("Pipeline did not return any stage updates.")

            st.session_state.agent_state = final_state
            st.session_state.agent_error = None
            st.session_state.agent_stage = "completed"
            st.session_state.agent_stage_history = stage_history
            progress.progress(1.0, text="All node stages completed")
            stage_box.markdown(
                "<div class='stage-panel'>"
                + "".join(
                    f"<p>{stage_completion_html('✓', item['node'], item['label'], item['elapsed_s'], item['step_num'])}</p>"
                    for item in stage_history
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.session_state.agent_error = f"Agent error: {e}"
            st.session_state.agent_state = None
            st.session_state.agent_stage = "error"
            st.session_state.agent_stage_history = []
            progress.empty()
            stage_box.empty()

if st.session_state.agent_error:
    st.error(st.session_state.agent_error)

state = st.session_state.agent_state
current_needs = coord.load_needs()
current_resources = coord.load_resources()
volunteer_matches = coord.match_needs_to_resources(current_needs, current_resources)
recent_approvals = coord.read_approvals(limit=50)

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
    st.markdown(
        f"<span class='pill'>Pipeline stage: {stage_label(st.session_state.get('agent_stage', 'completed'))}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    stage_history = st.session_state.get("agent_stage_history", [])
    if stage_history:
        with st.expander("Pipeline stage timings", expanded=False):
            for item in stage_history:
                st.markdown(
                    stage_completion_html("", item["node"], item["label"], item["elapsed_s"], item["step_num"]),
                    unsafe_allow_html=True,
                )
            st.dataframe(
                [{
                    "Stage": item["step_num"],
                    "Node": item["node"],
                    "Label": item["label"],
                    "Elapsed (s)": item["elapsed_s"],
                } for item in stage_history],
                width="stretch",
                hide_index=True,
            )

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

    # ── Executive analytics strip ────────────────────────────────────
    cards = charts.incident_summary_cards(state, volunteer_matches, recent_approvals)
    exec_cols = st.columns(3)
    for col, (title, value, subtitle) in zip(exec_cols, cards[:3]):
        with col.container(border=True):
            st.metric(title, value)
            st.caption(subtitle)

    viz1, viz2, viz3 = st.columns([1.05, 1, 1])
    with viz1.container(border=True):
        gauge_fig = charts.plotly_gauge(float(urgency.get("score", 0) or 0), "Incident urgency")
        if gauge_fig is not None:
            st.plotly_chart(gauge_fig, width="stretch", key="gauge_exec_strip")
        else:
            st.metric("Incident urgency", f"{urgency.get('score', 'N/A')}/100", urgency.get("level", ""))
    with viz2.container(border=True):
        status_counts = charts.volunteer_status_counts(volunteer_matches)
        status_fig = charts.plotly_donut(
            status_counts,
            "Volunteer worklist status",
            charts.STATUS_COLORS,
        )
        if status_fig is not None:
            st.plotly_chart(status_fig, width="stretch", key="status_exec_strip")
        else:
            render_count_fallback("Volunteer worklist status", status_counts, "Count")
    with viz3.container(border=True):
        risk_fig = charts.plotly_urgency_mix(state)
        if risk_fig is not None:
            st.plotly_chart(risk_fig, width="stretch", key="risk_exec_strip")
        else:
            breakdown = (state.get("urgency") or {}).get("breakdown", {})
            render_count_fallback("Urgency driver mix", breakdown, "Points")

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Response Report", "📊 Analytics", "🏥 Resources", "🗺️ Route & Map",
        "🌊 CWC Stations", "🧊 GLOF Watch", "🧾 Export", "⚙️ Agent Log"
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
        if ac1.button("✅ Approve & mark sent", key=f"agentsend_{agent_rid}", width="stretch"):
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
        if ac2.button("🚫 Reject / escalate", key=f"agentrej_{agent_rid}", width="stretch"):
            coord.log_approval({"request_id": agent_rid, "action": "rejected",
                                "coordinator_email": st.session_state.agent_coordinator_email,
                                "reason": "human rejected"})
            st.session_state.setdefault("approved_msgs", {})[agent_rid] = "rejected"
            st.rerun()
        if decided:
            st.success(f"Coordinator decision logged: **{decided.upper()}**")

        st.warning("⚠️ NOTE: This is AI-assisted disaster information. Always verify with HPSDMA and local authorities. Call 1078 (NDMA) for official support.")

    # ── TAB 2: Analytics ─────────────────────────────────────────────
    with tab2:
        st.markdown("### Analytics")
        top = st.columns([1, 1, 1])
        with top[0].container(border=True):
            gauge_fig = charts.plotly_gauge(float(urgency.get("score", 0) or 0), "Incident urgency")
            if gauge_fig is not None:
                st.plotly_chart(gauge_fig, width="stretch", key="gauge_analytics_tab")
            else:
                st.metric("Incident urgency", f"{urgency.get('score', 'N/A')}/100", urgency.get("level", ""))

        need_mix_counts = charts.volunteer_need_counts(volunteer_matches)
        if need_mix_counts:
            with top[1].container(border=True):
                need_mix_fig = charts.plotly_pie(
                    need_mix_counts,
                    "Need categories",
                    charts.NEED_COLORS,
                )
                if need_mix_fig is not None:
                    st.plotly_chart(need_mix_fig, width="stretch", key="need_mix_analytics_tab")
                else:
                    render_count_fallback("Need categories", need_mix_counts, "Count")

        provider_type_counts = charts.provider_type_mix(current_resources)
        if provider_type_counts:
            with top[2].container(border=True):
                provider_mix_fig = charts.plotly_pie(
                    provider_type_counts,
                    "Provider mix",
                    {
                        "Government": "#0f766e",
                        "Ngo": "#2563eb",
                        "Volunteer": "#7c3aed",
                        "Private": "#ea580c",
                        "Unknown": "#64748b",
                    },
                )
                if provider_mix_fig is not None:
                    st.plotly_chart(provider_mix_fig, width="stretch", key="provider_mix_analytics_tab")
                else:
                    render_count_fallback("Provider mix", provider_type_counts, "Count")

        mid = st.columns(2)
        source_counts = charts.need_source_counts(current_needs)
        if source_counts:
            with mid[0].container(border=True):
                source_fig = charts.plotly_pie(
                    source_counts,
                    "Need intake sources",
                    {
                        "Field Report": "#1565c0",
                        "Phone Call": "#2e7d32",
                        "Tweet": "#fb8c00",
                    },
                )
                if source_fig is not None:
                    st.plotly_chart(source_fig, width="stretch", key="source_analytics_tab")
                else:
                    render_count_fallback("Need intake sources", source_counts, "Count")

        status_counts = charts.need_status_counts(current_needs)
        if status_counts:
            with mid[1].container(border=True):
                status_fig = charts.plotly_pie(
                    status_counts,
                    "Need lifecycle status",
                    {
                        "Open": "#e53935",
                        "Matched": "#fb8c00",
                        "Fulfilled": "#2e7d32",
                        "Unknown": "#607d8b",
                    },
                )
                if status_fig is not None:
                    st.plotly_chart(status_fig, width="stretch", key="status_analytics_tab")
                else:
                    render_count_fallback("Need lifecycle status", status_counts, "Count")

        treemap_fig = charts.plotly_treemap_needs(current_needs)
        if treemap_fig is not None:
            with st.container(border=True):
                st.plotly_chart(treemap_fig, width="stretch", key="treemap_analytics_tab")
        else:
            with st.container(border=True):
                district_counts = charts.district_need_counts(current_needs)
                render_count_fallback("Demand concentration by district", district_counts, "Open requests")

        flow_left, flow_right = st.columns(2)
        sunburst_fig = charts.plotly_sunburst_worklist(volunteer_matches)
        if sunburst_fig is not None:
            with flow_left.container(border=True):
                st.plotly_chart(sunburst_fig, width="stretch", key="sunburst_analytics_tab")
        else:
            with flow_left.container(border=True):
                approval_counts = charts.approval_action_counts(recent_approvals)
                render_count_fallback("Recent human decisions", approval_counts, "Count")

        risk_counts = charts.risk_signal_counts(state)
        risk_mix_fig = charts.plotly_pie(
            risk_counts,
            "Live risk signal mix",
            {
                key: (
                    "#c03a2b" if key.startswith("Urgency:")
                    else "#ea580c" if key.startswith("Weather:")
                    else "#7c3aed" if key.startswith("Wildfire:")
                    else "#0f766e"
                )
                for key in risk_counts
            } if risk_counts else None,
        )
        if risk_mix_fig is not None:
            with flow_right.container(border=True):
                st.plotly_chart(risk_mix_fig, width="stretch", key="risk_mix_analytics_tab")
        else:
            with flow_right.container(border=True):
                render_count_fallback("Live risk signals", risk_counts, "Signals")

        summary_left, summary_right = st.columns(2)
        district_counts = charts.district_need_counts(current_needs)
        if district_counts:
            with summary_left.container(border=True):
                district_mix_fig = charts.plotly_pie(
                    district_counts,
                    "Demand share by district",
                    {
                        "Kullu": "#1d4f91",
                        "Kangra": "#1565c0",
                        "Mandi": "#2e7d32",
                        "Chamba": "#8e24aa",
                        "Hamirpur": "#fb8c00",
                        "Una": "#607d8b",
                    },
                )
                if district_mix_fig is not None:
                    st.plotly_chart(district_mix_fig, width="stretch", key="district_mix_analytics_tab")
                else:
                    render_count_fallback("Demand share by district", district_counts, "Open requests")

        approval_counts = charts.approval_action_counts(recent_approvals)
        if approval_counts:
            with summary_right.container(border=True):
                approval_mix_fig = charts.plotly_pie(
                    approval_counts,
                    "Human decision split",
                    {
                        "approved_sent": "#2e7d32",
                        "rejected": "#c62828",
                        "edited_not_sent": "#fb8c00",
                        "unknown": "#607d8b",
                    },
                )
                if approval_mix_fig is not None:
                    st.plotly_chart(approval_mix_fig, width="stretch", key="approval_mix_analytics_tab")
                else:
                    render_count_fallback("Human decision split", approval_counts, "Count")

    # ── TAB 3: Resources ─────────────────────────────────────────────
    with tab3:
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

    # ── TAB 4: Route & Map ───────────────────────────────────────────
    with tab4:
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

    # ── TAB 5: CWC Stations ──────────────────────────────────────────
    with tab5:
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

    # ── TAB 6: GLOF Watch ────────────────────────────────────────────
    with tab6:
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

    # ── TAB 7: Export ────────────────────────────────────────────────
    with tab7:
        st.subheader("Exports")
        st.write("Generate a shareable incident PDF containing the final response report, urgency summary, route/risk notes, volunteer matching snapshot, and recent human decisions.")
        st.caption("The PDF is built from the live app state and uses the same analytics data shown above.")
        if st.button("Generate incident PDF", type="primary", width="stretch"):
            try:
                st.session_state.pdf_bytes = build_pdf(state, volunteer_matches, recent_approvals)
                st.success("PDF ready for download.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
        if st.session_state.pdf_bytes:
            st.download_button(
                "Download incident PDF",
                data=st.session_state.pdf_bytes,
                file_name=f"resq_incident_{state.get('district','hp').lower()}.pdf",
                mime="application/pdf",
                width="stretch",
            )

        st.markdown("---")
        st.write("Presentation artifact")
        deck_path = Path(__file__).resolve().parents[1] / "docs" / "ResQ_Capstone_Presentation.pptx"
        if deck_path.exists():
            with open(deck_path, "rb") as f:
                st.download_button(
                    "Download capstone presentation",
                    data=f.read(),
                    file_name="ResQ_Capstone_Presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    width="stretch",
                )
        else:
            st.info("Capstone presentation file is not present in this checkout.")

    # ── TAB 8: Agent Log ─────────────────────────────────────────────
    with tab8:
        st.markdown("### ⚙️ LangGraph Agent Execution Log")
        if stage_history:
            st.markdown("**Stage timings**")
            for item in stage_history:
                st.markdown(
                    f"• {stage_completion_html('', item['node'], item['label'], item['elapsed_s'])}",
                    unsafe_allow_html=True,
                )
            st.markdown("---")
        for log in state.get("node_log", []):
            icon = "⚠️" if "⚠" in log else "✅"
            highlighted = html.escape(log)
            for node_name in STAGE_ORDER:
                highlighted = highlighted.replace(node_name, stage_node_html(node_name))
            st.markdown(f"{icon} {highlighted}", unsafe_allow_html=True)

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
