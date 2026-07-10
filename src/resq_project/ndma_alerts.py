"""
NDMA SACHET headline alerts (Himachal Pradesh).

Fetches the live NDMA CAP RSS feed (IMD/SDMA warnings) and keeps only the
last N days. Each CAP alert already publishes an official English-language
<cap:info> block alongside the Hindi one — tools.fetch_ndma_alerts reads
that English headline directly from the source, so no machine translation
is needed here. This module just caches the fetch and renders it as a
scrolling headline ticker + an expandable detail list with source links.
"""

import streamlit as st

from resq_project.tools import fetch_ndma_alerts
from resq_project.i18n import t, current_language


@st.cache_data(show_spinner=False, ttl=1800)
def get_alerts(days: int = 7) -> list[dict]:
    """Cached 30 min so this runs roughly once per app load rather than on
    every Streamlit rerun."""
    return fetch_ndma_alerts(days=days)


def _headline(a: dict) -> str:
    """Pick the headline in the user's language. When Hindi is selected,
    use the CAP alert's own original Hindi <cap:info> text directly — no
    translation needed since the source already publishes both languages."""
    if current_language() == "Hindi":
        return a["title"]
    return a["title_en"]


def render_headline_ticker(alerts: list[dict]) -> None:
    """Scrolling headline strip + an expandable list with full detail."""
    if not alerts:
        st.info(t("📰 No NDMA alerts for Himachal Pradesh in the last 7 days."))
        return

    ticker_text = "     📰     ".join(
        f"{a['pub_display']} — {_headline(a)}" for a in alerts
    )
    st.markdown(
        f"""
        <div style='background:#1e3a5f;border-radius:8px;padding:10px 0;
                    overflow:hidden;white-space:nowrap;margin-bottom:12px;'>
          <div style='display:inline-block;padding-left:100%;
                      animation:ndma-scroll 45s linear infinite;
                      color:#f8fafc;font-size:14px;font-weight:500;'>
            {ticker_text}
          </div>
        </div>
        <style>
        @keyframes ndma-scroll {{
          0%   {{ transform: translate(0, 0); }}
          100% {{ transform: translate(-100%, 0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(t("📰 NDMA Alerts — Himachal Pradesh (last 7 days, {n})", n=len(alerts)), expanded=False):
        for a in alerts:
            badge = f" · {a['severity']}" if a.get("severity") else ""
            st.markdown(f"**{a['pub_display']}**{badge} — {_headline(a)}")
            if a["title_en"] == a["title"]:
                st.caption(t("⚠️ English version unavailable from source — showing original text."))
            if a.get("area_desc"):
                st.caption(t("Area: {v}", v=a['area_desc']))
            st.caption(
                t("Source: NDMA SACHET · {author} · [View CAP alert]({link})",
                  author=a.get('author', ''), link=a['link'])
            )
            st.markdown("---")
