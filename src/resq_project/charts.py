"""
Charts and summary aggregations for the disaster-management dashboard and PDF.
"""

from __future__ import annotations

import io
from collections import Counter
from typing import Dict, Iterable, List, Optional

STATUS_COLORS = {
    "MATCHED": "#168a7a",
    "PARTIAL": "#f5a623",
    "UNMATCHED": "#c03a2b",
}
NEED_COLORS = {
    "Medical": "#d94841",
    "Shelter": "#2e6fa5",
    "Food": "#f59e0b",
    "Water": "#14b8a6",
    "Rescue": "#7c3aed",
    "Transport": "#4f46e5",
    "Evacuation": "#0f766e",
}
RISK_COLORS = {
    "CRITICAL": "#c03a2b",
    "HIGH": "#ea580c",
    "MODERATE": "#ca8a04",
    "LOW": "#16a34a",
}


def _needs_from_state(state: dict) -> list[str]:
    needs = state.get("needs") or []
    if needs:
        return list(needs)
    return [str((state.get("priority_resource") or {}).get("resource_type", "Assistance")).title()]


def volunteer_status_counts(matches: Iterable[dict]) -> Dict[str, int]:
    c = Counter(m.get("status", "UNMATCHED") for m in matches)
    return {k: c.get(k, 0) for k in ("MATCHED", "PARTIAL", "UNMATCHED") if c.get(k, 0)}


def volunteer_need_counts(matches: Iterable[dict]) -> Dict[str, int]:
    c = Counter(str((m.get("need") or {}).get("category", "Unknown")) for m in matches)
    return dict(c)


def provider_type_mix(resources: Iterable[dict]) -> Dict[str, int]:
    c = Counter(str(r.get("provider_type", "unknown")).title() for r in resources)
    return dict(c)


def need_source_counts(needs: Iterable[dict]) -> Dict[str, int]:
    c = Counter(str(n.get("source", "unknown")).replace("_", " ").title() for n in needs)
    return dict(c)


def need_status_counts(needs: Iterable[dict]) -> Dict[str, int]:
    c = Counter(str(n.get("status", "unknown")).title() for n in needs)
    return dict(c)


def district_need_counts(needs: Iterable[dict], limit: int = 8) -> Dict[str, int]:
    c = Counter(str(n.get("district", n.get("location", "Unknown"))).title() for n in needs)
    return dict(c.most_common(limit))


def provider_type_counts(state: dict) -> Dict[str, int]:
    return {
        "Hospitals": len(state.get("hospitals", [])),
        "Shelters": len(state.get("shelters", [])),
    }


def monitoring_asset_counts(state: dict) -> Dict[str, int]:
    return {
        "CWC Stations": len(state.get("cwc_stations", [])),
        "Glacial Lakes": len(state.get("glacial_lakes", [])),
    }


def risk_signal_counts(state: dict) -> Dict[str, int]:
    urgency = (state.get("urgency") or {}).get("level", "LOW")
    alert = state.get("imd_alert_level", "GREEN")
    wildfire = (state.get("wildfire_risk") or {}).get("level", "MINIMAL")
    glof = (state.get("glof_alert") or {}).get("level", "NONE")
    return {
        f"Urgency: {urgency}": 1,
        f"Weather: {alert}": 1,
        f"Wildfire: {wildfire}": 1,
        f"GLOF: {glof}": 1,
    }


def route_distance_counts(state: dict) -> Dict[str, float]:
    counts = {}
    for route in state.get("routes", []):
        if route.get("distance_km") is None:
            continue
        counts[route.get("name", "Resource")] = float(route["distance_km"])
    if not counts and state.get("route", {}).get("distance_km") is not None:
        counts[state.get("route", {}).get("name", "Priority resource")] = float(state["route"]["distance_km"])
    return counts


def approval_action_counts(approvals: Iterable[dict]) -> Dict[str, int]:
    c = Counter(str(a.get("action", "unknown")) for a in approvals)
    return dict(c)


def per_category_status_counts(matches: Iterable[dict]) -> Dict[str, Dict[str, int]]:
    rows: Dict[str, Dict[str, int]] = {}
    for match in matches:
        category = str((match.get("need") or {}).get("category", "Unknown"))
        status = str(match.get("status", "UNMATCHED"))
        rows.setdefault(category, {"MATCHED": 0, "PARTIAL": 0, "UNMATCHED": 0})
        rows[category][status] = rows[category].get(status, 0) + 1
    return rows


def incident_summary_cards(state: dict, matches: Iterable[dict], approvals: Iterable[dict]) -> list[tuple[str, str, str]]:
    urgency = state.get("urgency", {}) or {}
    wf = state.get("wildfire_risk", {}) or {}
    matches = list(matches)
    approvals = list(approvals)
    return [
        ("Urgency", f"{urgency.get('score', 'N/A')}/100", urgency.get("level", "N/A")),
        ("Matched providers", str(sum(1 for m in matches if m.get("status") == "MATCHED")), f"/ {len(matches)} worklist"),
        ("Road risk warnings", str(sum(1 for r in state.get("road_risks", []) if r.get("currently_risky"))), "active corridors"),
        ("Human decisions", str(len(approvals)), "audit records"),
        ("Wildfire", wf.get("level", "N/A"), f"{wf.get('count_10km', 0)} hotspots ≤10km"),
        ("Primary needs", ", ".join(_needs_from_state(state)), state.get("district", "")),
    ]


def _plotly_graph_objects():
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    return go


def plotly_available() -> bool:
    return _plotly_graph_objects() is not None


def plotly_donut(counts: Dict[str, int], title: str, color_map: Optional[Dict[str, str]] = None):
    if not counts or sum(counts.values()) <= 0:
        return None
    go = _plotly_graph_objects()
    if go is None:
        return None
    # Sort keys descending by count for readability
    keys = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    colors = [(color_map or {}).get(k) for k in keys] if color_map else None
    fig = go.Figure(
        data=[go.Pie(
            labels=keys,
            values=[counts[k] for k in keys],
            hole=0.48,
            marker=dict(colors=colors),
            sort=False,
            textinfo="label+value+percent",
        )]
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 20}},
        font={"size": 15},
        margin=dict(t=58, b=10, l=10, r=10),
        height=360,
    )
    return fig


def plotly_pie(
    counts: Dict[str, int],
    title: str,
    color_map: Optional[Dict[str, str]] = None,
    label_map: Optional[Dict[str, str]] = None,
):
    if not counts or sum(counts.values()) <= 0:
        return None
    go = _plotly_graph_objects()
    if go is None:
        return None
    # Sort keys descending by count for readability
    keys = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    labels = [(label_map or {}).get(k, k) for k in keys]
    colors = [(color_map or {}).get(k) for k in keys] if color_map else None
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=[counts[k] for k in keys],
            marker=dict(colors=colors),
            sort=False,
            textinfo="label+value+percent",
        )]
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 20}},
        font={"size": 15},
        margin=dict(t=58, b=10, l=10, r=10),
        height=360,
    )
    return fig


def plotly_bar(counts: Dict[str, float], title: str, color: str = "#2e6fa5", x_title: str = "Value"):
    if not counts:
        return None
    go = _plotly_graph_objects()
    if go is None:
        return None
    # Sort keys ascending so the largest bar appears at the top of the horizontal layout
    keys = sorted(counts.keys(), key=lambda k: counts[k])
    values = [counts[k] for k in keys]
    text_values = [_format_value_label(value) for value in values]
    fig = go.Figure([go.Bar(
        x=values,
        y=keys,
        orientation="h",
        marker=dict(color=color),
        text=text_values,
        textposition="auto",
    )])
    fig.update_layout(
        title={"text": title, "font": {"size": 19}},
        font={"size": 14},
        xaxis_title=x_title,
        margin=dict(t=56, b=10, l=10, r=10),
        height=max(300, 44 * len(keys) + 76),
    )
    return fig


def plotly_gauge(score: float, title: str):
    go = _plotly_graph_objects()
    if go is None:
        return None

    if score >= 75:
        bar_color = RISK_COLORS["CRITICAL"]
    elif score >= 50:
        bar_color = RISK_COLORS["HIGH"]
    elif score >= 30:
        bar_color = RISK_COLORS["MODERATE"]
    else:
        bar_color = RISK_COLORS["LOW"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " /100", "font": {"size": 38}},
        title={"text": title, "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color, "thickness": 0.35},
            "steps": [
                {"range": [0, 30], "color": "#ecfdf5"},
                {"range": [30, 50], "color": "#fef9c3"},
                {"range": [50, 75], "color": "#ffedd5"},
                {"range": [75, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": bar_color, "width": 4}, "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(height=300, margin=dict(t=56, b=10, l=20, r=20), font={"size": 15})
    return fig


def plotly_timeline_breakdown(state: dict, title: str = "Urgency factor breakdown"):
    breakdown = (state.get("urgency") or {}).get("breakdown", {})
    if not breakdown:
        return None
    go = _plotly_graph_objects()
    if go is None:
        return None
    keys = list(breakdown.keys())
    vals = [breakdown[k] for k in keys]
    fig = go.Figure([go.Bar(
        x=keys,
        y=vals,
        marker=dict(color=["#2e6fa5", "#168a7a", "#f5a623", "#c03a2b", "#7c3aed"][: len(keys)]),
    )])
    fig.update_layout(
        title={"text": title, "font": {"size": 19}},
        font={"size": 14},
        yaxis_title="Points",
        margin=dict(t=56, b=30, l=10, r=10),
        height=340,
    )
    return fig


def plotly_urgency_mix(state: dict, title: str = "Urgency driver mix"):
    breakdown = (state.get("urgency") or {}).get("breakdown", {})
    if not breakdown:
        return None
    color_map = {
        "severity": "#c03a2b",
        "people": "#ea580c",
        "access": "#ca8a04",
        "vulnerable": "#7c3aed",
        "weather": "#0f766e",
    }
    label_map = {
        "severity": "Severity",
        "people": "People affected",
        "access": "Access blocked",
        "vulnerable": "Vulnerable groups",
        "weather": "Weather pressure",
    }
    normalized = {label_map.get(k, k.title()): v for k, v in breakdown.items()}
    colors = {label_map.get(k, k.title()): color_map.get(k, "#64748b") for k in breakdown}
    return plotly_donut(normalized, title, colors)


def plotly_treemap_needs(needs: Iterable[dict], title: str = "Needs map (district -> category)") -> object:
    go = _plotly_graph_objects()
    if go is None:
        return None
    rows = []
    for n in needs:
        district = str(n.get("district", "Unknown")).title()
        category = str(n.get("category", "Unknown"))
        status = str(n.get("status", "Unknown")).title()
        try:
            quantity = max(float(n.get("quantity", 1) or 1), 1)
        except (TypeError, ValueError):
            quantity = 1
        rows.append((district, category, status, quantity))
    if not rows:
        return None

    district_totals: Dict[str, float] = {}
    category_totals: Dict[tuple[str, str], float] = {}
    status_totals: Dict[tuple[str, str, str], float] = {}
    for district, category, status, quantity in rows:
        district_totals[district] = district_totals.get(district, 0.0) + quantity
        category_totals[(district, category)] = category_totals.get((district, category), 0.0) + quantity
        status_totals[(district, category, status)] = status_totals.get((district, category, status), 0.0) + quantity

    labels = ["Needs"]
    parents = [""]
    values = [sum(district_totals.values())]
    ids = ["root"]
    colors = ["#102a43"]

    for district, total in district_totals.items():
        labels.append(district)
        parents.append("root")
        values.append(total)
        ids.append(f"district::{district}")
        colors.append("#1d4f91")

    for (district, category), total in category_totals.items():
        labels.append(category)
        parents.append(f"district::{district}")
        values.append(total)
        ids.append(f"category::{district}::{category}")
        colors.append(NEED_COLORS.get(category, "#64748b"))

    status_palette = {
        "Open": "#e53935",
        "Matched": "#fb8c00",
        "Fulfilled": "#2e7d32",
        "Unknown": "#607d8b",
    }
    for (district, category, status), total in status_totals.items():
        labels.append(status)
        parents.append(f"category::{district}::{category}")
        values.append(total)
        ids.append(f"status::{district}::{category}::{status}")
        colors.append(status_palette.get(status, "#607d8b"))

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        ids=ids,
        branchvalues="total",
        marker=dict(colors=colors),
        textinfo="label+value",
        hovertemplate="%{label}<br>Quantity: %{value}<extra></extra>",
    ))
    fig.update_layout(title={"text": title, "font": {"size": 20}}, font={"size": 15}, height=420, margin=dict(t=58, b=10, l=10, r=10))
    return fig


def plotly_stacked_status(matches: Iterable[dict], title: str = "Worklist outcomes by category") -> object:
    go = _plotly_graph_objects()
    if go is None:
        return None
    grouped = per_category_status_counts(matches)
    if not grouped:
        return None
    categories = list(grouped.keys())
    fig = go.Figure()
    for status in ("MATCHED", "PARTIAL", "UNMATCHED"):
        fig.add_bar(
            name=status.title(),
            x=categories,
            y=[grouped[c].get(status, 0) for c in categories],
            marker_color=STATUS_COLORS.get(status, "#94a3b8"),
        )
    fig.update_layout(
        title={"text": title, "font": {"size": 19}},
        font={"size": 14},
        barmode="stack",
        height=380,
        margin=dict(t=56, b=30, l=10, r=10),
        yaxis_title="Needs",
    )
    return fig


def plotly_sunburst_worklist(matches: Iterable[dict], title: str = "Worklist flow (category -> status)") -> object:
    go = _plotly_graph_objects()
    if go is None:
        return None
    grouped = per_category_status_counts(matches)
    if not grouped:
        return None

    labels = ["Worklist"]
    parents = [""]
    values = [sum(sum(v.values()) for v in grouped.values())]
    ids = ["root"]
    colors = ["#102a43"]

    for category, status_counts in grouped.items():
        total = sum(status_counts.values())
        cat_id = f"category::{category}"
        labels.append(category)
        parents.append("root")
        values.append(total)
        ids.append(cat_id)
        colors.append(NEED_COLORS.get(category, "#64748b"))
        for status, count in status_counts.items():
            if not count:
                continue
            labels.append(status.title())
            parents.append(cat_id)
            values.append(count)
            ids.append(f"{cat_id}::{status}")
            colors.append(STATUS_COLORS.get(status, "#94a3b8"))

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        ids=ids,
        branchvalues="total",
        marker=dict(colors=colors),
        insidetextorientation="radial",
        hovertemplate="%{label}<br>Count: %{value}<extra></extra>",
    ))
    fig.update_layout(title={"text": title, "font": {"size": 20}}, font={"size": 15}, height=420, margin=dict(t=58, b=10, l=10, r=10))
    return fig


def _mpl_setup():
    try:
        import matplotlib
    except Exception:
        return False

    matplotlib.use("Agg")
    return True


def donut_png(counts: Dict[str, int], title: str, color_map: Optional[Dict[str, str]] = None) -> Optional[bytes]:
    if not counts:
        return None
    if not _mpl_setup():
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # Sort descending by count
    keys = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    values = [counts[k] for k in keys]
    colors = [(color_map or {}).get(k) for k in keys] if color_map else None

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    wedges, texts, autotexts = ax.pie(
        values,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(width=0.42),
        colors=colors,
        textprops=dict(fontsize=8)
    )
    ax.legend(wedges, keys, loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=8, frameon=False)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("equal")
    return _fig_to_png(fig)


def pie_png(
    counts: Dict[str, int],
    title: str,
    color_map: Optional[Dict[str, str]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> Optional[bytes]:
    if not counts:
        return None
    if not _mpl_setup():
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # Sort descending by count
    keys = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    labels = [(label_map or {}).get(k, k) for k in keys]
    values = [counts[k] for k in keys]
    colors = [(color_map or {}).get(k) for k in keys] if color_map else None

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    wedges, texts, autotexts = ax.pie(
        values,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
        textprops=dict(fontsize=8)
    )
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=8, frameon=False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("equal")
    return _fig_to_png(fig)


def bar_png(counts: Dict[str, float], title: str, color: str = "#2e6fa5") -> Optional[bytes]:
    if not counts:
        return None
    if not _mpl_setup():
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # Sort ascending so largest value is at the top of the horizontal bar chart
    keys = sorted(counts.keys(), key=lambda k: counts[k])
    values = [counts[k] for k in keys]
    fig, ax = plt.subplots(figsize=(4.8, max(2.6, 0.4 * len(keys) + 1.2)))
    ax.barh(keys, values, color=color)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    for i, value in enumerate(values):
        ax.text(value, i, f" {_format_value_label(value)}", va="center", fontsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def gauge_png(score: float, title: str) -> bytes:
    if not _mpl_setup():
        return b""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return b""

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.axis("off")
    color = "#16a34a"
    if score >= 75:
        color = "#c03a2b"
    elif score >= 50:
        color = "#ea580c"
    elif score >= 30:
        color = "#ca8a04"
    ax.text(0.5, 0.75, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.45, f"{round(score)}/100", ha="center", va="center", fontsize=28, fontweight="bold", color=color)
    ax.text(0.5, 0.18, "LOW  ·  MODERATE  ·  HIGH  ·  CRITICAL", ha="center", va="center", fontsize=8, color="#4b5563")
    return _fig_to_png(fig)


def _fig_to_png(fig) -> bytes:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return b""

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _format_value_label(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return f"{value:.1f}"
