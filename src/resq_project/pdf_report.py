"""
PDF export for incident-response runs and volunteer-coordination analytics.
"""

from __future__ import annotations

import html
import io
import re
from datetime import datetime, timezone
from typing import Dict, List

from resq_project import charts


def build_pdf(state: Dict, volunteer_matches: List[Dict], approvals: List[Dict]) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
        from reportlab.graphics.charts.piecharts import Pie
        from reportlab.graphics.shapes import Circle, Drawing, Rect, String
        from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        return _build_simple_pdf(state, volunteer_matches, approvals)

    def styles():
        ss = getSampleStyleSheet()
        ss.add(ParagraphStyle("Hero", parent=ss["Title"], fontSize=26, spaceAfter=8))
        ss.add(ParagraphStyle("Meta", parent=ss["Normal"], fontSize=10.5, textColor=colors.grey))
        ss.add(ParagraphStyle("SecH", parent=ss["Heading2"], fontSize=16, spaceBefore=16, spaceAfter=7, textColor=colors.HexColor("#102a43")))
        ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=11.5, leading=15, alignment=TA_LEFT))
        ss.add(ParagraphStyle("ResqCode", parent=ss["Code"], fontSize=9.2, leading=11.5, backColor=colors.HexColor("#f8fafc"), borderPadding=5, leftIndent=6))
        return ss

    ss = styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        title="ResQ Incident Response Report",
    )
    story = []

    story.append(Paragraph("ResQ Disaster Relief Incident Report", ss["Hero"]))
    story.append(Paragraph(
        f"District: {html.escape(str(state.get('district', 'N/A')))}  |  Disaster: {html.escape(str(state.get('disaster_type', 'N/A')))}",
        ss["Meta"],
    ))
    story.append(Paragraph(
        f"Generated {html.escape(datetime.now(timezone.utc).isoformat())}  |  Model: {html.escape(str(state.get('llm_model_label', 'N/A')))}",
        ss["Meta"],
    ))
    story.append(Spacer(1, 10))

    cards = charts.incident_summary_cards(state, volunteer_matches, approvals)
    row = []
    for title, value, subtitle in cards[:3]:
        row.append(Paragraph(
            f'<font size="20" color="#102a43"><b>{html.escape(value)}</b></font><br/><font size="10">{html.escape(title)} · {html.escape(subtitle)}</font>',
            ss["Body"],
        ))
    story.append(Table([row], colWidths=[5.1 * cm] * len(row), style=TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe2ea")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe2ea")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ])))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Visual Summary", ss["SecH"]))
    visuals = []
    
    gauge_bytes = charts.gauge_png(float((state.get("urgency") or {}).get("score", 0) or 0), "Incident urgency")
    if gauge_bytes:
        visuals.append(_png_image(gauge_bytes, 7.8, ImageReader))
        
    need_bytes = charts.pie_png(charts.volunteer_need_counts(volunteer_matches), "Need categories", charts.NEED_COLORS)
    if need_bytes:
        visuals.append(_png_image(need_bytes, 7.8, ImageReader))
        
    provider_bytes = charts.pie_png(charts.provider_type_mix([
        m.get("best_match", {}).get("resource", {})
        for m in volunteer_matches
        if m.get("best_match")
    ]), "Provider mix")
    if provider_bytes:
        visuals.append(_png_image(provider_bytes, 7.8, ImageReader))
        
    status_bytes = charts.donut_png(charts.volunteer_status_counts(volunteer_matches), "Worklist status", charts.STATUS_COLORS)
    if status_bytes:
        visuals.append(_png_image(status_bytes, 7.8, ImageReader))
        
    visual_objs = [visual for visual in visuals if visual]
    for i in range(0, len(visual_objs), 2):
        row_imgs = visual_objs[i:i + 2]
        story.append(Table([row_imgs], colWidths=[8 * cm] * len(row_imgs)))
        story.append(Spacer(1, 6))

    story.append(Paragraph("Actionable Response", ss["SecH"]))
    story.extend(_markdown_to_flowables(state.get("final_report", "No report generated."), ss))

    priority = state.get("priority_resource") or {}
    if priority:
        story.append(Paragraph("Priority resource", ss["SecH"]))
        story.append(Paragraph(
            f"<b>{html.escape(str(priority.get('name', 'N/A')))}</b><br/>"
            f"Type: {html.escape(str(priority.get('resource_type', priority.get('type', 'N/A'))))}<br/>"
            f"District: {html.escape(str(priority.get('district', state.get('district', 'N/A'))))}<br/>"
            f"Contact: {html.escape(str(priority.get('contact', 'N/A')))}",
            ss["Body"],
        ))

    route = state.get("route") or {}
    story.append(Paragraph("Routing & risk", ss["SecH"]))
    route_text = (
        f"Distance: {route.get('distance_km', 'N/A')} km<br/>"
        f"Duration: {route.get('duration_min', 'N/A')} min<br/>"
        f"Warning: {html.escape(str(state.get('route_warning', 'None')))}"
    )
    story.append(Paragraph(route_text, ss["Body"]))

    story.append(Paragraph("Volunteer worklist snapshot", ss["SecH"]))
    if volunteer_matches:
        for match in volunteer_matches[:8]:
            need = match.get("need") or {}
            best = match.get("best_match") or {}
            provider = (best.get("resource") or {}).get("provider_name", "No provider")
            story.append(Paragraph(
                f"<b>{html.escape(str(need.get('request_id', 'N/A')))}</b> · "
                f"{html.escape(str(need.get('category', 'N/A')))} · "
                f"{html.escape(str(match.get('status', 'N/A')))}<br/>"
                f"Location: {html.escape(str(need.get('location', 'N/A')))}<br/>"
                f"Provider: {html.escape(str(provider))}",
                ss["Body"],
            ))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No volunteer worklist matches available.", ss["Body"]))

    if approvals:
        story.append(Paragraph("Recent human decisions", ss["SecH"]))
        for approval in reversed(approvals[-8:]):
            story.append(Paragraph(
                f"{html.escape(str(approval.get('timestamp', ''))[:19])} · "
                f"{html.escape(str(approval.get('request_id', 'N/A')))} · "
                f"<b>{html.escape(str(approval.get('action', 'N/A')))}</b>",
                ss["Body"],
            ))

    story.append(Paragraph("Agent trace", ss["SecH"]))
    story.append(Preformatted("\n".join(state.get("node_log", []) or ["No node log recorded."]), ss["ResqCode"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _png_image(png_bytes: bytes, width_cm: float, image_reader) -> object:
    from reportlab.lib.units import cm
    from reportlab.platypus import Image

    reader = image_reader(io.BytesIO(png_bytes))
    width_px, height_px = reader.getSize()
    width = width_cm * cm
    height = width * (height_px / width_px) if width_px else 5.6 * cm
    return Image(io.BytesIO(png_bytes), width=width, height=height)


def _reportlab_gauge_drawing(score: float, title: str):
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Circle, Drawing, Rect, String

    score = max(0.0, min(100.0, float(score or 0)))
    drawing = Drawing(230, 170)
    drawing.add(Rect(0, 0, 230, 170, fillColor=colors.HexColor("#f8fafc"), strokeColor=colors.HexColor("#dbe2ea"), rx=10, ry=10))
    drawing.add(String(115, 148, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=13, fillColor=colors.HexColor("#102a43")))

    if score >= 75:
        ring = "#c03a2b"
        band = "CRITICAL"
    elif score >= 50:
        ring = "#ea580c"
        band = "HIGH"
    elif score >= 30:
        ring = "#ca8a04"
        band = "MODERATE"
    else:
        ring = "#16a34a"
        band = "LOW"

    drawing.add(Circle(115, 88, 42, fillColor=colors.HexColor("#ffffff"), strokeColor=colors.HexColor(ring), strokeWidth=8))
    drawing.add(String(115, 94, f"{round(score)}/100", textAnchor="middle", fontName="Helvetica-Bold", fontSize=21, fillColor=colors.HexColor("#102a43")))
    drawing.add(String(115, 72, band, textAnchor="middle", fontName="Helvetica-Bold", fontSize=11, fillColor=colors.HexColor(ring)))
    drawing.add(String(115, 25, "LOW  •  MODERATE  •  HIGH  •  CRITICAL", textAnchor="middle", fontName="Helvetica", fontSize=8.5, fillColor=colors.HexColor("#52606d")))
    return drawing


def _reportlab_pie_drawing(counts: Dict[str, int], title: str, color_map: Dict[str, str]):
    if not counts or sum(counts.values()) <= 0:
        return None

    from reportlab.lib import colors
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing, Rect, String

    drawing = Drawing(230, 170)
    drawing.add(Rect(0, 0, 230, 170, fillColor=colors.HexColor("#f8fafc"), strokeColor=colors.HexColor("#dbe2ea"), rx=10, ry=10))
    drawing.add(String(115, 148, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=13, fillColor=colors.HexColor("#102a43")))

    pie = Pie()
    pie.x = 16
    pie.y = 20
    pie.width = 110
    pie.height = 110
    pie.data = [max(int(value), 0) for value in counts.values()]
    pie.labels = [f"{label} ({value})" for label, value in counts.items()]
    pie.sideLabels = True
    pie.simpleLabels = False
    pie.slices.strokeWidth = 0.6
    pie.slices.strokeColor = colors.white

    for idx, key in enumerate(counts.keys()):
        pie.slices[idx].fillColor = colors.HexColor(color_map.get(key, "#64748b"))
        pie.slices[idx].labelRadius = 1.15
        pie.slices[idx].fontName = "Helvetica"
        pie.slices[idx].fontSize = 8.5

    drawing.add(pie)

    legend_y = 118
    for idx, (label, value) in enumerate(list(counts.items())[:4]):
        swatch_y = legend_y - idx * 22
        drawing.add(Rect(145, swatch_y, 10, 10, fillColor=colors.HexColor(color_map.get(label, "#64748b")), strokeColor=colors.HexColor(color_map.get(label, "#64748b"))))
        drawing.add(String(160, swatch_y + 2, f"{label}: {value}", fontName="Helvetica", fontSize=9.2, fillColor=colors.HexColor("#243b53")))
    return drawing


def _markdown_to_flowables(text: str, styles) -> list[object]:
    from reportlab.platypus import Paragraph, Spacer

    flowables: list[object] = []
    for raw_line in str(text or "No report generated.").splitlines():
        line = raw_line.strip()
        if not line:
            flowables.append(Spacer(1, 4))
            continue
        if line.startswith("## "):
            flowables.append(Paragraph(_inline_markup(line[3:]), styles["SecH"]))
            continue
        if line.startswith("### "):
            flowables.append(Paragraph(_inline_markup(line[4:]), styles["SecH"]))
            continue
        if line.startswith("- "):
            flowables.append(Paragraph(f"&bull; {_inline_markup(line[2:])}", styles["Body"]))
            continue
        flowables.append(Paragraph(_inline_markup(line), styles["Body"]))
    return flowables


def _inline_markup(text: str) -> str:
    escaped = html.escape(str(text))
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def _build_simple_pdf(state: Dict, volunteer_matches: List[Dict], approvals: List[Dict]) -> bytes:
    cards = charts.incident_summary_cards(state, volunteer_matches, approvals)
    chart_blocks = [
        ("Need categories", charts.volunteer_need_counts(volunteer_matches), charts.NEED_COLORS),
        ("Provider mix", charts.provider_type_mix([m.get("best_match", {}).get("resource", {}) for m in volunteer_matches if m.get("best_match")]), {
            "Government": "#0f766e",
            "Ngo": "#2563eb",
            "Volunteer": "#7c3aed",
            "Private": "#ea580c",
            "Unknown": "#64748b",
        }),
        ("Worklist status", charts.volunteer_status_counts(volunteer_matches), charts.STATUS_COLORS),
        ("Human decisions", charts.approval_action_counts(approvals), {
            "approved_sent": "#2e7d32",
            "rejected": "#c62828",
            "edited_not_sent": "#fb8c00",
            "unknown": "#607d8b",
        }),
    ]
    worklist_lines = []
    for match in volunteer_matches[:8]:
        need = match.get("need") or {}
        best = match.get("best_match") or {}
        provider = (best.get("resource") or {}).get("provider_name", "No provider")
        worklist_lines.append(
            f"{need.get('request_id', 'N/A')} | {need.get('category', 'N/A')} | {match.get('status', 'N/A')} | {provider}"
        )
    if not worklist_lines:
        worklist_lines = ["No volunteer worklist matches available."]

    decision_lines = [
        f"{str(a.get('timestamp', ''))[:19]} | {a.get('request_id', 'N/A')} | {a.get('action', 'N/A')}"
        for a in approvals[-8:]
    ] or ["No human decisions logged."]

    trace_lines = state.get("node_log", []) or ["No node log recorded."]

    return _styled_pdf_bytes(
        title="ResQ Disaster Relief Incident Report",
        subtitle=(
            f"District: {state.get('district', 'N/A')}   |   Disaster: {state.get('disaster_type', 'N/A')}   |   "
            f"Model: {state.get('llm_model_label', 'N/A')}"
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
        cards=cards[:4],
        chart_blocks=chart_blocks,
        response_lines=_plain_text_lines(state.get("final_report", "No report generated.")),
        route_lines=[
            f"Distance: {(state.get('route') or {}).get('distance_km', 'N/A')} km",
            f"Duration: {(state.get('route') or {}).get('duration_min', 'N/A')} min",
            f"Route warning: {state.get('route_warning', 'None')}",
        ],
        worklist_lines=worklist_lines,
        decision_lines=decision_lines,
        trace_lines=trace_lines,
    )


def _plain_text_lines(text: str) -> list[str]:
    output = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            output.append("")
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        output.append(line)
    return output or ["No report generated."]


def _simple_pdf_bytes(lines: List[str]) -> bytes:
    page_width = 595
    page_height = 842
    left = 50
    top = 790
    line_height = 16
    lines_per_page = 44

    pages = [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)] or [["No content"]]
    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_obj = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    content_ids = []

    for page_lines in pages:
        content = ["BT", f"/F1 11 Tf", f"{left} {top} Td"]
        for idx, line in enumerate(page_lines):
            safe = _pdf_escape(line)
            if idx == 0:
                content.append(f"({safe}) Tj")
            else:
                content.append(f"0 -{line_height} Td ({safe}) Tj")
        content.append("ET")
        content_stream = "\n".join(content).encode("latin-1", "replace")
        content_id = add_object(
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("latin-1")
            + content_stream
            + b"\nendstream"
        )
        content_ids.append(content_id)
        page_ids.append(add_object(b""))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1"))

    for idx, page_id in enumerate(page_ids):
        objects[page_id - 1] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_ids[idx]} 0 R >>"
        ).encode("latin-1")

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("latin-1"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("latin-1")
    )
    return bytes(output)


def _pdf_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _styled_pdf_bytes(
    title: str,
    subtitle: str,
    generated_at: str,
    cards: List[tuple[str, str, str]],
    chart_blocks: List[tuple[str, Dict[str, int], Dict[str, str]]],
    response_lines: List[str],
    route_lines: List[str],
    worklist_lines: List[str],
    decision_lines: List[str],
    trace_lines: List[str],
) -> bytes:
    page_width = 595
    page_height = 842
    margin = 36
    card_w = 126
    card_h = 66
    section_gap = 18

    pages: List[List[str]] = [[]]
    current_y = 0

    def new_page() -> None:
        nonlocal current_y
        pages.append([])
        current_y = page_height - margin

    def ops() -> List[str]:
        return pages[-1]

    def ensure_space(height: int) -> None:
        nonlocal current_y
        if current_y - height < margin:
            new_page()

    def draw_text(x: int, y: int, text: str, size: int = 10, color: str = "0 0 0", bold: bool = False) -> None:
        font = "Helvetica-Bold" if bold else "Helvetica"
        ops().append(f"BT /{font} {size} Tf {color} rg 1 0 0 1 {x} {y} Tm ({_pdf_escape(text)}) Tj ET")

    def draw_rect(x: int, y: int, w: int, h: int, fill_rgb: str, stroke_rgb: str | None = None) -> None:
        ops().append(f"q {fill_rgb} rg")
        if stroke_rgb:
            ops().append(f" {stroke_rgb} RG 1 w")
        ops().append(f" {x} {y} {w} {h} re B Q" if stroke_rgb else f" {x} {y} {w} {h} re f Q")

    def section_header(label: str) -> None:
        nonlocal current_y
        ensure_space(28)
        draw_text(margin, current_y, label, size=14, color="0.06 0.16 0.26", bold=True)
        current_y -= 18

    def draw_card(x: int, y: int, title_text: str, value_text: str, subtitle_text: str, color_hex: str) -> None:
        rgb = _hex_to_rgb(color_hex)
        draw_rect(x, y, card_w, card_h, "0.97 0.98 0.99", "0.84 0.88 0.91")
        draw_rect(x, y + card_h - 6, card_w, 6, rgb)
        draw_text(x + 10, y + 36, value_text, size=22, color="0.06 0.16 0.26", bold=True)
        draw_text(x + 10, y + 20, title_text, size=10, color="0.35 0.39 0.44")
        draw_text(x + 10, y + 9, subtitle_text[:24], size=9, color="0.48 0.53 0.58")

    def draw_mix_block(x: int, y: int, width: int, height: int, title_text: str, counts: Dict[str, int], color_map: Dict[str, str]) -> None:
        draw_rect(x, y, width, height, "0.985 0.988 0.992", "0.85 0.89 0.93")
        draw_text(x + 10, y + height - 16, title_text, size=12, color="0.06 0.16 0.26", bold=True)
        if not counts:
            draw_text(x + 10, y + height - 34, "No data", size=10, color="0.45 0.5 0.55")
            return
        total = sum(counts.values()) or 1
        row_y = y + height - 38
        for label_text, value in list(counts.items())[:5]:
            rgb = _hex_to_rgb(color_map.get(label_text, "#64748b"))
            draw_rect(x + 10, row_y - 1, 10, 10, rgb)
            pct = round((value / total) * 100)
            draw_text(x + 26, row_y, f"{label_text[:16]}  {value} ({pct}%)", size=10, color="0.22 0.27 0.33")
            row_y -= 16

    def draw_lines(lines: List[str], body_size: int = 10, leading: int = 14) -> None:
        nonlocal current_y
        for line in lines:
            for wrapped in _wrap_text(line, 78):
                ensure_space(leading)
                draw_text(margin, current_y, wrapped, size=body_size, color="0.12 0.16 0.2")
                current_y -= leading

    current_y = page_height - margin
    draw_rect(0, page_height - 92, page_width, 92, "0.06 0.16 0.26")
    draw_text(margin, page_height - 48, title, size=23, color="1 1 1", bold=True)
    draw_text(margin, page_height - 67, subtitle[:92], size=10, color="0.83 0.9 0.94")
    draw_text(margin, page_height - 80, f"Generated {generated_at}", size=9, color="0.72 0.82 0.87")
    current_y = page_height - 120

    card_colors = ["#1565c0", "#6a1b9a", "#c62828", "#2e7d32"]
    for idx, (label, value, subtitle_text) in enumerate(cards):
        draw_card(margin + idx * (card_w + 8), current_y - card_h, label, value, subtitle_text, card_colors[idx % len(card_colors)])
    current_y -= card_h + section_gap

    section_header("Visual Summary")
    draw_mix_block(margin, current_y - 112, 250, 112, chart_blocks[0][0], chart_blocks[0][1], chart_blocks[0][2])
    draw_mix_block(margin + 272, current_y - 112, 250, 112, chart_blocks[1][0], chart_blocks[1][1], chart_blocks[1][2])
    current_y -= 128
    draw_mix_block(margin, current_y - 112, 250, 112, chart_blocks[2][0], chart_blocks[2][1], chart_blocks[2][2])
    draw_mix_block(margin + 272, current_y - 112, 250, 112, chart_blocks[3][0], chart_blocks[3][1], chart_blocks[3][2])
    current_y -= 128

    section_header("Actionable Response")
    draw_lines(response_lines[:18])
    current_y -= 4

    section_header("Routing & Risk")
    draw_lines(route_lines, body_size=10, leading=13)
    current_y -= 4

    section_header("Volunteer Worklist Snapshot")
    draw_lines(worklist_lines[:10], body_size=9, leading=12)
    current_y -= 4

    section_header("Recent Human Decisions")
    draw_lines(decision_lines[:8], body_size=9, leading=12)
    current_y -= 4

    section_header("Agent Trace")
    draw_lines(trace_lines[:14], body_size=9, leading=12)

    return _assemble_pdf_from_pages(pages, page_width, page_height)


def _assemble_pdf_from_pages(pages: List[List[str]], page_width: int, page_height: int) -> bytes:
    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_regular = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    page_ids = []
    content_ids = []

    for page_ops in pages:
        content_stream = "\n".join(page_ops).encode("latin-1", "replace")
        content_id = add_object(
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("latin-1")
            + content_stream
            + b"\nendstream"
        )
        content_ids.append(content_id)
        page_ids.append(add_object(b""))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1"))

    for idx, page_id in enumerate(page_ids):
        objects[page_id - 1] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /Helvetica {font_regular} 0 R /Helvetica-Bold {font_bold} 0 R >> >> "
            f"/Contents {content_ids[idx]} 0 R >>"
        ).encode("latin-1")

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))
    return _write_pdf_objects(objects, catalog_id)


def _write_pdf_objects(objects: List[bytes], catalog_id: int) -> bytes:
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("latin-1"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("latin-1")
    )
    return bytes(output)


def _hex_to_rgb(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return f"{r:.3f} {g:.3f} {b:.3f}"


def _wrap_text(text: str, width: int) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
