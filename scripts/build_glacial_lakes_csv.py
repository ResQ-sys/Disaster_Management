"""
Extract Himachal Pradesh glacial-lake records from the CWC monthly monitoring
report PDF and write them to source_data/hp_glacial_lakes.csv.

Source: "Monthly Monitoring Report of Glacial Lakes & Water Bodies in the
Himalayan Region of Indian River Basins - September 2025" (Central Water
Commission, M&CC Directorate). GLOF-relevant, HP-only.

The PDF tables come in a few inconsistent layouts, so we anchor extraction on
the coordinate columns (which every lake row has) — either DMS
(32° 29' 47.04"  77° 33' 5.76") or decimal (32.33590 76.33200) — and read the
lake id before and the basin/river/district/area columns after.
"""

import re
import csv
import math
from pathlib import Path

from pypdf import PdfReader

# Source PDF and output CSV both live under source_data/
REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = REPO_ROOT / "source_data" / "Monitoring_Report_September_2025 .pdf"
OUT_CSV = REPO_ROOT / "source_data" / "hp_glacial_lakes.csv"

REPORT_SOURCE = "CWC Monthly Monitoring Report of Glacial Lakes - September 2025"

# HP districts (for parsing + nearest-centroid inference when absent)
HP_DISTRICT_CENTERS = {
    "KANGRA": (32.10, 76.27), "MANDI": (31.71, 76.93), "SHIMLA": (31.10, 77.17),
    "KULLU": (31.95, 77.11), "SOLAN": (30.91, 77.10), "SIRMOUR": (30.56, 77.66),
    "BILASPUR": (31.34, 76.76), "HAMIRPUR": (31.68, 76.52), "CHAMBA": (32.55, 76.12),
    "UNA": (31.47, 76.27), "KINNAUR": (31.59, 78.44), "LAHUL AND SPITI": (32.55, 77.60),
}
DISTRICT_ALIASES = {
    "LAHUL & SPITI": "LAHUL AND SPITI", "LAHUL AND SPITI": "LAHUL AND SPITI",
    "LAHUL": "LAHUL AND SPITI", "SIRMAUR": "SIRMOUR",
}
KNOWN_RIVERS = ["Chandra Bhaga", "Chandra", "Bhaga", "Chenab", "Sutlej", "Satluj",
                "Ravi", "Beas", "Spiti", "Baspa", "Parvati", "Pin"]

# Coordinate anchors
DMS = r"(\d{1,3})°\s*(\d{1,2})'\s*([\d.]+)\"?"
DMS_PAIR = re.compile(DMS + r"\s+" + DMS)
DEC_PAIR = re.compile(r"(?<![\d.])(\d{2}\.\d{3,6})\s+(\d{2}\.\d{3,6})(?![\d.])")

# A logical table row starts with "<Sno> <LakeID>" where LakeID is one of the
# report's id formats (01_52H_004 | 0152D0703190 | 0253I0700018 | 1774 SDC).
ROW_START = re.compile(
    r"^\s*\d{1,4}\s+(?:01_\d+[A-Z]_\d+|\d{9,}|[0-9][0-9A-Z]{9,}|\d{3,4}\s+(?:NRSC|SDC))",
    re.IGNORECASE,
)


def dms_to_dec(d, m, s):
    return round(int(d) + int(m) / 60 + float(s) / 3600, 5)


def nearest_district(lat, lon):
    best, bestd = "", float("inf")
    for name, (dlat, dlon) in HP_DISTRICT_CENTERS.items():
        dist = math.hypot(lat - dlat, lon - dlon)
        if dist < bestd:
            bestd, best = dist, name
    return best


def clean_district(raw):
    raw = raw.upper().strip()
    return DISTRICT_ALIASES.get(raw, raw if raw in HP_DISTRICT_CENTERS else "")


def logical_rows(text):
    """Reconstruct logical table rows from a page's lines: a row starts at a
    "<Sno> <LakeID>" line; wrapped continuation lines are merged into it."""
    rows, current = [], None
    for line in text.splitlines():
        if not line.strip():
            continue
        if ROW_START.match(line):
            if current:
                rows.append(current)
            current = line.strip()
        elif current is not None:
            current += " " + line.strip()
    if current:
        rows.append(current)
    return rows


def parse_row(row):
    """Parse one logical HP table row → record dict, or None."""
    # Coordinates (DMS preferred, else decimal)
    m = DMS_PAIR.search(row)
    if m:
        lat = dms_to_dec(m.group(1), m.group(2), m.group(3))
        lon = dms_to_dec(m.group(4), m.group(5), m.group(6))
        coord_end = m.end()
    else:
        m = DEC_PAIR.search(row)
        if not m:
            return None
        lat, lon = float(m.group(1)), float(m.group(2))
        coord_end = m.end()
    if not (30.0 <= lat <= 33.5 and 75.5 <= lon <= 79.5):
        return None

    lake_id = row.split()[1] if len(row.split()) > 1 else ""
    after = row[coord_end:]

    basin = next((b for b in ("Indus", "Ganga", "Brahmaputra") if b in after), "")

    river = ""
    for rv in KNOWN_RIVERS:
        if re.search(rv, after, re.IGNORECASE):
            river = "Chandra Bhaga" if rv in ("Chandra", "Bhaga") else rv
            river = "Sutlej" if river == "Satluj" else river
            break

    district = ""
    for dname in list(HP_DISTRICT_CENTERS) + list(DISTRICT_ALIASES):
        if re.search(re.escape(dname), after, re.IGNORECASE):
            district = clean_district(dname)
            if district:
                break
    if not district:
        district = nearest_district(lat, lon)

    # Last integer of the row = % change vs base year (report's convention)
    tail = after.split("Pradesh", 1)[-1]
    nums = re.findall(r"-?\d+", tail)
    pct = int(nums[-1]) if nums else None

    status = ("unknown" if pct is None else
              "increase" if pct > 0 else
              "decrease" if pct < 0 else "no_change")

    return {
        "lake_id": lake_id,
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "basin": basin,
        "river": river,
        "district": district,
        "area_pct_change": "" if pct is None else pct,
        "status": status,
        "state": "Himachal Pradesh",
        "monitored_period": "September 2025",
        "source": REPORT_SOURCE,
    }


def extract():
    reader = PdfReader(str(PDF_PATH))
    records = {}
    for page in reader.pages:
        text = page.extract_text() or ""
        if "himachal" not in text.lower():
            continue
        for row in logical_rows(text):
            if "himachal" not in row.lower():
                continue
            rec = parse_row(row)
            if rec:
                records[rec["lake_id"]] = rec
    return list(records.values())


def main():
    rows = extract()
    rows.sort(key=lambda r: (r["district"], r["lake_id"]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["lake_id", "latitude", "longitude", "basin", "river", "district",
              "area_pct_change", "status", "state", "monitored_period", "source"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    inc = sum(1 for r in rows if r["status"] == "increase")
    print(f"Wrote {len(rows)} HP glacial-lake records → {OUT_CSV}")
    print(f"  increasing (GLOF watch): {inc}")
    from collections import Counter
    print("  by district:", dict(Counter(r["district"] for r in rows)))
    for r in rows[:12]:
        print("   ", r["lake_id"], r["latitude"], r["longitude"], r["river"] or "-",
              r["district"], r["status"], r["area_pct_change"])


if __name__ == "__main__":
    main()
