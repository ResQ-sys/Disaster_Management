"""
Operations-map helpers: place a free-text need/resource location on the map.

Locations in needs.csv / resources.csv / tweets are town-level descriptions
("Kullu bus stand", "Manali ward 3"). A built-in gazetteer of HP towns keeps
the operations map instant and fully offline; Nominatim geocoding is only a
fallback for places the gazetteer doesn't know.
"""

from typing import Optional

# Town → (lat, lon). District HQs plus the towns used in the demo datasets.
HP_PLACE_COORDS = {
    "shimla": (31.104, 77.173), "mandi": (31.708, 76.931), "kullu": (31.958, 77.109),
    "manali": (32.243, 77.189), "solan": (30.905, 77.097), "kangra": (32.099, 76.271),
    "dharamshala": (32.219, 76.323), "bilaspur": (31.331, 76.757), "chamba": (32.556, 76.126),
    "una": (31.464, 76.269), "hamirpur": (31.684, 76.522), "nahan": (30.561, 77.295),
    "keylong": (32.571, 77.032), "rampur": (31.449, 77.633), "sundernagar": (31.532, 76.905),
    "bhuntar": (31.888, 77.154), "aut": (31.749, 77.106), "baijnath": (32.052, 76.649),
    "palampur": (32.110, 76.537), "reckong peo": (31.538, 78.272), "kaza": (32.227, 78.072),
    "kalpa": (31.536, 78.258), "theog": (31.121, 77.348), "karsog": (31.383, 77.200),
    "jogindernagar": (31.986, 76.788), "sarkaghat": (31.699, 76.735), "nagrota": (32.056, 76.386),
    "dalhousie": (32.539, 75.971), "banjar": (31.639, 77.344), "sainj": (31.771, 77.311),
    "kasol": (32.010, 77.315), "jibhi": (31.593, 77.345), "nurpur": (32.297, 75.883),
    "paonta sahib": (30.437, 77.624), "ghumarwin": (31.443, 76.716), "chirgaon": (31.281, 77.858),
    "pandoh": (31.669, 77.057), "kotkhai": (31.117, 77.539), "arki": (31.152, 76.965),
    "dehra": (31.867, 76.216),
}


def locate(place_text: str) -> Optional[tuple]:
    """Best-effort (lat, lon, source) for a free-text place description.

    Gazetteer lookup first (offline, instant): the longest known town name
    appearing in the text wins ("paonta sahib" beats "paonta"). Falls back to
    Nominatim geocoding, returning None if that fails too.
    """
    text = (place_text or "").strip().lower()
    if not text:
        return None

    hit = None
    for town, coords in HP_PLACE_COORDS.items():
        if town in text and (hit is None or len(town) > len(hit[0])):
            hit = (town, coords)
    if hit:
        town, (lat, lon) = hit
        return lat, lon, f"gazetteer · {town.title()}"

    from resq_project.tools import geocode_place
    geo = geocode_place(f"{place_text}, Himachal Pradesh, India")
    if geo:
        return geo[0], geo[1], "geocoded (OSM)"
    return None
