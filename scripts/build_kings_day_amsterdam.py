"""One-time builder for the King's Day Amsterdam pack.

Fetches the official Gemeente Amsterdam map GeoJSON
(https://kaart.amsterdam.nl/koningsdag-2026 → /api/maps/894.json), extracts
the categories we care about (EHBO posts, toilets, P+R), and writes
`data/kings_day_amsterdam.json`. The runtime layer in
`scrapers/kings_day_nl.py` merges this with the hardcoded RAG docs.

Run from the repo root:
    python -m scripts.build_kings_day_amsterdam
"""
import html
import json
import re
import sys
from pathlib import Path

import requests

MAP_API = "https://kaart.amsterdam.nl/api/maps/894.json"
MAP_LINK = "https://kaart.amsterdam.nl/koningsdag-2026"
HEADERS = {"User-Agent": "CrisisAppDataPackBuilder/1.0"}

# Map Gemeente category names → our POI `kind` slug.
# Add more categories here to ship them with the pack.
CATEGORY_TO_KIND = {
    "EHBO posten": "ehbo",
    "Toiletten": "toilet",
    "P+R": "park_ride",
}

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "kings_day_amsterdam.json"


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def _coord_from_feature(feat: dict) -> tuple[float, float] | None:
    """Return (lat, lng) from a GeoJSON Point or MultiPoint feature."""
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates")
    if not coords:
        return None
    # GeoJSON is [lng, lat]
    if geom.get("type") == "Point":
        lng, lat = coords[0], coords[1]
    elif geom.get("type") == "MultiPoint" and coords:
        lng, lat = coords[0][0], coords[0][1]
    else:
        return None
    return float(lat), float(lng)


def fetch_features() -> list[dict]:
    print(f"Fetching {MAP_API}...")
    resp = requests.get(MAP_API, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("features", [])


def extract_pois(features: list[dict]) -> list[dict]:
    pois: list[dict] = []
    for feat in features:
        props = feat.get("properties", {})
        cat = props.get("category", "")
        kind = CATEGORY_TO_KIND.get(cat)
        if kind is None:
            continue
        coord = _coord_from_feature(feat)
        if coord is None:
            continue
        lat, lng = coord
        address_parts = [props.get("street"), props.get("city")]
        address = ", ".join(p for p in address_parts if p) or None
        pois.append({
            "name": props.get("title", "").strip(),
            "kind": kind,
            "lat": lat,
            "lng": lng,
            "address": address,
            "description": _strip_html(props.get("description", "")) or None,
            "hours": None,  # Gemeente embeds hours in description; leave for now
            "source": props.get("link") or MAP_LINK,
        })
    return pois


def build_ehbo_rag_doc(pois: list[dict]) -> dict | None:
    """Single doc summarising all EHBO posts so the on-device LLM can answer
    'where is the nearest first-aid post?' even before the map is consulted."""
    ehbo = [p for p in pois if p["kind"] == "ehbo"]
    if not ehbo:
        return None
    lines = [
        "Official EHBO (first-aid) posts on King's Day Amsterdam. For "
        "non-life-threatening issues, go to the nearest post instead of "
        "calling 112. All posts are open 11:30–23:30 unless otherwise stated.",
        "",
    ]
    for p in sorted(ehbo, key=lambda x: x["name"]):
        line = f"- {p['name']}"
        if p["address"]:
            line += f" — {p['address']}"
        lines.append(line)
    return {
        "title": "EHBO First-Aid Posts — King's Day Amsterdam (Official)",
        "category": "Navigation",
        "content": "\n".join(lines),
        "tags": ["ehbo", "first_aid_post", "navigation", "kings_day"],
        "priority": "high",
        "severity": "info",
        "source": MAP_LINK,
    }


def build() -> None:
    print("=== Building King's Day Amsterdam pack ===\n")
    features = fetch_features()
    print(f"  {len(features)} features in source GeoJSON")

    pois = extract_pois(features)
    by_kind: dict[str, int] = {}
    for p in pois:
        by_kind[p["kind"]] = by_kind.get(p["kind"], 0) + 1
    print(f"  Extracted {len(pois)} POIs:")
    for k, n in sorted(by_kind.items()):
        print(f"    {n:3d}  {k}")

    documents = []
    ehbo_doc = build_ehbo_rag_doc(pois)
    if ehbo_doc:
        documents.append(ehbo_doc)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pack_id": "kings_day_amsterdam",
        "version": "1.0",
        "documents": documents,
        "points_of_interest": pois,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUTPUT}")
    print(f"  documents: {len(documents)}")
    print(f"  points_of_interest: {len(pois)}")


if __name__ == "__main__":
    try:
        build()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
