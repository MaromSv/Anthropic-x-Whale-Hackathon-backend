"""King's Day Amsterdam pack — event-specific hazards, rules, and protocols
layered on top of the Amsterdam basic pack.

Two data sources:
  1. Hardcoded RAG docs below (NS rules, crowd crush, canal rescue, street
     alcohol limit) — facts the LLM should always have for this event.
  2. `data/kings_day_amsterdam.json` produced by
     `scripts/build_kings_day_amsterdam.py`, which fetches the Gemeente
     Amsterdam map (EHBO posts, toilets, P+R) and writes structured POIs +
     an auto-generated 'where are the EHBO posts' RAG doc.

The runtime merges both. Clinical content (alcohol intoxication, drowning
treatment) intentionally lives in `core_medical` only — no duplication here.
"""
import json
from functools import lru_cache
from pathlib import Path

from models.schemas import POI, RAGDocument

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "kings_day_amsterdam.json"


def _ns_rules_doc() -> RAGDocument:
    return RAGDocument(
        title="NS Train Rules — King's Day & King's Night",
        category="Rules",
        content=(
            "Alcohol ban: from 19:00 on April 26 to 07:00 on April 28, "
            "travellers may NOT bring alcohol to NS stations or onto trains. "
            "Station shops will not sell alcohol during this window. "
            "Closed stations: Amsterdam Science Park and Amsterdam RAI are "
            "closed all of King's Day. Utrecht Centraal Jaarbeurszijde "
            "entrances close from 01:00 King's Night — enter via Centrumzijde. "
            "Bikes: not allowed on trains on King's Day (April 27). On the "
            "R-net Alphen aan den Rijn–Gouda route, bikes are banned both "
            "King's Day and King's Night. "
            "International: Eurocity Direct runs to/from Almere Centrum "
            "(not Lelystad Centrum) on April 27. Allow extra travel time to "
            "Schiphol due to heavy city traffic."
        ),
        tags=["train", "ns", "alcohol", "travel", "kings_day"],
        priority="high",
        severity="info",
        source="https://www.ns.nl/en/featured/kings-day",
    )


def _hardcoded_docs() -> list[RAGDocument]:
    return [
        _ns_rules_doc(),
        RAGDocument(
            title="Crowd Crush — What To Do",
            category="Hazards",
            content=(
                "Hands up by the chest like a boxer to protect your ribs. Move "
                "diagonally across the flow of the crowd, not against it. Stay "
                "on your feet at all costs — if you fall, curl into a ball with "
                "hands over your head and try to rise during any surge."
            ),
            tags=["crowd_crush", "stampede", "kings_day"],
            priority="high",
            severity="life_threatening",
        ),
        RAGDocument(
            title="Canal Water Rescue — King's Day Context",
            category="Hazards",
            content=(
                "King's Day sees many canal falls due to crowding and alcohol. "
                "Do not jump in. Throw a life ring or anything that floats. "
                "Call 112 — Dutch law protects you, even if drugs are involved. "
                "Keep eye contact with the person and point so rescuers can locate. "
                "Climb-out ladders are placed every ~50 m along most Amsterdam canals."
            ),
            tags=["canal", "drowning", "kings_day"],
            priority="high",
            severity="life_threatening",
        ),
        RAGDocument(
            title="Street Alcohol Limit — King's Day Amsterdam",
            category="Rules",
            content=(
                "On King's Day, Amsterdam supermarkets in stadsdeel Centrum "
                "(within the S100 ring) are restricted to selling ONE alcoholic "
                "drink per customer per transaction, and may not display chilled "
                "alcohol. Carrying more than one unit (a sixpack or crate counts "
                "as multiple) on the street is prohibited. Rules apply 06:00 to "
                "midnight on April 27."
            ),
            tags=["alcohol", "rules", "kings_day"],
            priority="normal",
            severity="info",
            source="https://www.amsterdam.nl/en/leisure/kingsday/",
        ),
        RAGDocument(
            title="Non-Emergency First Aid — Use EHBO, Not 112",
            category="Navigation",
            content=(
                "If the situation is NOT life-threatening, do not call 112. "
                "Walk to the nearest EHBO (Red Cross) post — they handle minor "
                "injuries, intoxication triage, and dehydration. The official "
                "list of post locations is included in this pack as map points. "
                "For anything life-threatening (unconscious, not breathing, "
                "severe bleeding, severe overdose), call 112 immediately."
            ),
            tags=["ehbo", "navigation", "kings_day", "triage"],
            priority="high",
            severity="info",
        ),
        RAGDocument(
            title="Boat Safety — King's Day Canals",
            category="Hazards",
            content=(
                "King's Day canals are extremely crowded with party boats. "
                "Legal limit: maximum 12 people plus a skipper per boat — "
                "overloaded boats capsize, and falls into the canal during "
                "crowding are a leading cause of drownings. Skippers face "
                "fines (€80–€800) for violations. Wear something visible. "
                "If your boat capsizes or someone falls overboard: do not jump "
                "in after them, throw anything that floats, call 112, keep "
                "eye contact and point so rescuers can locate. Cold-water "
                "shock can incapacitate strong swimmers in seconds."
            ),
            tags=["boat", "canal", "drowning", "kings_day"],
            priority="high",
            severity="life_threatening",
            source="https://www.amsterdam.nl/en/leisure/kingsday/",
        ),
    ]


@lru_cache(maxsize=1)
def _load_built_pack() -> dict:
    if not DATA_FILE.exists():
        return {"documents": [], "points_of_interest": []}
    return json.loads(DATA_FILE.read_text())


def build_kings_day_data() -> list[RAGDocument]:
    docs = list(_hardcoded_docs())
    for raw in _load_built_pack().get("documents", []):
        docs.append(RAGDocument(**raw))
    return docs


def build_kings_day_pois() -> list[POI]:
    return [POI(**raw) for raw in _load_built_pack().get("points_of_interest", [])]
