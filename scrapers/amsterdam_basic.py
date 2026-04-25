"""Amsterdam basic pack — general info for anyone in the city.

Emergency contacts (NL), nearest hospitals / Huisartsenpost, tram safety,
canal awareness, pickpocket / theft reporting, etc.
"""
from models.schemas import RAGDocument


def build_amsterdam_basic_data() -> list[RAGDocument]:
    return [
        RAGDocument(
            title="Drug Overdose — Police Amnesty (Netherlands)",
            category="Legal/Medical",
            content=(
                "CRITICAL: In the Netherlands you will NOT be arrested or "
                "prosecuted for seeking medical help for illegal drug use. "
                "If someone is overdosing on MDMA, GHB, cocaine, ketamine, or "
                "any other substance, call 112 immediately. Be honest with the "
                "paramedics about exactly what was taken and how much — their "
                "only job is to save the person's life, not to involve the "
                "police. Hesitating to call out of fear of arrest is the "
                "single most common reason drug overdoses become fatal."
            ),
            tags=["drugs", "overdose", "police", "mdma", "ghb", "legal", "amnesty"],
            priority="high",
            severity="urgent",
            source="https://www.jellinek.nl",
        ),
        RAGDocument(
            title="Emergency Numbers — Netherlands",
            category="Contacts",
            content=(
                "112 — Life-threatening emergencies (police, ambulance, fire). "
                "0900-8844 — Non-emergency police. "
                "113 — Suicide prevention. "
                "088-003-0600 — Huisartsenpost (out-of-hours GP, Amsterdam region)."
            ),
            tags=["emergency", "phone", "112"],
            severity="info",
        ),
        RAGDocument(
            title="Major Hospitals in Amsterdam",
            category="Navigation",
            content=(
                "OLVG (Oost & West) — large general hospitals with 24/7 ER. "
                "Amsterdam UMC (AMC, Zuidoost / VUmc, Zuid) — university hospitals, "
                "trauma centers. For minor issues out-of-hours, contact Huisartsenpost first."
            ),
            tags=["hospital", "ER", "spoedeisende_hulp"],
            severity="info",
        ),
        RAGDocument(
            title="Canal Safety",
            category="Hazards",
            content=(
                "Amsterdam canals are 2–3 m deep with steep walls and cold water "
                "year-round. If someone falls in: do not jump in after them — throw "
                "a flotation device, call 112, point to them so rescuers can locate. "
                "Climb-out ladders are placed every ~50 m along most canals."
            ),
            tags=["canal", "drowning", "water_rescue"],
            severity="life_threatening",
        ),
    ]
