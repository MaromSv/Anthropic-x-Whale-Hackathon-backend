"""Tag lists for the RAG fast-lane.

The on-device RAG pipeline first filters documents by tag (e.g. only search
within `kings_day_priority` for likely festival emergencies). If similarity
is low, it falls back to the full document set.

Add new event/context lists here as the app expands (warzone, hiking, etc.).
"""

# Statistically most likely emergencies at King's Day Amsterdam:
# heavy alcohol + drug use, canal falls, crowd density, broken glass.
KINGS_DAY_HOTLIST = {
    # Substances
    "Alcohol intoxication",
    "Ecstasy (MDMA) toxicity",
    "GHB toxicity",
    "Cocaine toxicity",
    "Amphetamine toxicity",
    "Nitrous oxide toxicity",
    # Behavioral
    "Agitated or combative patient",
    "Panic attack",
    # Environmental
    "Hypothermia",
    "Heat exhaustion",
    "Drowning",
    # Trauma
    "Laceration repair",
}


def kings_day_tags_for(title: str) -> tuple[str, list[str]]:
    """Return (priority, tags) for a given WikEM page title."""
    if title in KINGS_DAY_HOTLIST:
        return "high", ["kings_day_priority"]
    return "normal", []
