"""Core medical pack — universal first-aid + emergency medicine knowledge.

The actual content is built offline by `scripts/build_core_medical.py`,
which scrapes WikEM, applies tags, and writes `data/core_medical.json`.
This module just loads that file at request time.
"""
import json
from functools import lru_cache
from pathlib import Path

from models.schemas import RAGDocument

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "core_medical.json"


@lru_cache(maxsize=1)
def _load_payload() -> dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run: python -m scripts.build_core_medical"
        )
    return json.loads(DATA_FILE.read_text())


def build_core_medical_data() -> list[RAGDocument]:
    payload = _load_payload()
    return [RAGDocument(**doc) for doc in payload["documents"]]
