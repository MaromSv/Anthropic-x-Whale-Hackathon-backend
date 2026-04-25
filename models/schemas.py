from pydantic import BaseModel, Field
from typing import List, Optional


class RAGDocument(BaseModel):
    """A single chunk of knowledge the on-device LLM will retrieve over."""
    title: str
    category: str  # e.g. "Toxicology", "Trauma", "Hazards", "Contacts"
    content: str
    tags: List[str] = Field(default_factory=list)
    priority: str = "normal"  # "high" → search-fast-lane on relevant queries
    severity: Optional[str] = None  # "info" | "urgent" | "life_threatening"
    source: Optional[str] = None  # URL or human-readable origin


class PackManifest(BaseModel):
    """Lightweight description of a pack — used by the listing endpoint."""
    pack_id: str
    name: str
    description: str
    version: str
    location: Optional[str] = None
    is_core: bool = False  # core packs are installed by default on the device


class POI(BaseModel):
    """A geo-pinned point of interest — EHBO post, hospital, toilet, P+R, etc.

    Used by the Android app to plot pins on the map *and* to answer
    'where is the nearest X' questions via the on-device RAG layer.
    """
    name: str
    kind: str  # "ehbo" | "toilet" | "hospital" | "park_ride" | "diversion" | ...
    lat: float
    lng: float
    address: Optional[str] = None
    description: Optional[str] = None
    hours: Optional[str] = None
    source: Optional[str] = None


class DataPack(BaseModel):
    pack_id: str
    version: str
    location: Optional[str] = None
    documents: List[RAGDocument]
    points_of_interest: List[POI] = Field(default_factory=list)
