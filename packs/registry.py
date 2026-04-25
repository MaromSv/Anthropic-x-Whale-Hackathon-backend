"""Single source of truth for all available data packs.

Adding a new pack = add one entry here and a builder function. The API
routes in main.py iterate this registry — no other file needs to change.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from models.schemas import POI, PackManifest, RAGDocument
from scrapers.amsterdam_basic import build_amsterdam_basic_data
from scrapers.core_medical import build_core_medical_data
from scrapers.kings_day_nl import build_kings_day_data, build_kings_day_pois


@dataclass(frozen=True)
class PackSpec:
    manifest: PackManifest
    builder: Callable[[], List[RAGDocument]]
    poi_builder: Optional[Callable[[], List[POI]]] = None


REGISTRY: Dict[str, PackSpec] = {
    "core_medical": PackSpec(
        manifest=PackManifest(
            pack_id="core_medical",
            name="Core Medical",
            description="Universal first-aid knowledge — installed by default.",
            version="1.0",
            location=None,
            is_core=True,
        ),
        builder=build_core_medical_data,
    ),
    "amsterdam_basic": PackSpec(
        manifest=PackManifest(
            pack_id="amsterdam_basic",
            name="Amsterdam Basics",
            description="Emergency contacts, hospitals, and city hazards for Amsterdam.",
            version="1.0",
            location="Amsterdam, NL",
            is_core=False,
        ),
        builder=build_amsterdam_basic_data,
    ),
    "kings_day_amsterdam": PackSpec(
        manifest=PackManifest(
            pack_id="kings_day_amsterdam",
            name="King's Day Amsterdam",
            description="Event-specific protocols + EHBO post locations from the Gemeente Amsterdam map.",
            version="1.0",
            location="Amsterdam, NL",
            is_core=False,
        ),
        builder=build_kings_day_data,
        poi_builder=build_kings_day_pois,
    ),
}


def list_manifests() -> List[PackManifest]:
    return [spec.manifest for spec in REGISTRY.values()]


def get_pack_spec(pack_id: str) -> PackSpec | None:
    return REGISTRY.get(pack_id)
