from typing import List

from fastapi import FastAPI, HTTPException

from models.schemas import DataPack, PackManifest
from packs.registry import get_pack_spec, list_manifests

app = FastAPI(title="Crisis App Backend")


@app.get("/")
def health_check():
    return {"status": "online", "message": "Backend is ready to serve data packs."}


@app.get("/packs", response_model=List[PackManifest])
def list_packs():
    """List all available packs. The Android app calls this to populate
    the download screen. Core packs (is_core=True) should auto-install."""
    return list_manifests()


@app.get("/packs/{pack_id}", response_model=DataPack)
def get_data_pack(pack_id: str):
    """Return the full document set for a pack. Called when the user taps
    'Download' (or on first launch for core packs)."""
    spec = get_pack_spec(pack_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Pack '{pack_id}' not found")

    return DataPack(
        pack_id=spec.manifest.pack_id,
        version=spec.manifest.version,
        location=spec.manifest.location,
        documents=spec.builder(),
        points_of_interest=spec.poi_builder() if spec.poi_builder else [],
    )
