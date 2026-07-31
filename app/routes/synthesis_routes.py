"""
FastAPI router for Track 4 — Situation Report synthesis.
"""
import json

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.pipeline.synthesis import get_synthesis_engine

router = APIRouter(prefix="/api/synthesis", tags=["Synthesis (Track 4)"])


@router.post("/{event_id}")
async def generate(event_id: str):
    """Build the evidence packet and (if an API key is set) generate the SitRep."""
    try:
        return get_synthesis_engine().generate_sitrep(event_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{event_id}")
async def get_sitrep(event_id: str):
    """Return the persisted SitRep for an event (run POST first)."""
    path = settings.paths.dataset_dir / event_id / "sitrep.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No SitRep for {event_id}; POST to generate")
    return json.loads(path.read_text(encoding="utf-8"))
