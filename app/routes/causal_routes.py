"""
FastAPI router for Track 2 — Causal Engine endpoints.
"""
import json

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.pipeline.causal import get_causal_engine

router = APIRouter(prefix="/api/causal", tags=["Causal (Track 2)"])


@router.post("/analyze/{event_id}")
async def analyze_event(event_id: str):
    """Run target-centric PCMCI+ causal discovery on an event's kinematics."""
    try:
        return get_causal_engine().analyze_event(event_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{event_id}")
async def get_causal_graph(event_id: str):
    """Return the persisted causal graph for an event (run analyze first)."""
    path = settings.paths.dataset_dir / event_id / "causal_graph.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No causal graph for {event_id}; POST analyze first")
    return json.loads(path.read_text(encoding="utf-8"))
