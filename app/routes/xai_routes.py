"""
FastAPI routes for serving XAI (GradCAM + SHAP) artifacts.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter(prefix="/api/xai", tags=["XAI"])


@router.get("/{event_id}/gradcam")
async def list_gradcam_frames(event_id: str):
    """Return a list of available GradCAM heatmap filenames for an event."""
    gradcam_dir = settings.paths.dataset_dir / event_id / "xai" / "gradcam"
    if not gradcam_dir.exists():
        raise HTTPException(status_code=404, detail="No GradCAM data for this event")
    frames = sorted(f.name for f in gradcam_dir.glob("*.jpg"))
    return {"event_id": event_id, "frames": frames}


@router.get("/{event_id}/gradcam/{filename}")
async def get_gradcam_frame(event_id: str, filename: str):
    """Serve a single GradCAM overlay image."""
    path = settings.paths.dataset_dir / event_id / "xai" / "gradcam" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="GradCAM frame not found")
    return FileResponse(str(path), media_type="image/jpeg")


@router.get("/{event_id}/shap/plot")
async def get_shap_plot(event_id: str):
    """Serve the SHAP summary bar-chart image."""
    path = settings.paths.dataset_dir / event_id / "xai" / "shap_summary.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No SHAP plot for this event")
    return FileResponse(str(path), media_type="image/png")


@router.get("/{event_id}/shap/values")
async def get_shap_values(event_id: str):
    """Return SHAP feature-importance values as JSON."""
    path = settings.paths.dataset_dir / event_id / "xai" / "shap_importance.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No SHAP data for this event")
    return json.loads(path.read_text(encoding="utf-8"))
