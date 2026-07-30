"""
FastAPI router for camera-ingestion Step 2 — live feed monitoring.
"""
from fastapi import APIRouter, HTTPException

from app.database import get_video_source
from app.pipeline.monitor import get_feed_manager

router = APIRouter(prefix="/api/feeds", tags=["Feeds (Camera Ingestion)"])


@router.get("")
async def list_feeds():
    """Status of all feed monitors (frames read, vehicles indexed, events triggered)."""
    return {"feeds": get_feed_manager().status()}


@router.post("/{video_id}/start")
async def start_feed(video_id: str):
    """Start continuous monitoring of a registered video source (file path or stream URL)."""
    source = get_video_source(video_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {video_id}")
    started = get_feed_manager().start(video_id, source["File_Path"], source.get("Label", ""))
    if not started:
        raise HTTPException(status_code=409, detail=f"Feed {video_id} is already running")
    return {"status": "started", "video_id": video_id, "source": source["File_Path"]}


@router.post("/{video_id}/stop")
async def stop_feed(video_id: str):
    """Signal a feed monitor to stop (it finishes its current frame and shuts down)."""
    if not get_feed_manager().stop(video_id):
        raise HTTPException(status_code=404, detail=f"No monitor for {video_id}")
    return {"status": "stopping", "video_id": video_id}
