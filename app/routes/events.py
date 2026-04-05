"""
REST API endpoints for the Track 1 pipeline.

Provides routes to trigger pipeline runs, list events, fetch event details,
and download event artifacts (CSV, video).
"""

import uuid
import logging
import shutil
from pathlib import Path

import cv2
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.models import PipelineRequest, PipelineResponse, EventDetail, EventList, VideoSource, VideoSourceList
from app.database import (
    get_event, list_events, update_event_status, insert_event,
    insert_video_source, get_video_source, get_video_source_by_path,
    list_video_sources, list_events_for_source, update_event_reasoning, reset_registry,
    delete_event, delete_video_source_if_orphaned,
)
from app.pipeline.ingestion import scan_for_events
from app.pipeline.perception import process_event
from app.pipeline.handoff import finalize_event
from app.pipeline.reasoning import generate_reasoning_report, save_reasoning_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Track 1"])


def _assert_video_readable(video_path: str) -> None:
    """Fail fast if OpenCV cannot read the requested source video."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Video file exists but cannot be opened by the backend. "
                    "Use a video path inside the project workspace, such as the dataset folder, "
                    "or re-encode the file to a standard MP4/H.264 format."
                ),
            )
        ok, _ = cap.read()
        if not ok:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Video file was found, but the backend could not decode frames from it. "
                    "Try moving it into the project workspace or re-encoding it to a standard MP4/H.264 file."
                ),
            )
    finally:
        cap.release()


def _run_pipeline(video_path: str, event_id: str) -> None:
    """
    Execute the full 3-phase pipeline as a background task.
    Handles errors gracefully and updates event status on failure.
    """
    try:
        # ── Phase 0: Ingestion ───────────────────────────────────────
        logger.info("[%s] Phase 0: Scanning video %s", event_id, video_path)
        event_blocks = scan_for_events(video_path)

        if not event_blocks:
            logger.warning("[%s] No events detected — nothing to process", event_id)
            update_event_status(event_id, "Failed")
            return

        # Process only the first triggered event for prototype
        block = event_blocks[0]

        # ── Phase 1: Perception ──────────────────────────────────────
        logger.info("[%s] Phase 1: Running detection & tracking", event_id)
        perception_result = process_event(event_id, block)

        # ── Phase 2: Handoff ─────────────────────────────────────────
        logger.info("[%s] Phase 2: Finalizing event", event_id)
        output = finalize_event(event_id, perception_result, block.trigger_time_sec, source_video_path=video_path)

        logger.info("[%s] Phase 3: Generating reasoning report", event_id)
        report = generate_reasoning_report(
            event_id=event_id,
            csv_path=output["csv_path"],
            trigger_time=block.trigger_time_sec,
            video_path=output["video_path"],
            crops_dir=output["crops_dir"],
        )
        reasoning_path = save_reasoning_report(report)
        update_event_reasoning(
            event_id=event_id,
            reasoning_json_path=reasoning_path,
            reasoning_summary=report.summary,
            status="Reasoned",
        )

        logger.info("[%s] Pipeline complete. Output: %s", event_id, output)

    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s", event_id, exc, exc_info=True)
        try:
            update_event_status(event_id, "Failed")
        except Exception:
            pass  # Best-effort status update


@router.post("/pipeline/run", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Trigger the Track 1 pipeline on a video file.
    Processing runs in the background; returns immediately with an event_id.
    """
    video_path = request.video_path
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    _assert_video_readable(video_path)

    # Auto-register as a Video Source if not already known
    source = get_video_source_by_path(video_path)
    if not source:
        vid = f"VID_{uuid.uuid4().hex[:8].upper()}"
        label = Path(video_path).stem
        insert_video_source(vid, label, video_path)
        video_id = vid
    else:
        video_id = source["Video_ID"]

    event_id = f"EVT_{uuid.uuid4().hex[:12].upper()}"
    logger.info("Pipeline triggered: event_id=%s, video=%s, source=%s", event_id, video_path, video_id)

    # Synchronous insert so that frontend GET /events/{event_id} doesn't 404
    insert_event(
        event_id=event_id,
        trigger_time=0.0,
        video_path="",
        csv_path="",
        crops_dir="",
        duration_s=0.0,
        status="Processing",
        source_video_path=video_path,
    )
    # Link event to source
    from app.database import _get_connection
    with _get_connection() as conn:
        conn.execute("UPDATE Master_Event_Log SET Video_ID = ? WHERE Event_ID = ?", (video_id, event_id))

    background_tasks.add_task(_run_pipeline, video_path, event_id)

    return PipelineResponse(
        event_id=event_id,
        status="processing",
        message=f"Pipeline started for {Path(video_path).name}. Poll GET /api/events/{event_id} for status.",
    )


@router.get("/events", response_model=EventList)
async def get_events():
    """List all registered events."""
    events = list_events()
    return EventList(events=[EventDetail(**e) for e in events])


@router.get("/events/{event_id}", response_model=EventDetail)
async def get_event_detail(event_id: str):
    """Get details of a specific event."""
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    return EventDetail(**event)


@router.delete("/events/{event_id}")
async def remove_event(event_id: str):
    """Delete an event and its generated artifacts."""
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    paths_to_delete: list[Path] = []
    for key in ("Raw_Video_Path", "Causal_CSV_Path", "Crops_Dir_Path", "Reasoning_JSON_Path"):
        raw = event.get(key)
        if raw:
            paths_to_delete.append(Path(raw))

    event_dir = settings.paths.dataset_dir / event_id
    if event_dir.exists():
        shutil.rmtree(event_dir, ignore_errors=True)
    else:
        for path in paths_to_delete:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)

    video_id = event.get("Video_ID")
    delete_event(event_id)
    delete_video_source_if_orphaned(video_id)
    return {"status": "ok", "event_id": event_id}


@router.get("/events/{event_id}/csv")
async def download_csv(event_id: str):
    """Download the causal CSV for an event."""
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    csv_path = Path(event["Causal_CSV_Path"])
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found on disk")

    return FileResponse(
        path=str(csv_path),
        media_type="text/csv",
        filename=csv_path.name,
    )


@router.get("/events/{event_id}/video")
async def download_video(event_id: str):
    """Download the archival video for an event."""
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    video_path = Path(event["Raw_Video_Path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
    )


@router.get("/events/{event_id}/crops")
async def list_crops(event_id: str):
    """List all available crop images for an event."""
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    crops_dir = Path(event["Crops_Dir_Path"])
    if not crops_dir.exists():
        return {"crops": []}

    crops = [f.name for f in crops_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    return {"crops": crops}


@router.get("/events/{event_id}/crops/{filename}")
async def get_crop(event_id: str, filename: str):
    """Serve a specific crop image."""
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    crops_dir = Path(event["Crops_Dir_Path"])
    file_path = crops_dir / filename
    
    # Security check to prevent path traversal
    try:
        resolved_file = file_path.resolve(strict=False)
        resolved_dir = crops_dir.resolve(strict=True)
        if not str(resolved_file).startswith(str(resolved_dir)):
            raise HTTPException(status_code=403, detail="Invalid file path")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid file path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Crop not found")

    return FileResponse(path=str(file_path), media_type="image/jpeg")


@router.get("/events/{event_id}/source-video")
async def stream_source_video(event_id: str):
    """Serve the full source video file for an event."""
    event = get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    source_path = event.get("Source_Video_Path")
    if not source_path or not Path(source_path).exists():
        raise HTTPException(status_code=404, detail="Source video not available")

    return FileResponse(
        path=source_path,
        media_type="video/mp4",
        filename=Path(source_path).name,
    )


# ── Video Sources (camera feeds) ─────────────────────────────────────────────

@router.get("/sources", response_model=VideoSourceList)
async def get_sources():
    """List all registered video sources / camera feeds."""
    sources = list_video_sources()
    return VideoSourceList(sources=[VideoSource(**s) for s in sources])


@router.get("/sources/{video_id}")
async def get_source_detail(video_id: str):
    """Get a specific video source."""
    source = get_video_source(video_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {video_id}")
    return VideoSource(**source)


@router.get("/sources/{video_id}/stream")
async def stream_source(video_id: str):
    """Stream the video file for a source/feed."""
    source = get_video_source(video_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {video_id}")
    file_path = source["File_Path"]
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    return FileResponse(path=file_path, media_type="video/mp4", filename=Path(file_path).name)


@router.get("/sources/{video_id}/events", response_model=EventList)
async def get_source_events(video_id: str):
    """List all events extracted from a specific video source."""
    source = get_video_source(video_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source not found: {video_id}")
    events = list_events_for_source(video_id)
    return EventList(events=[EventDetail(**e) for e in events])


@router.post("/reset")
async def reset_pipeline_state():
    """Delete all generated events, analyses, sources, vector data, and logs."""
    reset_registry()

    dataset_dir = settings.paths.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for child in dataset_dir.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)

    log_file = settings.paths.log_dir / "track1.log"
    if log_file.exists():
        log_file.unlink(missing_ok=True)

    settings.paths.log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reset pipeline state: cleared dataset, registry, and logs")
    return {"status": "ok", "message": "All generated pipeline state has been deleted."}
