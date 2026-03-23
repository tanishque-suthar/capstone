"""
Pydantic request/response schemas for the Track 1 API.
"""

from pydantic import BaseModel, Field


class PipelineRequest(BaseModel):
    """Request body for POST /api/pipeline/run."""
    video_path: str = Field(
        ...,
        description="Absolute path to the input .mp4 video file.",
        examples=["D:/videos/intersection_01.mp4"],
    )


class PipelineResponse(BaseModel):
    """Immediate response after triggering the pipeline."""
    event_id: str | None = Field(
        None,
        description="Unique event identifier. Null if no event was detected.",
    )
    status: str = Field(
        ...,
        description="Current pipeline status: 'processing', 'no_event', or 'error'.",
    )
    message: str = Field(
        ...,
        description="Human-readable status message.",
    )


class EventDetail(BaseModel):
    """Full detail of a registered event (mirrors Master_Event_Log)."""
    Event_ID: str
    Trigger_Time: float
    Raw_Video_Path: str
    Causal_CSV_Path: str
    Crops_Dir_Path: str
    Duration_s: float | None = None
    Status: str
    Source_Video_Path: str | None = None


class EventList(BaseModel):
    """Wrapper for listing events."""
    events: list[EventDetail]
