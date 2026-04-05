"""
Pydantic request/response schemas for the Track 1 API.
"""

from pydantic import BaseModel, Field


class PipelineRequest(BaseModel):
    """Request body for POST /api/pipeline/run."""
    video_path: str = Field(
        ...,
        description="Absolute path to the input .mp4 video file.",
        examples=["path/to/videos/intersection_01.mp4"],
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
    Video_ID: str | None = None
    Privacy_Applied: int | None = 0
    Encrypted: int | None = 0


class EventList(BaseModel):
    """Wrapper for listing events."""
    events: list[EventDetail]


class VideoSource(BaseModel):
    """A registered video source / camera feed."""
    Video_ID: str
    Label: str
    File_Path: str
    Added_At: float


class VideoSourceList(BaseModel):
    """Wrapper for listing video sources."""
    sources: list[VideoSource]


# ── Privacy & Audit Models ────────────────────────────────────────────────────

class PrivacyStatus(BaseModel):
    """Current privacy configuration and runtime statistics."""
    face_blur_enabled: bool
    plate_redaction_enabled: bool
    edge_processing: bool = True  # Always true — all processing is local
    encryption_enabled: bool
    encryption_key_configured: bool
    faces_blurred: int = 0
    plates_redacted: int = 0
    frames_processed: int = 0
    crops_processed: int = 0


class AuditEntry(BaseModel):
    """Single audit log entry."""
    ID: int
    Timestamp: float
    Action: str
    Actor: str
    Resource: str | None = None
    Details: str | None = None
    Checksum: str | None = None


class AuditLogResponse(BaseModel):
    """Paginated audit log response."""
    entries: list[AuditEntry]
    total: int
    limit: int
    offset: int

