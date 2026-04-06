"""
FastAPI router for Privacy & Audit endpoints.

Provides:
- GET /api/privacy/status — Current privacy config and statistics
- GET /api/privacy/audit — Paginated audit log entries
- GET /api/privacy/audit/export — Export full audit log as CSV
"""

import csv
import io
import logging
import time

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models import PrivacyStatus, AuditEntry, AuditLogResponse
from app.pipeline.privacy import get_privacy_stats
from app.pipeline.encryption import is_encryption_available
from app.audit import query_audit_log, count_audit_entries

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/privacy", tags=["Privacy & Audit"])


@router.get("/status", response_model=PrivacyStatus)
async def privacy_status():
    """
    Return current privacy configuration and runtime statistics.
    Shows face blurring, plate redaction, edge processing, and encryption status.
    """
    stats = get_privacy_stats()
    enc_available = is_encryption_available()

    return PrivacyStatus(
        face_blur_enabled=settings.privacy.face_blur_enabled,
        plate_redaction_enabled=settings.privacy.plate_redaction_enabled,
        edge_processing=True,  # Always local — no cloud calls
        encryption_enabled=settings.encryption.enabled,
        encryption_key_configured=enc_available,
        faces_blurred=stats.faces_blurred,
        plates_redacted=stats.plates_redacted,
        frames_processed=stats.frames_processed,
        crops_processed=stats.crops_processed,
    )


@router.get("/audit", response_model=AuditLogResponse)
async def get_audit_log(
    limit: int = Query(50, ge=1, le=500, description="Number of entries to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    action: str | None = Query(None, description="Filter by action type"),
    actor: str | None = Query(None, description="Filter by actor"),
):
    """
    Return paginated audit log entries from SQLite.
    Supports filtering by action type and actor.
    """
    entries = query_audit_log(
        limit=limit,
        offset=offset,
        action_filter=action,
        actor_filter=actor,
    )
    total = count_audit_entries(action_filter=action)

    return AuditLogResponse(
        entries=[AuditEntry(**e) for e in entries],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/audit/export")
async def export_audit_log():
    """
    Export the full audit log as a downloadable CSV file.
    """
    # Fetch all entries (large limit)
    entries = query_audit_log(limit=100_000, offset=0)

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["ID", "Timestamp", "Action", "Actor", "Resource", "Details", "Checksum"],
    )
    writer.writeheader()
    for entry in entries:
        writer.writerow(entry)

    output.seek(0)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=audit_log_{timestamp}.csv"
        },
    )
