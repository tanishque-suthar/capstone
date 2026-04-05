"""
SQLite event registry — Master_Event_Log + Video_Sources tables.
"""

import json
import sqlite3
import logging
import time
import uuid
from pathlib import Path
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)

_CREATE_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS Master_Event_Log (
    Event_ID          TEXT PRIMARY KEY,
    Trigger_Time      REAL NOT NULL,
    Raw_Video_Path    TEXT NOT NULL,
    Causal_CSV_Path   TEXT NOT NULL,
    Crops_Dir_Path    TEXT NOT NULL,
    Duration_s        REAL,
    Status            TEXT NOT NULL DEFAULT 'Extracted',
    Source_Video_Path TEXT,
    Video_ID          TEXT
);
"""

_CREATE_SOURCES_SQL = """
CREATE TABLE IF NOT EXISTS Video_Sources (
    Video_ID   TEXT PRIMARY KEY,
    Label      TEXT NOT NULL,
    File_Path  TEXT NOT NULL UNIQUE,
    Added_At   REAL NOT NULL
);
"""

_CREATE_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS Audit_Log (
    ID          INTEGER PRIMARY KEY AUTOINCREMENT,
    Timestamp   REAL NOT NULL,
    Action      TEXT NOT NULL,
    Actor       TEXT NOT NULL DEFAULT 'system',
    Resource    TEXT,
    Details     TEXT,
    Checksum    TEXT
);
"""


@contextmanager
def _get_connection():
    """Yield a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(settings.paths.db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and run migrations."""
    with _get_connection() as conn:
        conn.execute(_CREATE_EVENTS_SQL)
        conn.execute(_CREATE_SOURCES_SQL)
        conn.execute(_CREATE_AUDIT_SQL)

        # Migration: add Source_Video_Path if missing (existing DBs)
        try:
            conn.execute("ALTER TABLE Master_Event_Log ADD COLUMN Source_Video_Path TEXT")
            logger.info("Migrated: added Source_Video_Path column")
        except Exception:
            pass

        # Migration: add Video_ID FK if missing
        try:
            conn.execute("ALTER TABLE Master_Event_Log ADD COLUMN Video_ID TEXT")
            logger.info("Migrated: added Video_ID column")
        except Exception:
            pass

        # Migration: add Privacy_Applied flag
        try:
            conn.execute("ALTER TABLE Master_Event_Log ADD COLUMN Privacy_Applied INTEGER DEFAULT 0")
            logger.info("Migrated: added Privacy_Applied column")
        except Exception:
            pass

        # Migration: add Encrypted flag
        try:
            conn.execute("ALTER TABLE Master_Event_Log ADD COLUMN Encrypted INTEGER DEFAULT 0")
            logger.info("Migrated: added Encrypted column")
        except Exception:
            pass

        # Migration: seed Video_Sources from existing events
        rows = conn.execute(
            "SELECT DISTINCT Source_Video_Path FROM Master_Event_Log WHERE Source_Video_Path IS NOT NULL AND Source_Video_Path != ''"
        ).fetchall()
        for row in rows:
            src = row[0]
            existing = conn.execute("SELECT Video_ID FROM Video_Sources WHERE File_Path = ?", (src,)).fetchone()
            if not existing:
                vid = f"VID_{uuid.uuid4().hex[:8].upper()}"
                label = Path(src).stem
                conn.execute(
                    "INSERT INTO Video_Sources (Video_ID, Label, File_Path, Added_At) VALUES (?, ?, ?, ?)",
                    (vid, label, src, time.time()),
                )
                # Back-fill Video_ID on existing events
                conn.execute(
                    "UPDATE Master_Event_Log SET Video_ID = ? WHERE Source_Video_Path = ?",
                    (vid, src),
                )
                logger.info("Migrated source video '%s' → %s", label, vid)

    logger.info("Database initialized at %s", settings.paths.db_path)


def insert_event(
    event_id: str,
    trigger_time: float,
    video_path: str,
    csv_path: str,
    crops_dir: str,
    duration_s: float | None = None,
    status: str = "Extracted",
    source_video_path: str | None = None,
) -> None:
    """
    Insert a completed event into the registry.
    On failure, retries once, then falls back to a JSON file.
    """
    sql = """
    INSERT INTO Master_Event_Log
        (Event_ID, Trigger_Time, Raw_Video_Path, Causal_CSV_Path, Crops_Dir_Path, Duration_s, Status, Source_Video_Path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (event_id, trigger_time, video_path, csv_path, crops_dir, duration_s, status, source_video_path)

    for attempt in range(2):
        try:
            with _get_connection() as conn:
                conn.execute(sql, params)
            logger.info("Registered event %s with status=%s", event_id, status)
            return
        except sqlite3.Error as exc:
            logger.warning("DB insert attempt %d failed for %s: %s", attempt + 1, event_id, exc)

    # Fallback: write to JSON
    fallback_path = Path(crops_dir).parent / "event_meta.json"
    meta = {
        "Event_ID": event_id,
        "Trigger_Time": trigger_time,
        "Raw_Video_Path": video_path,
        "Causal_CSV_Path": csv_path,
        "Crops_Dir_Path": crops_dir,
        "Duration_s": duration_s,
        "Status": status,
        "Source_Video_Path": source_video_path,
    }
    fallback_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.error("DB insert failed after retries. Wrote fallback JSON to %s", fallback_path)


def update_event_status(event_id: str, status: str) -> None:
    """Update the Status column for an existing event."""
    with _get_connection() as conn:
        conn.execute(
            "UPDATE Master_Event_Log SET Status = ? WHERE Event_ID = ?",
            (status, event_id),
        )
    logger.info("Updated event %s status to %s", event_id, status)


def update_event_details(
    event_id: str,
    trigger_time: float,
    video_path: str,
    csv_path: str,
    crops_dir: str,
    duration_s: float | None = None,
    status: str = "Extracted",
    source_video_path: str | None = None,
) -> None:
    """Update event path details after processing."""
    sql = """
    UPDATE Master_Event_Log
    SET Trigger_Time = ?,
        Raw_Video_Path = ?,
        Causal_CSV_Path = ?,
        Crops_Dir_Path = ?,
        Duration_s = ?,
        Status = ?,
        Source_Video_Path = ?
    WHERE Event_ID = ?
    """
    params = (trigger_time, video_path, csv_path, crops_dir, duration_s, status, source_video_path, event_id)
    with _get_connection() as conn:
        conn.execute(sql, params)
    logger.info("Updated details for event %s, status=%s", event_id, status)


def get_event(event_id: str) -> dict | None:
    """Return a single event as a dict, or None if not found."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM Master_Event_Log WHERE Event_ID = ?", (event_id,)
        ).fetchone()
    return dict(row) if row else None


def list_events() -> list[dict]:
    """Return all events as a list of dicts."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM Master_Event_Log ORDER BY Trigger_Time DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def list_events_for_source(video_id: str) -> list[dict]:
    """Return events linked to a specific video source."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM Master_Event_Log WHERE Video_ID = ? ORDER BY Trigger_Time ASC",
            (video_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Video_Sources CRUD ────────────────────────────────────────────────────────

def insert_video_source(video_id: str, label: str, file_path: str) -> None:
    """Register a new video source / camera feed."""
    with _get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO Video_Sources (Video_ID, Label, File_Path, Added_At) VALUES (?, ?, ?, ?)",
            (video_id, label, file_path, time.time()),
        )
    logger.info("Registered video source %s (%s)", video_id, label)


def get_video_source(video_id: str) -> dict | None:
    with _get_connection() as conn:
        row = conn.execute("SELECT * FROM Video_Sources WHERE Video_ID = ?", (video_id,)).fetchone()
    return dict(row) if row else None


def get_video_source_by_path(file_path: str) -> dict | None:
    with _get_connection() as conn:
        row = conn.execute("SELECT * FROM Video_Sources WHERE File_Path = ?", (file_path,)).fetchone()
    return dict(row) if row else None


def list_video_sources() -> list[dict]:
    with _get_connection() as conn:
        rows = conn.execute("SELECT * FROM Video_Sources ORDER BY Added_At DESC").fetchall()
    return [dict(r) for r in rows]
