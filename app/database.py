"""
SQLite event registry — Master_Event_Log table.
Schema defined in context.md §3.1 Item 1.
"""

import json
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS Master_Event_Log (
    Event_ID       TEXT PRIMARY KEY,
    Trigger_Time   REAL NOT NULL,
    Raw_Video_Path TEXT NOT NULL,
    Causal_CSV_Path TEXT NOT NULL,
    Crops_Dir_Path TEXT NOT NULL,
    Duration_s     REAL,
    Status         TEXT NOT NULL DEFAULT 'Extracted'
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
    """Create the Master_Event_Log table if it doesn't exist."""
    with _get_connection() as conn:
        conn.execute(_CREATE_TABLE_SQL)
    logger.info("Database initialized at %s", settings.paths.db_path)


def insert_event(
    event_id: str,
    trigger_time: float,
    video_path: str,
    csv_path: str,
    crops_dir: str,
    duration_s: float | None = None,
    status: str = "Extracted",
) -> None:
    """
    Insert a completed event into the registry.
    On failure, retries once, then falls back to a JSON file.
    """
    sql = """
    INSERT INTO Master_Event_Log
        (Event_ID, Trigger_Time, Raw_Video_Path, Causal_CSV_Path, Crops_Dir_Path, Duration_s, Status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    params = (event_id, trigger_time, video_path, csv_path, crops_dir, duration_s, status)

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
