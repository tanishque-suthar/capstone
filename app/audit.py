"""
Audit Logging — Structured, tamper-evident access logging.

Every data access, creation, deletion, and privacy operation is logged to:
1. SQLite `Audit_Log` table (structured, queryable)
2. Append-only `logs/audit.log` file (tamper-evident backup)

Each entry contains: timestamp, action, actor, resource, details, checksum.
"""

import hashlib
import json
import logging
import sqlite3
import time
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

from app.config import settings

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Enumeration of auditable actions."""
    DATA_CREATED = "DATA_CREATED"
    DATA_ACCESSED = "DATA_ACCESSED"
    DATA_DELETED = "DATA_DELETED"
    PRIVACY_APPLIED = "PRIVACY_APPLIED"
    ENCRYPTION_APPLIED = "ENCRYPTION_APPLIED"
    DECRYPTION_PERFORMED = "DECRYPTION_PERFORMED"
    PIPELINE_STARTED = "PIPELINE_STARTED"
    PIPELINE_COMPLETED = "PIPELINE_COMPLETED"
    PIPELINE_FAILED = "PIPELINE_FAILED"


# ── Audit File Logger ─────────────────────────────────────────────────────────

_audit_file_handler: logging.FileHandler | None = None
_audit_logger: logging.Logger | None = None


def _get_audit_logger() -> logging.Logger:
    """Get or create the dedicated audit file logger."""
    global _audit_logger, _audit_file_handler

    if _audit_logger is not None:
        return _audit_logger

    _audit_logger = logging.getLogger("audit")
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False  # Don't bubble to root logger

    # Don't add duplicate handlers
    if not _audit_logger.handlers:
        log_path = settings.paths.base_dir / settings.audit.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _audit_file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        _audit_file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s")
        )
        _audit_logger.addHandler(_audit_file_handler)

    return _audit_logger


# ── Database helpers ──────────────────────────────────────────────────────────

@contextmanager
def _get_audit_connection():
    """Yield a SQLite connection for audit writes."""
    conn = sqlite3.connect(str(settings.paths.db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _compute_entry_checksum(
    timestamp: float, action: str, actor: str, resource: str, details: str
) -> str:
    """
    Compute SHA-256 checksum over the entry fields for tamper evidence.
    """
    payload = f"{timestamp}|{action}|{actor}|{resource}|{details}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def log_audit(
    action: AuditAction | str,
    resource: str = "",
    actor: str = "system",
    details: dict | None = None,
) -> None:
    """
    Log an audit entry to both SQLite and the audit log file.

    Args:
        action: The type of action being audited.
        resource: The resource being acted upon (e.g., file path, event ID).
        actor: Who performed the action (IP address, "system", etc.).
        details: Additional metadata as a dict (serialized to JSON).
    """
    if not settings.audit.enabled:
        return

    timestamp = time.time()
    action_str = action.value if isinstance(action, AuditAction) else str(action)
    details_str = json.dumps(details) if details else "{}"
    checksum = _compute_entry_checksum(timestamp, action_str, actor, resource, details_str)

    # 1. Write to SQLite
    try:
        with _get_audit_connection() as conn:
            conn.execute(
                """INSERT INTO Audit_Log (Timestamp, Action, Actor, Resource, Details, Checksum)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (timestamp, action_str, actor, resource, details_str, checksum),
            )
    except sqlite3.OperationalError as e:
        # Table might not exist yet (first call before init_db)
        logger.warning("Audit DB write failed (table may not exist yet): %s", e)

    # 2. Write to append-only log file
    try:
        audit_log = _get_audit_logger()
        log_line = f"{action_str} | actor={actor} | resource={resource} | sha256={checksum[:16]}… | {details_str}"
        audit_log.info(log_line)
    except Exception as e:
        logger.warning("Audit file write failed: %s", e)


def query_audit_log(
    limit: int = 50,
    offset: int = 0,
    action_filter: str | None = None,
    actor_filter: str | None = None,
) -> list[dict]:
    """
    Query audit log entries from SQLite.

    Returns list of dicts with keys: id, timestamp, action, actor, resource, details, checksum.
    """
    sql = "SELECT * FROM Audit_Log"
    params: list = []
    conditions = []

    if action_filter:
        conditions.append("Action = ?")
        params.append(action_filter)
    if actor_filter:
        conditions.append("Actor = ?")
        params.append(actor_filter)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY Timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        with _get_audit_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def count_audit_entries(action_filter: str | None = None) -> int:
    """Count total audit entries, optionally filtered by action."""
    sql = "SELECT COUNT(*) FROM Audit_Log"
    params: list = []

    if action_filter:
        sql += " WHERE Action = ?"
        params.append(action_filter)

    try:
        with _get_audit_connection() as conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0
