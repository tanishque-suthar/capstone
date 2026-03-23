"""
Phase 0 — Heuristic Ingestion.

Scans a video file for anomaly events using MOG2 background subtraction.
On trigger, captures a 10-second clip (4s pre-buffer + 6s post-trigger).

Reference: context.md §4 Phase 0, §5.2, §6.1
"""

import logging
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EventFrameBlock:
    """Container for a triggered event's frames and metadata."""
    trigger_time_sec: float
    source_fps: float
    pre_frames: list[bytes] = field(default_factory=list)   # JPEG-encoded
    post_frames: list[bytes] = field(default_factory=list)  # JPEG-encoded

    @property
    def all_frames(self) -> list[bytes]:
        return self.pre_frames + self.post_frames

    @property
    def duration_sec(self) -> float:
        total = len(self.all_frames)
        return total / self.source_fps if self.source_fps > 0 else 0.0


def _encode_frame(frame: np.ndarray) -> bytes:
    """JPEG-encode a frame to reduce memory footprint."""
    ok, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.video.jpeg_quality]
    )
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return buf.tobytes()


def _foreground_ratio(mask: np.ndarray) -> float:
    """Fraction of foreground pixels in a MOG2 mask."""
    return float(np.count_nonzero(mask)) / mask.size


def scan_for_events(video_path: str) -> list[EventFrameBlock]:
    """
    Scan a video file and return a list of triggered EventFrameBlocks.

    Steps:
        1. Open video, read source FPS.
        2. Maintain a JPEG-compressed rolling buffer (deque) of pre_buffer_seconds.
        3. Warmup: first 60s, accumulate foreground ratios.
        4. Compute adaptive threshold = percentile(95) * 1.5, floored at 0.05.
        5. Scan: on threshold breach, freeze buffer + capture post-trigger frames.
        6. Enforce cooldown between events.
    """
    cfg_v = settings.video
    cfg_t = settings.threshold

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        raise IOError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pre_buffer_size = int(cfg_v.pre_buffer_seconds * source_fps)
    post_frame_count = int(cfg_v.post_trigger_seconds * source_fps)
    warmup_frame_count = int(cfg_t.warmup_seconds * source_fps)
    cooldown_frame_count = int(cfg_t.cooldown_seconds * source_fps)

    logger.info(
        "Scanning %s | src_fps=%.1f | buffer=%d frames | post=%d frames",
        video_path, source_fps, pre_buffer_size, post_frame_count,
    )

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    buffer: deque[bytes] = deque(maxlen=pre_buffer_size)
    warmup_ratios: list[float] = []
    threshold: float | None = None
    events: list[EventFrameBlock] = []
    cooldown_remaining = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fg_mask = bg_sub.apply(frame)
        ratio = _foreground_ratio(fg_mask)
        encoded = _encode_frame(frame)

        # ── Warmup phase ─────────────────────────────────────────────────
        if frame_idx < warmup_frame_count:
            warmup_ratios.append(ratio)
            buffer.append(encoded)
            frame_idx += 1

            if frame_idx == warmup_frame_count:
                raw_threshold = np.percentile(warmup_ratios, cfg_t.percentile) * cfg_t.multiplier
                threshold = max(raw_threshold, cfg_t.absolute_floor)
                logger.info(
                    "Warmup complete (frame %d). Threshold=%.4f (raw=%.4f, floor=%.4f)",
                    frame_idx, threshold, raw_threshold, cfg_t.absolute_floor,
                )
            continue

        # ── Cooldown ─────────────────────────────────────────────────────
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            buffer.append(encoded)
            frame_idx += 1
            continue

        # ── Scan phase ───────────────────────────────────────────────────
        buffer.append(encoded)

        if threshold is not None and ratio > threshold:
            trigger_time = frame_idx / source_fps
            logger.info(
                "EVENT TRIGGERED at frame %d (t=%.2fs) | ratio=%.4f > threshold=%.4f",
                frame_idx, trigger_time, ratio, threshold,
            )

            # Freeze pre-buffer
            pre_frames = list(buffer)

            # Capture post-trigger frames
            post_frames: list[bytes] = []
            for _ in range(post_frame_count):
                ok, pf = cap.read()
                if not ok:
                    break
                post_frames.append(_encode_frame(pf))
                frame_idx += 1

            event = EventFrameBlock(
                trigger_time_sec=trigger_time,
                source_fps=source_fps,
                pre_frames=pre_frames,
                post_frames=post_frames,
            )
            events.append(event)
            logger.info(
                "Captured event: %d pre + %d post frames (%.1fs)",
                len(pre_frames), len(post_frames), event.duration_sec,
            )

            cooldown_remaining = cooldown_frame_count
            buffer.clear()

        frame_idx += 1

    cap.release()

    if not events:
        logger.warning("No events triggered in %s", video_path)

    return events
