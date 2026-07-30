"""
Phase 0 — Heuristic Ingestion.

Scans a frame source (a video file *or* a live stream) for anomaly events
using MOG2 background subtraction. On trigger, captures a 10-second clip
(4s pre-buffer + 6s post-trigger).

The core trigger logic lives in :class:`FrameEventDetector`, a push-driven
state machine fed one frame at a time. Both the batch scanner
(:func:`scan_for_events`) and the live feed worker drive the same detector,
so file and stream ingestion share identical event semantics.

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


class FrameEventDetector:
    """
    Stateful, push-driven anomaly-event detector.

    Feed frames one at a time via :meth:`process_frame`; it returns an
    :class:`EventFrameBlock` on the frame that completes an event, else None.
    It is agnostic to the frame source, so it works identically for a finite
    video file and an unbounded live stream.

    Lifecycle per frame:
        WARMUP    — accumulate foreground ratios for ``warmup_seconds`` to
                    compute the adaptive threshold.
        SCANNING  — maintain the pre-buffer; on threshold breach, freeze the
                    pre-buffer and begin capturing post-trigger frames.
        CAPTURING — collect ``post_frame_count`` frames, then emit the event.
        COOLDOWN  — ignore triggers for ``cooldown_seconds`` after an event.
    """

    def __init__(self, source_fps: float):
        cfg_v = settings.video
        cfg_t = settings.threshold

        self.source_fps = source_fps if source_fps and source_fps > 0 else 30.0
        self.pre_buffer_size = int(cfg_v.pre_buffer_seconds * self.source_fps)
        self.post_frame_count = int(cfg_v.post_trigger_seconds * self.source_fps)
        self.warmup_frame_count = int(cfg_t.warmup_seconds * self.source_fps)
        self.cooldown_frame_count = int(cfg_t.cooldown_seconds * self.source_fps)
        self._cfg_t = cfg_t

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.buffer: deque[bytes] = deque(maxlen=self.pre_buffer_size)
        self.warmup_ratios: list[float] = []
        self.threshold: float | None = None
        self.frame_idx = 0
        self.cooldown_remaining = 0

        # Post-trigger capture state
        self._capturing = False
        self._pre_frames: list[bytes] | None = None
        self._post_frames: list[bytes] | None = None
        self._trigger_time = 0.0

    @property
    def warmed_up(self) -> bool:
        return self.threshold is not None

    def process_frame(self, frame: np.ndarray) -> EventFrameBlock | None:
        """Advance the state machine by one frame; return a completed event or None."""
        cfg_t = self._cfg_t
        fg_mask = self.bg_sub.apply(frame)
        ratio = _foreground_ratio(fg_mask)
        encoded = _encode_frame(frame)

        # ── Capturing post-trigger frames ────────────────────────────────
        if self._capturing:
            self._post_frames.append(encoded)
            self.frame_idx += 1
            if len(self._post_frames) >= self.post_frame_count:
                return self._emit_event()
            return None

        # ── Warmup phase ─────────────────────────────────────────────────
        if self.frame_idx < self.warmup_frame_count:
            self.warmup_ratios.append(ratio)
            self.buffer.append(encoded)
            self.frame_idx += 1
            if self.frame_idx == self.warmup_frame_count:
                raw_threshold = (
                    np.percentile(self.warmup_ratios, cfg_t.percentile) * cfg_t.multiplier
                )
                self.threshold = max(raw_threshold, cfg_t.absolute_floor)
                logger.info(
                    "Warmup complete (frame %d). Threshold=%.4f (raw=%.4f, floor=%.4f)",
                    self.frame_idx, self.threshold, raw_threshold, cfg_t.absolute_floor,
                )
            return None

        # ── Cooldown ─────────────────────────────────────────────────────
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.buffer.append(encoded)
            self.frame_idx += 1
            return None

        # ── Scan phase ───────────────────────────────────────────────────
        self.buffer.append(encoded)
        if self.threshold is not None and ratio > self.threshold:
            self._trigger_time = self.frame_idx / self.source_fps
            logger.info(
                "EVENT TRIGGERED at frame %d (t=%.2fs) | ratio=%.4f > threshold=%.4f",
                self.frame_idx, self._trigger_time, ratio, self.threshold,
            )
            self._pre_frames = list(self.buffer)  # freeze pre-buffer (incl. trigger frame)
            self._post_frames = []
            self._capturing = True

        self.frame_idx += 1
        return None

    def flush(self) -> EventFrameBlock | None:
        """
        Emit a partial event if a capture was in progress when the source ended.
        Used by the batch scanner at EOF; live workers never call this.
        """
        if self._capturing and self._post_frames:
            return self._emit_event()
        return None

    def _emit_event(self) -> EventFrameBlock:
        event = EventFrameBlock(
            trigger_time_sec=self._trigger_time,
            source_fps=self.source_fps,
            pre_frames=self._pre_frames or [],
            post_frames=self._post_frames or [],
        )
        logger.info(
            "Captured event: %d pre + %d post frames (%.1fs)",
            len(event.pre_frames), len(event.post_frames), event.duration_sec,
        )
        self._capturing = False
        self._pre_frames = None
        self._post_frames = None
        self.buffer.clear()
        self.cooldown_remaining = self.cooldown_frame_count
        return event


def scan_for_events(video_path: str) -> list[EventFrameBlock]:
    """
    Scan a finite video file and return all triggered EventFrameBlocks.

    Drives a :class:`FrameEventDetector` over every frame, then flushes any
    partial event captured at end-of-file. Behavior is unchanged from the
    original monolithic scanner.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        raise IOError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = FrameEventDetector(source_fps)

    logger.info(
        "Scanning %s | src_fps=%.1f | buffer=%d frames | post=%d frames",
        video_path, source_fps, detector.pre_buffer_size, detector.post_frame_count,
    )

    events: list[EventFrameBlock] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        event = detector.process_frame(frame)
        if event is not None:
            events.append(event)

    tail = detector.flush()
    if tail is not None:
        events.append(tail)

    cap.release()

    if not events:
        logger.warning("No events triggered in %s", video_path)

    return events
