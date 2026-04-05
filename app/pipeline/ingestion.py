"""
Phase 0 — Heuristic Ingestion.

Scans a video file for anomaly events using a rolling baseline over
foreground activity, motion change, scene entropy, and blob size.
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
    anomaly_score: float = 0.0
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


def _scene_entropy(gray: np.ndarray) -> float:
    """Normalized grayscale entropy in [0, 1]."""
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / np.log2(32))


def _largest_blob_ratio(mask: np.ndarray) -> float:
    """Largest connected foreground region divided by full frame area."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest / mask.size


def _extract_features(
    frame: np.ndarray,
    fg_mask: np.ndarray,
    previous_gray: np.ndarray | None,
) -> tuple[dict[str, float], np.ndarray]:
    """Build a compact anomaly feature vector for one frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_ratio = _foreground_ratio(fg_mask)

    if previous_gray is None:
        motion_score = 0.0
    else:
        frame_delta = cv2.absdiff(gray, previous_gray)
        motion_score = float(frame_delta.mean() / 255.0)

    return {
        "foreground_ratio": fg_ratio,
        "motion_score": motion_score,
        "entropy_score": _scene_entropy(gray),
        "blob_score": _largest_blob_ratio(fg_mask),
    }, gray


def _robust_zscore(value: float, history: deque[float]) -> float:
    """Median/MAD z-score that stays stable when the baseline is noisy."""
    if len(history) < 10:
        return 0.0

    sample = np.asarray(history, dtype=float)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    scale = max(1.4826 * mad, 1e-3)
    return abs(value - median) / scale


def _compute_anomaly_score(
    features: dict[str, float],
    baseline: dict[str, deque[float]],
) -> tuple[float, dict[str, float]]:
    """Return the strongest anomaly evidence across all tracked features."""
    zscores = {
        name: _robust_zscore(value, baseline[name])
        for name, value in features.items()
    }
    weighted_score = max(
        zscores["foreground_ratio"],
        zscores["motion_score"],
        zscores["blob_score"],
        zscores["entropy_score"] * 0.75,
    )
    return weighted_score, zscores


def _update_baseline(
    baseline: dict[str, deque[float]],
    features: dict[str, float],
) -> None:
    """Append one feature vector into the rolling baseline."""
    for name, value in features.items():
        baseline[name].append(value)


def _build_fallback_event(
    source_fps: float,
    trigger_time: float,
    score: float,
    pre_frames: list[bytes],
    post_frames: list[bytes],
) -> EventFrameBlock:
    """Create an event block from the best non-triggered window."""
    return EventFrameBlock(
        trigger_time_sec=max(0.0, trigger_time),
        source_fps=source_fps,
        anomaly_score=score,
        pre_frames=pre_frames,
        post_frames=post_frames,
    )


def scan_for_events(video_path: str) -> list[EventFrameBlock]:
    """
    Scan a video file and return a list of triggered EventFrameBlocks.

    Detection strategy:
        1. Build a rolling visual baseline during warmup.
        2. Compare each new frame against that baseline using robust z-scores.
        3. Trigger only when anomaly evidence persists across a short streak.
        4. Keep the old foreground threshold as a safety backstop.
        5. Freeze pre-buffer + capture post-trigger frames, then enforce cooldown.
    """
    cfg_v = settings.video
    cfg_t = settings.threshold

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        raise IOError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration_s = (total_frames / source_fps) if source_fps > 0 and total_frames > 0 else 0.0
    pre_buffer_size = int(cfg_v.pre_buffer_seconds * source_fps)
    post_frame_count = int(cfg_v.post_trigger_seconds * source_fps)
    cooldown_frame_count = int(cfg_t.cooldown_seconds * source_fps)
    rolling_window_size = max(30, int(cfg_t.rolling_baseline_seconds * source_fps))
    adaptive_warmup_s = min(max(3.0, video_duration_s * 0.25), min(10.0, cfg_t.warmup_seconds))
    warmup_frame_count = int(adaptive_warmup_s * source_fps)

    if total_frames > 0:
        warmup_frame_count = min(warmup_frame_count, max(10, total_frames - 1))

    logger.info(
        "Scanning %s | src_fps=%.1f | duration=%.1fs | buffer=%d frames | post=%d frames | baseline=%d frames | warmup=%.1fs",
        video_path,
        source_fps,
        video_duration_s,
        pre_buffer_size,
        post_frame_count,
        rolling_window_size,
        warmup_frame_count / source_fps if source_fps > 0 else 0.0,
    )

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    buffer: deque[bytes] = deque(maxlen=pre_buffer_size)
    baseline = {
        "foreground_ratio": deque(maxlen=rolling_window_size),
        "motion_score": deque(maxlen=rolling_window_size),
        "entropy_score": deque(maxlen=rolling_window_size),
        "blob_score": deque(maxlen=rolling_window_size),
    }
    warmup_ratios: list[float] = []
    threshold: float | None = None
    events: list[EventFrameBlock] = []
    cooldown_remaining = 0
    trigger_streak = 0
    previous_gray: np.ndarray | None = None
    frame_idx = 0
    best_fallback_score = float("-inf")
    best_fallback_trigger_time = 0.0
    best_fallback_pre_frames: list[bytes] = []
    best_fallback_post_frames: list[bytes] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        fg_mask = bg_sub.apply(frame)
        features, previous_gray = _extract_features(frame, fg_mask, previous_gray)
        ratio = features["foreground_ratio"]
        encoded = _encode_frame(frame)

        # ── Warmup phase ─────────────────────────────────────────────────
        if frame_idx < warmup_frame_count:
            warmup_ratios.append(ratio)
            _update_baseline(baseline, features)
            buffer.append(encoded)
            frame_idx += 1

            if frame_idx == warmup_frame_count:
                raw_threshold = np.percentile(warmup_ratios, cfg_t.percentile) * cfg_t.multiplier
                threshold = max(raw_threshold, cfg_t.absolute_floor)
                logger.info(
                    "Warmup complete (frame %d). Foreground threshold=%.4f (raw=%.4f, floor=%.4f)",
                    frame_idx,
                    threshold,
                    raw_threshold,
                    cfg_t.absolute_floor,
                )
            continue

        # ── Cooldown ─────────────────────────────────────────────────────
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            _update_baseline(baseline, features)
            buffer.append(encoded)
            frame_idx += 1
            continue

        # ── Scan phase ───────────────────────────────────────────────────
        buffer.append(encoded)
        anomaly_score, zscores = _compute_anomaly_score(features, baseline)
        motion_gate = (
            features["motion_score"] >= cfg_t.min_motion_score
            or ratio >= cfg_t.absolute_floor
            or features["blob_score"] >= cfg_t.absolute_floor
        )
        foreground_backstop = threshold is not None and ratio > threshold
        anomaly_trigger = anomaly_score >= cfg_t.anomaly_zscore_threshold and motion_gate
        fallback_score = anomaly_score
        if foreground_backstop:
            fallback_score = max(fallback_score, ratio / max(threshold or 1.0, 1e-6))

        if fallback_score > best_fallback_score and buffer:
            best_fallback_score = fallback_score
            best_fallback_trigger_time = frame_idx / source_fps if source_fps > 0 else 0.0
            best_fallback_pre_frames = list(buffer)
            best_fallback_post_frames = []

        if anomaly_trigger or foreground_backstop:
            trigger_streak += 1
        else:
            trigger_streak = 0
            _update_baseline(baseline, features)

        if trigger_streak >= cfg_t.min_trigger_streak:
            trigger_time = frame_idx / source_fps
            effective_score = anomaly_score if anomaly_trigger else ratio / max(threshold or 1.0, 1e-6)
            logger.info(
                "ANOMALY TRIGGERED at frame %d (t=%.2fs) | score=%.2f | fg=%.4f | motion=%.4f | entropy=%.4f | blob=%.4f | z=%s",
                frame_idx,
                trigger_time,
                effective_score,
                ratio,
                features["motion_score"],
                features["entropy_score"],
                features["blob_score"],
                {k: round(v, 2) for k, v in zscores.items()},
            )

            pre_frames = list(buffer)
            post_frames: list[bytes] = []
            frame_idx += 1

            # ── Dynamic Capture Logic ────────────────────────────────────
            # Record at least post_frame_count, then continue until score < maintenance_zscore
            # or we hit max_post_trigger_seconds cap.
            max_p_frames = int(cfg_v.max_post_trigger_seconds * source_fps)
            min_p_frames = post_frame_count
            p_idx = 0

            while p_idx < max_p_frames:
                ok, pf = cap.read()
                if not ok:
                    break

                post_frames.append(_encode_frame(pf))
                frame_idx += 1
                p_idx += 1

                # Evaluate activity to decide whether to stop
                if p_idx >= min_p_frames:
                    p_fg_mask = bg_sub.apply(pf)
                    p_features, previous_gray = _extract_features(pf, p_fg_mask, previous_gray)
                    p_anomaly_score, _ = _compute_anomaly_score(p_features, baseline)

                    # Update baseline briefly even during capture to prevent score drift
                    _update_baseline(baseline, p_features)

                    if p_anomaly_score < cfg_t.maintenance_zscore:
                        logger.info(
                            "Dynamic capture end: anomaly score %.2f < maintenance threshold %.2f (total post frames: %d)",
                            p_anomaly_score, cfg_t.maintenance_zscore, p_idx
                        )
                        break
                else:
                    # Still in the mandatory minimum window; just update background model
                    p_fg_mask = bg_sub.apply(pf)
                    p_features, previous_gray = _extract_features(pf, p_fg_mask, previous_gray)
                    _update_baseline(baseline, p_features)

            if p_idx >= max_p_frames:
                logger.info("Dynamic capture end: reached max safety limit of %ds", cfg_v.max_post_trigger_seconds)

            event = EventFrameBlock(
                trigger_time_sec=trigger_time,
                source_fps=source_fps,
                anomaly_score=effective_score,
                pre_frames=pre_frames,
                post_frames=post_frames,
            )
            events.append(event)
            logger.info(
                "Captured anomaly clip: %d pre + %d post frames (%.1fs, score=%.2f)",
                len(pre_frames),
                len(post_frames),
                event.duration_sec,
                event.anomaly_score,
            )

            cooldown_remaining = cooldown_frame_count
            trigger_streak = 0
            buffer.clear()
            previous_gray = None
            continue

        frame_idx += 1

    cap.release()

    if threshold is None and warmup_ratios:
        raw_threshold = np.percentile(warmup_ratios, cfg_t.percentile) * cfg_t.multiplier
        threshold = max(raw_threshold, cfg_t.absolute_floor)
        logger.info(
            "Late threshold fallback applied at end of scan. Foreground threshold=%.4f (raw=%.4f, floor=%.4f)",
            threshold,
            raw_threshold,
            cfg_t.absolute_floor,
        )

    if not events:
        if best_fallback_pre_frames:
            logger.info(
                "No anomaly trigger found; using best-scoring fallback window for %s at t=%.2fs (score=%.2f)",
                video_path,
                best_fallback_trigger_time,
                best_fallback_score,
            )
            cap = cv2.VideoCapture(video_path)
            all_pre_frames: list[bytes] = []
            all_post_frames: list[bytes] = []
            
            if cap.isOpened():
                # We want pre_buffer_seconds before best_fallback_trigger_time
                start_time = max(0.0, best_fallback_trigger_time - cfg_v.pre_buffer_seconds)
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)
                
                # We will read at most (pre + post) frames
                total_fallback_frames = int((cfg_v.pre_buffer_seconds + cfg_v.post_trigger_seconds) * source_fps)
                pre_count = int(cfg_v.pre_buffer_seconds * source_fps)
                
                for i in range(total_fallback_frames):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    
                    encoded = _encode_frame(frame)
                    if i < pre_count:
                        all_pre_frames.append(encoded)
                    else:
                        all_post_frames.append(encoded)
                cap.release()

            fallback_event = _build_fallback_event(
                source_fps=source_fps,
                trigger_time=best_fallback_trigger_time,
                score=max(0.0, best_fallback_score),
                pre_frames=all_pre_frames,
                post_frames=all_post_frames,
            )

            events.append(fallback_event)
        else:
            logger.warning("No anomaly events triggered in %s", video_path)

    return events
