"""
Phase 2 — Handoff & Logging.

Interpolates occluded tracks, exports CSV, saves entity crops,
writes archival video, and registers the event in SQLite.

Reference: context.md §4 Phase 2, §6.3, §6.4, §6.5
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from app.config import settings
from app.database import update_event_details, update_event_status
from app.pipeline.perception import PerceptionResult, DetectionMeta

logger = logging.getLogger(__name__)

# Columns to interpolate (spatial & bounding-box)
_INTERP_COLS = ["BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2", "Pos_X_m", "Pos_Y_m"]


def _smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Savitzky-Golay smoothing of a 1-D world-position series.

    No-ops when disabled (window < 5), too short, or containing NaNs (large gaps
    that were NaN-padded rather than interpolated) — savgol requires finite input.
    """
    if window < 5 or arr.size <= window or not np.all(np.isfinite(arr)):
        return arr
    if window % 2 == 0:
        window += 1
    return savgol_filter(arr, window, polyorder=2)


def _interpolate_tracks(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """
    For each Object_ID, fill missing frames within its lifespan.
    - Gaps ≤ max_gap: linearly interpolate spatial/bbox columns, recalculate velocity.
    - Gaps > max_gap: NaN-pad.
    """
    if df.empty:
        return df

    cfg_v = settings.video
    dt = 1.0 / cfg_v.target_fps
    t_start = -cfg_v.pre_buffer_seconds
    total_frames = cfg_v.total_frames

    # Full frame index for the clip
    all_frame_ids = list(range(total_frames))

    interpolated_parts: list[pd.DataFrame] = []

    for obj_id in df["Object_ID"].unique():
        obj_df = df[df["Object_ID"] == obj_id].copy()
        obj_df = obj_df.set_index("Frame_ID")

        # Span of this object's life
        first_frame = obj_df.index.min()
        last_frame = obj_df.index.max()
        lifespan_frames = list(range(first_frame, last_frame + 1))

        # Re-index to full lifespan
        obj_full = obj_df.reindex(lifespan_frames)

        # Carry forward constant columns
        obj_full["Event_ID"] = obj_full["Event_ID"].ffill().bfill()
        obj_full["Object_ID"] = obj_id
        obj_full["Class"] = obj_full["Class"].ffill().bfill()

        # Timestamps for each frame
        obj_full["Timestamp"] = [
            round(t_start + f * dt, 2) for f in obj_full.index
        ]

        # Identify contiguous gaps
        is_missing = obj_df.reindex(lifespan_frames)["Pos_X_m"].isna()
        gap_groups = (is_missing != is_missing.shift()).cumsum()

        for gap_id, group in is_missing.groupby(gap_groups):
            if not group.iloc[0]:
                # Not a gap
                continue
            gap_len = len(group)
            gap_indices = group.index.tolist()

            if gap_len <= max_gap:
                # Interpolate
                for col in _INTERP_COLS:
                    obj_full[col] = obj_full[col].interpolate(method="linear")
            else:
                # NaN-pad (already NaN by reindex)
                pass

        # Smooth world positions before differencing to suppress bbox-jitter /
        # perspective spikes, then cap velocity at a physical maximum. Monocular
        # BEV projection is jitter-sensitive (small pixel wobble → large metric
        # motion, worst far from the camera), so raw per-step velocity is noisy.
        win = settings.interpolation.velocity_smooth_window
        px = _smooth_series(obj_full["Pos_X_m"].to_numpy(dtype=float), win)
        py = _smooth_series(obj_full["Pos_Y_m"].to_numpy(dtype=float), win)
        obj_full["Pos_X_m"] = px
        obj_full["Pos_Y_m"] = py

        dx = pd.Series(px, index=obj_full.index).diff()
        dy = pd.Series(py, index=obj_full.index).diff()
        vel = (np.sqrt(dx**2 + dy**2) / dt).clip(upper=settings.interpolation.max_speed_mps)
        obj_full["Velocity_mps"] = vel
        obj_full.loc[obj_full.index[0], "Velocity_mps"] = 0.0

        obj_full["Frame_ID"] = obj_full.index
        interpolated_parts.append(obj_full.reset_index(drop=True))

    if not interpolated_parts:
        return df

    result = pd.concat(interpolated_parts, ignore_index=True)

    # Ensure correct column order
    col_order = [
        "Event_ID", "Timestamp", "Frame_ID", "Object_ID", "Class",
        "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2",
        "Pos_X_m", "Pos_Y_m", "Velocity_mps",
    ]
    result = result[col_order]
    result.sort_values(["Object_ID", "Frame_ID"], inplace=True)

    return result


def _select_best_crop(
    detections: list[DetectionMeta],
) -> dict[str, DetectionMeta]:
    """
    For each Object_ID, pick the best frame for cropping.
    Primary: highest confidence after area filter. Fallback: largest area.
    """
    from collections import defaultdict

    by_object: dict[str, list[DetectionMeta]] = defaultdict(list)
    for det in detections:
        by_object[det.object_id].append(det)

    best: dict[str, DetectionMeta] = {}
    min_area_ratio = settings.crop.min_area_ratio

    for obj_id, dets in by_object.items():
        # Compute areas
        areas = [
            (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) for d in dets
        ]
        max_area = max(areas) if areas else 1
        area_threshold = max_area * min_area_ratio

        # Filter by area
        candidates = [
            (d, a) for d, a in zip(dets, areas) if a >= area_threshold
        ]

        if candidates:
            # Highest confidence among area-filtered
            chosen = max(candidates, key=lambda x: x[0].confidence)[0]
        else:
            # Fallback: largest area regardless of confidence
            chosen = dets[areas.index(max(areas))]

        best[obj_id] = chosen

    return best


def _filter_short_tracks(
    df: pd.DataFrame,
    detections: list[DetectionMeta],
    min_frames: int,
) -> tuple[pd.DataFrame, list[DetectionMeta], list[str]]:
    """
    Drop tracks (Object_IDs) observed in fewer than ``min_frames`` frames.

    Removes ghost/fragment tracks — the short-lived duplicate IDs that BoT-SORT
    spawns from flickering detections — from both the causal CSV and the crop
    set, without suppressing detections during tracking. Counts are taken from
    raw observations (pre-interpolation), so interpolated gap-fills can't inflate
    a short track past the threshold.

    Returns the filtered DataFrame, the filtered detections, and the list of
    dropped Object_IDs.
    """
    if df.empty or min_frames <= 1:
        return df, detections, []

    counts = df.groupby("Object_ID").size()
    kept_ids = set(counts[counts >= min_frames].index)
    dropped_ids = sorted(oid for oid in counts.index if oid not in kept_ids)

    if not dropped_ids:
        return df, detections, []

    filtered_df = df[df["Object_ID"].isin(kept_ids)].copy()
    filtered_dets = [d for d in detections if d.object_id in kept_ids]

    logger.info(
        "Min-lifespan filter: dropped %d/%d track(s) seen in <%d frames: %s",
        len(dropped_ids), len(counts), min_frames, ", ".join(dropped_ids),
    )
    return filtered_df, filtered_dets, dropped_ids


def finalize_event(
    event_id: str,
    result: PerceptionResult,
    trigger_time: float,
    source_video_path: str | None = None,
) -> dict:
    """
    Complete Phase 2:
        1. Interpolate tracks
        2. Export CSV
        3. Write archival video
        4. Save entity crops
        5. Register event in SQLite

    Returns a dict of output paths.
    """
    cfg = settings
    event_dir = cfg.paths.dataset_dir / event_id
    crops_dir = event_dir / "entity_crops"
    event_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Min-lifespan filter (drop ghost/fragment tracks) ─────────────
    raw_df, detections, _dropped = _filter_short_tracks(
        result.df, result.detections, cfg.interpolation.min_track_frames
    )

    # ── 1. Interpolation ────────────────────────────────────────────────
    df = _interpolate_tracks(
        raw_df, max_gap=cfg.interpolation.max_gap_frames
    )

    # ── 2. CSV export ───────────────────────────────────────────────────
    csv_filename = f"{event_id}_causal_data.csv"
    csv_path = event_dir / csv_filename
    df.to_csv(csv_path, index=False)
    logger.info("Exported CSV: %s (%d rows)", csv_path, len(df))

    # ── 3. Archival video ───────────────────────────────────────────────
    video_filename = f"{event_id}.mp4"
    video_path = event_dir / video_filename
    frames = result.decoded_frames

    if frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, cfg.video.target_fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        logger.info("Wrote video: %s (%d frames)", video_path, len(frames))
    else:
        logger.warning("No frames to write for event %s", event_id)

    # ── 4. Entity crops ─────────────────────────────────────────────────
    best_crops = _select_best_crop(detections)
    crops_saved = 0

    for obj_id, det in best_crops.items():
        if det.frame_idx < len(frames):
            frame = frames[det.frame_idx]
            x1, y1, x2, y2 = det.bbox
            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_filename = f"{event_id}_{obj_id}_crop.jpg"
                cv2.imwrite(str(crops_dir / crop_filename), crop)
                crops_saved += 1

    logger.info("Saved %d entity crops to %s", crops_saved, crops_dir)

    # ── 5. Register in SQLite ───────────────────────────────────────────
    duration = len(frames) / cfg.video.target_fps if frames else 0.0

    try:
        update_event_details(
            event_id=event_id,
            trigger_time=trigger_time,
            video_path=str(video_path),
            csv_path=str(csv_path),
            crops_dir=str(crops_dir),
            duration_s=duration,
            status="Extracted",
            source_video_path=source_video_path,
        )
    except Exception as exc:
        logger.error("Failed to register event %s: %s", event_id, exc)
        update_event_status(event_id, "Failed")

    logger.info("Phase 2 complete for event %s", event_id)

    return {
        "event_id": event_id,
        "video_path": str(video_path),
        "csv_path": str(csv_path),
        "crops_dir": str(crops_dir),
        "duration_s": duration,
    }
