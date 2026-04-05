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

from app.config import settings
from app.database import update_event_details, update_event_status
from app.pipeline.perception import PerceptionResult, DetectionMeta

logger = logging.getLogger(__name__)

# Columns to interpolate (spatial & bounding-box)
_INTERP_COLS = ["BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2", "Pos_X_m", "Pos_Y_m"]


def _compute_velocity_series(df: pd.DataFrame) -> pd.Series:
    """Recompute velocity after interpolation while suppressing impossible spikes."""
    if df.empty:
        return pd.Series(dtype=float)

    dt = df["Timestamp"].diff()
    dx = df["Pos_X_m"].diff()
    dy = df["Pos_Y_m"].diff()

    velocity = np.sqrt(dx**2 + dy**2) / dt.replace(0, np.nan)
    velocity = velocity.astype(float)
    velocity.iloc[0] = 0.0

    max_speed = settings.speed.max_reasonable_mps
    velocity = velocity.mask(~np.isfinite(velocity))
    velocity = velocity.mask(velocity > max_speed)
    velocity = velocity.ffill().fillna(0.0)

    return velocity.clip(lower=0.0, upper=max_speed)


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

        # Recalculate velocity from (possibly interpolated) positions
        obj_full["Velocity_mps"] = _compute_velocity_series(obj_full)

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
    frames: list[np.ndarray],
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
    max_overlap_iou = settings.crop.max_overlap_iou
    min_crop_size_px = settings.crop.min_crop_size_px

    frame_lookup: dict[int, list[DetectionMeta]] = defaultdict(list)
    for det in detections:
        frame_lookup[det.frame_idx].append(det)

    def bbox_area(det: DetectionMeta) -> int:
        return max(1, det.bbox[2] - det.bbox[0]) * max(1, det.bbox[3] - det.bbox[1])

    def overlap_iou(det: DetectionMeta) -> float:
        from app.pipeline.perception import _bbox_iou

        others = [
            _bbox_iou(det.bbox, other.bbox)
            for other in frame_lookup.get(det.frame_idx, [])
            if other.object_id != det.object_id
        ]
        return max(others) if others else 0.0

    def touches_border(det: DetectionMeta) -> bool:
        if det.frame_idx >= len(frames):
            return True
        frame = frames[det.frame_idx]
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox
        return x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1

    def crop_is_clean(det: DetectionMeta) -> bool:
        width = det.bbox[2] - det.bbox[0]
        height = det.bbox[3] - det.bbox[1]
        if width < min_crop_size_px or height < min_crop_size_px:
            return False
        if touches_border(det):
            return False
        return overlap_iou(det) <= max_overlap_iou

    for obj_id, dets in by_object.items():
        # Compute areas
        areas = [bbox_area(d) for d in dets]
        max_area = max(areas) if areas else 1
        area_threshold = max_area * min_area_ratio

        # Filter by area
        candidates = [
            (d, a) for d, a in zip(dets, areas) if a >= area_threshold
        ]
        clean_candidates = [(d, a) for d, a in candidates if crop_is_clean(d)]

        if clean_candidates:
            chosen = max(
                clean_candidates,
                key=lambda x: (x[0].confidence, x[1], -overlap_iou(x[0])),
            )[0]
        elif candidates:
            chosen = max(
                candidates,
                key=lambda x: (not touches_border(x[0]), -overlap_iou(x[0]), x[0].confidence, x[1]),
            )[0]
        else:
            chosen = max(
                dets,
                key=lambda d: (not touches_border(d), -overlap_iou(d), bbox_area(d), d.confidence),
            )

        best[obj_id] = chosen

    return best


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

    # ── 1. Interpolation ────────────────────────────────────────────────
    df = _interpolate_tracks(
        result.df, max_gap=cfg.interpolation.max_gap_frames
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
    best_crops = _select_best_crop(result.detections, frames)
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

    # ── 4b. XAI Artifacts (GradCAM + SHAP) ──────────────────────────────
    xai_result: dict = {}
    try:
        from app.pipeline.xai import generate_xai_artifacts
        from app.pipeline.reasoning import (
            _find_interactions,
            _build_anomalies,
            _frame_level_features,
        )

        interactions = _find_interactions(df)
        anomalies = _build_anomalies(df, interactions)
        feature_df = _frame_level_features(df, interactions, anomalies)

        xai_result = generate_xai_artifacts(
            event_id=event_id,
            frames=frames,
            perception_df=df,
            anomalies=anomalies,
            feature_df=feature_df,
            output_dir=event_dir,
        )
        logger.info("Generated XAI artifacts for %s: %s", event_id, list(xai_result.keys()))
    except Exception as exc:
        logger.warning("XAI generation failed (non-fatal) for %s: %s", event_id, exc)

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
