"""
Phase 1 — Perception & Transformation.

Decodes event frames, runs YOLO + BoT-SORT tracking, applies homography
transform, and computes velocity. Outputs a structured DataFrame.

Reference: context.md §4 Phase 1, §5.1, §5.3, §6.2
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import tempfile
import yaml
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from app.config import settings
from app.pipeline.ingestion import EventFrameBlock

logger = logging.getLogger(__name__)

# COCO class index → readable label
_COCO_LABELS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class DetectionMeta:
    """Per-detection metadata used for crop selection in Phase 2."""
    object_id: str
    frame_idx: int
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class PerceptionResult:
    """Output of Phase 1: DataFrame + metadata for Phase 2."""
    df: pd.DataFrame
    detections: list[DetectionMeta] = field(default_factory=list)
    decoded_frames: list[np.ndarray] = field(default_factory=list)


def _resolve_yolo_model() -> str:
    """
    Return the model path for the configured backend.

    For backend="openvino", use the exported IR directory if present, otherwise
    fall back to the PyTorch weights with a warning (e.g. on a fresh clone where
    the IR hasn't been built yet — run scripts/export_openvino.py).
    """
    cfg_y = settings.yolo
    if cfg_y.backend == "openvino":
        ov_dir = settings.paths.base_dir / cfg_y.openvino_model_dir
        if ov_dir.exists():
            logger.info("YOLO backend=openvino, model=%s, device=%s", ov_dir, cfg_y.device)
            return str(ov_dir)
        logger.warning(
            "backend=openvino but IR not found at %s; falling back to PyTorch %s "
            "(run scripts/export_openvino.py to enable OpenVINO)",
            ov_dir, cfg_y.model_name,
        )
    else:
        logger.info("YOLO backend=pytorch, model=%s, device=%s", cfg_y.model_name, cfg_y.device)
    return cfg_y.model_name


def _load_homography() -> np.ndarray | None:
    """Load the 3×3 homography matrix, or None if unavailable."""
    h_path = settings.paths.homography_path
    if h_path.exists():
        H = np.load(str(h_path))
        logger.info("Loaded homography matrix from %s", h_path)
        return H
    logger.warning("No homography file at %s — spatial columns will be NaN", h_path)
    return None


def _decode_frames(encoded_frames: list[bytes]) -> list[np.ndarray]:
    """Decode JPEG buffers into arrays."""
    decoded = []
    for f in encoded_frames:
        buf = np.frombuffer(f, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            decoded.append(frame)
    return decoded


def _pixel_to_world(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography to an Nx2 array of pixel coordinates.
    Returns Nx2 array of world coordinates (meters).
    """
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


def process_event(event_id: str, frame_block: EventFrameBlock) -> PerceptionResult:
    """
    Run YOLO + BoT-SORT on a downsampled event clip and build a flat DataFrame.

    Returns a PerceptionResult containing:
        - df: DataFrame with columns per context.md §3.1 Item 4
        - detections: per-detection metadata for crop selection
        - decoded_frames: for crop extraction and video export
    """
    cfg_y = settings.yolo
    cfg_tr = settings.tracker
    cfg_v = settings.video
    dt = 1.0 / cfg_v.target_fps  # 0.1s

    # ── Decode all frames for accurate tracking ───────────────────
    full_frames = _decode_frames(frame_block.all_frames)
    source_fps = frame_block.source_fps if frame_block.source_fps > 0 else 30.0
    step = max(1, round(source_fps / cfg_v.target_fps))
    logger.info("Decoded %d frames (source FPS: %.1f). Downsampling by step %d to %.1f FPS", 
                len(full_frames), source_fps, step, cfg_v.target_fps)

    # ── Generate Custom Tracker YAML ──────────────────────────────────────
    tracker_yaml = Path(settings.paths.config_dir) / "custom_tracker.yaml"
    with open(tracker_yaml, "w") as f:
        yaml.dump({
            "tracker_type": "botsort",
            "track_high_thresh": cfg_tr.track_high_thresh,
            "track_low_thresh": cfg_tr.track_low_thresh,
            "new_track_thresh": cfg_tr.new_track_thresh,
            "track_buffer": cfg_tr.track_buffer,
            "match_thresh": cfg_tr.match_thresh,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "with_reid": cfg_tr.with_reid,
            "model": "auto"
        }, f)

    # ── Load model & homography ──────────────────────────────────────────
    model = YOLO(_resolve_yolo_model())
    H = _load_homography()

    # ── Track ────────────────────────────────────────────────────────────
    records: list[dict] = []
    detection_metas: list[DetectionMeta] = []
    target_frames: list[np.ndarray] = []
    failed_frames = 0

    # Pre-compute timestamp offset: pre-buffer seconds before trigger
    t_start = -settings.video.pre_buffer_seconds
    dt_target = 1.0 / cfg_v.target_fps

    for full_idx, frame in enumerate(full_frames):
        is_target = (full_idx % step == 0)
        
        target_fidx = full_idx // step
        timestamp = round(t_start + target_fidx * dt_target, 2)

        try:
            results = model.track(
                frame,
                persist=True,
                conf=cfg_y.confidence,
                classes=list(cfg_y.class_whitelist),
                tracker=str(tracker_yaml),
                device=cfg_y.device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("Inference failed on frame %d: %s", full_idx, exc)
            failed_frames += 1
            continue

        if not is_target:
            continue
            
        target_frames.append(frame)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes
        for box in boxes:
            # Skip detections without a track ID
            if box.id is None:
                continue

            track_id = int(box.id.item())
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            class_label = _COCO_LABELS.get(cls_idx, f"class_{cls_idx}")
            object_id = f"V_{track_id:02d}"

            # Bottom-center of bounding box
            bc_x = (x1 + x2) / 2.0
            bc_y = float(y2)

            # Spatial transform
            if H is not None:
                world = _pixel_to_world(np.array([[bc_x, bc_y]]), H)
                pos_x_m, pos_y_m = float(world[0, 0]), float(world[0, 1])
            else:
                pos_x_m, pos_y_m = float("nan"), float("nan")

            records.append({
                "Event_ID": event_id,
                "Timestamp": timestamp,
                "Frame_ID": target_fidx,
                "Object_ID": object_id,
                "Class": class_label,
                "BBox_X1": x1,
                "BBox_Y1": y1,
                "BBox_X2": x2,
                "BBox_Y2": y2,
                "Pos_X_m": pos_x_m,
                "Pos_Y_m": pos_y_m,
                "Velocity_mps": 0.0,  # Placeholder — computed below
            })

            detection_metas.append(DetectionMeta(
                object_id=object_id,
                frame_idx=target_fidx,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
            ))

    # ── Check failure rate ───────────────────────────────────────────────
    total_frames = len(full_frames)
    if total_frames > 0 and (failed_frames / total_frames) > 0.2:
        raise RuntimeError(
            f"Inference failed on {failed_frames}/{total_frames} frames (>20%) — aborting event"
        )

    # ── Build DataFrame & compute velocity ───────────────────────────────
    if not records:
        logger.warning("No detections in event %s", event_id)
        return PerceptionResult(
            df=pd.DataFrame(columns=[
                "Event_ID", "Timestamp", "Frame_ID", "Object_ID", "Class",
                "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2",
                "Pos_X_m", "Pos_Y_m", "Velocity_mps",
            ]),
            detections=detection_metas,
            decoded_frames=target_frames,
        )

    df = pd.DataFrame(records)
    df.sort_values(["Object_ID", "Frame_ID"], inplace=True)

    # Velocity = euclidean distance / dt between consecutive positions of same object
    for obj_id in df["Object_ID"].unique():
        mask = df["Object_ID"] == obj_id
        obj_df = df.loc[mask]

        dx = obj_df["Pos_X_m"].diff()
        dy = obj_df["Pos_Y_m"].diff()
        velocity = np.sqrt(dx**2 + dy**2) / dt
        velocity.iloc[0] = 0.0  # No velocity for first observation

        df.loc[mask, "Velocity_mps"] = velocity.values

    logger.info(
        "Phase 1 complete for %s: %d unique tracks, %d rows",
        event_id, df["Object_ID"].nunique(), len(df),
    )

    return PerceptionResult(
        df=df,
        detections=detection_metas,
        decoded_frames=target_frames,
    )
