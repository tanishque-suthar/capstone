"""
Phase 1 — Perception & Transformation.

Decodes event frames, runs YOLO + BoT-SORT tracking, applies homography
transform, and computes velocity. Outputs a structured DataFrame.

Reference: context.md §4 Phase 1, §5.1, §5.3, §6.2
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

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
_CLASS_PREFIX = {
    "person": "P",
    "bicycle": "B",
    "car": "C",
    "motorcycle": "M",
    "bus": "U",
    "truck": "T",
}
_VEHICLE_CLASSES = {"car", "truck", "bus"}
_TWO_WHEELER_CLASSES = {"motorcycle", "bicycle"}


@dataclass
class DetectionMeta:
    """Per-detection metadata used for crop selection in Phase 2."""
    object_id: str
    frame_idx: int
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class FrameDetection:
    """Single tracked detection before ID reconciliation."""
    frame_idx: int
    timestamp: float
    raw_track_id: int
    class_label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    pos_x_m: float
    pos_y_m: float

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return float(max(1, self.bbox[2] - self.bbox[0]))

    @property
    def height(self) -> float:
        return float(max(1, self.bbox[3] - self.bbox[1]))

    @property
    def has_world_position(self) -> bool:
        return np.isfinite(self.pos_x_m) and np.isfinite(self.pos_y_m)


@dataclass
class TrackSegment:
    """Contiguous segment of a raw tracker ID."""
    raw_track_id: int
    class_label: str
    detections: list[FrameDetection] = field(default_factory=list)

    @property
    def first(self) -> FrameDetection:
        return self.detections[0]

    @property
    def last(self) -> FrameDetection:
        return self.detections[-1]

    @property
    def dominant_class(self) -> str:
        counts: dict[str, int] = {}
        for det in self.detections:
            counts[det.class_label] = counts.get(det.class_label, 0) + 1
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


@dataclass
class PerceptionResult:
    """Output of Phase 1: DataFrame + metadata for Phase 2."""
    df: pd.DataFrame
    detections: list[DetectionMeta] = field(default_factory=list)
    decoded_frames: list[np.ndarray] = field(default_factory=list)


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


def _load_detector(model_name: str) -> YOLO:
    """Load the preferred detector, falling back to the tiny checkpoint if needed."""
    try:
        return YOLO(model_name)
    except Exception as exc:
        if model_name != "yolo11n.pt":
            logger.warning("Failed to load %s, falling back to yolo11n.pt: %s", model_name, exc)
            return YOLO("yolo11n.pt")
        raise


def _pixel_to_world(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography to an Nx2 array of pixel coordinates.
    Returns Nx2 array of world coordinates (meters).
    """
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute IoU for two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1) * max(1, y2 - y1)


def _class_family(label: str) -> str:
    if label in _VEHICLE_CLASSES:
        return "vehicle"
    if label in _TWO_WHEELER_CLASSES:
        return "two_wheeler"
    return label


def _classes_compatible(a: str, b: str) -> bool:
    return _class_family(a) == _class_family(b)


def _preferred_class(a: str, b: str) -> str:
    priority = {
        "person": 5,
        "motorcycle": 4,
        "bicycle": 3,
        "car": 4,
        "truck": 3,
        "bus": 2,
    }
    return a if priority.get(a, 0) >= priority.get(b, 0) else b


def _is_valid_detection(det: FrameDetection) -> bool:
    """Reject tiny or implausible detections that commonly cause ID noise."""
    area = _bbox_area(det.bbox)
    width = det.width
    height = det.height
    aspect = width / max(1.0, height)

    if det.class_label == "person":
        return det.confidence >= 0.5 and area >= 1200 and 0.2 <= aspect <= 1.0 and height >= 48
    if det.class_label == "motorcycle":
        return det.confidence >= 0.5 and area >= 700 and 0.35 <= aspect <= 2.8
    if det.class_label in {"car", "truck", "bus"}:
        return det.confidence >= 0.45 and area >= 1600 and 0.6 <= aspect <= 4.5 and width >= 35
    if det.class_label == "bicycle":
        return det.confidence >= 0.45 and area >= 500 and 0.2 <= aspect <= 3.0
    return True


def _center_distance_px(a: FrameDetection, b: FrameDetection) -> float:
    """Pixel-space fallback distance between detections."""
    return float(np.hypot(a.center_x - b.center_x, a.center_y - b.center_y))


def _speed_between(a: FrameDetection, b: FrameDetection) -> float | None:
    """World-space speed estimate between two detections."""
    dt = b.timestamp - a.timestamp
    if dt <= 0 or not (a.has_world_position and b.has_world_position):
        return None
    dist = float(np.hypot(b.pos_x_m - a.pos_x_m, b.pos_y_m - a.pos_y_m))
    return dist / dt


def _pixel_motion_is_reasonable(a: FrameDetection, b: FrameDetection) -> bool:
    """
    Fallback continuity check when no homography is available.
    We allow larger jumps for larger boxes and larger frame gaps.
    """
    frame_gap = max(1, b.frame_idx - a.frame_idx)
    max_box_dim = max(a.width, a.height, b.width, b.height)
    return _center_distance_px(a, b) <= max_box_dim * 2.5 * frame_gap


def _deduplicate_frame_detections(detections: list[FrameDetection]) -> list[FrameDetection]:
    """Remove duplicate same-frame detections that describe the same object with different classes."""
    if not detections:
        return detections

    deduped: list[FrameDetection] = []
    by_frame: dict[int, list[FrameDetection]] = {}
    for det in detections:
        by_frame.setdefault(det.frame_idx, []).append(det)

    for frame_idx in sorted(by_frame):
        frame_dets = sorted(by_frame[frame_idx], key=lambda det: det.confidence, reverse=True)
        kept: list[FrameDetection] = []
        for det in frame_dets:
            duplicate = False
            for existing in kept:
                iou = _bbox_iou(det.bbox, existing.bbox)
                if iou < 0.7:
                    continue
                if not _classes_compatible(det.class_label, existing.class_label):
                    continue
                duplicate = True
                if det.confidence > existing.confidence:
                    kept.remove(existing)
                    kept.append(
                        FrameDetection(
                            frame_idx=det.frame_idx,
                            timestamp=det.timestamp,
                            raw_track_id=det.raw_track_id,
                            class_label=_preferred_class(det.class_label, existing.class_label),
                            confidence=det.confidence,
                            bbox=det.bbox,
                            pos_x_m=det.pos_x_m,
                            pos_y_m=det.pos_y_m,
                        )
                    )
                else:
                    existing.class_label = _preferred_class(existing.class_label, det.class_label)
                break
            if not duplicate:
                kept.append(det)
        deduped.extend(kept)

    return deduped


def _suppress_composite_vehicle_detections(detections: list[FrameDetection]) -> list[FrameDetection]:
    """
    Suppress large vehicle boxes that simply wrap a collision cluster of smaller objects.
    This targets failure cases where a car + motorcycle pair is also detected as one bus/truck.
    """
    if not detections:
        return detections

    filtered: list[FrameDetection] = []
    by_frame: dict[int, list[FrameDetection]] = {}
    for det in detections:
        by_frame.setdefault(det.frame_idx, []).append(det)

    for frame_idx in sorted(by_frame):
        frame_dets = by_frame[frame_idx]
        suppressed: set[int] = set()
        for idx, det in enumerate(frame_dets):
            if det.class_label not in _VEHICLE_CLASSES:
                continue
            det_area = _bbox_area(det.bbox)
            contained = []
            for jdx, other in enumerate(frame_dets):
                if idx == jdx:
                    continue
                ox = (other.bbox[0] + other.bbox[2]) / 2.0
                oy = (other.bbox[1] + other.bbox[3]) / 2.0
                inside = det.bbox[0] <= ox <= det.bbox[2] and det.bbox[1] <= oy <= det.bbox[3]
                if not inside:
                    continue
                if _bbox_area(other.bbox) >= det_area * 0.9:
                    continue
                contained.append(other)

            if len(contained) < 2:
                continue
            distinct_families = {_class_family(item.class_label) for item in contained}
            has_two_wheeler = any(item.class_label in _TWO_WHEELER_CLASSES for item in contained)
            max_other_conf = max(item.confidence for item in contained)
            if has_two_wheeler and len(distinct_families) >= 2 and det.confidence <= max_other_conf + 0.12:
                suppressed.add(idx)

        filtered.extend(det for idx, det in enumerate(frame_dets) if idx not in suppressed)

    return filtered


def _should_split_segment(previous: FrameDetection, current: FrameDetection) -> bool:
    """Break a raw track when continuity becomes physically implausible."""
    tracker_cfg = settings.tracker
    if not _classes_compatible(current.class_label, previous.class_label):
        return True

    frame_gap = current.frame_idx - previous.frame_idx
    if frame_gap <= 0:
        return True
    if frame_gap > tracker_cfg.max_idle_gap_frames:
        return True

    speed = _speed_between(previous, current)
    if speed is not None:
        return speed > tracker_cfg.max_reconnect_speed_mps

    return not _pixel_motion_is_reasonable(previous, current)


def _should_merge_segments(previous: TrackSegment, current: TrackSegment) -> bool:
    """Join fragmented segments that likely belong to the same object."""
    if not _classes_compatible(previous.dominant_class, current.dominant_class):
        return False

    tracker_cfg = settings.tracker
    prev_last = previous.last
    curr_first = current.first
    frame_gap = curr_first.frame_idx - prev_last.frame_idx
    if frame_gap <= 0 or frame_gap > tracker_cfg.max_idle_gap_frames:
        return False

    speed = _speed_between(prev_last, curr_first)
    if speed is not None:
        return speed <= tracker_cfg.max_reconnect_speed_mps

    iou = _bbox_iou(prev_last.bbox, curr_first.bbox)
    return iou >= 0.05 or _pixel_motion_is_reasonable(prev_last, curr_first)


def _split_raw_tracks(detections: list[FrameDetection]) -> list[TrackSegment]:
    """Split raw tracker IDs into physically consistent segments."""
    if not detections:
        return []

    segments: list[TrackSegment] = []
    by_raw_id: dict[int, list[FrameDetection]] = {}
    for det in detections:
        by_raw_id.setdefault(det.raw_track_id, []).append(det)

    for raw_track_id in sorted(by_raw_id):
        ordered = sorted(by_raw_id[raw_track_id], key=lambda d: d.frame_idx)
        current_segment = TrackSegment(
            raw_track_id=raw_track_id,
            class_label=ordered[0].class_label,
            detections=[ordered[0]],
        )

        for det in ordered[1:]:
            if _should_split_segment(current_segment.last, det):
                segments.append(current_segment)
                current_segment = TrackSegment(
                    raw_track_id=raw_track_id,
                    class_label=det.class_label,
                    detections=[det],
                )
            else:
                current_segment.detections.append(det)

        segments.append(current_segment)

    return sorted(segments, key=lambda seg: (seg.first.frame_idx, seg.raw_track_id))


def _reconcile_track_ids(detections: list[FrameDetection]) -> tuple[dict[tuple[int, int], str], dict[str, str]]:
    """
    Build stable canonical object IDs by merging short-lived track fragments.
    The key is (raw_track_id, frame_idx) because a raw ID can be split into segments.
    """
    segments = _split_raw_tracks(detections)
    if not segments:
        return {}, {}

    canonical_segments: list[TrackSegment] = []
    assignments: dict[tuple[int, int], str] = {}
    class_counts: dict[str, int] = {}
    segment_ids: list[str] = []

    for segment in segments:
        best_idx: int | None = None
        best_score = float("inf")

        for idx, canonical in enumerate(canonical_segments):
            if not _should_merge_segments(canonical, segment):
                continue

            speed = _speed_between(canonical.last, segment.first)
            if speed is not None:
                score = speed
            else:
                score = _center_distance_px(canonical.last, segment.first)

            if score < best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            canonical_segments.append(
                TrackSegment(
                    raw_track_id=segment.raw_track_id,
                    class_label=segment.dominant_class,
                    detections=list(segment.detections),
                )
            )
            prefix = _CLASS_PREFIX.get(segment.dominant_class, "O")
            class_counts[prefix] = class_counts.get(prefix, 0) + 1
            canonical_id = f"{prefix}_{class_counts[prefix]:02d}"
            segment_ids.append(canonical_id)
        else:
            canonical_segments[best_idx].detections.extend(segment.detections)
            canonical_segments[best_idx].class_label = canonical_segments[best_idx].dominant_class
            canonical_id = segment_ids[best_idx]

        for det in segment.detections:
            assignments[(det.raw_track_id, det.frame_idx)] = canonical_id

    final_classes = {segment_ids[idx]: segment.dominant_class for idx, segment in enumerate(canonical_segments)}
    return assignments, final_classes


def _merge_person_fragments(df: pd.DataFrame, detections: list[DetectionMeta], class_map: dict[str, str]) -> tuple[pd.DataFrame, list[DetectionMeta], dict[str, str]]:
    """Merge short person fragments into nearby longer person tracks."""
    person_ids = [obj_id for obj_id, label in class_map.items() if label == "person"]
    if len(person_ids) < 2 or df.empty:
        return df, detections, class_map

    replacement: dict[str, str] = {}
    for obj_id in person_ids:
        obj_df = df[df["Object_ID"] == obj_id]
        if obj_df["Frame_ID"].nunique() >= 8:
            continue
        best_target = None
        best_score = float("inf")
        for target_id in person_ids:
            if target_id == obj_id:
                continue
            target_df = df[df["Object_ID"] == target_id]
            if target_df["Frame_ID"].nunique() < obj_df["Frame_ID"].nunique():
                continue
            common = obj_df.merge(target_df, on="Frame_ID", suffixes=("_a", "_b"))
            if common.empty:
                continue
            center_dist = np.mean(
                np.hypot(
                    ((common["BBox_X1_a"] + common["BBox_X2_a"]) / 2) - ((common["BBox_X1_b"] + common["BBox_X2_b"]) / 2),
                    ((common["BBox_Y1_a"] + common["BBox_Y2_a"]) / 2) - ((common["BBox_Y1_b"] + common["BBox_Y2_b"]) / 2),
                )
            )
            if center_dist < 45 and center_dist < best_score:
                best_score = center_dist
                best_target = target_id
        if best_target:
            replacement[obj_id] = best_target

    if not replacement:
        return df, detections, class_map

    df = df.copy()
    df["Object_ID"] = df["Object_ID"].replace(replacement)
    detections = [
        DetectionMeta(
            object_id=replacement.get(det.object_id, det.object_id),
            frame_idx=det.frame_idx,
            confidence=det.confidence,
            bbox=det.bbox,
        )
        for det in detections
    ]
    class_map = {replacement.get(obj_id, obj_id): label for obj_id, label in class_map.items() if obj_id not in replacement}
    return df, detections, class_map


def _filter_low_quality_tracks(df: pd.DataFrame, detections: list[DetectionMeta]) -> tuple[pd.DataFrame, list[DetectionMeta]]:
    """Drop short-lived, low-confidence track fragments before export."""
    if df.empty:
        return df, detections

    tracker_cfg = settings.tracker
    keep_ids: set[str] = set()
    stats = (
        df.groupby(["Object_ID", "Class"])
        .agg(
            frame_count=("Frame_ID", "nunique"),
            mean_area=("BBox_X2", lambda s: 0.0),
        )
        .reset_index()
    )

    for _, row in stats.iterrows():
        obj_id = str(row["Object_ID"])
        obj_df = df[df["Object_ID"] == obj_id]
        frame_count = int(obj_df["Frame_ID"].nunique())
        mean_conf = np.mean([d.confidence for d in detections if d.object_id == obj_id]) if detections else 0.0
        mean_area = float(np.mean([
            _bbox_area((int(r.BBox_X1), int(r.BBox_Y1), int(r.BBox_X2), int(r.BBox_Y2)))
            for r in obj_df.itertuples()
        ]))
        class_label = str(obj_df["Class"].mode().iloc[0])

        min_frames = tracker_cfg.min_track_frames
        if class_label in {"truck", "bus"}:
            min_frames = 2

        if frame_count >= min_frames and (mean_conf >= tracker_cfg.min_track_confidence or mean_area >= 2500):
            keep_ids.add(obj_id)

    filtered_df = df[df["Object_ID"].isin(keep_ids)].copy()
    filtered_detections = [d for d in detections if d.object_id in keep_ids]
    return filtered_df, filtered_detections


def _compute_velocity_series(df: pd.DataFrame) -> pd.Series:
    """Compute per-object velocity with real time deltas and spike suppression."""
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


def _track_frames(
    model: YOLO,
    frames: list[np.ndarray],
    tracker_yaml: Path,
    cfg_y,
) -> tuple[list, int]:
    """Run the tracker over the whole clip; fall back to per-frame mode if needed."""
    try:
        results = list(model.track(
            frames,
            persist=True,
            conf=cfg_y.confidence,
            classes=list(cfg_y.class_whitelist),
            tracker=str(tracker_yaml),
            verbose=False,
            stream=True,  # Process sequentially to save memory
        ))
        return results, 0
    except Exception as exc:
        logger.warning("Batch tracking failed, falling back to per-frame mode: %s", exc)

    results = []
    failed_frames = 0
    for frame_idx, frame in enumerate(frames):
        try:
            frame_results = model.track(
                frame,
                persist=True,
                conf=cfg_y.confidence,
                classes=list(cfg_y.class_whitelist),
                tracker=str(tracker_yaml),
                verbose=False,
            )
            results.append(frame_results[0] if frame_results else None)
        except Exception as exc:
            logger.warning("Inference failed on frame %d: %s", frame_idx, exc)
            results.append(None)
            failed_frames += 1

    return results, failed_frames


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
            "proximity_thresh": cfg_tr.proximity_thresh,
            "appearance_thresh": cfg_tr.appearance_thresh,
            "with_reid": False,
            "model": "auto"
        }, f)

    # ── Load model & homography ──────────────────────────────────────────
    model = _load_detector(cfg_y.model_name)
    H = _load_homography()

    # ── Track ────────────────────────────────────────────────────────────
    frame_detections: list[FrameDetection] = []
    target_frames: list[np.ndarray] = []

    # Pre-compute timestamp offset: pre-buffer seconds before trigger
    t_start = -settings.video.pre_buffer_seconds
    dt_target = 1.0 / cfg_v.target_fps

    # Downsample frames first to save 6x on tracking compute
    for full_idx, frame in enumerate(full_frames):
        if full_idx % step == 0:
            target_frames.append(frame)

    tracking_results, failed_frames = _track_frames(model, target_frames, tracker_yaml, cfg_y)

    for target_fidx, frame in enumerate(target_frames):
        result = tracking_results[target_fidx] if target_fidx < len(tracking_results) else None
        timestamp = round(t_start + target_fidx * dt_target, 2)

        if result is None or result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes
        for box in boxes:
            if box.id is None:
                continue

            raw_track_id = int(box.id.item())
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            class_label = _COCO_LABELS.get(cls_idx, f"class_{cls_idx}")

            # Bottom-center of bounding box
            bc_x = (x1 + x2) / 2.0
            bc_y = float(y2)

            # Spatial transform
            if H is not None:
                world = _pixel_to_world(np.array([[bc_x, bc_y]]), H)
                pos_x_m, pos_y_m = float(world[0, 0]), float(world[0, 1])
            else:
                pos_x_m, pos_y_m = float("nan"), float("nan")

            frame_detections.append(
                FrameDetection(
                    frame_idx=target_fidx,
                    timestamp=timestamp,
                    raw_track_id=raw_track_id,
                    class_label=class_label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    pos_x_m=pos_x_m,
                    pos_y_m=pos_y_m,
                )
            )

    # ── Check failure rate ───────────────────────────────────────────────
    total_frames = len(full_frames)
    if total_frames > 0 and (failed_frames / total_frames) > 0.2:
        raise RuntimeError(
            f"Inference failed on {failed_frames}/{total_frames} frames (>20%) — aborting event"
        )

    frame_detections = [det for det in frame_detections if _is_valid_detection(det)]
    frame_detections = _deduplicate_frame_detections(frame_detections)
    frame_detections = _suppress_composite_vehicle_detections(frame_detections)

    # ── Build DataFrame & compute velocity ───────────────────────────────
    if not frame_detections:
        logger.warning("No detections in event %s", event_id)
        return PerceptionResult(
            df=pd.DataFrame(columns=[
                "Event_ID", "Timestamp", "Frame_ID", "Object_ID", "Class",
                "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2",
                "Pos_X_m", "Pos_Y_m", "Velocity_mps",
            ]),
            detections=[],
            decoded_frames=target_frames,
        )

    canonical_ids, canonical_classes = _reconcile_track_ids(frame_detections)
    records: list[dict] = []
    detection_metas: list[DetectionMeta] = []

    for det in frame_detections:
        object_id = canonical_ids.get((det.raw_track_id, det.frame_idx))
        if object_id is None:
            continue

        x1, y1, x2, y2 = det.bbox
        records.append({
            "Event_ID": event_id,
            "Timestamp": det.timestamp,
            "Frame_ID": det.frame_idx,
            "Object_ID": object_id,
            "Class": canonical_classes.get(object_id, det.class_label),
            "BBox_X1": x1,
            "BBox_Y1": y1,
            "BBox_X2": x2,
            "BBox_Y2": y2,
            "Pos_X_m": det.pos_x_m,
            "Pos_Y_m": det.pos_y_m,
            "Velocity_mps": 0.0,
        })
        detection_metas.append(
            DetectionMeta(
                object_id=object_id,
                frame_idx=det.frame_idx,
                confidence=det.confidence,
                bbox=det.bbox,
            )
        )

    df = pd.DataFrame(records)
    df.sort_values(["Object_ID", "Frame_ID", "Timestamp"], inplace=True)

    df, detection_metas, canonical_classes = _merge_person_fragments(df, detection_metas, canonical_classes)
    if not df.empty:
        df["Class"] = df["Object_ID"].map(canonical_classes).fillna(df["Class"])
        df = df.sort_values(["Object_ID", "Frame_ID", "Timestamp"]).drop_duplicates(
            subset=["Object_ID", "Frame_ID"],
            keep="first",
        )

    for obj_id in df["Object_ID"].unique():
        mask = df["Object_ID"] == obj_id
        obj_df = df.loc[mask]
        df.loc[mask, "Velocity_mps"] = _compute_velocity_series(obj_df).values

    df, detection_metas = _filter_low_quality_tracks(df, detection_metas)

    logger.info(
        "Phase 1 complete for %s: %d unique tracks, %d rows",
        event_id, df["Object_ID"].nunique(), len(df),
    )

    return PerceptionResult(
        df=df,
        detections=detection_metas,
        decoded_frames=target_frames,
    )
