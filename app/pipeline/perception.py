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
    class_label: str = ""
    timestamp: float = 0.0
    pos_x_m: float = float("nan")
    pos_y_m: float = float("nan")
    appearance_embedding: np.ndarray | None = None


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
    appearance_embedding: np.ndarray | None = None

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

    @property
    def appearance_embedding(self) -> np.ndarray | None:
        embeddings = [det.appearance_embedding for det in self.detections if det.appearance_embedding is not None]
        if not embeddings:
            return None
        return _normalize_embedding(np.mean(np.stack(embeddings), axis=0))


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


def _load_detector(model_name: str, candidates: tuple[str, ...]) -> YOLO:
    """Load the strongest available detector from the configured candidate list."""
    ordered_candidates: list[str] = []
    for candidate in (model_name, *candidates):
        if candidate not in ordered_candidates:
            ordered_candidates.append(candidate)

    last_exc: Exception | None = None
    for candidate in ordered_candidates:
        try:
            logger.info("Loading detector checkpoint: %s", candidate)
            return YOLO(candidate)
        except Exception as exc:
            logger.warning("Failed to load detector %s: %s", candidate, exc)
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No detector candidates configured")


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


def _normalize_embedding(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


def _appearance_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    a_norm = _normalize_embedding(a)
    b_norm = _normalize_embedding(b)
    if a_norm is None or b_norm is None:
        return None
    return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))


def _extract_appearance_embedding(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Create a lightweight appearance signature from the crop."""
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w - 1, bbox[0]))
    y1 = max(0, min(frame_h - 1, bbox[1]))
    x2 = max(x1 + 1, min(frame_w, bbox[2]))
    y2 = max(y1 + 1, min(frame_h, bbox[3]))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
        return None

    resized = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 64, 160)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)
    aspect = np.array([crop.shape[1] / max(1.0, crop.shape[0])], dtype=np.float32)
    area_ratio = np.array([crop.shape[0] * crop.shape[1] / max(1.0, frame_h * frame_w)], dtype=np.float32)
    embedding = np.concatenate([hist, edge_density, aspect, area_ratio]).astype(np.float32)
    return _normalize_embedding(embedding)


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

    appearance_sim = _appearance_similarity(previous.appearance_embedding, current.appearance_embedding)
    if appearance_sim is not None and appearance_sim < tracker_cfg.min_appearance_similarity:
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


def _build_detection_lookup(detections: list[DetectionMeta]) -> dict[tuple[str, int], DetectionMeta]:
    return {(det.object_id, det.frame_idx): det for det in detections}


def _track_embedding(detections: list[DetectionMeta], object_id: str) -> np.ndarray | None:
    embeddings = [det.appearance_embedding for det in detections if det.object_id == object_id and det.appearance_embedding is not None]
    if not embeddings:
        return None
    return _normalize_embedding(np.mean(np.stack(embeddings), axis=0))


def _track_motion_is_consistent(df_a: pd.DataFrame, df_b: pd.DataFrame, max_gap_frames: int) -> bool:
    if df_a.empty or df_b.empty:
        return False

    if int(df_a["Frame_ID"].max()) <= int(df_b["Frame_ID"].min()):
        earlier, later = df_a, df_b
    elif int(df_b["Frame_ID"].max()) <= int(df_a["Frame_ID"].min()):
        earlier, later = df_b, df_a
    else:
        return False

    frame_gap = int(later["Frame_ID"].min()) - int(earlier["Frame_ID"].max())
    if frame_gap <= 0 or frame_gap > max_gap_frames:
        return False

    prev_row = earlier.sort_values("Frame_ID").iloc[-1]
    next_row = later.sort_values("Frame_ID").iloc[0]
    prev_det = FrameDetection(
        frame_idx=int(prev_row["Frame_ID"]),
        timestamp=float(prev_row["Timestamp"]),
        raw_track_id=-1,
        class_label=str(prev_row["Class"]),
        confidence=1.0,
        bbox=(int(prev_row["BBox_X1"]), int(prev_row["BBox_Y1"]), int(prev_row["BBox_X2"]), int(prev_row["BBox_Y2"])),
        pos_x_m=float(prev_row["Pos_X_m"]),
        pos_y_m=float(prev_row["Pos_Y_m"]),
    )
    next_det = FrameDetection(
        frame_idx=int(next_row["Frame_ID"]),
        timestamp=float(next_row["Timestamp"]),
        raw_track_id=-1,
        class_label=str(next_row["Class"]),
        confidence=1.0,
        bbox=(int(next_row["BBox_X1"]), int(next_row["BBox_Y1"]), int(next_row["BBox_X2"]), int(next_row["BBox_Y2"])),
        pos_x_m=float(next_row["Pos_X_m"]),
        pos_y_m=float(next_row["Pos_Y_m"]),
    )

    speed = _speed_between(prev_det, next_det)
    if speed is not None:
        return speed <= settings.tracker.max_reconnect_speed_mps
    return _pixel_motion_is_reasonable(prev_det, next_det)


def _merge_person_fragments(df: pd.DataFrame, detections: list[DetectionMeta], class_map: dict[str, str]) -> tuple[pd.DataFrame, list[DetectionMeta], dict[str, str]]:
    """Merge short person fragments into nearby longer person tracks."""
    person_ids = [obj_id for obj_id, label in class_map.items() if label == "person"]
    if len(person_ids) < 2 or df.empty:
        return df, detections, class_map

    detection_lookup = _build_detection_lookup(detections)
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
            appearance_scores = [
                _appearance_similarity(
                    detection_lookup.get((obj_id, int(row.Frame_ID)), DetectionMeta("", 0, 0.0, (0, 0, 0, 0))).appearance_embedding,
                    detection_lookup.get((target_id, int(row.Frame_ID)), DetectionMeta("", 0, 0.0, (0, 0, 0, 0))).appearance_embedding,
                )
                for row in common.itertuples()
            ]
            appearance_scores = [score for score in appearance_scores if score is not None]
            mean_appearance = float(np.mean(appearance_scores)) if appearance_scores else None
            if center_dist < 45 and (mean_appearance is None or mean_appearance >= 0.65) and center_dist < best_score:
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
            class_label=det.class_label,
            timestamp=det.timestamp,
            pos_x_m=det.pos_x_m,
            pos_y_m=det.pos_y_m,
            appearance_embedding=det.appearance_embedding,
        )
        for det in detections
    ]
    class_map = {replacement.get(obj_id, obj_id): label for obj_id, label in class_map.items() if obj_id not in replacement}
    return df, detections, class_map


def _merge_duplicate_tracks(df: pd.DataFrame, detections: list[DetectionMeta], class_map: dict[str, str]) -> tuple[pd.DataFrame, list[DetectionMeta], dict[str, str]]:
    """Collapse duplicate canonical tracks that overlap heavily in the same frames."""
    if df.empty:
        return df, detections, class_map

    tracker_cfg = settings.tracker
    detection_lookup = _build_detection_lookup(detections)
    replacement: dict[str, str] = {}
    object_ids = sorted(df["Object_ID"].unique())

    for idx, obj_a in enumerate(object_ids):
        if obj_a in replacement:
            continue
        df_a = df[df["Object_ID"] == obj_a]
        class_a = class_map.get(obj_a, str(df_a["Class"].mode().iloc[0]))
        for obj_b in object_ids[idx + 1:]:
            if obj_b in replacement:
                continue
            df_b = df[df["Object_ID"] == obj_b]
            class_b = class_map.get(obj_b, str(df_b["Class"].mode().iloc[0]))
            if not _classes_compatible(class_a, class_b):
                continue

            common = df_a.merge(df_b, on="Frame_ID", suffixes=("_a", "_b"))
            if common.shape[0] < 5:
                continue

            ious = []
            center_dists = []
            appearance_scores = []
            for row in common.itertuples():
                bbox_a = (int(row.BBox_X1_a), int(row.BBox_Y1_a), int(row.BBox_X2_a), int(row.BBox_Y2_a))
                bbox_b = (int(row.BBox_X1_b), int(row.BBox_Y1_b), int(row.BBox_X2_b), int(row.BBox_Y2_b))
                ious.append(_bbox_iou(bbox_a, bbox_b))
                center_a = ((row.BBox_X1_a + row.BBox_X2_a) / 2.0, (row.BBox_Y1_a + row.BBox_Y2_a) / 2.0)
                center_b = ((row.BBox_X1_b + row.BBox_X2_b) / 2.0, (row.BBox_Y1_b + row.BBox_Y2_b) / 2.0)
                center_dists.append(float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])))
                appearance_sim = _appearance_similarity(
                    detection_lookup.get((obj_a, int(row.Frame_ID)), DetectionMeta("", 0, 0.0, (0, 0, 0, 0))).appearance_embedding,
                    detection_lookup.get((obj_b, int(row.Frame_ID)), DetectionMeta("", 0, 0.0, (0, 0, 0, 0))).appearance_embedding,
                )
                if appearance_sim is not None:
                    appearance_scores.append(appearance_sim)

            mean_iou = float(np.mean(ious)) if ious else 0.0
            mean_center_dist = float(np.mean(center_dists)) if center_dists else float("inf")
            mean_appearance = float(np.mean(appearance_scores)) if appearance_scores else None
            geometry_match = mean_iou >= 0.7 or mean_center_dist <= 20.0
            appearance_match = mean_appearance is None or mean_appearance >= tracker_cfg.min_duplicate_appearance_similarity
            if geometry_match and appearance_match:
                keep = obj_a if df_a["Frame_ID"].nunique() >= df_b["Frame_ID"].nunique() else obj_b
                drop = obj_b if keep == obj_a else obj_a
                replacement[drop] = keep

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
            class_label=det.class_label,
            timestamp=det.timestamp,
            pos_x_m=det.pos_x_m,
            pos_y_m=det.pos_y_m,
            appearance_embedding=det.appearance_embedding,
        )
        for det in detections
    ]
    class_map = {replacement.get(obj_id, obj_id): label for obj_id, label in class_map.items() if obj_id not in replacement}
    df = df.sort_values(["Object_ID", "Frame_ID", "Timestamp"]).drop_duplicates(
        subset=["Object_ID", "Frame_ID"],
        keep="first",
    )
    return df, detections, class_map


def _stitch_global_tracks(df: pd.DataFrame, detections: list[DetectionMeta], class_map: dict[str, str]) -> tuple[pd.DataFrame, list[DetectionMeta], dict[str, str]]:
    """Merge non-overlapping fragmented tracks using appearance and trajectory consistency."""
    if df.empty:
        return df, detections, class_map

    tracker_cfg = settings.tracker
    object_ids = sorted(df["Object_ID"].unique(), key=lambda obj_id: int(df[df["Object_ID"] == obj_id]["Frame_ID"].min()))
    embeddings = {obj_id: _track_embedding(detections, obj_id) for obj_id in object_ids}
    replacement: dict[str, str] = {}

    for idx, obj_a in enumerate(object_ids):
        if obj_a in replacement:
            continue
        df_a = df[df["Object_ID"] == obj_a]
        class_a = class_map.get(obj_a, str(df_a["Class"].mode().iloc[0]))
        for obj_b in object_ids[idx + 1:]:
            if obj_b in replacement:
                continue
            df_b = df[df["Object_ID"] == obj_b]
            class_b = class_map.get(obj_b, str(df_b["Class"].mode().iloc[0]))
            if not _classes_compatible(class_a, class_b):
                continue
            if not _track_motion_is_consistent(df_a, df_b, tracker_cfg.stitch_max_gap_frames):
                continue

            appearance_sim = _appearance_similarity(embeddings.get(obj_a), embeddings.get(obj_b))
            if appearance_sim is not None and appearance_sim < tracker_cfg.stitch_min_appearance_similarity:
                continue

            keep = obj_a if df_a["Frame_ID"].nunique() >= df_b["Frame_ID"].nunique() else obj_b
            drop = obj_b if keep == obj_a else obj_a
            replacement[drop] = keep

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
            class_label=det.class_label,
            timestamp=det.timestamp,
            pos_x_m=det.pos_x_m,
            pos_y_m=det.pos_y_m,
            appearance_embedding=det.appearance_embedding,
        )
        for det in detections
    ]
    class_map = {replacement.get(obj_id, obj_id): label for obj_id, label in class_map.items() if obj_id not in replacement}
    df = df.sort_values(["Object_ID", "Frame_ID", "Timestamp"]).drop_duplicates(
        subset=["Object_ID", "Frame_ID"],
        keep="first",
    )
    return df, detections, class_map


def _smooth_track_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce jitter before velocity and reasoning are computed."""
    if df.empty:
        return df

    window = max(1, settings.tracker.smoothing_window)
    if window <= 1:
        return df

    smoothed = df.copy()
    bbox_cols = ["BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"]
    metric_cols = ["Pos_X_m", "Pos_Y_m"]

    for obj_id in smoothed["Object_ID"].unique():
        mask = smoothed["Object_ID"] == obj_id
        obj_df = smoothed.loc[mask].sort_values("Frame_ID")
        for col in bbox_cols:
            series = obj_df[col].rolling(window=window, min_periods=1, center=True).median().round().astype(int)
            smoothed.loc[obj_df.index, col] = series.values
        for col in metric_cols:
            series = obj_df[col].rolling(window=window, min_periods=1, center=True).median()
            smoothed.loc[obj_df.index, col] = series.values

    return smoothed


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
        results = model.track(
            frames,
            persist=True,
            conf=cfg_y.confidence,
            classes=list(cfg_y.class_whitelist),
            tracker=str(tracker_yaml),
            verbose=False,
        )
        return list(results), 0
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
            "with_reid": cfg_tr.with_reid,
            "model": cfg_tr.reid_model,
        }, f)

    # ── Load model & homography ──────────────────────────────────────────
    model = _load_detector(cfg_y.model_name, cfg_y.detector_candidates)
    H = _load_homography()

    # ── Track ────────────────────────────────────────────────────────────
    frame_detections: list[FrameDetection] = []
    target_frames: list[np.ndarray] = []

    # Pre-compute timestamp offset: pre-buffer seconds before trigger
    t_start = -settings.video.pre_buffer_seconds
    dt_target = 1.0 / cfg_v.target_fps

    tracking_results, failed_frames = _track_frames(model, full_frames, tracker_yaml, cfg_y)

    for full_idx, frame in enumerate(full_frames):
        is_target = (full_idx % step == 0)
        if not is_target:
            continue

        result = tracking_results[full_idx] if full_idx < len(tracking_results) else None
        target_fidx = full_idx // step
        timestamp = round(t_start + target_fidx * dt_target, 2)
        target_frames.append(frame)

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
                    appearance_embedding=_extract_appearance_embedding(frame, (x1, y1, x2, y2)),
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
                class_label=canonical_classes.get(object_id, det.class_label),
                timestamp=det.timestamp,
                pos_x_m=det.pos_x_m,
                pos_y_m=det.pos_y_m,
                appearance_embedding=det.appearance_embedding,
            )
        )

    df = pd.DataFrame(records)
    df.sort_values(["Object_ID", "Frame_ID", "Timestamp"], inplace=True)

    df, detection_metas, canonical_classes = _merge_person_fragments(df, detection_metas, canonical_classes)
    df, detection_metas, canonical_classes = _merge_duplicate_tracks(df, detection_metas, canonical_classes)
    df, detection_metas, canonical_classes = _stitch_global_tracks(df, detection_metas, canonical_classes)
    if not df.empty:
        df["Class"] = df["Object_ID"].map(canonical_classes).fillna(df["Class"])
        df = df.sort_values(["Object_ID", "Frame_ID", "Timestamp"]).drop_duplicates(
            subset=["Object_ID", "Frame_ID"],
            keep="first",
        )
        df = _smooth_track_geometry(df)

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
