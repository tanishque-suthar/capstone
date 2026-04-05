"""
Privacy-Compliant Edge AI — Face Blurring & License Plate Redaction.

All processing runs locally on-device (edge-level). No cloud API calls.

Face detection: MediaPipe Face Detection (CPU-optimized, <5 MB model).
License plate detection: OpenCV contour-based heuristic (aspect ratio + edge density).
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ── Statistics tracker ────────────────────────────────────────────────────────

@dataclass
class PrivacyStats:
    """Accumulated statistics for the current session."""
    faces_blurred: int = 0
    plates_redacted: int = 0
    frames_processed: int = 0
    crops_processed: int = 0


# Module-level stats singleton
_stats = PrivacyStats()


def get_privacy_stats() -> PrivacyStats:
    """Return current session privacy statistics."""
    return _stats


# ── Face Detection (MediaPipe Task API v0.10+) ────────────────────────────────

_face_detector = None


def _get_face_detector():
    """Lazy-load MediaPipe face detector (edge-optimized, Task API)."""
    global _face_detector
    if _face_detector is None:
        try:
            import mediapipe as mp
            from pathlib import Path

            # Look for the .tflite model in config/
            model_path = settings.paths.config_dir / "blaze_face_short_range.tflite"
            if not model_path.exists():
                # Try to download if not present
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading face detection model to %s ...", model_path)
                urllib.request.urlretrieve(url, str(model_path))
                logger.info("Downloaded face detection model (%d bytes)", model_path.stat().st_size)

            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=settings.privacy.face_detection_confidence,
            )
            _face_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
            logger.info("MediaPipe face detector initialized via Task API (edge-local)")
        except Exception as exc:
            logger.warning("Failed to initialize face detector: %s — face blurring disabled", exc)
            _face_detector = False  # Sentinel: tried and failed
    return _face_detector if _face_detector is not False else None


def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in a BGR frame using MediaPipe Task API.
    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    detector = _get_face_detector()
    if detector is None:
        return []

    import mediapipe as mp

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    boxes = []
    if result.detections:
        for det in result.detections:
            bb = det.bounding_box
            x1 = max(0, bb.origin_x)
            y1 = max(0, bb.origin_y)
            x2 = min(w, bb.origin_x + bb.width)
            y2 = min(h, bb.origin_y + bb.height)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

    return boxes


def blur_faces(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Apply Gaussian blur to detected face regions."""
    k = settings.privacy.face_blur_strength
    # Ensure kernel size is odd
    if k % 2 == 0:
        k += 1

    result = frame.copy()
    for (x1, y1, x2, y2) in boxes:
        roi = result[y1:y2, x1:x2]
        if roi.size > 0:
            blurred = cv2.GaussianBlur(roi, (k, k), 30)
            result[y1:y2, x1:x2] = blurred

    return result


# ── License Plate Detection (OpenCV Heuristic) ───────────────────────────────

def detect_license_plates(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect potential license plate regions using contour-based heuristics.

    Strategy:
    1. Convert to grayscale, apply bilateral filter for edge preservation
    2. Canny edge detection
    3. Find contours with rectangular shape
    4. Filter by aspect ratio (typical LP: 2:1 to 5:1) and minimum area

    This is a lightweight edge-local approach — no model download needed.
    """
    min_confidence = settings.privacy.plate_detection_confidence
    h, w = frame.shape[:2]
    min_plate_area = (w * h) * 0.001   # Minimum 0.1% of frame area
    max_plate_area = (w * h) * 0.05    # Maximum 5% of frame area (avoid full-frame FPs)

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)

    # Morphological closing to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_plate_area or area > max_plate_area:
            continue

        # Approximate polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # License plates are typically rectangular (4 vertices)
        if len(approx) >= 4 and len(approx) <= 6:
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect_ratio = bw / max(bh, 1)

            # Typical LP aspect ratios: 2:1 to 5:1
            if 2.0 <= aspect_ratio <= 5.5:
                # Verify edge density within the candidate region
                roi_edges = edges[y:y+bh, x:x+bw]
                if roi_edges.size > 0:
                    edge_density = np.count_nonzero(roi_edges) / roi_edges.size
                    # License plates have high edge density (text characters)
                    if edge_density > 0.15 * min_confidence:
                        plates.append((x, y, x + bw, y + bh))

    # Non-maximum suppression (simple: remove overlapping boxes)
    plates = _nms_boxes(plates, overlap_thresh=0.3)

    return plates


def _nms_boxes(
    boxes: list[tuple[int, int, int, int]], overlap_thresh: float = 0.3
) -> list[tuple[int, int, int, int]]:
    """Simple non-maximum suppression for bounding boxes."""
    if not boxes:
        return []

    boxes_arr = np.array(boxes, dtype=float)
    x1, y1, x2, y2 = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        remaining = np.where(iou <= overlap_thresh)[0]
        order = order[remaining + 1]

    return [boxes[i] for i in keep]


def redact_plates(
    frame: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Fill detected license plate regions with solid color."""
    color = settings.privacy.plate_redaction_color
    result = frame.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness=-1)  # Filled
    return result


# ── Orchestration ─────────────────────────────────────────────────────────────

def apply_privacy_filters(frame: np.ndarray) -> np.ndarray:
    """
    Apply all enabled privacy filters to a single frame.
    Operates in-place-safe (returns a new array).

    This is the main entry point for privacy processing.
    """
    cfg = settings.privacy
    result = frame

    # Face blurring
    if cfg.face_blur_enabled:
        face_boxes = detect_faces(result)
        if face_boxes:
            result = blur_faces(result, face_boxes)
            _stats.faces_blurred += len(face_boxes)

    # License plate redaction
    if cfg.plate_redaction_enabled:
        plate_boxes = detect_license_plates(result)
        if plate_boxes:
            result = redact_plates(result, plate_boxes)
            _stats.plates_redacted += len(plate_boxes)

    _stats.frames_processed += 1
    return result


def process_frames_for_privacy(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Batch-apply privacy filters to a list of frames.
    Used for processing archival video before writing to disk.
    """
    if not settings.privacy.face_blur_enabled and not settings.privacy.plate_redaction_enabled:
        logger.info("Privacy filters disabled — skipping frame processing")
        return frames

    logger.info("Applying privacy filters to %d frames (edge-local processing)...", len(frames))
    processed = []
    for i, frame in enumerate(frames):
        processed.append(apply_privacy_filters(frame))
        if (i + 1) % 25 == 0:
            logger.debug("Privacy processed %d/%d frames", i + 1, len(frames))

    logger.info(
        "Privacy processing complete: %d faces blurred, %d plates redacted across %d frames",
        _stats.faces_blurred, _stats.plates_redacted, len(frames),
    )
    return processed


def process_crop_for_privacy(crop: np.ndarray) -> np.ndarray:
    """
    Apply privacy filters to a single entity crop image.
    Used before saving crop JPGs to disk.
    """
    result = apply_privacy_filters(crop)
    _stats.crops_processed += 1
    return result
