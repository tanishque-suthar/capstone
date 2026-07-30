"""
Camera ingestion — Step 2: continuous live-feed monitoring with hybrid indexing.

A FeedMonitor runs one read loop per source in a background thread. Each frame is
fed to BOTH:
  - FrameEventDetector (Phase 0) -> triggers full event processing (perception +
    handoff + causal-ready CSV), the same rich path as batch ingestion; and
  - ContinuousTracker -> tracks EVERY vehicle across the feed and indexes each one's
    best crop into the RAG vector DB when its track ends (the "hybrid" all-vehicle
    corpus, not just event vehicles).

In-process, one thread per feed, managed by a FeedManager singleton.
"""
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.config import settings
from app.database import insert_event
from app.pipeline.ingestion import FrameEventDetector
from app.pipeline.perception import process_event, _resolve_yolo_model
from app.pipeline.handoff import finalize_event

logger = logging.getLogger(__name__)


def _tracker_yaml_path() -> str:
    """Write the BoT-SORT config from settings.tracker to config/ and return its path."""
    cfg_tr = settings.tracker
    path = Path(settings.paths.config_dir) / "custom_tracker.yaml"
    with open(path, "w") as f:
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
            "model": "auto",
        }, f)
    return str(path)


class ContinuousTracker:
    """
    Streams YOLO + BoT-SORT over a feed, keeps the best crop per track, and emits a
    finished vehicle (track_id + crop) once its track has been unseen for
    ``feed.track_end_frames`` sampled frames. This is the all-vehicle index engine.
    """

    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(_resolve_yolo_model())
        self.tracker_yaml = _tracker_yaml_path()
        self.best: dict[int, dict] = {}       # track_id -> {score, crop, frame}
        self.last_seen: dict[int, int] = {}   # track_id -> sampled frame index
        self.step_idx = -1                    # sampled-frame counter

    def process(self, frame: np.ndarray) -> list[dict]:
        """Advance one sampled frame; return list of finished-track crops."""
        self.step_idx += 1
        cfg_y = settings.yolo
        r = self.model.track(frame, persist=True, conf=cfg_y.confidence,
                             classes=list(cfg_y.class_whitelist), tracker=self.tracker_yaml,
                             device=cfg_y.device, verbose=False)

        if r and r[0].boxes is not None and r[0].boxes.id is not None:
            boxes = r[0].boxes
            for box, tid, conf in zip(boxes.xyxy.tolist(), boxes.id.tolist(), boxes.conf.tolist()):
                tid = int(tid)
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                area = (x2 - x1) * (y2 - y1)
                if area < settings.feed.min_crop_area_px:
                    continue
                score = float(conf) * area  # prefer confident, large (near) crops
                self.last_seen[tid] = self.step_idx
                if tid not in self.best or score > self.best[tid]["score"]:
                    self.best[tid] = {"score": score, "crop": frame[y1:y2, x1:x2].copy(),
                                      "frame": self.step_idx}

        return self._collect_finished()

    def _collect_finished(self) -> list[dict]:
        end_after = settings.feed.track_end_frames
        finished = []
        for tid in [t for t, seen in self.last_seen.items() if self.step_idx - seen > end_after]:
            data = self.best.pop(tid, None)
            self.last_seen.pop(tid, None)
            if data is not None and data["crop"].size > 0:
                finished.append({"track_id": tid, "crop": data["crop"]})
        return finished

    def flush(self) -> list[dict]:
        """Emit every remaining tracked vehicle (call when the feed ends)."""
        finished = [{"track_id": tid, "crop": d["crop"]}
                    for tid, d in self.best.items() if d["crop"].size > 0]
        self.best.clear()
        self.last_seen.clear()
        return finished


@dataclass
class FeedStats:
    frames_read: int = 0
    vehicles_indexed: int = 0
    events_triggered: int = 0
    reconnects: int = 0
    started_at: float = 0.0
    running: bool = True
    last_error: str | None = None


class FeedMonitor(threading.Thread):
    """Background per-feed worker: event detection + continuous all-vehicle indexing."""

    def __init__(self, video_id: str, source: str, label: str = ""):
        super().__init__(daemon=True, name=f"feed-{video_id}")
        self.video_id = video_id
        self.source = source
        self.label = label or video_id
        self.stats = FeedStats()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _is_stream(source: str) -> bool:
        """A live stream (reconnect on drop) vs a finite local file (stop at EOF)."""
        s = str(source).lower()
        return s.startswith(("rtsp://", "http://", "https://", "udp://", "tcp://")) or s.isdigit()

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self) -> None:
        cfg_v = settings.feed
        self.stats.started_at = time.time()
        is_stream = self._is_stream(self.source)
        attempts = 0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(int(self.source) if str(self.source).isdigit() else self.source)
            if not cap.isOpened():
                attempts += 1
                self.stats.last_error = f"cannot open source (attempt {attempts})"
                logger.warning("[%s] %s", self.video_id, self.stats.last_error)
                if not is_stream or attempts >= cfg_v.max_reconnect_attempts:
                    break
                self._wait(cfg_v.reconnect_delay_s)
                continue
            attempts = 0
            self._consume(cap)
            cap.release()
            # A finite file is done at EOF; a live stream that returns EOF is a drop → reconnect.
            if self._stop.is_set() or not is_stream:
                break
            self.stats.reconnects += 1
            self._wait(cfg_v.reconnect_delay_s)

        self.stats.running = False
        logger.info("[%s] monitor stopped (%d frames, %d vehicles, %d events)",
                    self.video_id, self.stats.frames_read, self.stats.vehicles_indexed,
                    self.stats.events_triggered)

    def _consume(self, cap: cv2.VideoCapture) -> None:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / settings.video.target_fps))
        pw, ph = settings.feed.process_width, settings.feed.process_height
        detector = FrameEventDetector(src_fps)
        tracker = ContinuousTracker() if settings.feed.index_all_vehicles else None
        fi = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            self.stats.frames_read += 1
            if pw and ph and (frame.shape[1] != pw or frame.shape[0] != ph):
                frame = cv2.resize(frame, (pw, ph))  # normalize to processing/calibration resolution

            event = detector.process_frame(frame)
            if event is not None:
                self._handle_event(event)

            if tracker is not None and fi % step == 0:
                self._index_vehicles(tracker.process(frame))
            fi += 1

        if tracker is not None:
            self._index_vehicles(tracker.flush())

    # ── event path (reuse batch pipeline) ────────────────────────────────────
    def _handle_event(self, event) -> None:
        event_id = f"EVT_{uuid.uuid4().hex[:12].upper()}"
        try:
            insert_event(event_id, 0.0, "", "", "", 0.0, "Processing", self.source)
            result = process_event(event_id, event)
            finalize_event(event_id, result, event.trigger_time_sec, source_video_path=self.source)
            self.stats.events_triggered += 1
            logger.info("[%s] event %s processed", self.video_id, event_id)
        except Exception as exc:
            logger.error("[%s] event %s failed: %s", self.video_id, event_id, exc, exc_info=True)

    # ── hybrid all-vehicle indexing ──────────────────────────────────────────
    def _index_vehicles(self, finished: list[dict]) -> None:
        if not finished:
            return
        from PIL import Image
        from app.pipeline.rag import get_rag_pipeline

        crops_dir = settings.paths.dataset_dir / "feeds" / self.video_id
        crops_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for f in finished:
            fname = f"{self.video_id}_V_{f['track_id']:04d}_{int(time.time()*1000)}.jpg"
            path = crops_dir / fname
            cv2.imwrite(str(path), f["crop"])
            rgb = cv2.cvtColor(f["crop"], cv2.COLOR_BGR2RGB)
            rel = path.relative_to(settings.paths.base_dir).as_posix()
            items.append((f["track_id"], Image.fromarray(rgb), rel))
        try:
            n = get_rag_pipeline().index_vehicles(f"FEED_{self.video_id}", items)
            self.stats.vehicles_indexed += n
        except Exception as exc:
            logger.error("[%s] indexing %d vehicles failed: %s", self.video_id, len(items), exc)

    def _wait(self, seconds: float) -> None:
        self._stop.wait(timeout=seconds)


class FeedManager:
    """Singleton registry of running FeedMonitor threads."""

    def __init__(self):
        self._monitors: dict[str, FeedMonitor] = {}
        self._lock = threading.Lock()

    def start(self, video_id: str, source: str, label: str = "") -> bool:
        with self._lock:
            m = self._monitors.get(video_id)
            if m and m.is_alive():
                return False
            monitor = FeedMonitor(video_id, source, label)
            self._monitors[video_id] = monitor
            monitor.start()
            logger.info("Started feed monitor %s (%s)", video_id, source)
            return True

    def stop(self, video_id: str) -> bool:
        with self._lock:
            m = self._monitors.get(video_id)
            if not m:
                return False
            m.stop()
            return True

    def status(self) -> list[dict]:
        with self._lock:
            out = []
            for vid, m in self._monitors.items():
                s = m.stats
                out.append({
                    "video_id": vid, "label": m.label, "source": m.source,
                    "running": m.is_alive() and s.running,
                    "frames_read": s.frames_read, "vehicles_indexed": s.vehicles_indexed,
                    "events_triggered": s.events_triggered, "reconnects": s.reconnects,
                    "last_error": s.last_error,
                })
            return out

    def stop_all(self) -> None:
        with self._lock:
            for m in self._monitors.values():
                m.stop()


_manager: FeedManager | None = None


def get_feed_manager() -> FeedManager:
    global _manager
    if _manager is None:
        _manager = FeedManager()
    return _manager
