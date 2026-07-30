"""
Central configuration for the Track 1 pipeline.
All constants are derived from context.md Sections 5-6.
"""

from pathlib import Path
from dataclasses import dataclass, field

# ── Project root (one level above app/) ──────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VideoConfig:
    """Timing and FPS constants for event clips."""
    pre_buffer_seconds: float = 4.0
    post_trigger_seconds: float = 6.0
    target_fps: int = 10
    jpeg_quality: int = 85

    @property
    def clip_duration(self) -> float:
        return self.pre_buffer_seconds + self.post_trigger_seconds

    @property
    def total_frames(self) -> int:
        return int(self.clip_duration * self.target_fps)


@dataclass(frozen=True)
class ThresholdConfig:
    """MOG2 entropy-based event trigger parameters."""
    warmup_seconds: float = 120.0
    percentile: float = 95.0
    multiplier: float = 1.5
    absolute_floor: float = 0.05
    cooldown_seconds: float = 15.0


@dataclass(frozen=True)
class YOLOConfig:
    """Detection model parameters."""
    model_name: str = "yolo11n.pt"
    confidence: float = 0.35
    # COCO class indices: car=2, motorcycle=3, bus=5, truck=7, bicycle=1, person=0
    class_whitelist: tuple[int, ...] = (0, 1, 2, 3, 5, 7)


@dataclass(frozen=True)
class TrackerConfig:
    """BoT-SORT tracker parameters.

    Tuned to reduce duplicate/fragmented Object_IDs. match_thresh reverted from
    a non-standard 0.99 (caused ID switches) to the 0.8 default; new_track_thresh
    raised to suppress ghost tracks. ReID was verified active but gave no benefit
    on our footage, so it stays off. See handoff.py for the complementary
    min-lifespan track filter.
    """
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.5
    track_buffer: int = 60
    match_thresh: float = 0.8
    with_reid: bool = False


@dataclass(frozen=True)
class InterpolationConfig:
    """Occluded-track interpolation policy."""
    max_gap_frames: int = 10  # 1.0 second at 10 FPS


@dataclass(frozen=True)
class CropConfig:
    """Entity crop selection parameters."""
    min_area_ratio: float = 0.4


@dataclass(frozen=True)
class RAGConfig:
    """RAG pipeline and LanceDB configuration."""
    model_name: str = "google/siglip-base-patch16-224"
    db_uri: str = "dataset/lancedb"
    table_name: str = "entity_crops"


@dataclass(frozen=True)
class PathConfig:
    """All filesystem paths."""
    base_dir: Path = BASE_DIR
    dataset_dir: Path = field(default_factory=lambda: BASE_DIR / "dataset")
    config_dir: Path = field(default_factory=lambda: BASE_DIR / "config")
    log_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")
    db_path: Path = field(default_factory=lambda: BASE_DIR / "event_registry.db")
    homography_path: Path = field(default_factory=lambda: BASE_DIR / "config" / "homography.npy")
    lancedb_path: Path = field(default_factory=lambda: BASE_DIR / "dataset" / "lancedb")


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    video: VideoConfig = field(default_factory=VideoConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# ── Singleton config instance ────────────────────────────────────────────────
settings = PipelineConfig()
