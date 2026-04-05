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
    """BoT-SORT tracker parameters."""
    track_high_thresh: float = 0.2
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.3
    track_buffer: int = 60
    match_thresh: float = 0.99


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
class PrivacyConfig:
    """Privacy processing settings — edge-level face/plate redaction."""
    face_blur_enabled: bool = True
    face_blur_strength: int = 51          # Gaussian kernel size (must be odd)
    plate_redaction_enabled: bool = True
    plate_redaction_color: tuple[int, int, int] = (0, 0, 0)  # Black fill
    face_detection_confidence: float = 0.5
    plate_detection_confidence: float = 0.5


@dataclass(frozen=True)
class EncryptionConfig:
    """Encrypted storage settings (AES-256-GCM at rest)."""
    enabled: bool = False                 # Off by default for dev
    algorithm: str = "AES-256-GCM"


@dataclass(frozen=True)
class AuditConfig:
    """Audit logging settings."""
    enabled: bool = True
    log_file: str = "logs/audit.log"


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
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# ── Singleton config instance ────────────────────────────────────────────────
settings = PipelineConfig()
