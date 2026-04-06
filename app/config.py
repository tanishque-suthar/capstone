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
    rolling_baseline_seconds: float = 30.0
    percentile: float = 95.0
    multiplier: float = 1.5
    absolute_floor: float = 0.05
    cooldown_seconds: float = 15.0
    anomaly_zscore_threshold: float = 6.0
    min_trigger_streak: int = 3
    min_motion_score: float = 0.015


@dataclass(frozen=True)
class YOLOConfig:
    """Detection model parameters."""
    model_name: str = "yolo11l.pt"
    detector_candidates: tuple[str, ...] = ("yolo11l.pt", "yolo11m.pt", "yolo11n.pt")
    confidence: float = 0.35
    # COCO class indices: car=2, motorcycle=3, bus=5, truck=7, bicycle=1, person=0
    class_whitelist: tuple[int, ...] = (0, 1, 2, 3, 5, 7)


@dataclass(frozen=True)
class TrackerConfig:
    """BoT-SORT tracker parameters."""
    track_high_thresh: float = 0.3
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.35
    track_buffer: int = 90
    match_thresh: float = 0.8
    proximity_thresh: float = 0.65
    appearance_thresh: float = 0.8
    max_idle_gap_frames: int = 15
    max_reconnect_speed_mps: float = 55.0
    min_track_frames: int = 3
    min_track_confidence: float = 0.45
    with_reid: bool = True
    reid_model: str = "auto"
    min_appearance_similarity: float = 0.72
    min_duplicate_appearance_similarity: float = 0.82
    stitch_max_gap_frames: int = 20
    stitch_min_appearance_similarity: float = 0.7
    smoothing_window: int = 3


@dataclass(frozen=True)
class SpeedConfig:
    """Velocity sanity limits for traffic footage."""
    max_reasonable_mps: float = 55.0


@dataclass(frozen=True)
class ReasoningConfig:
    """Thresholds for anomaly explanation and causal discovery."""
    min_vehicle_speed_mps: float = 1.5
    abrupt_stop_drop_mps: float = 3.0
    abrupt_stop_final_mps: float = 1.0
    acceleration_window_frames: int = 3
    interaction_distance_m: float = 8.0
    collision_distance_m: float = 3.0
    synchronized_stop_window_s: float = 1.5
    min_event_confidence: float = 0.45
    min_answer_confidence: float = 0.72
    causal_max_lag_frames: int = 15
    causal_min_effect_support: int = 1
    causal_min_score: float = 0.6
    pcmci_enabled: bool = True
    pcmci_min_rows: int = 30
    pcmci_tau_max_frames: int = 8
    pcmci_pc_alpha: float = 0.2
    pcmci_alpha_level: float = 0.1
    pcmci_max_relations: int = 6
    min_track_frames: int = 5
    min_person_track_frames: int = 8


@dataclass(frozen=True)
class VLMConfig:
    """Optional multimodal enrichment settings."""
    enabled: bool = False
    caption_model_name: str = "Salesforce/blip-image-captioning-base"
    max_images: int = 3
    min_caption_confidence: float = 0.35


@dataclass(frozen=True)
class InterpolationConfig:
    """Occluded-track interpolation policy."""
    max_gap_frames: int = 10  # 1.0 second at 10 FPS


@dataclass(frozen=True)
class CropConfig:
    """Entity crop selection parameters."""
    min_area_ratio: float = 0.4
    max_overlap_iou: float = 0.35
    min_crop_size_px: int = 24


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
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# ── Singleton config instance ────────────────────────────────────────────────
settings = PipelineConfig()
