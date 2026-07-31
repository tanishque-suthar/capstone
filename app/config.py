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
    # Inference backend: "openvino" (FP16 IR, ~4-5x faster on Intel CPU) or "pytorch".
    # OpenVINO falls back to the .pt weights if the exported IR dir is absent.
    backend: str = "openvino"
    openvino_model_dir: str = "yolo11n_openvino_model"
    # Pin the inference device. Must be explicit: a CUDA-built torch with no usable
    # GPU otherwise mis-detects one and raises "Invalid device id".
    device: str = "cpu"


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
    min_track_frames: int = 3  # drop tracks observed in fewer frames (ghost/fragment filter)
    velocity_smooth_window: int = 11  # Savitzky-Golay window (odd) for world positions before differencing; 0/<5 disables
    max_speed_mps: float = 40.0       # physical speed cap (~144 km/h) — clamps residual BEV projection spikes


@dataclass(frozen=True)
class ProjectionConfig:
    """
    Reliable-region gate for the bird's-eye-view projection.

    Monocular homography extrapolates non-linearly outside the calibration quad
    (near the horizon, 1px of jitter → metres of error), so world positions far
    from the calibrated region are unreliable. Positions outside these bounds are
    set to NaN and excluded from kinematics/causal analysis. Bounds are generous
    around the quad (X∈[0,3.5], Y∈[0,14] for the current calibration).
    """
    min_depth_m: float = -10.0        # world Y (longitudinal) lower bound
    max_depth_m: float = 35.0         # world Y beyond this is extrapolation garbage
    max_abs_lateral_m: float = 25.0   # |world X| beyond this → NaN


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
class CausalConfig:
    """Track 2 — target-centric PCMCI+ causal discovery parameters."""
    tau_max: int = 5              # max lag in frames (0.5s at 10 FPS)
    pc_alpha: float = 0.05        # PCMCI+ significance level
    min_series_len: int = 20      # min valid frames for an object to be a candidate
    lane_tolerance_m: float = 4.0 # lateral tolerance for "lead vehicle" (same-lane) detection
    max_plausible_speed_mps: float = 25.0  # reject targets whose peak speed exceeds this (~90 km/h; above = projection/tracking spike in surveillance BEV)


@dataclass(frozen=True)
class SynthesisConfig:
    """Track 4 — LLM situation-report synthesis.

    Provider-agnostic: targets any OpenAI-compatible /chat/completions endpoint
    (hosted now, a local/edge server later — just change base_url). The API key is
    read from the environment variable named by api_key_env (never stored in code).
    """
    provider: str = "openai_compatible"
    # Defaults to Google Gemini's OpenAI-compatible endpoint (free tier via AI Studio).
    # Swap base_url/model for any other OpenAI-compatible provider or a local edge server.
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    model: str = "gemini-flash-latest"  # alias → current Gemini flash (avoids model-deprecation breakage)
    api_key_env: str = "LLM_API_KEY"
    max_tokens: int = 2500  # generous: "thinking" Gemini flash models spend tokens on internal reasoning
    temperature: float = 0.3
    max_entities: int = 10        # cap entities described in the evidence packet
    min_entity_frames: int = 8    # ignore fleeting tracks in the packet


@dataclass(frozen=True)
class FeedConfig:
    """Continuous live-monitoring + hybrid all-vehicle indexing."""
    index_all_vehicles: bool = True   # hybrid: index every tracked vehicle, not just event ones
    process_width: int = 640          # downscale incoming frames to this before processing (0 = native)
    process_height: int = 360         # keep at the homography calibration resolution; also bounds 4K memory
    track_end_frames: int = 15        # sampled frames a track may be unseen before it's finalized
    min_crop_area_px: int = 500       # ignore tiny far-field crops (noise)
    reconnect_delay_s: float = 2.0    # wait before reopening a dropped stream
    max_reconnect_attempts: int = 5   # give up on a stream after this many consecutive failures


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
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# ── Singleton config instance ────────────────────────────────────────────────
settings = PipelineConfig()
