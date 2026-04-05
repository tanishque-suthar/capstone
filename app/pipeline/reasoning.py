"""
Structured causal and multimodal reasoning over tracked event data.

This layer combines:
- local causal graph features derived from the event time-series,
- anomaly explanations built from motion and interaction evidence,
- optional VLM-based captions for richer scene grounding,
- confidence gating so the system can say when evidence is insufficient.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
VULNERABLE_CLASSES = {"person", "bicycle", "motorcycle"}


@dataclass
class ObjectSummary:
    object_id: str
    class_label: str
    first_seen_s: float
    last_seen_s: float
    mean_speed_mps: float
    peak_speed_mps: float
    abrupt_stop_count: int
    path_length_m: float


@dataclass
class AnomalyExplanation:
    kind: str
    timestamp_s: float
    objects: list[str]
    reason: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class CausalHypothesis:
    label: str
    answer: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class CausalRelation:
    cause: str
    effect: str
    score: float
    lag_frames: int | None = None
    p_value: float | None = None
    method: str = "heuristic"
    evidence: list[str] = field(default_factory=list)


@dataclass
class MultimodalFinding:
    source: str
    description: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class ConfidenceGate:
    sufficient: bool
    confidence: float
    threshold: float
    explanation: str


@dataclass
class CausalEngineStatus:
    requested: str
    used: str
    used_pcmci: bool
    fallback_used: bool
    relation_count: int
    explanation: str


@dataclass
class ReasoningReport:
    event_id: str
    trigger_time: float
    summary: str
    objects: list[ObjectSummary]
    anomalies: list[AnomalyExplanation]
    hypotheses: list[CausalHypothesis]
    causal_graph: list[CausalRelation] = field(default_factory=list)
    multimodal_findings: list[MultimodalFinding] = field(default_factory=list)
    causal_engine: CausalEngineStatus = field(
        default_factory=lambda: CausalEngineStatus(
            requested="pcmci+",
            used="heuristic",
            used_pcmci=False,
            fallback_used=True,
            relation_count=0,
            explanation="No causal relations were generated yet.",
        )
    )
    confidence_gate: ConfidenceGate = field(
        default_factory=lambda: ConfidenceGate(
            sufficient=False,
            confidence=0.0,
            threshold=settings.reasoning.min_answer_confidence,
            explanation="No reasoning evidence available.",
        )
    )


class OptionalCaptioner:
    """Lazy optional image captioning helper."""

    def __init__(self) -> None:
        self._pipe = None
        self._load_attempted = False

    def _ensure_loaded(self):
        if self._load_attempted:
            return self._pipe
        self._load_attempted = True

        if not settings.vlm.enabled:
            return None

        try:
            from transformers import pipeline

            self._pipe = pipeline(
                "image-to-text",
                model=settings.vlm.caption_model_name,
            )
            logger.info("Loaded optional VLM captioner: %s", settings.vlm.caption_model_name)
        except Exception as exc:
            logger.warning("Optional VLM captioner unavailable: %s", exc)
            self._pipe = None

        return self._pipe

    def caption(self, image_path: Path) -> tuple[str, float] | None:
        pipe = self._ensure_loaded()
        if pipe is None:
            return None

        try:
            result = pipe(str(image_path))
        except Exception as exc:
            logger.warning("Captioning failed for %s: %s", image_path, exc)
            return None

        if not result:
            return None

        item = result[0]
        text = item.get("generated_text", "").strip()
        score = float(item.get("score", settings.vlm.min_caption_confidence))
        if not text:
            return None
        return text, score


_captioner = OptionalCaptioner()


def _load_event_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    return df.sort_values(["Object_ID", "Frame_ID"]).copy()


def _filter_robust_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only tracks with enough temporal support for grounded reasoning."""
    if df.empty:
        return df

    cfg = settings.reasoning
    keep_ids: list[str] = []
    for obj_id, obj_df in df.groupby("Object_ID"):
        class_label = str(obj_df["Class"].mode().iloc[0])
        min_frames = cfg.min_person_track_frames if class_label == "person" else cfg.min_track_frames
        if int(obj_df["Frame_ID"].nunique()) >= min_frames:
            keep_ids.append(obj_id)
    return df[df["Object_ID"].isin(keep_ids)].copy()


def _safe_float(value: float | int | np.floating | None) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def _object_path_length(obj_df: pd.DataFrame) -> float:
    dx = obj_df["Pos_X_m"].diff()
    dy = obj_df["Pos_Y_m"].diff()
    return _safe_float(np.sqrt(dx**2 + dy**2).fillna(0.0).sum())


def _detect_abrupt_stops(obj_df: pd.DataFrame) -> pd.DataFrame:
    cfg = settings.reasoning
    working = obj_df.copy()
    prev = working["Velocity_mps"].shift(cfg.acceleration_window_frames)
    working["prev_speed"] = prev.fillna(working["Velocity_mps"].shift(1))
    working["speed_drop"] = working["prev_speed"] - working["Velocity_mps"]
    mask = (
        (working["Class"].isin(VEHICLE_CLASSES))
        & (working["prev_speed"] >= cfg.min_vehicle_speed_mps)
        & (working["speed_drop"] >= cfg.abrupt_stop_drop_mps)
        & (working["Velocity_mps"] <= cfg.abrupt_stop_final_mps)
    )
    return working.loc[mask].copy()


def _summarize_objects(df: pd.DataFrame) -> list[ObjectSummary]:
    summaries: list[ObjectSummary] = []
    for obj_id, obj_df in df.groupby("Object_ID"):
        speeds = obj_df["Velocity_mps"].fillna(0.0)
        summaries.append(
            ObjectSummary(
                object_id=obj_id,
                class_label=str(obj_df["Class"].mode().iloc[0]),
                first_seen_s=_safe_float(obj_df["Timestamp"].min()),
                last_seen_s=_safe_float(obj_df["Timestamp"].max()),
                mean_speed_mps=_safe_float(speeds.mean()),
                peak_speed_mps=_safe_float(speeds.max()),
                abrupt_stop_count=int(_detect_abrupt_stops(obj_df).shape[0]),
                path_length_m=_object_path_length(obj_df),
            )
        )
    return sorted(summaries, key=lambda item: (item.class_label, item.object_id))


def _frame_positions(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    positions: dict[int, pd.DataFrame] = {}
    for frame_id, frame_df in df.groupby("Frame_ID"):
        valid = frame_df[np.isfinite(frame_df["Pos_X_m"]) & np.isfinite(frame_df["Pos_Y_m"])].copy()
        positions[int(frame_id)] = valid
    return positions


def _find_interactions(df: pd.DataFrame) -> list[dict]:
    cfg = settings.reasoning
    interactions: list[dict] = []
    for frame_id, frame_df in _frame_positions(df).items():
        rows = frame_df.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a = rows[i]
                b = rows[j]
                distance = float(np.hypot(a["Pos_X_m"] - b["Pos_X_m"], a["Pos_Y_m"] - b["Pos_Y_m"]))
                if distance > cfg.interaction_distance_m:
                    continue
                interactions.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": _safe_float(a["Timestamp"]),
                        "object_a": a["Object_ID"],
                        "class_a": a["Class"],
                        "object_b": b["Object_ID"],
                        "class_b": b["Class"],
                        "distance_m": distance,
                    }
                )
    return interactions


def _nearest_interaction(
    interactions: list[dict],
    obj_id: str,
    timestamp_s: float,
    class_filter: set[str] | None = None,
) -> dict | None:
    best = None
    best_key = (float("inf"), float("inf"))
    for item in interactions:
        if obj_id not in {item["object_a"], item["object_b"]}:
            continue
        other_class = item["class_b"] if item["object_a"] == obj_id else item["class_a"]
        if class_filter is not None and other_class not in class_filter:
            continue
        key = (abs(item["timestamp"] - timestamp_s), item["distance_m"])
        if key < best_key:
            best = item
            best_key = key
    return best


def _format_interaction_reason(stop_row: pd.Series, interaction: dict | None) -> tuple[str, float, list[str]]:
    obj_id = str(stop_row["Object_ID"])
    timestamp = _safe_float(stop_row["Timestamp"])
    speed_before = _safe_float(stop_row.get("prev_speed", 0.0))
    speed_after = _safe_float(stop_row["Velocity_mps"])
    evidence = [f"{obj_id} dropped from {speed_before:.1f} m/s to {speed_after:.1f} m/s near t={timestamp:.1f}s."]

    if interaction is None:
        return (
            f"{obj_id} stopped abruptly, most likely because traffic ahead changed unexpectedly or an unseen obstacle forced braking.",
            0.45,
            evidence,
        )

    other_object = interaction["object_b"] if interaction["object_a"] == obj_id else interaction["object_a"]
    other_class = interaction["class_b"] if interaction["object_a"] == obj_id else interaction["class_a"]
    evidence.append(
        f"{obj_id} was within {interaction['distance_m']:.1f} m of {other_class} {other_object} at t={interaction['timestamp']:.1f}s."
    )

    if other_class == "person":
        return (
            f"{obj_id} appears to have stopped for a pedestrian entering or occupying its path.",
            0.64,
            evidence,
        )
    if other_class in {"bicycle", "motorcycle"}:
        return (
            f"{obj_id} appears to have stopped to avoid a nearby {other_class} crossing or cutting into its path.",
            0.58,
            evidence,
        )
    if other_class in VEHICLE_CLASSES:
        return (
            f"{obj_id} appears to have braked because another vehicle ahead or beside it created an immediate conflict.",
            0.52,
            evidence,
        )
    return (
        f"{obj_id} stopped in response to a nearby {other_class} entering a potential conflict zone.",
        0.48,
        evidence,
    )


def _build_anomalies(df: pd.DataFrame, interactions: list[dict]) -> list[AnomalyExplanation]:
    anomalies: list[AnomalyExplanation] = []
    stop_rows: list[pd.Series] = []

    for _, obj_df in df.groupby("Object_ID"):
        stops = _detect_abrupt_stops(obj_df)
        for _, stop_row in stops.iterrows():
            stop_rows.append(stop_row)
            interaction = _nearest_interaction(
                interactions,
                str(stop_row["Object_ID"]),
                _safe_float(stop_row["Timestamp"]),
            )
            reason, confidence, evidence = _format_interaction_reason(stop_row, interaction)
            anomalies.append(
                AnomalyExplanation(
                    kind="abrupt_stop",
                    timestamp_s=_safe_float(stop_row["Timestamp"]),
                    objects=[str(stop_row["Object_ID"])],
                    reason=reason,
                    confidence=confidence,
                    evidence=evidence,
                )
            )

    stop_times = sorted(_safe_float(row["Timestamp"]) for row in stop_rows)
    if stop_times:
        window = settings.reasoning.synchronized_stop_window_s
        for anchor in stop_times:
            near = [row for row in stop_rows if abs(_safe_float(row["Timestamp"]) - anchor) <= window]
            if len(near) < 2:
                continue

            involved = sorted({str(row["Object_ID"]) for row in near})
            interaction = None
            for obj_id in involved:
                interaction = _nearest_interaction(interactions, obj_id, anchor, VULNERABLE_CLASSES)
                if interaction is not None:
                    break

            if interaction is not None:
                other_class = interaction["class_b"] if interaction["object_a"] in involved else interaction["class_a"]
                reason = f"Multiple vehicles slowed or stopped together because a nearby {other_class} created a shared hazard."
                confidence = 0.54
            else:
                reason = "Multiple vehicles slowed or stopped together, which is consistent with sudden congestion or an emerging obstruction ahead."
                confidence = 0.42

            anomalies.append(
                AnomalyExplanation(
                    kind="synchronized_stop",
                    timestamp_s=anchor,
                    objects=involved,
                    reason=reason,
                    confidence=confidence,
                    evidence=[f"{len(involved)} vehicles showed abrupt stopping within {window:.1f}s around t={anchor:.1f}s."],
                )
            )
            break

    anomalies.sort(key=lambda item: (-item.confidence, item.timestamp_s))
    return anomalies


def _build_signal_frames(df: pd.DataFrame, anomalies: list[AnomalyExplanation], interactions: list[dict]) -> dict[str, list[int]]:
    signals = {
        "pedestrian_proximity": [],
        "two_wheeler_proximity": [],
        "vehicle_conflict": [],
        "abrupt_vehicle_stop": [],
        "synchronized_stop": [],
    }

    for item in interactions:
        if item["class_a"] == "person" or item["class_b"] == "person":
            signals["pedestrian_proximity"].append(item["frame_id"])
        elif item["class_a"] in {"bicycle", "motorcycle"} or item["class_b"] in {"bicycle", "motorcycle"}:
            signals["two_wheeler_proximity"].append(item["frame_id"])
        elif item["class_a"] in VEHICLE_CLASSES and item["class_b"] in VEHICLE_CLASSES:
            signals["vehicle_conflict"].append(item["frame_id"])

    frame_lookup = df[["Timestamp", "Frame_ID"]].drop_duplicates()
    for anomaly in anomalies:
        match = frame_lookup.iloc[(frame_lookup["Timestamp"] - anomaly.timestamp_s).abs().argsort()[:1]]
        if match.empty:
            continue
        frame_id = int(match["Frame_ID"].iloc[0])
        if anomaly.kind == "abrupt_stop":
            signals["abrupt_vehicle_stop"].append(frame_id)
        elif anomaly.kind == "synchronized_stop":
            signals["synchronized_stop"].append(frame_id)

    return {name: sorted(set(values)) for name, values in signals.items()}


def _frame_level_features(df: pd.DataFrame, interactions: list[dict], anomalies: list[AnomalyExplanation]) -> pd.DataFrame:
    frame_rows: list[dict[str, float]] = []
    anomaly_frames: dict[str, set[int]] = {"abrupt_stop": set(), "synchronized_stop": set()}

    frame_lookup = df[["Timestamp", "Frame_ID"]].drop_duplicates()
    for anomaly in anomalies:
        match = frame_lookup.iloc[(frame_lookup["Timestamp"] - anomaly.timestamp_s).abs().argsort()[:1]]
        if match.empty:
            continue
        frame_id = int(match["Frame_ID"].iloc[0])
        if anomaly.kind in anomaly_frames:
            anomaly_frames[anomaly.kind].add(frame_id)

    interaction_by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in interactions:
        interaction_by_frame.setdefault(int(item["frame_id"]), []).append(item)

    for frame_id, frame_df in sorted(df.groupby("Frame_ID"), key=lambda item: item[0]):
        valid = frame_df.copy()
        vehicles = valid[valid["Class"].isin(VEHICLE_CLASSES)]
        peds = valid[valid["Class"] == "person"]
        two_wheelers = valid[valid["Class"].isin({"bicycle", "motorcycle"})]

        speed_series = vehicles["Velocity_mps"].fillna(0.0)
        accel_series = vehicles["Velocity_mps"].diff().fillna(0.0)
        frame_interactions = interaction_by_frame.get(int(frame_id), [])

        vehicle_conflict = [item for item in frame_interactions if item["class_a"] in VEHICLE_CLASSES and item["class_b"] in VEHICLE_CLASSES]
        pedestrian_conflict = [item for item in frame_interactions if "person" in {item["class_a"], item["class_b"]}]
        two_wheeler_conflict = [item for item in frame_interactions if item["class_a"] in {"bicycle", "motorcycle"} or item["class_b"] in {"bicycle", "motorcycle"}]

        frame_rows.append(
            {
                "frame_id": float(frame_id),
                "timestamp_s": _safe_float(valid["Timestamp"].iloc[0]),
                "vehicle_count": float(len(vehicles)),
                "pedestrian_present": float(not peds.empty),
                "two_wheeler_present": float(not two_wheelers.empty),
                "mean_vehicle_speed": _safe_float(speed_series.mean()),
                "max_vehicle_speed": _safe_float(speed_series.max()),
                "max_vehicle_deceleration": _safe_float(np.maximum(0.0, -accel_series.min())),
                "vehicle_conflict": float(bool(vehicle_conflict)),
                "pedestrian_proximity": float(bool(pedestrian_conflict)),
                "two_wheeler_proximity": float(bool(two_wheeler_conflict)),
                "min_vehicle_distance": _safe_float(min((item["distance_m"] for item in vehicle_conflict), default=np.nan)),
                "abrupt_vehicle_stop": float(int(frame_id) in anomaly_frames["abrupt_stop"]),
                "synchronized_stop": float(int(frame_id) in anomaly_frames["synchronized_stop"]),
            }
        )

    feature_df = pd.DataFrame(frame_rows).sort_values("frame_id").reset_index(drop=True)
    if feature_df.empty:
        return feature_df

    feature_df["vehicle_density"] = feature_df["vehicle_count"].rolling(5, min_periods=1).mean()
    min_distance = feature_df["min_vehicle_distance"].replace(0.0, np.nan)
    feature_df["vehicle_distance_pressure"] = (1.0 / min_distance).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_df["speed_instability"] = feature_df["mean_vehicle_speed"].diff().abs().fillna(0.0)
    return feature_df.fillna(0.0)


def _score_causal_link(cause_frames: list[int], effect_frames: list[int]) -> tuple[float, list[tuple[int, int]]]:
    cfg = settings.reasoning
    if not cause_frames or not effect_frames:
        return 0.0, []

    matches: list[tuple[int, int]] = []
    for effect in effect_frames:
        possible = [cause for cause in cause_frames if 0 <= effect - cause <= cfg.causal_max_lag_frames]
        if possible:
            best = max(possible)
            matches.append((best, effect))

    if len(matches) < cfg.causal_min_effect_support:
        return 0.0, []

    score = len(matches) / max(1, len(effect_frames))
    return score, matches


def _build_causal_graph_heuristic(df: pd.DataFrame, anomalies: list[AnomalyExplanation], interactions: list[dict]) -> list[CausalRelation]:
    signals = _build_signal_frames(df, anomalies, interactions)
    candidates = [
        ("pedestrian_proximity", "abrupt_vehicle_stop"),
        ("two_wheeler_proximity", "abrupt_vehicle_stop"),
        ("vehicle_conflict", "abrupt_vehicle_stop"),
        ("vehicle_conflict", "synchronized_stop"),
    ]

    relations: list[CausalRelation] = []
    for cause, effect in candidates:
        score, matches = _score_causal_link(signals[cause], signals[effect])
        if score < settings.reasoning.causal_min_score:
            continue
        evidence = [
            f"{cause} preceded {effect} in {len(matches)} matched frame sequence(s) within {settings.reasoning.causal_max_lag_frames} frames."
        ]
        if matches:
            evidence.append(f"Example match: cause frame {matches[0][0]} -> effect frame {matches[0][1]}.")
        lag = matches[0][1] - matches[0][0] if matches else None
        relations.append(
            CausalRelation(
                cause=cause,
                effect=effect,
                score=score,
                lag_frames=lag,
                method="heuristic",
                evidence=evidence,
            )
        )

    relations.sort(key=lambda item: item.score, reverse=True)
    return relations


def _normalize_series(series: pd.Series) -> np.ndarray:
    values = series.astype(float).to_numpy()
    if values.size == 0:
        return values
    if np.allclose(values, values[0]):
        return np.zeros_like(values, dtype=float)
    std = float(np.nanstd(values))
    if std < 1e-8:
        return np.nan_to_num(values, nan=0.0)
    mean = float(np.nanmean(values))
    return np.nan_to_num((values - mean) / std, nan=0.0)


def _choose_pcmci_features(feature_df: pd.DataFrame) -> list[str]:
    preferred = [
        "pedestrian_proximity",
        "two_wheeler_proximity",
        "vehicle_conflict",
        "vehicle_distance_pressure",
        "vehicle_density",
        "mean_vehicle_speed",
        "speed_instability",
        "max_vehicle_deceleration",
        "abrupt_vehicle_stop",
        "synchronized_stop",
    ]
    selected: list[str] = []
    for name in preferred:
        if name not in feature_df.columns:
            continue
        values = feature_df[name].to_numpy(dtype=float)
        if np.nanmax(values) - np.nanmin(values) < 1e-8:
            continue
        selected.append(name)
    return selected


def _run_pcmci_discovery(feature_df: pd.DataFrame) -> list[CausalRelation]:
    cfg = settings.reasoning
    if not cfg.pcmci_enabled or feature_df.shape[0] < cfg.pcmci_min_rows:
        return []

    feature_names = _choose_pcmci_features(feature_df)
    if len(feature_names) < 3:
        return []

    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except Exception as exc:
        logger.info("PCMCI+ unavailable, falling back to heuristic graph: %s", exc)
        return []

    values = np.column_stack([_normalize_series(feature_df[name]) for name in feature_names])
    if values.shape[0] < cfg.pcmci_min_rows or values.shape[1] < 3:
        return []

    try:
        dataframe = pp.DataFrame(values, var_names=feature_names)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
        results = pcmci.run_pcmciplus(
            tau_max=min(cfg.pcmci_tau_max_frames, max(1, values.shape[0] // 4)),
            pc_alpha=cfg.pcmci_pc_alpha,
        )
    except Exception as exc:
        logger.warning("PCMCI+ failed on event features, using heuristic graph instead: %s", exc)
        return []

    graph = results.get("graph")
    val_matrix = results.get("val_matrix")
    p_matrix = results.get("p_matrix")
    if graph is None or val_matrix is None or p_matrix is None:
        return []

    relations: list[CausalRelation] = []
    alpha = cfg.pcmci_alpha_level
    tau_max = min(graph.shape[2] - 1, cfg.pcmci_tau_max_frames)

    for cause_idx, cause_name in enumerate(feature_names):
        for effect_idx, effect_name in enumerate(feature_names):
            if cause_idx == effect_idx:
                continue
            for lag in range(1, tau_max + 1):
                edge = str(graph[cause_idx, effect_idx, lag])
                p_value = float(p_matrix[cause_idx, effect_idx, lag])
                score = abs(float(val_matrix[cause_idx, effect_idx, lag]))
                if edge not in {"-->", "o-o", "o->"}:
                    continue
                if not np.isfinite(p_value) or p_value > alpha:
                    continue
                if score < cfg.causal_min_score:
                    continue
                evidence = [
                    f"PCMCI+ found {cause_name} leading {effect_name} by {lag} frame(s) with effect size {score:.2f}.",
                    f"Analytic significance test produced p={p_value:.3f} on {values.shape[0]} frame samples.",
                ]
                relations.append(
                    CausalRelation(
                        cause=cause_name,
                        effect=effect_name,
                        score=score,
                        lag_frames=lag,
                        p_value=p_value,
                        method="pcmci+",
                        evidence=evidence,
                    )
                )

    relations.sort(key=lambda item: (item.p_value if item.p_value is not None else 1.0, -item.score, item.lag_frames or 0))
    deduped: list[CausalRelation] = []
    seen_pairs: set[tuple[str, str]] = set()
    for relation in relations:
        key = (relation.cause, relation.effect)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped.append(relation)
        if len(deduped) >= cfg.pcmci_max_relations:
            break
    return deduped


def _merge_causal_graphs(primary: list[CausalRelation], fallback: list[CausalRelation]) -> list[CausalRelation]:
    merged: list[CausalRelation] = []
    seen: set[tuple[str, str]] = set()
    for relation in primary + fallback:
        key = (relation.cause, relation.effect)
        if key in seen:
            continue
        seen.add(key)
        merged.append(relation)
    return merged


def _build_causal_engine_status(
    pcmci_relations: list[CausalRelation],
    heuristic_relations: list[CausalRelation],
    merged_relations: list[CausalRelation],
) -> CausalEngineStatus:
    used_pcmci = bool(pcmci_relations)
    fallback_used = bool(heuristic_relations) and not used_pcmci

    if used_pcmci:
        explanation = (
            f"PCMCI+ ran successfully and produced {len(pcmci_relations)} causal relation(s)."
        )
        if heuristic_relations:
            explanation += f" {len(heuristic_relations)} heuristic relation(s) were kept as supplemental fallback evidence."
        used = "pcmci+"
    elif heuristic_relations:
        explanation = (
            f"PCMCI+ did not yield accepted relations for this event, so the report used {len(heuristic_relations)} heuristic causal relation(s) instead."
        )
        used = "heuristic"
        fallback_used = True
    else:
        explanation = "Neither PCMCI+ nor the heuristic fallback produced a causal relation for this event."
        used = "none"
        fallback_used = False

    return CausalEngineStatus(
        requested="pcmci+",
        used=used,
        used_pcmci=used_pcmci,
        fallback_used=fallback_used,
        relation_count=len(merged_relations),
        explanation=explanation,
    )


def _build_causal_graph(df: pd.DataFrame, anomalies: list[AnomalyExplanation], interactions: list[dict]) -> list[CausalRelation]:
    feature_df = _frame_level_features(df, interactions, anomalies)
    pcmci_relations = _run_pcmci_discovery(feature_df)
    heuristic_relations = _build_causal_graph_heuristic(df, anomalies, interactions)
    merged_relations = _merge_causal_graphs(pcmci_relations, heuristic_relations)
    status = _build_causal_engine_status(pcmci_relations, heuristic_relations, merged_relations)
    return merged_relations, status


def _select_multimodal_images(event_dir: Path, crops_dir: str | Path | None) -> list[Path]:
    images: list[Path] = []
    if crops_dir:
        crop_root = Path(crops_dir)
        if crop_root.exists():
            images.extend(sorted(crop_root.glob("*.jpg")))
    if not images:
        images.extend(sorted(event_dir.glob("*.jpg")))
    return images[: settings.vlm.max_images]


def _build_multimodal_findings(
    event_id: str,
    event_dir: Path,
    crops_dir: str | Path | None,
) -> list[MultimodalFinding]:
    if not settings.vlm.enabled:
        return []

    findings: list[MultimodalFinding] = []
    for image_path in _select_multimodal_images(event_dir, crops_dir):
        result = _captioner.caption(image_path)
        if result is None:
            continue
        caption, confidence = result
        findings.append(
            MultimodalFinding(
                source=image_path.name,
                description=caption,
                confidence=confidence,
                evidence=[f"Caption generated from {image_path.name}."],
            )
        )

    findings.sort(key=lambda item: item.confidence, reverse=True)
    return findings


def _build_hypotheses(
    df: pd.DataFrame,
    anomalies: list[AnomalyExplanation],
    interactions: list[dict],
    causal_graph: list[CausalRelation],
    multimodal_findings: list[MultimodalFinding],
) -> list[CausalHypothesis]:
    hypotheses: list[CausalHypothesis] = []
    cfg = settings.reasoning

    robust_classes = set(df["Class"].unique())

    def relation_score(cause: str, effect: str) -> float:
        scores = [item.score for item in causal_graph if item.cause == cause and item.effect == effect]
        return max(scores) if scores else 0.0

    collision_like = [
        item for item in interactions
        if item["class_a"] in VEHICLE_CLASSES
        and item["class_b"] in VEHICLE_CLASSES
        and item["distance_m"] <= cfg.collision_distance_m
    ]
    closest_collision = min(collision_like, key=lambda item: item["distance_m"]) if collision_like else None
    vehicle_conflict_score = max(
        relation_score("vehicle_conflict", "abrupt_vehicle_stop"),
        relation_score("vehicle_conflict", "synchronized_stop"),
    )
    if collision_like and vehicle_conflict_score >= cfg.causal_min_score and anomalies:
        closest = closest_collision
        pair_classes = {closest["class_a"], closest["class_b"]}
        evidence = [
            f"{closest['object_a']} and {closest['object_b']} came within {closest['distance_m']:.1f} m at t={closest['timestamp']:.1f}s."
        ]
        evidence.extend(item.evidence[0] for item in causal_graph[:1] if item.evidence)
        if pair_classes & {"motorcycle", "bicycle"}:
            answer = "The event most likely comes from a vehicle and two-wheeler conflict that rapidly closed distance and forced impact or emergency braking."
            label = "vehicle_two_wheeler_collision"
            confidence = 0.9
        else:
            answer = "The event most likely stems from a conflict between nearby vehicles that rapidly closed distance and forced emergency braking."
            label = "collision_or_near_collision"
            confidence = 0.78
        hypotheses.append(
            CausalHypothesis(
                label=label,
                answer=answer,
                confidence=confidence,
                evidence=evidence,
            )
        )

    pedestrian_related = [item for item in anomalies if "pedestrian" in item.reason.lower()]
    pedestrian_score = relation_score("pedestrian_proximity", "abrupt_vehicle_stop")
    person_near_collision = True
    if closest_collision is not None:
        person_near_collision = any(
            ("person" in {item["class_a"], item["class_b"]})
            and abs(item["timestamp"] - closest_collision["timestamp"]) <= 1.0
            and item["distance_m"] <= max(cfg.collision_distance_m, 4.0)
            for item in interactions
        )
    strongest_collision = max((item.confidence for item in hypotheses if "collision" in item.label), default=0.0)
    if (
        "person" in robust_classes
        and pedestrian_related
        and pedestrian_score >= cfg.causal_min_score
        and person_near_collision
        and strongest_collision < 0.85
    ):
        top = pedestrian_related[0]
        hypotheses.append(
            CausalHypothesis(
                label="pedestrian_conflict",
                answer="The strongest explanation is that a pedestrian entered a vehicle path and forced traffic to stop or yield.",
                confidence=min(max(top.confidence, pedestrian_score), max(0.0, strongest_collision - 0.05) if strongest_collision else 0.82),
                evidence=top.evidence + [f"Supporting causal link score: pedestrian_proximity -> abrupt_vehicle_stop = {pedestrian_score:.2f}."],
            )
        )

    cyclist_related = [item for item in anomalies if "bicycle" in item.reason.lower() or "motorcycle" in item.reason.lower()]
    two_wheeler_score = relation_score("two_wheeler_proximity", "abrupt_vehicle_stop")
    if robust_classes & {"motorcycle", "bicycle"} and cyclist_related and two_wheeler_score >= cfg.causal_min_score:
        top = cyclist_related[0]
        hypotheses.append(
            CausalHypothesis(
                label="two_wheeler_conflict",
                answer="A two-wheeler appears to have entered a conflict zone, prompting vehicles nearby to brake or stop.",
                confidence=max(top.confidence, two_wheeler_score),
                evidence=top.evidence + [f"Supporting causal link score: two_wheeler_proximity -> abrupt_vehicle_stop = {two_wheeler_score:.2f}."],
            )
        )

    if multimodal_findings:
        top_vlm = multimodal_findings[0]
        hypotheses.append(
            CausalHypothesis(
                label="vlm_scene_context",
                answer=f"Additional scene context from image understanding suggests: {top_vlm.description}",
                confidence=min(0.7, top_vlm.confidence),
                evidence=top_vlm.evidence,
            )
        )

    if not hypotheses and anomalies and causal_graph:
        top = anomalies[0]
        hypotheses.append(
            CausalHypothesis(
                label="traffic_disturbance",
                answer=top.reason,
                confidence=max(cfg.min_event_confidence, min(top.confidence, 0.55)),
                evidence=top.evidence,
            )
        )

    if not hypotheses:
        class_counts = Counter(df["Class"])
        common = ", ".join(f"{count} {label}" for label, count in class_counts.most_common(3))
        hypotheses.append(
            CausalHypothesis(
                label="no_strong_cause_detected",
                answer=f"No strong causal trigger was isolated from tracked motion alone. The scene mainly contains {common}.",
                confidence=0.25,
                evidence=["Tracked motion did not show a clear collision, crossing conflict, or synchronized stop pattern."],
            )
        )

    hypotheses.sort(key=lambda item: item.confidence, reverse=True)
    return hypotheses


def _build_confidence_gate(
    hypotheses: list[CausalHypothesis],
    anomalies: list[AnomalyExplanation],
    causal_graph: list[CausalRelation],
    multimodal_findings: list[MultimodalFinding],
) -> ConfidenceGate:
    threshold = settings.reasoning.min_answer_confidence
    top_confidence = hypotheses[0].confidence if hypotheses else 0.0
    evidence_sources = int(bool(anomalies)) + int(bool(causal_graph)) + int(bool(multimodal_findings))
    sufficient = top_confidence >= threshold and evidence_sources >= 2

    if sufficient:
        explanation = "The top explanation is supported by multiple evidence sources."
    elif top_confidence >= threshold:
        explanation = "A plausible explanation exists, but it is not corroborated by enough independent evidence streams."
    else:
        explanation = "Evidence is too weak or ambiguous to support a confident causal claim."

    return ConfidenceGate(
        sufficient=sufficient,
        confidence=top_confidence,
        threshold=threshold,
        explanation=explanation,
    )


def _build_summary(
    objects: list[ObjectSummary],
    anomalies: list[AnomalyExplanation],
    hypotheses: list[CausalHypothesis],
    gate: ConfidenceGate,
) -> str:
    class_counts = Counter(obj.class_label for obj in objects)
    object_phrase = ", ".join(f"{count} {label}" for label, count in class_counts.most_common(4))
    if gate.sufficient and hypotheses:
        anomaly_phrase = (
            f"Tracked motion produced {len(anomalies)} notable anomaly signal(s) across the event."
            if anomalies else
            "Tracked motion produced a usable causal signal."
        )
        top_hypothesis = hypotheses[0].answer
    elif anomalies:
        anomaly_phrase = "Tracked motion suggests a disturbance, but the evidence is not strong enough to identify a reliable root cause."
        top_hypothesis = "Evidence is currently insufficient for a confident root-cause claim."
    else:
        anomaly_phrase = "No strong anomaly was isolated from tracked motion."
        top_hypothesis = "Evidence is currently insufficient for a confident root-cause claim."
    return f"The event contains {object_phrase}. Main finding: {anomaly_phrase} Causal assessment: {top_hypothesis}"


def generate_reasoning_report(
    event_id: str,
    csv_path: str | Path,
    trigger_time: float,
    video_path: str | Path | None = None,
    crops_dir: str | Path | None = None,
) -> ReasoningReport:
    """Generate a structured reasoning report from the event CSV and optional image assets."""
    raw_df = _load_event_dataframe(csv_path)
    if raw_df.empty:
        return ReasoningReport(
            event_id=event_id,
            trigger_time=trigger_time,
            summary="No tracked objects were available, so no grounded reasoning could be produced.",
            objects=[],
            anomalies=[],
            hypotheses=[],
        )

    df = _filter_robust_objects(raw_df)
    if df.empty:
        return ReasoningReport(
            event_id=event_id,
            trigger_time=trigger_time,
            summary="Tracked objects were too brief or unstable to support grounded reasoning.",
            objects=[],
            anomalies=[],
            hypotheses=[
                CausalHypothesis(
                    label="insufficient_track_quality",
                    answer="The detected objects were not stable enough across frames to support a reliable causal explanation.",
                    confidence=0.15,
                    evidence=["All candidate tracks fell below the minimum support required for reasoning."],
                )
            ],
        )

    objects = _summarize_objects(df)
    interactions = _find_interactions(df)
    anomalies = _build_anomalies(df, interactions)
    causal_graph, causal_engine = _build_causal_graph(df, anomalies, interactions)
    event_dir = Path(csv_path).resolve().parent
    multimodal_findings = _build_multimodal_findings(event_id, event_dir, crops_dir)
    hypotheses = _build_hypotheses(df, anomalies, interactions, causal_graph, multimodal_findings)
    confidence_gate = _build_confidence_gate(hypotheses, anomalies, causal_graph, multimodal_findings)
    summary = _build_summary(objects, anomalies, hypotheses, confidence_gate)

    return ReasoningReport(
        event_id=event_id,
        trigger_time=trigger_time,
        summary=summary,
        objects=objects,
        anomalies=anomalies,
        hypotheses=hypotheses,
        causal_graph=causal_graph,
        multimodal_findings=multimodal_findings,
        causal_engine=causal_engine,
        confidence_gate=confidence_gate,
    )


def save_reasoning_report(report: ReasoningReport) -> str:
    event_dir = settings.paths.dataset_dir / report.event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    path = event_dir / f"{report.event_id}_reasoning.json"
    payload = {
        "event_id": report.event_id,
        "trigger_time": report.trigger_time,
        "summary": report.summary,
        "objects": [asdict(item) for item in report.objects],
        "anomalies": [asdict(item) for item in report.anomalies],
        "hypotheses": [asdict(item) for item in report.hypotheses],
        "causal_graph": [asdict(item) for item in report.causal_graph],
        "multimodal_findings": [asdict(item) for item in report.multimodal_findings],
        "causal_engine": asdict(report.causal_engine),
        "confidence_gate": asdict(report.confidence_gate),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved reasoning report for %s to %s", report.event_id, path)
    return str(path)


def load_reasoning_report(path: str | Path) -> ReasoningReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReasoningReport(
        event_id=payload["event_id"],
        trigger_time=float(payload["trigger_time"]),
        summary=payload["summary"],
        objects=[ObjectSummary(**item) for item in payload.get("objects", [])],
        anomalies=[AnomalyExplanation(**item) for item in payload.get("anomalies", [])],
        hypotheses=[CausalHypothesis(**item) for item in payload.get("hypotheses", [])],
        causal_graph=[CausalRelation(**item) for item in payload.get("causal_graph", [])],
        multimodal_findings=[MultimodalFinding(**item) for item in payload.get("multimodal_findings", [])],
        causal_engine=CausalEngineStatus(
            **payload.get(
                "causal_engine",
                {
                    "requested": "pcmci+",
                    "used": "heuristic",
                    "used_pcmci": any(item.get("method") == "pcmci+" for item in payload.get("causal_graph", [])),
                    "fallback_used": any(item.get("method") == "heuristic" for item in payload.get("causal_graph", [])),
                    "relation_count": len(payload.get("causal_graph", [])),
                    "explanation": "Legacy report loaded without explicit causal engine metadata.",
                },
            )
        ),
        confidence_gate=ConfidenceGate(
            **payload.get(
                "confidence_gate",
                {
                    "sufficient": False,
                    "confidence": 0.0,
                    "threshold": settings.reasoning.min_answer_confidence,
                    "explanation": "No confidence assessment recorded.",
                },
            )
        ),
    )


def _insufficient_evidence_answer(report: ReasoningReport) -> dict:
    evidence = []
    if report.hypotheses:
        evidence.extend(report.hypotheses[0].evidence[:2])
    if report.multimodal_findings:
        evidence.extend(item.description for item in report.multimodal_findings[:1])
    if not evidence:
        evidence = [report.confidence_gate.explanation]
    return {
        "answer": "Insufficient evidence to make a confident causal claim for this event.",
        "confidence": report.confidence_gate.confidence,
        "evidence": evidence,
    }


def answer_question(report: ReasoningReport, question: str) -> dict:
    """Answer a user question using the structured reasoning report."""
    q = question.strip().lower()

    if not report.confidence_gate.sufficient and any(word in q for word in {"why", "cause", "occur", "accident", "crash", "collision"}):
        return _insufficient_evidence_answer(report)

    if any(word in q for word in {"why", "cause", "occur", "accident", "crash", "collision"}):
        if report.hypotheses:
            top = report.hypotheses[0]
            return {
                "answer": top.answer,
                "confidence": top.confidence,
                "evidence": top.evidence,
            }

    if "anomal" in q or "what happened" in q or "what is happening" in q:
        if report.anomalies:
            top = report.anomalies[0]
            return {
                "answer": top.reason,
                "confidence": top.confidence,
                "evidence": top.evidence,
            }

    if "stop" in q or "brake" in q:
        stop_related = [
            item for item in report.anomalies
            if "stop" in item.kind or "stop" in item.reason.lower() or "brak" in item.reason.lower()
        ]
        if stop_related:
            top = stop_related[0]
            return {
                "answer": top.reason,
                "confidence": top.confidence,
                "evidence": top.evidence,
            }

    if "object" in q or "vehicle" in q or "who" in q:
        counts = Counter(obj.class_label for obj in report.objects)
        detail = ", ".join(f"{count} {label}" for label, count in counts.most_common())
        return {
            "answer": f"The event contains the following tracked object types: {detail}.",
            "confidence": 0.7 if report.objects else 0.2,
            "evidence": [f"Tracked objects: {detail}."],
        }

    if "caption" in q or "scene" in q or "see" in q:
        if report.multimodal_findings:
            top = report.multimodal_findings[0]
            return {
                "answer": top.description,
                "confidence": top.confidence,
                "evidence": top.evidence,
            }
        return _insufficient_evidence_answer(report)

    return {
        "answer": report.summary,
        "confidence": max(
            report.confidence_gate.confidence,
            report.hypotheses[0].confidence if report.hypotheses else 0.3,
        ),
        "evidence": (
            report.hypotheses[0].evidence
            if report.hypotheses
            else [report.confidence_gate.explanation]
        ),
    }
