"""
XAI Layer — GradCAM + SHAP explainability for event detections.

GradCAM:  Hook-based gradient-weighted class activation mapping on the
          YOLO backbone.  Produces per-frame spatial heatmaps showing
          *why* detections fired in specific image regions.

SHAP:     KernelSHAP on the frame-level tabular feature matrix (produced
          by reasoning.py) to show which behavioural signals most
          influenced the anomaly-detection logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.config import settings

logger = logging.getLogger(__name__)


# ─── GradCAM ────────────────────────────────────────────────────────────────


class YOLOGradCAM:
    """
    Compute Grad-CAM heatmaps for a YOLO model.

    Attaches forward / backward hooks to a target convolutional layer,
    runs a forward pass for a single frame, backpropagates the maximum
    detection confidence score, and produces a spatial heatmap.
    """

    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks(target_layer_name)

    # ── hook wiring ──────────────────────────────────────────────────────

    def _register_hooks(self, layer_name: str) -> None:
        """Navigate the nn.Module tree and attach hooks."""
        target = self.model
        for part in layer_name.split("."):
            if part.isdigit():
                target = target[int(part)]
            else:
                target = getattr(target, part)

        self._hook_handles.append(
            target.register_forward_hook(self._save_activation)
        )
        self._hook_handles.append(
            target.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    # ── heatmap generation ───────────────────────────────────────────────

    def generate(self, frame: np.ndarray) -> np.ndarray:
        """
        Return an H×W float32 heatmap (0-1 range) for a BGR frame.

        Workflow:
            1.  Letterbox-resize + normalise identical to YOLO preprocess.
            2.  Forward pass with grads enabled.
            3.  Back-prop from the max detection-confidence score.
            4.  Grad-weight the activation maps → spatial heatmap.
            5.  ReLU + normalise → resize to original frame.
        """
        device = next(self.model.model.parameters()).device
        img_h, img_w = frame.shape[:2]

        # ── 1. Preprocess (match YOLO's 640×640 letterbox) ───────────
        input_size = 640
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (input_size, input_size))
        tensor = (
            torch.from_numpy(resized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div(255.0)
            .to(device)
        )
        tensor.requires_grad_(True)

        # ── 2. Forward pass ──────────────────────────────────────────
        self.model.model.eval()
        preds = self.model.model(tensor)

        # ── 3. Extract confidence target for backprop ────────────────
        # Ultralytics returns a tuple/list; first element is the
        # raw predictions tensor of shape (B, num_preds, 4+num_classes).
        if isinstance(preds, (list, tuple)):
            raw = preds[0]
        else:
            raw = preds

        # objectness / class scores sit after the first 4 bbox values
        if raw.dim() == 3:
            scores = raw[0, :, 4:]           # (num_preds, num_classes)
        else:
            scores = raw[:, 4:]

        target_score = scores.max()

        # ── 4. Backward ─────────────────────────────────────────────
        self.model.model.zero_grad()
        target_score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            logger.warning("GradCAM hooks did not capture data — returning blank heatmap")
            return np.zeros((img_h, img_w), dtype=np.float32)

        # ── 5. Compute heatmap ───────────────────────────────────────
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # normalise to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # resize back to original frame size
        heatmap = (
            F.interpolate(cam, size=(img_h, img_w), mode="bilinear", align_corners=False)
            .squeeze()
            .cpu()
            .numpy()
        )
        return heatmap.astype(np.float32)

    # ── cleanup ──────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


# ─── Frame selection ─────────────────────────────────────────────────────────


def _select_gradcam_frames(
    df: pd.DataFrame,
    anomalies: list[Any],
    max_frames: int = 5,
) -> list[int]:
    """
    Pick the most informative frames for GradCAM:
        1. Frame closest to trigger time (t ≈ 0).
        2. Frames where anomalies were detected.
        3. Frames with the highest detection count.
    """
    selected: set[int] = set()

    if df.empty:
        return []

    # 1 — trigger frame (nearest to t=0)
    trigger_row = df.iloc[(df["Timestamp"] - 0.0).abs().argsort()[:1]]
    if not trigger_row.empty:
        selected.add(int(trigger_row["Frame_ID"].iloc[0]))

    # 2 — anomaly frames
    frame_lookup = df[["Timestamp", "Frame_ID"]].drop_duplicates()
    for anomaly in anomalies:
        ts = float(getattr(anomaly, "timestamp_s", 0.0))
        match = frame_lookup.iloc[
            (frame_lookup["Timestamp"] - ts).abs().argsort()[:1]
        ]
        if not match.empty:
            selected.add(int(match["Frame_ID"].iloc[0]))
        if len(selected) >= max_frames:
            break

    # 3 — busiest frames (highest detection count)
    if len(selected) < max_frames:
        counts = df.groupby("Frame_ID").size().sort_values(ascending=False)
        for frame_id in counts.index:
            selected.add(int(frame_id))
            if len(selected) >= max_frames:
                break

    return sorted(selected)[:max_frames]


# ─── GradCAM overlay writer ─────────────────────────────────────────────────


def _save_gradcam_overlay(
    frame: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> None:
    """Blend heatmap over original frame and save as JPEG."""
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, colormap)
    overlay = cv2.addWeighted(frame, 1.0 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ─── SHAP ────────────────────────────────────────────────────────────────────


_SHAP_FEATURE_EXCLUDE = {"frame_id", "timestamp_s"}


def _compute_shap_values(
    feature_df: pd.DataFrame,
    anomaly_labels: pd.Series,
) -> tuple[np.ndarray, list[str]]:
    """
    Use KernelSHAP with a lightweight logistic-regression surrogate
    to explain which frame-level features contribute most to anomaly
    detection.

    Returns
    -------
    shap_values : (n_frames, n_features) array
    feature_names : corresponding column names
    """
    from sklearn.linear_model import LogisticRegression
    import shap

    feature_cols = [
        c for c in feature_df.columns if c not in _SHAP_FEATURE_EXCLUDE
    ]
    X = feature_df[feature_cols].values.astype(np.float64)
    y = anomaly_labels.values.astype(int)

    # Fit lightweight surrogate
    surrogate = LogisticRegression(max_iter=500, class_weight="balanced")
    surrogate.fit(X, y)

    # KernelSHAP
    n_bg = min(settings.xai.shap_background_samples, len(X))
    background = shap.sample(pd.DataFrame(X, columns=feature_cols), n_bg)
    explainer = shap.KernelExplainer(surrogate.predict_proba, background)
    raw_shap = explainer.shap_values(X)

    # raw_shap can be a list [class_0, class_1] or a 3D array (samples, features, classes).
    if isinstance(raw_shap, list):
        shap_vals = np.array(raw_shap[1])
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_vals = raw_shap[:, :, 1]
    else:
        shap_vals = np.array(raw_shap)

    return shap_vals, feature_cols


def _save_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    max_features: int = 10,
) -> None:
    """
    Save:
        1. A horizontal bar-chart PNG of mean |SHAP| per feature.
        2. A JSON with numeric importance values.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-max_features:][::-1]

    # ── bar chart ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [feature_names[int(i)] for i in top_idx][::-1]
    values = mean_abs[top_idx][::-1]

    bars = ax.barh(labels, values, color="#6366f1", edgecolor="#4f46e5", linewidth=0.6)

    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance — Anomaly Detection", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    fig.savefig(str(output_dir / "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── JSON ─────────────────────────────────────────────────────────
    importance = {
        feature_names[int(i)]: round(float(mean_abs[i]), 6) for i in top_idx
    }
    (output_dir / "shap_importance.json").write_text(
        json.dumps(importance, indent=2), encoding="utf-8"
    )


# ─── Orchestrator ────────────────────────────────────────────────────────────


def generate_xai_artifacts(
    event_id: str,
    frames: list[np.ndarray],
    perception_df: pd.DataFrame,
    anomalies: list[Any],
    feature_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Generate all XAI artifacts for a single event.

    Parameters
    ----------
    event_id      : Unique event identifier.
    frames        : Decoded video frames (BGR, from PerceptionResult).
    perception_df : The interpolated tracking DataFrame.
    anomalies     : AnomalyExplanation objects from reasoning pipeline.
    feature_df    : Frame-level feature matrix from reasoning._frame_level_features().
    output_dir    : Root event directory (e.g. dataset/EVT_XXX).

    Returns
    -------
    dict with keys:
        gradcam_dir, gradcam_frames, shap_plot, shap_json
    """
    cfg = settings.xai
    xai_dir = output_dir / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}

    # ── GradCAM ──────────────────────────────────────────────────────
    if cfg.gradcam_enabled and frames:
        gradcam_dir = xai_dir / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)

        selected = _select_gradcam_frames(
            perception_df, anomalies, cfg.max_gradcam_frames
        )
        logger.info(
            "GradCAM: selected frames %s for event %s", selected, event_id
        )

        # Reload a fresh YOLO model for grad computation
        # (avoids mutating the perception model's state)
        from ultralytics import YOLO

        model = YOLO(settings.yolo.model_name)
        cam = YOLOGradCAM(model, cfg.gradcam_target_layer)

        saved: list[str] = []
        for frame_idx in selected:
            if frame_idx >= len(frames):
                continue
            try:
                heatmap = cam.generate(frames[frame_idx])
                out_path = gradcam_dir / f"gradcam_frame_{frame_idx:04d}.jpg"
                _save_gradcam_overlay(
                    frames[frame_idx],
                    heatmap,
                    out_path,
                    cfg.gradcam_alpha,
                    cfg.gradcam_colormap,
                )
                saved.append(out_path.name)
            except Exception as exc:
                logger.warning(
                    "GradCAM failed on frame %d: %s", frame_idx, exc
                )

        cam.cleanup()
        result["gradcam_dir"] = str(gradcam_dir)
        result["gradcam_frames"] = saved
        logger.info("GradCAM: saved %d overlays for %s", len(saved), event_id)

    # ── SHAP ─────────────────────────────────────────────────────────
    if cfg.shap_enabled and not feature_df.empty:
        try:
            # Build a binary anomaly label from the feature matrix
            anomaly_cols = [
                c
                for c in ("abrupt_vehicle_stop", "synchronized_stop")
                if c in feature_df.columns
            ]
            if anomaly_cols:
                anomaly_labels = (
                    feature_df[anomaly_cols].sum(axis=1).clip(0, 1)
                )
            else:
                anomaly_labels = pd.Series(
                    np.zeros(len(feature_df)), dtype=float
                )

            if anomaly_labels.sum() >= 1:
                shap_vals, feat_names = _compute_shap_values(
                    feature_df, anomaly_labels
                )
                _save_shap_summary(
                    shap_vals, feat_names, xai_dir, cfg.shap_max_features
                )
                result["shap_plot"] = str(xai_dir / "shap_summary.png")
                result["shap_json"] = str(xai_dir / "shap_importance.json")
                logger.info("SHAP: saved summary for %s", event_id)
            else:
                logger.info(
                    "SHAP: no anomaly frames found in %s — skipping", event_id
                )
        except Exception as exc:
            logger.warning("SHAP computation failed (non-fatal): %s", exc)

    return result
