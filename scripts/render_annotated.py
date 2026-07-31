"""
Render an annotated demo video for an event: overlay every tracked vehicle's
bounding box + Object_ID (+ class, speed) on the event's archival clip, marking
the anomaly-event window. Produces `{event_id}_annotated.mp4` — a preprocessed
video that visually shows the event and all vehicle IDs.

Usage: python scripts/render_annotated.py [EVENT_ID]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from app.config import settings


def _color(track_id: str) -> tuple:
    """Stable BGR color per Object_ID."""
    h = abs(hash(track_id))
    return (60 + h % 180, 60 + (h // 180) % 180, 60 + (h // 32400) % 180)


def render(event_id: str) -> Path:
    d = settings.paths.dataset_dir / event_id
    clip = d / f"{event_id}.mp4"
    csv = d / f"{event_id}_causal_data.csv"
    if not clip.exists() or not csv.exists():
        raise FileNotFoundError(f"Missing clip or CSV for {event_id}")

    df = pd.read_csv(csv)
    by_frame: dict[int, list] = {}
    for _, r in df.iterrows():
        by_frame.setdefault(int(r["Frame_ID"]), []).append(r)

    cap = cv2.VideoCapture(str(clip))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or settings.video.target_fps

    out_path = d / f"{event_id}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    fidx = 0
    n_boxes = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for r in by_frame.get(fidx, []):
            if any(pd.isna(r[c]) for c in ("BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2")):
                continue  # NaN-padded (large-gap) frame for this track
            x1, y1, x2, y2 = (int(r["BBox_X1"]), int(r["BBox_Y1"]),
                              int(r["BBox_X2"]), int(r["BBox_Y2"]))
            if x2 <= x1 or y2 <= y1:
                continue
            oid = str(r["Object_ID"])
            col = _color(oid)
            vel = r.get("Velocity_mps", float("nan"))
            label = f"{r['Class']} {oid}"
            if pd.notna(vel):
                label += f" | {float(vel):.1f}m/s"
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            n_boxes += 1

        # event banner
        t_rel = round(-settings.video.pre_buffer_seconds + fidx / fps, 1)
        cv2.rectangle(frame, (0, 0), (w, 22), (0, 0, 160), -1)
        cv2.putText(frame, f"EVENT {event_id}   t={t_rel:+.1f}s   tracks:{df.Object_ID.nunique()}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()
    print(f"Rendered {out_path} ({fidx} frames, {n_boxes} boxes, {df.Object_ID.nunique()} tracks)")
    return out_path


if __name__ == "__main__":
    event = sys.argv[1] if len(sys.argv) > 1 else "EVT_BE7DE8714797"
    render(event)
