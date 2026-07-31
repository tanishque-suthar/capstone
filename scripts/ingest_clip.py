"""
Ingest a pre-curated clip window directly as an event, bypassing the MOG2
warmup/trigger (Phase 0). For short incident clips where the event is already
known, this feeds a chosen window straight into perception + handoff.

Resamples the window to exactly target_fps (so sources whose fps isn't a clean
multiple of 10 keep correct 0.1s timing), builds an EventFrameBlock, and runs
process_event + finalize_event.

Usage: python scripts/ingest_clip.py "<video_path>" <trigger_sec> [pre] [post]
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app.config import settings
from app.database import init_db, insert_event
from app.pipeline.ingestion import EventFrameBlock, _encode_frame
from app.pipeline.perception import process_event
from app.pipeline.handoff import finalize_event


def ingest_window(video_path: str, trigger_sec: float,
                  pre: float | None = None, post: float | None = None) -> tuple[str, dict]:
    pre = settings.video.pre_buffer_seconds if pre is None else pre
    post = settings.video.post_trigger_seconds if post is None else post
    tfps = settings.video.target_fps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    def frame_at(t: float):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * src_fps))))
        ok, fr = cap.read()
        return fr if ok else None

    pre_frames, post_frames = [], []
    for k in range(int(round(pre * tfps))):
        fr = frame_at(trigger_sec - pre + k / tfps)
        if fr is not None:
            pre_frames.append(_encode_frame(fr))
    for k in range(int(round(post * tfps))):
        fr = frame_at(trigger_sec + k / tfps)
        if fr is not None:
            post_frames.append(_encode_frame(fr))
    cap.release()

    # source_fps = target_fps → process_event keeps every frame (already 10 fps, correct dt)
    block = EventFrameBlock(trigger_time_sec=trigger_sec, source_fps=float(tfps),
                            pre_frames=pre_frames, post_frames=post_frames)

    init_db()
    event_id = f"EVT_{uuid.uuid4().hex[:12].upper()}"
    insert_event(event_id, 0.0, "", "", "", 0.0, "Processing", video_path)
    result = process_event(event_id, block)
    out = finalize_event(event_id, result, trigger_sec, source_video_path=video_path)
    print(f"Ingested {event_id}: {len(pre_frames)} pre + {len(post_frames)} post frames "
          f"({result.df['Object_ID'].nunique()} tracks)")
    return event_id, out


if __name__ == "__main__":
    vp = sys.argv[1]
    trig = float(sys.argv[2]) if len(sys.argv) > 2 else 9.5
    ingest_window(vp, trig)
