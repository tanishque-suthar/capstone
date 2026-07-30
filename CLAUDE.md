# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A video surveillance analysis platform. `context.md` is the authoritative design spec: the full system is a four-track "Traffic Intersection Causal Framework," but **the implemented codebase is Track 1 (data engineering) plus Track 3 (RAG search)**. Tracks 2 (PCMCI+ causal engine) and 4 (LLM synthesis) are specified in `context.md` but not yet built. When touching pipeline behavior, cross-check `context.md` — the source often cites it (e.g. `Reference: context.md §4 Phase 1`), and its schemas/constants are treated as contract.

## Commands

Backend (from repo root; Windows PowerShell — activate `.venv\Scripts\activate`):
```bash
uvicorn app.main:app --reload --port 8000
```
Frontend (from `frontend/`):
```bash
npm run dev        # dev server on :3000
npm run build
npm run lint       # eslint (next lint config)
```

There is **no pytest suite**. The `test_*.py` and `debug_*.py` files at the repo root are standalone scripts run directly, and `tmp_*.py` are scratch scripts:
```bash
python test_rag.py            # verifies SigLIP + LanceDB init and text embedding
python test_pipeline_api.py   # hits a running server on :8000 (health, pipeline 404, events)
python debug_siglip.py        # SigLIP loading diagnostics
python config/homography_mat.py   # regenerate config/homography.npy from hardcoded point pairs
```

API docs at `http://localhost:8000/docs`. Dashboard at `http://localhost:3000`.

## Architecture

**Three-phase pipeline, run as a FastAPI background task.** A `POST /api/pipeline/run` with an absolute `video_path` inserts a `Processing` event row synchronously (so the frontend can poll it without a 404), then `_run_pipeline` in [app/routes/events.py](app/routes/events.py) executes the phases sequentially in the background:

1. **Phase 0 — Ingestion** ([app/pipeline/ingestion.py](app/pipeline/ingestion.py)): MOG2 background subtraction over the whole video. First 120s is a warmup that sets an adaptive foreground-ratio threshold; breaching it triggers a 10s event clip (4s pre-buffer from a rolling `deque` + 6s post-trigger), with a cooldown between events. Frames are JPEG-encoded in memory. **Only the first triggered event is processed** (prototype limitation).
2. **Phase 1 — Perception** ([app/pipeline/perception.py](app/pipeline/perception.py)): YOLOv11 (`yolo11n.pt`) + BoT-SORT tracking on every decoded frame, but records are kept only for downsampled target frames (source FPS → 10 FPS). Bottom-center of each bbox is projected to bird's-eye-view meters via the homography matrix, then velocity is computed per track. Returns a flat DataFrame + per-detection metadata + decoded frames.
3. **Phase 2 — Handoff** ([app/pipeline/handoff.py](app/pipeline/handoff.py)): interpolates occluded tracks (gaps ≤ `max_gap_frames` linearly interpolated, larger gaps NaN-padded to keep a uniform 100-frame grid), writes the causal CSV, encodes an archival `.mp4`, saves the single best crop per object, and updates the event row to `Extracted`.

**RAG (Track 3)** ([app/pipeline/rag.py](app/pipeline/rag.py)) is decoupled from the pipeline and lazy-initialized (weights load on first use, not import). `POST /api/rag/ingest/{event_id}` embeds that event's entity crops via SigLIP into LanceDB; `POST /api/rag/search` embeds a text query and does vector search. The pipeline does **not** auto-ingest into RAG — ingestion is a separate explicit call.

**Three storage layers**, all rooted at repo root via `PathConfig`:
- `event_registry.db` — SQLite. `Master_Event_Log` (events) + `Video_Sources` (camera feeds). WAL mode. `init_db()` runs idempotent `ALTER TABLE` migrations and back-fills on every startup.
- `dataset/{Event_ID}/` — per-event `.mp4`, `{Event_ID}_causal_data.csv`, and `entity_crops/{Event_ID}_{Object_ID}_crop.jpg`.
- `dataset/lancedb/` — LanceDB vector table `entity_crops`.

**Config** ([app/config.py](app/config.py)): a single frozen-dataclass `settings` singleton aggregating all tuning constants (video timing, MOG2 thresholds, YOLO classes, tracker params, paths). Change constants here, not inline. Note the tracker params here are duplicated: `perception.py` regenerates `config/custom_tracker.yaml` from `settings.tracker` on every run, so the root-level and `config/` yaml files are overwritten artifacts, not source of truth.

## Conventions and gotchas

- **Video paths must be absolute** on the host machine — the pipeline reads local files directly; there is no upload/storage of the source video.
- **Windows-first environment.** Paths, `ffmpeg` on PATH, and C++ build tools (for `lapx`/`sentencepiece`) are assumed. YOLO + SigLIP weights (~1GB) download on first run.
- The frontend `NEXT_PUBLIC_API_URL` defaults to `http://localhost:8000`; backend CORS is fully open (dev only).
- `frontend/src/lib/api.ts` declares privacy/audit endpoints (`/api/privacy/*`) that have **no backend implementation** — treat those client functions as unwired.
- IDs: events are `EVT_<12 hex>`, sources are `VID_<8 hex>`, tracked objects are `V_<nn>`.

## Frontend caveat (important)

`frontend/AGENTS.md` warns: this is **Next.js 16** (App Router, React 19), which has breaking changes from earlier versions. Before writing frontend code, read the relevant guide under `frontend/node_modules/next/dist/docs/` rather than relying on prior Next.js knowledge.
