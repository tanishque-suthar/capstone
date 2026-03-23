# Track 1 Pipeline — FastAPI Implementation Plan

Implementation of the Track 1 data engineering pipeline (as defined in [context.md](file:///d:/projects/capstone/context.md)) as a modular FastAPI application.

## Project Structure

```
d:\projects\capstone\
├── app/
│   ├── __init__.py            # Package init
│   ├── main.py                # FastAPI app, lifespan, logging setup
│   ├── config.py              # All constants from context.md §5–§6
│   ├── database.py            # SQLite setup + CRUD helpers
│   ├── models.py              # Pydantic request/response schemas
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingestion.py       # Phase 0: MOG2 buffer, event trigger
│   │   ├── perception.py      # Phase 1: YOLO + BoT-SORT, homography, velocity
│   │   └── handoff.py         # Phase 2: Interpolation, crops, CSV, DB insert
│   └── routes/
│       ├── __init__.py
│       └── events.py          # REST API endpoints
├── config/                     # homography.npy (user-provided)
├── dataset/                    # Output: {Event_ID}/ directories
├── logs/                       # track1.log
├── context.md
└── requirements.txt
```

---

## Proposed Changes

### Configuration Layer

#### [NEW] [config.py](file:///d:/projects/capstone/app/config.py)
Central configuration dataclass containing all constants from context.md Sections 5–6:
- **Video**: clip duration (10s), pre-buffer (4s), post-trigger (6s), target FPS (10)
- **MOG2/Threshold**: warmup duration (60s), percentile (95), multiplier (1.5), floor (0.05), cooldown (15s)
- **YOLO**: model path (`yolo11n.pt`), confidence (0.35), class whitelist indices
- **Tracker**: BoT-SORT parameters (high/low/new thresholds, buffer, match thresh)
- **Interpolation**: max gap (10 frames)
- **Crop**: min area ratio (0.4)
- **Paths**: base dataset dir, config dir, log dir, DB path

---

### Database Layer

#### [NEW] [database.py](file:///d:/projects/capstone/app/database.py)
- `init_db()` — creates `event_registry.db` with `Master_Event_Log` table matching the schema in context.md §3.1
- `insert_event(event_id, trigger_time, video_path, csv_path, crops_dir, status)` — INSERT with retry + JSON fallback
- `get_event(event_id)` — single event lookup
- `list_events()` — returns all events
- `update_event_status(event_id, status)` — for marking "Failed"

---

### Pydantic Models

#### [NEW] [models.py](file:///d:/projects/capstone/app/models.py)
- `PipelineRequest` — `video_path: str` (path to input `.mp4`)
- `PipelineResponse` — `event_id: str, status: str, message: str`
- `EventDetail` — mirrors `Master_Event_Log` columns
- `EventList` — list wrapper

---

### Pipeline — Phase 0: Ingestion

#### [NEW] [ingestion.py](file:///d:/projects/capstone/app/pipeline/ingestion.py)
Implements `scan_for_events(video_path: str) -> list[EventFrameBlock]`:

1. Open video via `cv2.VideoCapture`, get source FPS
2. Create `collections.deque(maxlen=pre_buffer_frames)` for JPEG-compressed rolling buffer
3. Create MOG2 background subtractor
4. **Warmup** (first 60s): accumulate foreground ratios, compute adaptive threshold
5. **Scan loop**: for each frame, compute foreground ratio, check against threshold
6. **On trigger**: freeze the deque (4s pre-buffer), read 6s more frames → assemble 10s `EventFrameBlock` (list of JPEG buffers + metadata)
7. Enforce 15s cooldown before next trigger
8. Returns list of `EventFrameBlock` objects (typically 1 for prototype)

---

### Pipeline — Phase 1: Perception

#### [NEW] [perception.py](file:///d:/projects/capstone/app/pipeline/perception.py)
Implements `process_event(event_id: str, frame_block: EventFrameBlock) -> pd.DataFrame`:

1. Decode JPEG buffers → raw frames, downsample to exactly 10 FPS (100 frames)
2. Load YOLO model (`yolo11n.pt`), run `model.track()` with `persist=True` and BoT-SORT config
3. For each detection: extract `Object_ID`, `Class`, bounding box, bottom-center pixel
4. Load homography matrix from `config/homography.npy` (or `NaN` fallback)
5. Apply `cv2.perspectiveTransform` to bottom-center → `Pos_X_m`, `Pos_Y_m`
6. Calculate `Velocity_mps` = Euclidean distance between consecutive spatial positions / Δt
7. Build DataFrame with all columns from context.md §3.1 Item 4
8. Store per-detection confidence + frame for Phase 2 crop selection

---

### Pipeline — Phase 2: Handoff

#### [NEW] [handoff.py](file:///d:/projects/capstone/app/pipeline/handoff.py)
Implements `finalize_event(event_id: str, df: pd.DataFrame, frame_block, detections_meta) -> dict`:

1. **Interpolation**: For each Object_ID, find gaps ≤10 frames → linear interpolate BBox/Pos columns, recalculate velocity. Gaps >10 frames → NaN-pad.
2. **CSV export**: Write to `dataset/{Event_ID}/{Event_ID}_causal_data.csv`
3. **Video export**: Write 10s clip to `dataset/{Event_ID}/{Event_ID}.mp4` at 10 FPS via `cv2.VideoWriter`
4. **Crops**: For each Object_ID, pick highest-confidence frame (with 40% area filter), save `{Event_ID}_{Object_ID}_crop.jpg`
5. **DB insert**: Call `database.insert_event(...)` with all paths, status="Extracted"
6. Returns dict of paths for API response

---

### FastAPI Routes

#### [NEW] [events.py](file:///d:/projects/capstone/app/routes/events.py)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/pipeline/run` | Accepts `PipelineRequest`, runs full pipeline as a background task, returns `event_id` |
| `GET` | `/api/events` | Lists all events from registry |
| `GET` | `/api/events/{event_id}` | Returns event detail + file paths |
| `GET` | `/api/events/{event_id}/csv` | Returns causal CSV as `FileResponse` |
| `GET` | `/api/events/{event_id}/video` | Returns event video as `FileResponse` |

---

### Application Entry Point

#### [NEW] [main.py](file:///d:/projects/capstone/app/main.py)
- FastAPI app with lifespan handler (init DB, create directories, configure logging)
- Mount routes from `events.py`
- Logging setup: dual handler (stderr + `logs/track1.log`), format from context.md §6.6
- Health endpoint at `GET /api/health`

---

#### [MODIFY] [requirements.txt](file:///d:/projects/capstone/requirements.txt)
Add `fastapi` and `uvicorn` for the API server layer.

---

## Verification Plan

### Automated Smoke Test
1. Start the server:
   ```
   cd d:\projects\capstone
   uvicorn app.main:app --port 8000
   ```
2. Hit health endpoint:
   ```
   curl http://localhost:8000/api/health
   ```
   Expected: `{"status": "ok"}`

3. List events (empty):
   ```
   curl http://localhost:8000/api/events
   ```
   Expected: `{"events": []}`

### Manual Integration Test
> [!IMPORTANT]
> This requires a sample traffic video file. Please provide a test `.mp4` file path and (optionally) a `config/homography.npy` file. Without the homography, spatial columns will be `NaN` (graceful fallback per spec).

1. Start the server as above
2. Trigger pipeline:
   ```
   curl -X POST http://localhost:8000/api/pipeline/run -H "Content-Type: application/json" -d "{\"video_path\": \"path/to/test.mp4\"}"
   ```
3. Check response contains `event_id` and `status: "processing"`
4. Poll `GET /api/events/{event_id}` until status changes to `"Extracted"`
5. Verify output artifacts exist:
   - `dataset/{Event_ID}/{Event_ID}.mp4` (10s clip)
   - `dataset/{Event_ID}/{Event_ID}_causal_data.csv` (100 rows × 11 columns)
   - `dataset/{Event_ID}/entity_crops/*.jpg` (one per tracked object)
6. Download CSV via `GET /api/events/{event_id}/csv` and verify schema matches context.md §3.1
