# Architecture Context: Traffic Intersection Causal Framework

## 1. Project Overview
The objective is to build a multimodal, edge-deployable intelligence framework that analyzes traffic intersection video. The system disaggregates raw video into kinematic data and semantic visual data, computes structural causal inferences (PCMCI+) to determine the root cause of traffic events (e.g., why a vehicle braked), and utilizes a text-only LLM to generate a Situation Report (SitRep).

To maintain mathematical rigor and reduce computational overhead, the architecture does not feed raw video to generative models for reasoning. It relies on a strict, decoupled data engineering pipeline.

## 2. System Architecture (The Four Tracks)
The project is divided into four tracks. **The current implementation phase is strictly focused on Track 1.**

* **Track 1 (Data Engineering):** Heuristic video ingestion, multi-object tracking, spatial transformation (Bird's Eye View), and generation of isolated temporal datasets.
* **Track 2 (Causal Engine):** Ingests tabular time-series data from Track 1 to build a causal graph using Tigramite/PCMCI+.
* **Track 3 (RAG Pipeline):** Ingests isolated entity image crops from Track 1, embedding them via SigLIP into LanceDB for semantic retrieval.
* **Track 4 (Synthesis):** Queries LanceDB and the Causal Engine, formatting the outputs into a text-only prompt for a standard LLM to generate the final report.

---

## 3. Current Implementation Task: Track 1 Pipeline

The immediate task is to write a Python script that acts as the perception and data structuring layer. It must read a local video file, isolate 10-second anomaly events, and output specific data artifacts required by Tracks 2, 3, and 4.

### 3.1. Required Storage Infrastructure
The script must populate four distinct storage layers per triggered event.

**1. The Event Registry (SQLite Database)**
* **File:** `event_registry.db`
* **Table:** `Master_Event_Log`
* **Schema:** `Event_ID` (PK), `Trigger_Time`, `Raw_Video_Path`, `Causal_CSV_Path`, `Crops_Dir_Path`, `Status` (Default: "Extracted").

**2. The Unstructured Data Lake (Local Disk)**
* **Structure:** `/dataset/{Event_ID}/`
* **Assets:** The 10-second raw `.mp4` video, the output CSV, and a `/entity_crops/` subdirectory.

**3. The Semantic Visual Assets (For Track 3)**
* **Format:** `.jpg` files saved in the `/entity_crops/` directory.
* **Constraint:** These are NOT full frames. They are isolated bounding box crops of tracked entities.
* **Naming Convention:** `{Event_ID}_{Object_ID}_crop.jpg`

**4. The Causal Time-Series Database (For Track 2)**
* **Format:** A strict `.csv` file saved as `{Event_ID}_causal_data.csv`.
* **Temporal Constraint:** Must be strictly downsampled to 10 FPS. Missing frames within an object's lifespan must be mathematically interpolated. Absent entities must be padded with `NaN` to maintain the uniform grid.
* **Schema:**
    * `Event_ID` (String)
    * `Timestamp` (Float: -4.0 to +6.0)
    * `Frame_ID` (Integer)
    * `Object_ID` (String: e.g., "V_02" from BoT-SORT)
    * `Class` (String: from YOLO)
    * `BBox_X1`, `BBox_Y1`, `BBox_X2`, `BBox_Y2` (Integers: 2D pixel coordinates)
    * `Pos_X_m`, `Pos_Y_m` (Floats: Bird's Eye View coordinates in meters)
    * `Velocity_mps` (Float: Calculated Euclidean velocity)

---

## 4. Execution Logic Flow

The Python script must implement the following sequential logic:

### Phase 0: Heuristic Ingestion
1.  Initialize a `collections.deque` to maintain a continuous 4-second rolling buffer of frames from a local `.mp4` file.
2.  Apply an OpenCV MOG2 background subtractor to calculate active pixel entropy.
3.  Upon breaching a predefined entropy threshold, trigger an event.
4.  Capture the 4-second buffer and the subsequent 6 seconds of video. Save this 10-second block to disk.

### Phase 1: Perception & Transformation
1.  Load the saved 10-second block and downsample it to exactly 10 FPS (100 frames).
2.  Pass frames through YOLOv11 for detection and BoT-SORT for persistent tracking (`Object_ID`).
3.  Extract the bottom-center pixel coordinate of each bounding box.
4.  Multiply this coordinate by a pre-calibrated Homography Matrix (`cv2.perspectiveTransform`) to output spatial meters (`Pos_X_m`, `Pos_Y_m`).
5.  Calculate `Velocity_mps` using the change in spatial meters over time ($\Delta t = 0.1s$).

### Phase 2: Handoff & Logging
1.  Apply mathematical interpolation for any temporarily occluded `Object_ID` to maintain the rigid 10 FPS time-series structure. Export the DataFrame to the CSV format defined in Section 3.
2.  For each unique `Object_ID`, extract the highest-confidence bounding box array slice and save it as a `.jpg` crop.
3.  Execute an SQL `INSERT` to log the completed event and all absolute file paths into the SQLite `Master_Event_Log`.

---

## 5. Calibration & Configuration

### 5.1. Homography Matrix
The Bird's Eye View transformation requires a 3×3 homography matrix that maps pixel coordinates to real-world meters. The matrix must be provided **per camera/intersection** and is assumed static for the duration of a video.

* **Calibration Method:** Manual four-point correspondence. Select four ground-plane points whose real-world positions (in meters) are known (e.g., lane markings, crosswalk corners). Use `cv2.getPerspectiveTransform(src_pts, dst_pts)` to compute the 3×3 matrix.
* **Storage:** Saved as a NumPy `.npy` file at `config/homography.npy`. Loaded once at script startup.
* **Fallback:** If no homography file exists, skip the spatial transform. Populate `Pos_X_m` and `Pos_Y_m` with `NaN` and log a warning. This allows the pipeline to still produce valid bounding-box and velocity-in-pixels data.
* **Validation:** After loading, transform a known calibration point and assert the output is within ±0.5 m of expected position. Fail fast with a clear error if validation fails.

### 5.2. Entropy Threshold (Event Trigger)
The MOG2 background subtractor produces a foreground mask each frame. The "active pixel entropy" is defined as the **foreground pixel ratio** (foreground pixels ÷ total pixels).

* **Warmup Period:** The first 60 seconds of video are used exclusively for warmup. During this window, record every frame's foreground ratio into a list.
* **Adaptive Threshold:** After warmup, compute the threshold as:
  ```
  threshold = percentile(warmup_ratios, 95) × 1.5
  ```
  This ensures the threshold adapts to the video's baseline activity level.
* **Minimum Absolute Floor:** The threshold must not fall below `0.05` (5% of pixels) to prevent triggering on camera noise.
* **Cooldown:** After an event triggers, enforce a **15-second cooldown** before the next event can trigger. This prevents overlapping captures.
* **No-Event Fallback:** If the video ends without any event triggering, log a warning and exit cleanly with status code 0, producing no output artifacts.

### 5.3. YOLO & BoT-SORT Configuration
* **Model:** `yolo11n.pt` (nano variant) — optimized for low-spec / edge hardware with minimal latency.
* **Class Whitelist:** Filter detections to COCO classes `[car, truck, bus, person, bicycle, motorcycle]`. All other classes are discarded before tracking.
* **Confidence Threshold:** `conf=0.35` — suppresses low-quality detections without losing distant vehicles.
* **Tracker:** Use Ultralytics' built-in BoT-SORT integration via `model.track()` with default parameters:
  * `track_high_thresh=0.3`
  * `track_low_thresh=0.05`
  * `new_track_thresh=0.4`
  * `track_buffer=30` (frames before a lost track is discarded)
  * `match_thresh=0.8`
* **Persistence:** Set `persist=True` across frames to maintain `Object_ID` continuity.

---

## 6. Runtime Policies

### 6.1. Rolling Buffer Memory Management
Raw frames must **not** be stored uncompressed in the deque.

* **Strategy:** Each frame is JPEG-encoded (`cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])`) before insertion into the deque. This reduces per-frame size from ~6 MB (1080p raw) to ~100–200 KB.
* **Expected Memory:** 4 seconds × 30 FPS × 150 KB ≈ **~18 MB** (vs. ~700 MB uncompressed).
* **Decode on Use:** Frames are decoded (`cv2.imdecode`) only when they are extracted for event processing.

### 6.2. Frame Pipeline (Eliminating Redundant I/O)
The 10-second event block follows a **write-once** strategy:

1. Phase 0 assembles the 10-second frame block **in memory** (list of JPEG-encoded buffers).
2. Phase 1 processes these in-memory frames directly (decode → YOLO → track → transform).
3. **After** Phase 1 completes successfully, the 10-second raw clip is written to disk as the archival `.mp4` using `cv2.VideoWriter` at 10 FPS (the downsampled rate).

This avoids the double I/O penalty of write-then-read.

### 6.3. Interpolation Policy for Occluded Tracks
* **Method:** Linear interpolation (`numpy.interp`) on each column independently (`BBox_X1`, `BBox_Y1`, ..., `Pos_X_m`, `Pos_Y_m`).
* **Maximum Gap:** 10 frames (1.0 second at 10 FPS). If an `Object_ID` is missing for more than 10 consecutive frames, the track is treated as **lost** — the gap is `NaN`-padded, not interpolated.
* **Velocity Recalculation:** After interpolation, `Velocity_mps` is recalculated from the interpolated spatial positions (not interpolated directly) to maintain physical consistency.

### 6.4. Entity Crop Selection
For each unique `Object_ID`, a single representative crop is saved:

* **Primary Criterion:** Select the frame where YOLO assigned the **highest confidence score** to that object.
* **Area Filter:** Discard candidate frames where the bounding box area is less than **40% of that object's maximum observed bounding box area**. This filters out heavily occluded or distant detections.
* **Fallback:** If all detections for an object fail the area filter, use the frame with the largest bounding box area regardless of confidence.

### 6.5. Error Handling
* **Corrupt/Unreadable Video:** If `cv2.VideoCapture.isOpened()` returns `False`, log the error and exit with code 1.
* **Video Shorter Than 10 Seconds:** If insufficient frames remain after a trigger to fill the 6-second post-buffer, capture whatever is available. Note the actual duration in a `Duration_s` column appended to the event registry row.
* **Model Inference Failure:** Wrap YOLO/tracker calls in try-except. On failure, log the frame number, skip the frame, and continue. If >20% of frames fail, abort the event and set `Status = "Failed"` in the registry.
* **SQLite Write Failure:** Retry once. If still failing, write the event metadata to a fallback JSON file at `dataset/{Event_ID}/event_meta.json`.

### 6.6. Logging & Observability
Use Python's `logging` module with the following configuration:

* **Log Level:** `INFO` by default, `DEBUG` via `--verbose` CLI flag.
* **Format:** `%(asctime)s | %(levelname)-7s | %(name)s | %(message)s`
* **Output:** Dual handler — `stderr` for console output, `logs/track1.log` for persistent log file.
* **Key Logged Metrics (INFO level):**
  * Warmup complete → computed threshold value
  * Event triggered → timestamp, entropy value
  * Phase 1 complete → unique tracks detected, total CSV rows
  * Phase 2 complete → crops saved, event ID registered

---

## 7. Dependencies

**Python version:** 3.10+

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥8.3 | YOLOv11 + BoT-SORT tracking |
| `opencv-python` | ≥4.8 | Video I/O, MOG2, homography |
| `numpy` | ≥1.24 | Array operations, interpolation |
| `pandas` | ≥2.0 | DataFrame construction, CSV export |
| `sqlite3` | stdlib | Event registry (built-in) |