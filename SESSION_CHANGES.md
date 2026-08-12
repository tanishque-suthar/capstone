# Session Changelog

Summary of all code changes made in this working session, on branch `akshit/code-review`
(22 commits on top of `ca4c58e "multiple camera feeds ui"`). Organized by work area, with
the rationale and key findings behind each change.

**Headline:** the system went from Tracks 1 + 3 implemented to **all four tracks working
end-to-end on a real incident clip**, plus tracking-quality fixes, an OpenVINO backend,
live-feed ingestion, and kept-current docs.

---

## 1. Repository hygiene & version control

| Commit | Change |
|---|---|
| `62a4620` | Fixed `.gitignore`: the generic Python `lib/` rule was also matching `frontend/src/lib/`, silently excluding the frontend API client. Anchored to `/lib/` `/lib64/` and tracked the previously-unversioned `api.ts` (285 lines). |
| `49ebd96` | Removed dead duplicate `events/[id]/VideoAnnotator.tsx` + its CSS (a stale pre-`forwardRef` copy imported by nothing; both pages use `@/components/VideoAnnotator`). |
| `58c70d9` | Added the initial `CLAUDE.md` (architecture + workflow guide). |

**Also discovered (no code change):** the privacy-compliance feature is *not* alive on this
branch — its full implementation lives only on the unmerged branch `feature/privacy-compliance`;
this branch has only dead `/api/privacy/*` client stubs in `api.ts` and stale `.pyc` leftovers.

---

## 2. Duplicate / fragmented vehicle IDs (tracking quality)

| Commit | Change |
|---|---|
| `8bb01a0` | Refactored the MOG2 trigger logic out of `scan_for_events` into a reusable, push-driven `FrameEventDetector` state machine — shared by batch and (later) live ingestion. Behavior unchanged. |
| `6adc610` | Tuned BoT-SORT to reduce fragmentation: `match_thresh` 0.99→0.8 (0.99 was non-standard and caused ID switches), `new_track_thresh` 0.3→0.5, `track_high_thresh` 0.2→0.25; added an explicit `with_reid` flag (verified active but *no benefit* on this footage → left off). |
| `6d3b42c` | Added a **min-lifespan filter** (`InterpolationConfig.min_track_frames`, default 3) in `handoff.py` — drops ghost/fragment tracks from both the CSV and the crop set, using raw pre-interpolation observation counts. |

**Finding:** a 10s clip was producing **59 `Object_ID`s** (11 present in a single frame). Tuning
cut unique IDs ~24% on saved clips; the deterministic filter removes the ghost tail without
suppressing live detections.

---

## 3. RAG de-duplication (Track 3)

| Commit | Change |
|---|---|
| `1718d1f` | Made `ingest_event_crops` **idempotent** (deletes an event's existing rows before re-adding) and added **search-time de-duplication** (`_dedup_hits` collapses exact `(event,object)` repeats and near-identical embeddings, keeping the closest match). |
| `388f812` | Added `scripts/dedup_lancedb.py` one-time maintenance to collapse duplicate rows left by the old non-idempotent ingest. |

**Finding:** one event had been ingested **5×** (255 rows for 51 crops). Cleanup reduced the
table from **373 → 169 rows** (one per entity). Gotcha logged: this lancedb version's
`list_tables()` is unreliable for membership checks — use `open_table`/`table_names()`.

---

## 4. OpenVINO inference backend

| Commit | Change |
|---|---|
| `e16e519` | Added an OpenVINO FP16 backend for YOLO: `yolo.backend` toggle (default `openvino`, graceful fallback to `.pt`), `scripts/export_openvino.py`, and a **pinned `device`** (the CUDA-built torch otherwise mis-detects a GPU → `Invalid device id`). |
| `712f980` | Docs update. |

**Finding:** OpenVINO **FP16 is ~4.8× faster on CPU** (32.6 vs 6.8 FPS, `device=cpu` pinned),
at ~3% near-threshold detection jitter. FP32 OpenVINO was *slower* than PyTorch — rejected.
Dependency: `openvino`.

---

## 5. Kinematics quality — homography, robust velocity, depth-gate

| Commit | Change |
|---|---|
| `464751f` | **Fixed the homography resolution mismatch** — the calibration quad was picked at a higher resolution and sat entirely off-screen on 640×360 frames (0% of detections inside it), producing −100 m/s² garbage. Recalibrated on the 640×360 plane, recorded the reference resolution in `homography.json`, and made `perception.py` scale detections to it. Added **robust velocity** in `handoff.py` (Savitzky-Golay smoothing of world positions + a physical speed cap). |
| `deaa43e` | Added the **depth-gate** (`ProjectionConfig`): NaN out world positions outside the reliable region around the calibration quad, since monocular BEV extrapolates unreliably far from the calibrated area. |

**Findings:** the resolution bug was the root cause of implausible speeds; after fixing it,
median speeds landed ~20 km/h (plausible). Residual accel spikes are inherent monocular-BEV
far-field jitter — mitigated by depth-gate + smoothing + cap, but kinematics remain
*approximate* (documented and flagged downstream).

---

## 6. Track 2 — Causal Engine (PCMCI+)

| Commit | Change |
|---|---|
| `deaa43e` | New `app/pipeline/causal.py`: **target-centric PCMCI+** causal discovery. Selects a target (largest sustained speed drop among vehicles *following* someone), builds a compact relative-kinematic variable set (target/lead/nearest speed + gap), runs PCMCI+ (tigramite) to find lagged drivers of the target's speed. **Speed-primary** by design (BEV acceleration too noisy). Exposed via `/api/causal/*`; persists `causal_graph.json`. |
| `2378ff7` | Added `test_causal.py` regression test (proves the engine recovers a known lead→follower link at the correct lag) and refined target selection to prefer *followers*. |
| `efa3d3d` | Docs. |

**Validation:** correctly returns *no* inter-vehicle causality on benign free-flow traffic
(only autoregression — the right null result), and recovers injected causality on synthetic
data. Dependencies: `tigramite`, `scikit-learn`, `scipy`.

---

## 7. Camera Ingestion Step 2 — live feeds with hybrid indexing

| Commit | Change |
|---|---|
| `d45bdc0` | New `app/pipeline/monitor.py`: `FeedMonitor` (background thread per source) whose single read loop feeds every frame to `FrameEventDetector` (full event pipeline) **and** sampled frames to a new `ContinuousTracker` that indexes **every** tracked vehicle's best crop into LanceDB (hybrid all-vehicle corpus under a `FEED_{video_id}` bucket). `FeedManager` + `/api/feeds` start/stop/status; stream-vs-file auto-detection (reconnect vs stop-at-EOF); `RAG.index_vehicles`. |
| `93e3f3b` | Downscale incoming frames to the processing resolution in the monitor (bounds 4K memory, matches calibration) + `scripts/render_annotated.py` demo renderer. |
| `5f7f633` | Docs. |

**Verified end-to-end** on a long source: an event fired *inside* the monitor (`Extracted`,
CSV written) while **334 vehicles were indexed concurrently**. Caveats: RTSP reconnect and
per-monitor model memory are coded but only file-source-verified; no frontend UI yet.

---

## 8. Incident-clip pipeline (per-scene homography + direct ingest)

| Commit | Change |
|---|---|
| `ed633f5` | Made `config/homography_mat.py` hold **per-scene calibrations** (`SCENES` dict + `ACTIVE` selector) — homography is camera-specific — and added the `KAMMAN_CAM5` incident scene. Added `scripts/ingest_clip.py` to process a **pre-curated clip window directly as an event**, bypassing the 120s MOG2 warmup and resampling to exact 10fps (so 25fps sources keep correct timing). Fixed `render_annotated.py` to skip NaN-padded bbox rows. |

**Context:** enabled testing on a real 30s collision clip (a maroon car striking a
motorcyclist). Confirmed the pipeline runs on it and the collider is trackable.

---

## 9. Track 4 — LLM Situation-Report synthesis

| Commit | Change |
|---|---|
| `0165501` | New `app/pipeline/synthesis.py`: distils event metadata + per-entity kinematics + **SigLIP zero-shot colour attributes** (new `RAG.zero_shot_batch`) + the causal graph into a **text-only evidence packet** (no raw imagery ever reaches the LLM), then generates a SitRep via any OpenAI-compatible endpoint. Exposed via `/api/synthesis/*`; persists `sitrep.md`/`sitrep.json`. Also added `causal.max_plausible_speed_mps` so target selection rejects projection speed spikes (was picking a 120 km/h artifact → now picks the real collider). |
| `6c0b361` | Defaulted to **Google Gemini** (`gemini-flash-latest`, free tier — avoids model deprecation), auto-load a repo-root `.env`, `LLM_MODEL`/`LLM_BASE_URL` env overrides, raised `max_tokens` for thinking models, and surface the provider's error body on HTTP failures. |
| `2f12a04` | Added a **collision-indicator** signal: flags a vehicle that decelerates sharply *and* whose track terminates mid-window as a likely collision, and names the nearest entity at that instant. On the incident event it correctly flags maroon car `V_07` (track ends 0.4s) adjacent to person `V_41` (2.1 m). The prompt leads with this. |

**Result:** produces a grounded, honest SitRep — correctly identifies the collider, reads the
causal result as "impact, not a braking chain," and flags unreliable kinematics. Provider-agnostic
by design (point `base_url` at a local model for a fully-edge deployment).

---

## 10. Documentation

`CLAUDE.md` was kept current throughout (`58c70d9`, `712f980`, `efa3d3d`, `5f7f633`, `1b925c1`) —
it now documents all four tracks, the live-feed subsystem, the OpenVINO backend, the depth-gate,
per-camera homography, and the `$LLM_API_KEY`/`.env` requirement.

---

## Dependencies added this session
`openvino`, `scipy`, `scikit-learn`, `tigramite`.

## New config groups (`app/config.py`)
`ProjectionConfig` (depth-gate), `CausalConfig` (PCMCI+ + plausibility), `SynthesisConfig`
(Track 4 LLM), `FeedConfig` (live monitoring), plus `TrackerConfig.with_reid`,
`InterpolationConfig` velocity-smoothing/min-track fields, and `YOLOConfig` backend/device.

## New API routers
`/api/causal/*` (Track 2), `/api/feeds/*` (live ingestion), `/api/synthesis/*` (Track 4).

## Known limitations / future work
- Kinematics are approximate (eyeballed 3.5 m-lane homography; monocular BEV far-field jitter).
- Causal engine models *reactive braking chains*, not collisions (a collision is correctly
  reported as "no braking chain / impact").
- RTSP reconnect and event-during-monitoring only file-source-verified.
- No frontend for Track 2 / Track 4 (API-only).
- Track 4 uses a cloud LLM (Gemini) for now; the interface is provider-agnostic so it can be
  pointed at a local/edge model to become fully edge-deployed.
