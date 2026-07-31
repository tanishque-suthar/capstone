"""
Track 4: Synthesis — LLM-generated Situation Report (SitRep).

Distils an event's STRUCTURED outputs (event metadata + per-entity kinematics from
the causal CSV + SigLIP zero-shot attributes + the Track 2 causal graph) into a
text-only evidence packet, then asks an LLM to write a concise situation report.

Architectural constraint (context.md §1): NO raw imagery ever reaches the LLM —
everything is structured text derived by the perception/causal/embedding layers.

Provider-agnostic: any OpenAI-compatible /chat/completions endpoint (hosted now,
local/edge later). The API key is read from the environment (never stored in code);
without it, the evidence packet + prompt are still built and persisted, and the
report is generated as soon as a key is present.
"""
import json
import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.database import get_event

logger = logging.getLogger(__name__)

_COLORS = ["white", "black", "grey", "silver", "red", "blue",
           "green", "yellow", "orange", "maroon", "brown"]
_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def _event_paths(event_id: str):
    d = settings.paths.dataset_dir / event_id
    return d, d / f"{event_id}_causal_data.csv", d / "causal_graph.json", d / "entity_crops"


def _entity_attributes(event_id: str, crops_dir: Path, object_ids: list[str]) -> dict:
    """SigLIP zero-shot colour per entity (best-effort; text-only enrichment)."""
    from PIL import Image
    imgs, ids = [], []
    for oid in object_ids:
        p = crops_dir / f"{event_id}_{oid}_crop.jpg"
        if p.exists():
            try:
                imgs.append(Image.open(p).convert("RGB")); ids.append(oid)
            except Exception:
                pass
    if not imgs:
        return {}
    try:
        from app.pipeline.rag import get_rag_pipeline
        labels = get_rag_pipeline().zero_shot_batch(imgs, [f"a {c} vehicle" for c in _COLORS])
        return {oid: lab.split()[1] for oid, lab in zip(ids, labels)}  # "a red vehicle" -> "red"
    except Exception as exc:
        logger.warning("SigLIP attribute enrichment failed: %s", exc)
        return {}


def _build_evidence(event_id: str) -> dict | None:
    d, csv_path, causal_path, crops_dir = _event_paths(event_id)
    ev = get_event(event_id)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)

    cfg = settings.synthesis
    # ── per-entity kinematics ────────────────────────────────────────────────
    entities = []
    for oid, sub in df.groupby("Object_ID"):
        cls = str(sub["Class"].dropna().iloc[0]) if sub["Class"].notna().any() else "object"
        v = sub["Velocity_mps"].to_numpy(dtype=float)
        vv = v[np.isfinite(v)]
        if vv.size < cfg.min_entity_frames:
            continue
        plausible = settings.causal.max_plausible_speed_mps
        uncertain = bool(float(vv.max()) > plausible)   # projection/tracking spike
        vc = np.clip(vv, 0.0, plausible)                # clamp so spikes don't reach the LLM
        drop = float((np.maximum.accumulate(vc) - vc).max())
        entities.append({
            "object_id": oid, "class": cls,
            "frames_tracked": int(vv.size),
            "speed_max_kmh": round(float(vc.max()) * 3.6, 1),
            "speed_mean_kmh": round(float(vc.mean()) * 3.6, 1),
            "decelerated": bool(drop > 3.0 and not uncertain),
            "speed_uncertain": uncertain,
            "entry_s": round(float(sub["Timestamp"].min()), 1),
            "exit_s": round(float(sub["Timestamp"].max()), 1),
        })
    entities.sort(key=lambda e: -e["frames_tracked"])
    entities = entities[:cfg.max_entities]

    # ── attributes (colour) for vehicle entities ─────────────────────────────
    veh_ids = [e["object_id"] for e in entities if e["class"] in _VEHICLE_CLASSES]
    attrs = _entity_attributes(event_id, crops_dir, veh_ids)
    for e in entities:
        if e["object_id"] in attrs:
            e["colour"] = attrs[e["object_id"]]

    # ── scene summary ────────────────────────────────────────────────────────
    per_obj_class = df.drop_duplicates("Object_ID")["Class"]
    class_counts = per_obj_class.value_counts().to_dict()
    n_persons = int(class_counts.get("person", 0))
    n_vehicles = int(sum(c for k, c in class_counts.items() if k in _VEHICLE_CLASSES))

    # ── causal findings (Track 2) ────────────────────────────────────────────
    causal = None
    if causal_path.exists():
        try:
            cg = json.loads(causal_path.read_text(encoding="utf-8"))
            drivers = cg.get("drivers_of_target_speed", [])
            external = [l for l in drivers if l.get("cause") != "tgt_speed"]
            causal = {
                "target_object": cg.get("target_object"),
                "target_class": cg.get("target_class"),
                "target_speed_drop_mps": cg.get("target_speed_drop_mps"),
                "external_drivers": external,
                "interpretation": (
                    "No inter-vehicle reactive-braking chain detected — the target's "
                    "speed change is explained by its own past only (consistent with an "
                    "impact or an independent manoeuvre, not a following response)."
                    if not external else
                    "Reactive coupling detected: " + "; ".join(
                        f"{l['cause']} at lag {l['lag']} (strength {l['strength']})" for l in external)
                ),
            }
        except Exception as exc:
            logger.warning("Could not read causal graph: %s", exc)

    return {
        "event": {
            "event_id": event_id,
            "source": (ev or {}).get("Source_Video_Path"),
            "trigger_time_s": (ev or {}).get("Trigger_Time"),
            "duration_s": (ev or {}).get("Duration_s"),
        },
        "scene": {
            "vehicles_tracked": n_vehicles, "persons_tracked": n_persons,
            "class_counts": class_counts, "window_s": [-settings.video.pre_buffer_seconds,
                                                       settings.video.post_trigger_seconds],
        },
        "entities": entities,
        "causal": causal,
    }


def _format_evidence(e: dict) -> str:
    lines = []
    ev, sc = e["event"], e["scene"]
    lines.append(f"EVENT {ev['event_id']}  (source: {ev.get('source')})")
    lines.append(f"Window: {sc['window_s'][0]}s to +{sc['window_s'][1]}s around the trigger "
                 f"(t=0 is the flagged moment).")
    lines.append(f"Scene: {sc['vehicles_tracked']} vehicles + {sc['persons_tracked']} persons tracked. "
                 f"Classes: {sc['class_counts']}.")
    lines.append("")
    lines.append("ENTITIES (kinematics from monocular bird's-eye-view; approximate):")
    for en in e["entities"]:
        col = (en["colour"] + " ") if en.get("colour") else ""
        note = ", DECELERATED sharply" if en["decelerated"] else ""
        if en.get("speed_uncertain"):
            note += " (speed estimate unreliable)"
        lines.append(
            f"  {en['object_id']}: {col}{en['class']} — present {en['entry_s']}s..{en['exit_s']}s, "
            f"mean {en['speed_mean_kmh']} km/h, peak {en['speed_max_kmh']} km/h{note}")
    lines.append("")
    if e["causal"]:
        c = e["causal"]
        lines.append("CAUSAL ASSESSMENT (Track 2 / PCMCI+):")
        lines.append(f"  Target: {c.get('target_class')} {c.get('target_object')} "
                     f"(speed drop {c.get('target_speed_drop_mps')} m/s).")
        lines.append(f"  {c.get('interpretation')}")
    else:
        lines.append("CAUSAL ASSESSMENT: not available (causal analysis not run).")
    return "\n".join(lines)


_SYSTEM = (
    "You are a traffic-incident analyst. Write a concise, factual Situation Report (SitRep) "
    "from the STRUCTURED perception and causal-inference data provided. Use ONLY the data given — "
    "do not invent vehicles, colours, actions, or outcomes beyond it, and do not speculate about "
    "fault or injuries. No imagery is available to you; reason from the structured evidence. "
    "Kinematics are approximate (monocular estimation) — treat them as indicative and flag "
    "low-confidence points. Structure the report as: Summary; Entities Involved; Kinematic Timeline; "
    "Causal Assessment; Confidence & Caveats. Keep it under ~250 words."
)


def _call_llm(system: str, user: str) -> str | None:
    cfg = settings.synthesis
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        return None
    body = {
        "model": cfg.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


class SynthesisEngine:
    def generate_sitrep(self, event_id: str) -> dict:
        evidence = _build_evidence(event_id)
        if evidence is None:
            return {"status": "error", "message": f"No causal CSV for {event_id}"}

        user = _format_evidence(evidence)
        try:
            report = _call_llm(_SYSTEM, user)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return {"status": "llm_error", "message": str(exc),
                    "evidence": evidence, "prompt": user}

        d, *_ = _event_paths(event_id)
        d.mkdir(parents=True, exist_ok=True)
        status = "ok" if report else "no_key"
        (d / "sitrep.json").write_text(json.dumps(
            {"status": status, "evidence": evidence, "prompt": user, "report": report}, indent=2),
            encoding="utf-8")
        if report:
            (d / "sitrep.md").write_text(report, encoding="utf-8")

        return {"status": status, "event_id": event_id, "report": report,
                "evidence": evidence, "prompt": user,
                **({"message": f"No API key in ${settings.synthesis.api_key_env}; evidence packet built. "
                               "Set the key and re-run to generate the report."} if not report else {})}


_engine: SynthesisEngine | None = None


def get_synthesis_engine() -> SynthesisEngine:
    global _engine
    if _engine is None:
        _engine = SynthesisEngine()
    return _engine
