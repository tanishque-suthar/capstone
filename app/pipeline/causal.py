"""
Track 2: Causal Engine — target-centric causal discovery over event kinematics.

Loads an event's causal CSV, identifies the event-defining vehicle (the largest
sustained speed drop = the one that "braked"), builds a compact set of
relative-kinematic variables around it, and runs PCMCI+ (tigramite) to find which
variables causally drive the target's speed changes, and at what time lag.

Speed-primary: on monocular BEV, speed is the reliable signal while acceleration is
a noisy derivative, so the variables are speeds and gaps — not accelerations.

NOTE: an event clip is ~100 timesteps, which is short for causal discovery. Results
are ranked hypotheses, not proof.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.database import get_event

logger = logging.getLogger(__name__)

MISSING = 999.0  # tigramite missing-value flag


def _load_event_df(event_id: str) -> pd.DataFrame | None:
    """Load an event's causal CSV, from the DB path or the conventional location."""
    ev = get_event(event_id)
    csv_path: Path | None = None
    if ev and ev.get("Causal_CSV_Path"):
        p = Path(ev["Causal_CSV_Path"])
        csv_path = p if p.exists() else None
    if csv_path is None:
        cand = settings.paths.dataset_dir / event_id / f"{event_id}_causal_data.csv"
        csv_path = cand if cand.exists() else None
    if csv_path is None:
        return None
    return pd.read_csv(csv_path)


def _object_series(df: pd.DataFrame) -> dict:
    """{Object_ID: DataFrame indexed by Frame_ID with position/velocity/class}."""
    out = {}
    for oid, sub in df.groupby("Object_ID"):
        out[oid] = sub.set_index("Frame_ID")[
            ["Pos_X_m", "Pos_Y_m", "Velocity_mps", "Class"]
        ].sort_index()
    return out


def _lead_present_fraction(target_oid: str, series: dict, frames: list, lane_tol: float) -> float:
    """Fraction of the target's valid frames that have a same-lane vehicle ahead."""
    s = series[target_oid]
    u = _forward_dir(s)
    tp_all = s.reindex(frames)[["Pos_X_m", "Pos_Y_m"]].to_numpy(dtype=float)
    others = {oid: o for oid, o in series.items() if oid != target_oid}
    valid = with_lead = 0
    for k, f in enumerate(frames):
        tp = tp_all[k]
        if not np.isfinite(tp).all():
            continue
        valid += 1
        for o in others.values():
            if f not in o.index:
                continue
            row = o.loc[f]
            op = np.array([row["Pos_X_m"], row["Pos_Y_m"]], dtype=float)
            if not np.isfinite(op).all():
                continue
            r = op - tp
            if float(r @ u) > 0 and abs(float(r[0] * u[1] - r[1] * u[0])) < lane_tol:
                with_lead += 1
                break
    return with_lead / valid if valid else 0.0


def _select_target(series: dict, min_len: int, frames: list, lane_tol: float):
    """
    Pick the analysis target: the vehicle whose braking can plausibly be *explained*.

    Prefers vehicles that are following someone (a lead is present ≥30% of their
    frames) — a reactor's speed drop can have an in-scene cause, whereas the
    frontmost braker cannot. Among the preferred pool, take the largest sustained
    speed drop. Falls back to the largest drop overall if nobody has a lead.
    """
    candidates = []
    for oid, s in series.items():
        v = s["Velocity_mps"].to_numpy(dtype=float)
        valid = np.isfinite(v)
        if valid.sum() < min_len:
            continue
        if not np.isfinite(s["Pos_X_m"].to_numpy(dtype=float)).any():
            continue  # need near-field (non-gated) positions
        vv = v[valid]
        if float(np.nanmax(vv)) > settings.causal.max_plausible_speed_mps:
            continue  # implausible peak speed → projection/tracking spike, not a real vehicle
        drop = float((np.maximum.accumulate(vv) - vv).max())
        lead_frac = _lead_present_fraction(oid, series, frames, lane_tol)
        candidates.append((oid, drop, lead_frac))

    if not candidates:
        return None
    followers = [c for c in candidates if c[2] >= 0.3]
    pool = followers if followers else candidates
    oid, drop, lead_frac = max(pool, key=lambda c: c[1])
    return oid, drop, lead_frac


def _forward_dir(s: pd.DataFrame) -> np.ndarray:
    """Unit vector of the target's overall travel direction, from first→last valid position."""
    p = s[["Pos_X_m", "Pos_Y_m"]].to_numpy(dtype=float)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 2:
        return np.array([0.0, 1.0])
    d = p[-1] - p[0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.array([0.0, 1.0])


def _build_variables(target_oid: str, series: dict, frames: list, lane_tol: float) -> dict:
    """
    Build target-centric time series over `frames`:
      tgt_speed  — target's speed (the effect)
      lead_gap   — forward distance to nearest same-lane vehicle ahead
      lead_speed — that lead vehicle's speed
      nn_gap     — distance to nearest vehicle (any direction)
      nn_speed   — that nearest vehicle's speed
    """
    tgt = series[target_oid].reindex(frames)
    u = _forward_dir(series[target_oid])
    tgt_pos = tgt[["Pos_X_m", "Pos_Y_m"]].to_numpy(dtype=float)
    tgt_speed = tgt["Velocity_mps"].to_numpy(dtype=float)

    n = len(frames)
    lead_gap = np.full(n, np.nan); lead_speed = np.full(n, np.nan)
    nn_gap = np.full(n, np.nan);   nn_speed = np.full(n, np.nan)
    others = {oid: s for oid, s in series.items() if oid != target_oid}

    for k, f in enumerate(frames):
        tp = tgt_pos[k]
        if not np.isfinite(tp).all():
            continue
        best_lead = (np.inf, np.nan)
        best_nn = (np.inf, np.nan)
        for s in others.values():
            if f not in s.index:
                continue
            row = s.loc[f]
            op = np.array([row["Pos_X_m"], row["Pos_Y_m"]], dtype=float)
            if not np.isfinite(op).all():
                continue
            osp = float(row["Velocity_mps"]) if np.isfinite(row["Velocity_mps"]) else np.nan
            r = op - tp
            dist = float(np.linalg.norm(r))
            if dist < best_nn[0]:
                best_nn = (dist, osp)
            fwd = float(r @ u)                          # forward component (ahead > 0)
            lat = abs(float(r[0] * u[1] - r[1] * u[0]))  # lateral offset
            if fwd > 0 and lat < lane_tol and fwd < best_lead[0]:
                best_lead = (fwd, osp)
        if np.isfinite(best_lead[0]):
            lead_gap[k], lead_speed[k] = best_lead
        if np.isfinite(best_nn[0]):
            nn_gap[k], nn_speed[k] = best_nn

    return {"tgt_speed": tgt_speed, "lead_gap": lead_gap, "lead_speed": lead_speed,
            "nn_gap": nn_gap, "nn_speed": nn_speed}


def _assemble(cols: dict) -> tuple[np.ndarray, list[str]]:
    """
    Keep tgt_speed plus variables that are present ≥50% and non-constant; restrict to
    frames where tgt_speed is finite; interpolate short gaps; flag residual missing.
    """
    keep_rows = np.isfinite(cols["tgt_speed"])
    names, arrays = [], []
    for name, arr in cols.items():
        a = arr[keep_rows]
        finite = np.isfinite(a)
        if name != "tgt_speed":
            if finite.mean() < 0.5:
                continue
            vals = a[finite]
            if vals.size and np.nanstd(vals) < 1e-6:
                continue  # constant → ParCorr can't use it
        names.append(name)
        arrays.append(a)

    d = pd.DataFrame(np.column_stack(arrays), columns=names)
    d = d.interpolate(limit=5, limit_direction="both").fillna(MISSING)
    return d.to_numpy(dtype=float), names


class CausalEngine:
    """Stateless target-centric causal analysis over a single event's CSV."""

    def analyze_event(self, event_id: str) -> dict:
        df = _load_event_df(event_id)
        if df is None or df.empty:
            return {"status": "error", "message": f"No causal CSV for {event_id}"}

        cfg = settings.causal
        series = _object_series(df)
        frames = sorted(int(f) for f in df["Frame_ID"].unique())
        target = _select_target(series, cfg.min_series_len, frames, cfg.lane_tolerance_m)
        if target is None:
            return {"status": "no_target",
                    "message": "No object with enough valid near-field data to analyze."}
        target_oid, drop, lead_frac = target

        cols = _build_variables(target_oid, series, frames, cfg.lane_tolerance_m)
        data, names = _assemble(cols)

        if data.shape[0] < cfg.min_series_len or len(names) < 2:
            return {"status": "insufficient",
                    "message": f"Only {data.shape[0]} usable timesteps / {len(names)} variables."}

        # ── PCMCI+ (lazy import to keep server startup light) ────────────────
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr

        dataframe = pp.DataFrame(data, var_names=names, missing_flag=MISSING)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
        res = pcmci.run_pcmciplus(tau_max=cfg.tau_max, pc_alpha=cfg.pc_alpha)
        graph, val = res["graph"], res["val_matrix"]

        tj = names.index("tgt_speed")
        links = []
        for i in range(len(names)):
            for tau in range(graph.shape[2]):
                if i == tj and tau == 0:
                    continue
                if graph[i, tj, tau] == "-->":
                    links.append({"cause": names[i], "lag": int(tau),
                                  "strength": round(float(val[i, tj, tau]), 3)})
        links.sort(key=lambda l: -abs(l["strength"]))

        cls = series[target_oid]["Class"].dropna()
        result = {
            "status": "ok",
            "event_id": event_id,
            "target_object": target_oid,
            "target_class": str(cls.iloc[0]) if not cls.empty else None,
            "target_speed_drop_mps": round(drop, 2),
            "target_lead_fraction": round(lead_frac, 2),
            "variables": names,
            "n_timesteps": int(data.shape[0]),
            "tau_max": cfg.tau_max,
            "drivers_of_target_speed": links,
            "note": "PCMCI+ over a short (~100-step) clip — treat links as ranked hypotheses, not proof.",
        }
        self._persist(event_id, result)
        return result

    @staticmethod
    def _persist(event_id: str, result: dict) -> None:
        out = settings.paths.dataset_dir / event_id / "causal_graph.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Causal graph for %s: %d driver(s) of target speed",
                    event_id, len(result.get("drivers_of_target_speed", [])))


_engine: CausalEngine | None = None


def get_causal_engine() -> CausalEngine:
    global _engine
    if _engine is None:
        _engine = CausalEngine()
    return _engine
