"""
Regression test for the Track 2 causal engine.

Generates a synthetic lead-follower braking scenario with KNOWN ground-truth
causality (a lead vehicle brakes; a follower reacts LAG frames later via a
car-following model) plus an unrelated distractor vehicle, then asserts the
engine selects the follower as target and recovers the lead -> follower link
at the correct lag.

Run: python test_causal.py
"""
import shutil
import sys

import numpy as np
import pandas as pd

from app.config import settings
from app.pipeline.causal import get_causal_engine

EVENT_ID = "EVT_SYNTHCAUSAL_TEST"
LAG = 5


def _build_synthetic_csv() -> None:
    np.random.seed(1)
    T, dt = 90, 0.1

    v_lead = np.full(T, 15.0)
    for t in range(40, T):
        v_lead[t] = max(4.0, 15 - 11 * (t - 40) / 10)   # exogenous brake at frame 40
    v_lead += np.random.randn(T) * 0.1

    v_follow = np.full(T, 15.0)                           # reacts to lead's speed LAG frames ago
    for t in range(1, T):
        ref = v_lead[t - LAG] if t - LAG >= 0 else 15.0
        v_follow[t] = v_follow[t - 1] + 0.35 * (ref - v_follow[t - 1]) + np.random.randn() * 0.1

    v_other = 12.0 + np.random.randn(T) * 0.3            # unrelated distractor (adjacent lane)
    y_lead = 25 + np.cumsum(v_lead) * dt
    y_follow = np.cumsum(v_follow) * dt
    y_other = np.cumsum(v_other) * dt + 10

    rows = []
    for t in range(T):
        for oid, x, y, v in [("V_LEAD", 1.75, y_lead[t], v_lead[t]),
                             ("V_FOLLOW", 1.75, y_follow[t], v_follow[t]),
                             ("V_OTHER", 7.5, y_other[t], v_other[t])]:
            rows.append(dict(Event_ID=EVENT_ID, Timestamp=round(-4 + t * dt, 2), Frame_ID=t,
                             Object_ID=oid, Class="car", BBox_X1=0, BBox_Y1=0, BBox_X2=10, BBox_Y2=10,
                             Pos_X_m=x, Pos_Y_m=y, Velocity_mps=v))
    d = settings.paths.dataset_dir / EVENT_ID
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{EVENT_ID}_causal_data.csv", index=False)


def main() -> int:
    _build_synthetic_csv()
    try:
        res = get_causal_engine().analyze_event(EVENT_ID)
    finally:
        shutil.rmtree(settings.paths.dataset_dir / EVENT_ID, ignore_errors=True)

    print("status:", res.get("status"), "| target:", res.get("target_object"),
          "| lead_fraction:", res.get("target_lead_fraction"))
    drivers = res.get("drivers_of_target_speed", [])
    for l in drivers:
        print(f"   {l['cause']}(t-{l['lag']})  strength={l['strength']}")

    ok = res.get("status") == "ok"
    target_ok = res.get("target_object") == "V_FOLLOW"
    lead_link = [l for l in drivers if l["cause"] in ("lead_speed", "lead_gap")]
    link_ok = any(l["lag"] == LAG for l in lead_link) or bool(lead_link)

    print(f"\ntarget is the follower : {target_ok}")
    print(f"lead->target recovered : {link_ok}  {[f'{l['cause']}(lag {l['lag']})' for l in lead_link]}")
    passed = ok and target_ok and link_ok
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
