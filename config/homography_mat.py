"""
Homography calibration for bird's-eye-view (metric) projection.

Homography is CAMERA-SPECIFIC: each camera/scene needs its own calibration. The
`src` points are in the coordinate space of the frames the pipeline processes
(640x360). Scale assumes a standard 3.5 m lane width; the longitudinal depth `L`
of the calibration quad is tuned so tracked vehicle speeds come out physically
plausible (median ~20 km/h on this footage). Swap in a real ground-truth distance
to make the metric scale exact.

Set ACTIVE to the scene being processed and rerun to regenerate homography.npy +
homography.json. (A proper multi-camera setup would store calibration per
Video_Source rather than a single active scene — future work.)
"""
import json
from pathlib import Path

import cv2
import numpy as np

REF_WIDTH, REF_HEIGHT = 640, 360

# src = near-left, near-right, far-right, far-left (pixels on the 640x360 frame),
# spanning one lane laterally along two adjacent lane lines.
SCENES = {
    # Original benign intersection (testVideo.mp4)
    "testvideo": {
        "src": [[335, 236], [455, 240], [486, 132], [420, 130]],
        "lane_width_m": 3.5, "longitudinal_m": 14.0,
    },
    # Cyberabad Traffic Police KAMMAN_CAM5 — incident clip (car hits motorcyclist)
    "kamman_cam5": {
        "src": [[250, 305], [370, 305], [415, 170], [365, 170]],
        "lane_width_m": 3.5, "longitudinal_m": 14.0,
    },
}
ACTIVE = "kamman_cam5"


def main() -> None:
    scene = SCENES[ACTIVE]
    src = np.float32(scene["src"])
    lw, L = scene["lane_width_m"], scene["longitudinal_m"]
    dst = np.float32([[0, 0], [lw, 0], [lw, L], [0, L]])
    H = cv2.getPerspectiveTransform(src, dst)

    config_dir = Path(__file__).parent
    np.save(config_dir / "homography.npy", H)
    (config_dir / "homography.json").write_text(json.dumps({
        "scene": ACTIVE, "ref_width": REF_WIDTH, "ref_height": REF_HEIGHT,
        "lane_width_m": lw, "longitudinal_m": L,
        "src_pts": scene["src"], "dst_pts": dst.tolist(),
        "note": f"Calibrated on 640x360 for scene '{ACTIVE}'; 3.5m lane assumption.",
    }, indent=2), encoding="utf-8")
    print(f"Saved homography for scene '{ACTIVE}' (ref {REF_WIDTH}x{REF_HEIGHT}, lane={lw}m, depth={L}m)")
    print(H)


if __name__ == "__main__":
    main()
