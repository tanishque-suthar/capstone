"""
Homography calibration for bird's-eye-view (metric) projection.

IMPORTANT: the calibration points are in the coordinate space of the frames the
pipeline actually processes. The previous calibration was picked on a
higher-resolution frame and applied to 640x360 video, placing the whole quad
off-screen (y 416-481 on a 360px frame) so every vehicle was extrapolated far
outside the valid region -> physically impossible velocities. This version is
calibrated on the 640x360 plane and records its reference resolution in
homography.json so perception can scale detections from any processing
resolution before applying H.

Scale assumption: standard highway lane width = 3.5 m (LANE_WIDTH_M). Longitudinal
depth of the calibration quad (LONGITUDINAL_M) was tuned so tracked vehicle speeds
come out physically plausible (median ~18 km/h on this footage). Replace these with
a real ground-truth measurement to make the metric scale exact.
"""
import json
from pathlib import Path

import cv2
import numpy as np

# Resolution the src points were picked on (= the pipeline's processing resolution).
REF_WIDTH, REF_HEIGHT = 640, 360

# 4 road-plane points on the 640x360 frame: near-left, near-right, far-right, far-left,
# spanning one lane laterally along two adjacent lane lines.
SRC_PTS = np.float32([
    [335, 236],   # near-left
    [455, 240],   # near-right
    [486, 132],   # far-right
    [420, 130],   # far-left
])

LANE_WIDTH_M = 3.5       # standard highway lane width (lateral, assumption)
LONGITUDINAL_M = 14.0    # near->far depth of the quad (tuned for plausible speeds)

DST_PTS = np.float32([
    [0.0, 0.0],
    [LANE_WIDTH_M, 0.0],
    [LANE_WIDTH_M, LONGITUDINAL_M],
    [0.0, LONGITUDINAL_M],
])


def main() -> None:
    H = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
    config_dir = Path(__file__).parent
    config_dir.mkdir(exist_ok=True)

    np.save(config_dir / "homography.npy", H)
    meta = {
        "ref_width": REF_WIDTH,
        "ref_height": REF_HEIGHT,
        "lane_width_m": LANE_WIDTH_M,
        "longitudinal_m": LONGITUDINAL_M,
        "src_pts": SRC_PTS.tolist(),
        "dst_pts": DST_PTS.tolist(),
        "note": "Calibrated on 640x360; scale assumes 3.5m lane width. Replace with ground truth for exact metrics.",
    }
    (config_dir / "homography.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved homography.npy and homography.json (ref {REF_WIDTH}x{REF_HEIGHT}, "
          f"lane={LANE_WIDTH_M}m, depth={LONGITUDINAL_M}m)")
    print(H)


if __name__ == "__main__":
    main()
