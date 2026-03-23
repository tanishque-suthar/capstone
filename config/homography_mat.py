import cv2
import numpy as np
from pathlib import Path

# 1. Define 4 points in the VIDEO (pixels)
# Use a tool like ImageJ or even Paint to find these (x, y) coordinates
src_pts = np.float32([
    [421, 424],   # Point 1 (Top-Left)
    [523, 416],  # Point 2 (Top-Right)
    [500, 473],  # Point 3 (Bottom-Right)
    [401, 481]    # Point 4 (Bottom-Left)
])

# 2. Define where those same 4 points should be in METERS (Bird's Eye View)
# Example: A rectangle that is 4 meters wide and 10 meters deep
dst_pts = np.float32([
    [0, 0],       # Point 1 maps to (0,0)
    [1.574, 0],       # Point 2 maps to 4m right
    [1.574, 2.805],      # Point 3 maps 10m down
    [0, 2.805]       # Point 4 maps to (0,10)
])

# 3. Compute the Homography Matrix
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 4. Save it to the config folder
config_dir = Path("d:/projects/capstone/config")
config_dir.mkdir(exist_ok=True)
np.save(config_dir / "homography.npy", H)

print(f"Matrix saved to {config_dir / 'homography.npy'}")
print(H)
