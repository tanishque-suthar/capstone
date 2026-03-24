import cv2
import yaml
import tempfile
from ultralytics import YOLO

video_path = "dataset/EVT_2395AE3BD1EB/EVT_2395AE3BD1EB.mp4"
cap = cv2.VideoCapture(video_path)

frames = []
source_fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

# Use custom bytetrack YAML
custom_yaml_path = "custom_tracker.yaml"
with open(custom_yaml_path, "w") as f:
    yaml.dump({
        "tracker_type": "botsort",
        "track_high_thresh": 0.1,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.4,
        "track_buffer": 60,
        "match_thresh": 0.99,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.8,
        "with_reid": False,
        "model": "auto"
    }, f)


print(f"Tracking on {len(frames)} frames with custom YAML")
model = YOLO("yolo11n.pt")
results = model.track(frames, persist=True, tracker=custom_yaml_path, classes=[0,1,2,3,5,7], verbose=False)

track_ids = set()
for res in results:
    if res.boxes and res.boxes.id is not None:
        track_ids.update(res.boxes.id.tolist())
print(f"Total Unique IDs (Custom tracker): {len(track_ids)}")
