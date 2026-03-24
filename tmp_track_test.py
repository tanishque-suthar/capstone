import cv2
import numpy as np
from ultralytics import YOLO

def generate_test_frames():
    frames = []
    # Create a moving square that jumps by 20 pixels per frame (simulating low FPS)
    for i in range(20):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cx = 100 + i * 20
        cy = 100 + i * 20
        cv2.rectangle(img, (cx, cy), (cx + 50, cy + 50), (255, 255, 255), -1)
        frames.append(img)
    return frames

frames = generate_test_frames()

print("Using model.track in a loop:")
model1 = YOLO("yolov8n.pt")
for i, frame in enumerate(frames):
    res = model1.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
    if res[0].boxes and res[0].boxes.id is not None:
        print(f"Loop Frame {i}: ID {res[0].boxes.id.item()} BBox {res[0].boxes.xyxy[0].tolist()}")
    else:
        print(f"Loop Frame {i}: No ID")

print("\nUsing model.track on list of frames:")
model2 = YOLO("yolov8n.pt")
results = model2.track(frames, persist=True, tracker="botsort.yaml", verbose=False)
for i, res in enumerate(results):
    if res.boxes and res.boxes.id is not None:
        print(f"List Frame {i}: ID {res.boxes.id.item()} BBox {res.boxes.xyxy[0].tolist()}")
    else:
        print(f"List Frame {i}: No ID")
