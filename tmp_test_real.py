import cv2
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

# Do not downsample
print(f"Original frames: {len(frames)}")

print("--- USING botsort.yaml LOOP ---")
model = YOLO("yolo11n.pt")
track_ids_loop = set()
for i, frame in enumerate(frames):
    res = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0,1,2,3,5,7], verbose=False)
    if res[0].boxes and res[0].boxes.id is not None:
        ids = res[0].boxes.id.tolist()
        track_ids_loop.update(ids)
print(f"Total Unique IDs (Loop + botsort): {len(track_ids_loop)}")

print("--- USING botsort.yaml LIST ---")
model2 = YOLO("yolo11n.pt")
results = model2.track(frames, persist=True, tracker="botsort.yaml", classes=[0,1,2,3,5,7], verbose=False)
track_ids_list = set()
for res in results:
    if res.boxes and res.boxes.id is not None:
        ids = res.boxes.id.tolist()
        track_ids_list.update(ids)
print(f"Total Unique IDs (List + botsort): {len(track_ids_list)}")

print("--- USING bytetrack.yaml LOOP ---")
model3 = YOLO("yolo11n.pt")
track_ids_byte = set()
for i, frame in enumerate(frames):
    res = model3.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0,1,2,3,5,7], verbose=False)
    if res[0].boxes and res[0].boxes.id is not None:
        ids = res[0].boxes.id.tolist()
        track_ids_byte.update(ids)
print(f"Total Unique IDs (Loop + bytetrack): {len(track_ids_byte)}")
