import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import deque

model = YOLO('yolov8n.pt', task='detect')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

IOU_MATCH_THRESH = 0.3
HISTORY_FRAMES = 5
MIN_HITS = 3

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

class StableDetector:
    def __init__(self, history=HISTORY_FRAMES, min_hits=MIN_HITS):
        self.history = history
        self.min_hits = min_hits
        self.tracks = []

    def update(self, detections):
        matched = set()
        for det in detections:
            box = det[0]
            best_iou = IOU_MATCH_THRESH
            best_track = None
            for i, track in enumerate(self.tracks):
                if i in matched or len(track['history']) == 0:
                    continue
                last_box = track['history'][-1][0] if track['history'][-1] is not None else None
                if last_box is None:
                    continue
                iou = box_iou(box, last_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = i

            if best_track is not None:
                self.tracks[best_track]['history'].append(det)
                matched.add(best_track)
            else:
                self.tracks.append({'history': deque([det], maxlen=self.history)})
                matched.add(len(self.tracks) - 1)

        for i, track in enumerate(self.tracks):
            if i not in matched:
                track['history'].append(None)

        self.tracks = [t for t in self.tracks
                       if any(x is not None for x in t['history'])]

        stable = []
        for track in self.tracks:
            hits = [x for x in track['history'] if x is not None]
            if len(hits) >= self.min_hits:
                avg_box = np.array([h[0] for h in hits]).mean(axis=0).astype(int)
                latest = hits[-1]
                stable.append((avg_box, latest[1], latest[2], latest[3]))
        return stable

detector = StableDetector()
colors = {}
fps_history = deque(maxlen=30)

print("Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()

        results = model(
            frame,
            stream=True,
            conf=0.55,
            imgsz=320,
            max_det=20,
            iou=0.5,
            verbose=False
        )

        raw_dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int)
            ):
                raw_dets.append((box.astype(int), float(conf), cls_id, r.names[cls_id]))

        stable_dets = detector.update(raw_dets)

        annotated = frame.copy()
        for (box, conf, cls_id, cls_name) in stable_dets:
            x1, y1, x2, y2 = box
            if cls_id not in colors:
                np.random.seed(cls_id * 7)
                colors[cls_id] = tuple(np.random.randint(100, 255, 3).tolist())
            color = colors[cls_id]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1-lh-10), (x1+lw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # FPS counter
        fps_history.append(1 / (time.time() - t_start))
        fps = np.mean(fps_history)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Detections: {len(stable_dets)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Stable Detection Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()