from ultralytics import YOLO
import cv2
import sys

# Try without specifying task, let YOLO auto-detect
model = YOLO('best.pt')
print("Model task:", model.task)  # This will tell us what the model actually is

image_path = "test_img/2.jpg"
img = cv2.imread(image_path)

results = model(img, imgsz=320, verbose=True)

for r in results:
    print("Result type:", type(r))
    print("Boxes:", r.boxes)
    print("Probs:", r.probs)