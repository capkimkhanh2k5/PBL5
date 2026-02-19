import cv2
import os
import numpy as np

# Point these to your training data
images_dir = '../data/train/images'
labels_dir = 'datasets/train/labels'

image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls, cx, cy, bw, bh = map(float, parts[:5])

        # Convert YOLO normalized → pixel coords
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        # Flag suspiciously large boxes
        box_area_ratio = bw * bh
        color = (0, 255, 0)  # Green = normal
        if box_area_ratio > 0.7:
            color = (0, 0, 255)  # Red = suspiciously large
            print(f"WARNING: Large box in {img_file} — covers {box_area_ratio*100:.1f}% of image")

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"cls:{int(cls)} {box_area_ratio*100:.0f}%", 
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Label Check (press any key for next, 'q' to quit)", img)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()