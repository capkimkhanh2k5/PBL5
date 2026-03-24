import os
import cv2
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from torchvision import transforms
import numpy as np
from collections import deque
import time

# ============================================================
# 1. CẤU HÌNH CAMERA VÀ MODEL
# ============================================================
CAMERA_ID = 1  # Đã đổi sang ID 1 cho webcam ngoài
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "Train", "outputs", "best_model.pth"))

CLASS_ALIASES = {
    "battery": "Battery",
    "biological": "Biological",
    "general_waste": "General_Waste",
    "trash": "General_Waste",
    "glass": "Glass",
    "brown-glass": "Glass",
    "green-glass": "Glass",
    "white-glass": "Glass",
    "metal": "Metal",
    "paper_cardboard": "Paper_Cardboard",
    "paper-cardboard": "Paper_Cardboard",
    "paper": "Paper_Cardboard",
    "cardboard": "Paper_Cardboard",
    "plastic": "Plastic",
    "textiles": "General_Waste",
    "clothes": "General_Waste",
    "shoes": "General_Waste",
}


def normalize_class_name(class_name: str) -> str:
    key = str(class_name).strip().lower().replace(" ", "_")
    return CLASS_ALIASES.get(key, class_name)

BIN_GROUPS = {
    "ORGANIC": ["Biological"],
    "RECYCLABLE": ["Plastic", "Metal", "Paper_Cardboard", "Glass"],
    "Other": ["Battery", "General_Waste"]
}

# Định nghĩa lại Model (phải khớp với lúc train)
class WasteDetector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = mobilenet_v3_large(weights=None)
        self.features = base.features
        self.gem_pool = GeM(p=3)
        in_features = base.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.Hardswish(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.Hardswish(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.objectness = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.gem_pool(feat).flatten(1)
        return self.classifier(feat), self.objectness(feat)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1.0 / self.p)

# Load Model và Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint['classes']
model = WasteDetector(len(classes)).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# 2. KHỞI TẠO WEBCAM VÀ THÔNG SỐ XỬ LÝ
# ============================================================
cap = cv2.VideoCapture(CAMERA_ID)
# Cố định độ phân giải để đồng nhất với logic tính toán diện tích vật thể
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# State Machine
STATE_EMPTY, STATE_DETECTING, STATE_LOCKED = 0, 1, 2
state, empty_count, object_count = STATE_EMPTY, 0, 0
history = deque(maxlen=10)
locked_class, locked_bin = None, None
fps_history = deque(maxlen=30)

# ============================================================
# 3. VÒNG LẶP CHÍNH
# ============================================================
print(f"System ready on {device}. Camera ID: {CAMERA_ID}. Press 'q' to exit.")

try:
    while True:
        t_start = time.time()
        ret, frame_bgr = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated = frame_bgr.copy()

        # --- Lọc vật thể (Shadow & Background) ---
        mask = backSub.apply(frame_rgb, learningRate=(0.008 if state == STATE_EMPTY else 0))
        mask = np.where(mask == 255, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        h, w = mask.shape
        roi = mask[h//4:3*h//4, w//4:3*w//4] # Kiểm tra vùng trung tâm
        object_present = cv2.countNonZero(roi) > 1500

        # --- State Machine Logic ---
        if state == STATE_EMPTY:
            if object_present:
                object_count += 1
                if object_count >= 4:
                    state, history = STATE_DETECTING, deque(maxlen=10)
            else: object_count = 0
            cv2.putText(annotated, "STATUS: WAITING...", (10, 40), 2, 0.8, (200, 200, 200), 2)

        elif state == STATE_DETECTING:
            if not object_present:
                empty_count += 1
                if empty_count >= 10: state = STATE_EMPTY
            else:
                empty_count = 0
                # Phân loại vật thể
                tensor = transform(frame_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    conf, idx = probs.max(0)
                    top_class = normalize_class_name(classes[idx.item()])
                
                history.append(top_class if conf > 0.3 else "uncertain")
                
                # Biểu quyết (Voted Lock)
                votes = [c for c in history if c != "uncertain"]
                if votes:
                    candidate = max(set(votes), key=votes.count)
                    if votes.count(candidate) >= 7: # Đủ 7/10 frame giống nhau
                        locked_class = candidate
                        # Tìm thùng rác (Bin) tương ứng
                        locked_bin = "Other"
                        for b_name, group in BIN_GROUPS.items():
                            if locked_class in group:
                                locked_bin = b_name
                                break
                        state = STATE_LOCKED

                cv2.putText(annotated, f"DETECTING: {top_class} ({conf:.2f})", (10, 40), 2, 0.7, (0, 255, 0), 2)

        elif state == STATE_LOCKED:
            if not object_present:
                empty_count += 1
                if empty_count >= 10: 
                    state, locked_class, locked_bin = STATE_EMPTY, None, None
            else: empty_count = 0
            
            # --- HIỂN THỊ CẢ BIN VÀ CLASS KHI ĐÃ LOCK ---
            cv2.rectangle(annotated, (5, 5), (650, 80), (0, 0, 255), -1) # Vẽ nền đỏ cho thông báo
            cv2.putText(annotated, f"BIN: {locked_bin}", (20, 35), 2, 1.0, (255, 255, 255), 3)
            cv2.putText(annotated, f"CLASS: {locked_class}", (20, 70), 2, 0.8, (255, 255, 255), 2)

        # FPS & Show
        fps_history.append(1.0 / (time.time() - t_start + 1e-9))
        cv2.putText(annotated, f"FPS: {int(np.mean(fps_history))}", (10, h-10), 2, 0.6, (0, 255, 255), 1)
        
        cv2.imshow("WasteDetector - PC Test (Cam 1)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()