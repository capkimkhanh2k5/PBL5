import os
os.environ["DISPLAY"] = ":0"

import cv2
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from torchvision import transforms
from picamera2 import Picamera2
import numpy as np
from collections import deque
import time

# ============================================================
# ===================== MODEL DEFINITION =====================
# ============================================================

class WasteDetector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base           = mobilenet_v3_large(weights=None)
        self.features  = base.features
        self.avgpool   = base.avgpool
        in_features    = base.classifier[0].in_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

        self.objectness = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat   = self.features(x)
        feat   = self.avgpool(feat)
        feat   = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        obj    = self.objectness(feat)
        return logits, obj


# ============================================================
# ======================= LOAD MODEL =========================
# ============================================================

checkpoint  = torch.load('model.pth', map_location='cpu')
classes     = checkpoint['classes']
num_classes = len(classes)

model = WasteDetector(num_classes)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print("Classes:", classes)

# ============================================================
# ======================= TRANSFORM ==========================
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# ========================= CAMERA ===========================
# ============================================================

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (720, 720)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

# ============================================================
# ================= BACKGROUND SUBTRACTOR ====================
# ============================================================

# detectShadows=True → MOG2 đánh dấu bóng = 127, foreground thực = 255
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=40,
    detectShadows=True
)

kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# ============================================================
# ================= PARAMETERS ===============================
# ============================================================

MIN_CONTOUR_AREA      = 4000
MIN_FOREGROUND_PIXELS = 1500

EMPTY_CONFIRM_FRAMES  = 10
OBJECT_CONFIRM_FRAMES = 4

# Lock: trong WINDOW_SIZE frame gần nhất,
# top-1 class phải xuất hiện >= LOCK_VOTES lần (conf >= MIN_CONF)
WINDOW_SIZE = 10
LOCK_VOTES  = 7
MIN_CONF    = 0.30      # nới lỏng xuống 20%

# Shadow filter bằng HSV:
# Bóng thường làm giảm V (brightness) nhưng giữ nguyên H, S
# Pixel nào có V tăng hoặc S tăng mạnh so với background → object thật
# Đây là ngưỡng độ sáng tối thiểu của vùng foreground (trung bình V)
# Nếu vùng FG quá tối → khả năng cao là bóng
SHADOW_V_THRESH = 40    # nếu mean(V) của vùng foreground < ngưỡng này → bóng

# ---- States ----
STATE_EMPTY     = 0
STATE_DETECTING = 1
STATE_LOCKED    = 2

state        = STATE_EMPTY
empty_count  = 0
object_count = 0

history      = deque(maxlen=WINDOW_SIZE)
locked_class = None
fps_history  = deque(maxlen=30)

# ============================================================
# ===================== HELPER FUNCTIONS =====================
# ============================================================

def detect_object(frame: np.ndarray) -> bool:
    """
    Phát hiện vật thể thực sự với 2 lớp lọc bóng:
      Lớp 1 - MOG2: loại pixel shadow (giá trị 127)
      Lớp 2 - HSV brightness: nếu vùng foreground quá tối
                               → khả năng cao là bóng đổ, bỏ qua
    """
    lr  = 0.008 if state == STATE_EMPTY else 0
    raw = backSub.apply(frame, learningRate=lr)

    # Lớp 1: chỉ giữ foreground thực (255), loại bóng MOG2 (127)
    mask = np.where(raw == 255, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

    # Vùng trung tâm
    h, w   = mask.shape
    cx, cy = w // 2, h // 2
    mx, my = w // 4, h // 4
    center_mask = mask[cy - my : cy + my, cx - mx : cx + mx]

    if cv2.countNonZero(center_mask) < MIN_FOREGROUND_PIXELS:
        return False

    contours, _ = cv2.findContours(center_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max((cv2.contourArea(c) for c in contours), default=0)
    if max_area <= MIN_CONTOUR_AREA:
        return False

    # Lớp 2: kiểm tra độ sáng HSV của vùng foreground
    # Bóng thật sự làm tối vùng đó → V thấp
    # Vật thể thật thường có màu sắc riêng → V cao hơn
    center_frame = frame[cy - my : cy + my, cx - mx : cx + mx]
    hsv          = cv2.cvtColor(center_frame, cv2.COLOR_RGB2HSV)
    v_channel    = hsv[:, :, 2]

    # Chỉ lấy pixel V tại vùng foreground
    fg_v_values = v_channel[center_mask > 0]
    if len(fg_v_values) == 0:
        return False

    mean_v = float(np.mean(fg_v_values))

    # Nếu vùng foreground quá tối → đây là bóng, không phải object
    if mean_v < SHADOW_V_THRESH:
        return False

    return True


def classify(frame: np.ndarray):
    """Trả về (class_name, confidence, objectness)."""
    tensor = transform(frame).unsqueeze(0)
    with torch.no_grad():
        logits, obj = model(tensor)
        probs       = torch.softmax(logits, dim=1)[0]
        objectness  = torch.sigmoid(obj).item()

    conf, idx = probs.max(0)
    return classes[idx.item()], conf.item(), objectness


def try_lock() -> str | None:
    """
    Trong WINDOW_SIZE frame gần nhất:
    - Frame conf >= MIN_CONF → ghi tên class
    - Frame conf < MIN_CONF  → ghi 'uncertain' (không tính vote)
    Lock nếu 1 class có >= LOCK_VOTES phiếu.
    """
    if len(history) < WINDOW_SIZE:
        return None

    real_votes = [c for c in history if c != "uncertain"]
    if not real_votes:
        return None

    candidate = max(set(real_votes), key=real_votes.count)
    if real_votes.count(candidate) >= LOCK_VOTES:
        return candidate
    return None


# ============================================================
# ========================= MAIN LOOP ========================
# ============================================================

print("System ready. Press 'q' to exit.")

try:
    while True:
        t_start   = time.time()
        frame     = picam2.capture_array()
        annotated = frame.copy()

        object_present = detect_object(frame)

        # =====================================================
        # ================= STATE MACHINE =====================
        # =====================================================

        # ── EMPTY ────────────────────────────────────────────
        if state == STATE_EMPTY:

            if object_present:
                object_count += 1
                if object_count >= OBJECT_CONFIRM_FRAMES:
                    state        = STATE_DETECTING
                    object_count = 0
                    history.clear()
                    print("→ DETECTING")
            else:
                object_count = 0

            cv2.putText(annotated, "Disk empty - waiting...",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (200, 200, 200), 2)

        # ── DETECTING ────────────────────────────────────────
        elif state == STATE_DETECTING:

            if not object_present:
                empty_count += 1
                if empty_count >= EMPTY_CONFIRM_FRAMES:
                    state       = STATE_EMPTY
                    empty_count = 0
                    history.clear()
                    print("→ EMPTY (no object)")
            else:
                empty_count = 0

                top_class, top_conf, objectness = classify(frame)

                # Luôn append để giữ đúng cửa sổ WINDOW_SIZE frame
                if top_conf >= MIN_CONF:
                    history.append(top_class)
                else:
                    history.append("uncertain")

                # Đếm vote của top_class hiện tại trong cửa sổ
                real_votes = [c for c in history if c != "uncertain"]
                cur_votes  = real_votes.count(top_class) if real_votes else 0

                print(f"class={top_class} conf={top_conf:.2f} obj={objectness:.2f} "
                      f"votes={cur_votes}/{LOCK_VOTES} window={len(history)}/{WINDOW_SIZE}")

                # Thử lock
                candidate = try_lock()
                if candidate is not None:
                    locked_class = candidate
                    state        = STATE_LOCKED
                    print(f"→ LOCKED: {locked_class}")

                label = (f"Detecting: {top_class} ({top_conf*100:.1f}%)  "
                         f"votes {cur_votes}/{LOCK_VOTES}  [{len(history)}/{WINDOW_SIZE}]")
                cv2.putText(annotated, label,
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 0), 2)

        # ── LOCKED ───────────────────────────────────────────
        elif state == STATE_LOCKED:
            # Không classify nữa — chỉ chờ disk trống

            if not object_present:
                empty_count += 1
                if empty_count >= EMPTY_CONFIRM_FRAMES:
                    state        = STATE_EMPTY
                    empty_count  = 0
                    history.clear()
                    locked_class = None
                    print("→ EMPTY (object removed)")
            else:
                empty_count = 0

            cv2.putText(annotated, f"LOCKED: {locked_class}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 0, 255), 4)

        # =====================================================
        # ======================== FPS ========================
        # =====================================================

        fps_history.append(1.0 / (time.time() - t_start + 1e-9))
        cv2.putText(annotated,
                    f"FPS: {np.mean(fps_history):.1f}",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("WasteDetector", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()