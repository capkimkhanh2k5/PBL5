"""
TrainReal.py  —  v1.0  (Human-in-the-Loop Active Learning)
===========================================================
Mục đích: Thu thập dữ liệu thực tế có nhãn chính xác từ camera để phục vụ
          re-training model. Mỗi lần AI chốt kết quả (STATE_LOCKED), hệ
          thống dừng hình và hỏi người dùng: Đúng hay Sai?

Dựa trên: testPC.py v3.0 (ONNX + AGC + Square Crop)

THAY ĐỔI SO VỚI testPC.py v3.0
──────────────────────────────────────────────────────────────────────────────
[HITL]    Thêm state STATE_REVIEW giữa LOCKED và COOLDOWN.
          Video đóng băng, chờ phím Y/N từ người dùng.
          Phím Y: chốt nhãn AI → lưu.
          Phím N: hiển thị menu 1-7 để sửa nhãn → lưu.
          Phím S (Skip): bỏ qua mẫu này, không lưu.

[SAVE]    Lưu ảnh RAW (frame_bgr chưa qua AGC) dạng hình VUÔNG
          (square crop đúng như pipeline inference) vào:
              DataRealTest/<ClassName>/<class>_<timestamp_ms>.jpg
          KHÔNG lưu ảnh đã AGC — Train pipeline đã có RandomAdaptiveGammaJitter
          tự xử lý, lưu ảnh AGC sẽ gây double-processing.

[UI]      Overlay review hiển thị:
          - Ảnh crop vuông phóng to bên trái màn hình
          - Banner lớn: BIN / CLASS / Confidence / Objectness / Gamma
          - Hướng dẫn phím rõ ràng
          - Thanh xác nhận màu (xanh=Đúng, đỏ=Sai, vàng=Skip)

[STATS]   Thống kê session theo thời gian thực (góc phải):
          - Tổng mẫu đã review
          - Tỉ lệ AI đoán đúng (real-world accuracy)
          - Số mẫu đã lưu theo từng class

[LOG]     Ghi log CSV: DataRealTest/session_log.csv
          Mỗi dòng: timestamp, filename, ai_class, final_class, correct, conf, obj, gamma
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import cv2
import onnxruntime as ort
import numpy as np
from torchvision import transforms
import torch
import json
import math
import os
import csv
from collections import deque, defaultdict
import time
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.dirname(SCRIPT_DIR)

# ============================================================
# 1. CẤU HÌNH — GIỮ NGUYÊN testPC.py v3.0
# ============================================================

CAMERA_ID  = 0
ONNX_PATH  = os.path.join(MODEL_DIR, "Train", "outputs", "waste_detector.onnx")
META_PATH  = os.path.join(MODEL_DIR, "Train", "outputs", "model_meta.json")
IMG_SIZE   = 384

N_TTA          = 1
INFER_EVERY_N  = 2

CONF_THRESH    = 0.82
OBJ_THRESH     = 0.35

VOTE_WINDOW      = 10
VOTE_MIN         = 5
DOMINANT_RATIO   = 0.65
VOTE_TIMEOUT_SEC = 10.0

WARMUP_SEC            = 3.0
OBJECT_CONFIRM_FRAMES = 2
EMPTY_DETECT_FRAMES   = 12
EMPTY_LOCKED_FRAMES   = 15
MIN_LOCK_HOLD_FRAMES  = 15
COOLDOWN_FRAMES       = 20

ROI_X1_RATIO = 0.18
ROI_X2_RATIO = 0.82
ROI_Y1_RATIO = 0.18
ROI_Y2_RATIO = 0.82

MOG2_PIXEL_THRESH   = 1000
DIFF_PIXEL_THRESH   = 800
DIFF_GRAY_THRESH    = 18
CROP_PAD            = 40

STABILITY_DIFF_THRESH   = 8.0
STABLE_FRAMES_REQUIRED  = 4
VOTE_DELAY_SEC          = 2.0

CONTOUR_AREA_THRESH = 3000
FROZEN_DIFF_THRESH  = 12.0

EMA_ALPHA           = 0.35
SNAPSHOT_EMA_ALPHA  = 0.05

BIN_GROUPS = {
    "ORGANIC":    ["Biological"],
    "RECYCLABLE": ["Plastic", "Metal", "Paper_Cardboard", "Glass"],
    "HAZARDOUS":  ["Battery"],
    "OTHER":      ["General_Waste"],
}
BIN_COLORS = {
    "ORGANIC":    (34, 139, 34),
    "RECYCLABLE": (30, 144, 255),
    "HAZARDOUS":  (0, 0, 220),
    "OTHER":      (100, 100, 100),
}

# ============================================================
# 2. CẤU HÌNH THU THẬP DỮ LIỆU  (Human-in-the-Loop)
# ============================================================

DATA_DIR    = os.path.join(SCRIPT_DIR, "DataRealTest")           # Thư mục gốc lưu dữ liệu
LOG_FILE    = os.path.join(DATA_DIR, "session_log.csv")

# Chất lượng JPEG lưu ảnh (95 = giữ tối đa chi tiết cho training)
SAVE_JPEG_QUALITY = 95

# ============================================================
# 3. ĐỌC MODEL META
# ============================================================

_meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        _meta = json.load(f)
    print(f"[INFO] Loaded meta: {META_PATH}")
else:
    print(f"[WARN] {META_PATH} not found — dùng giá trị mặc định")

classes    = _meta.get('classes', ['Battery','Biological','General_Waste',
                                    'Glass','Metal','Paper_Cardboard','Plastic'])
img_size   = _meta.get('img_size',   IMG_SIZE)
AGC_TARGET = _meta.get('agc_target', 128)
AGC_MIN    = _meta.get('agc_gamma_min', 0.4)
AGC_MAX    = _meta.get('agc_gamma_max', 3.0)

NUM_CLASSES = len(classes)

print(f"[INFO] Classes   : {classes}")
print(f"[INFO] img_size  : {img_size}")
print(f"[INFO] AGC       : target={AGC_TARGET}  clip=[{AGC_MIN}, {AGC_MAX}]")

# ============================================================
# 4. KHỞI TẠO THƯ MỤC  DataRealTest/<ClassName>/
# ============================================================

os.makedirs(DATA_DIR, exist_ok=True)
for c in classes:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)
print(f"[INIT] DataRealTest/ structure ready ({NUM_CLASSES} class folders)")

# Khởi tạo / mở file log CSV
_log_is_new = not os.path.exists(LOG_FILE)
_log_fh     = open(LOG_FILE, 'a', newline='', encoding='utf-8')
_log_writer = csv.writer(_log_fh)
if _log_is_new:
    _log_writer.writerow(['timestamp_ms', 'filename', 'ai_class',
                          'final_class', 'correct', 'conf', 'obj', 'gamma'])
    _log_fh.flush()
print(f"[INIT] Session log: {LOG_FILE}")

# ============================================================
# 5. THỐNG KÊ SESSION
# ============================================================

session_stats = {
    'total_reviewed': 0,
    'ai_correct':     0,
    'skipped':        0,
    'saved_per_class': defaultdict(int),
}


# ============================================================
# 5b. ON-SCREEN BUTTON SYSTEM  (Touch / Mouse friendly)
# ============================================================

# Registry: list of dicts {x1,y1,x2,y2,action}  — rebuilt each REVIEW frame
_btn_registry: list = []
_btn_click_action: list = [None]   # dùng list để closure có thể ghi vào

def _mouse_callback(event, x, y, flags, param):
    """Xử lý click chuột / touch — ánh xạ tọa độ → action."""
    if event == cv2.EVENT_LBUTTONDOWN:
        for btn in _btn_registry:
            if btn['x1'] <= x <= btn['x2'] and btn['y1'] <= y <= btn['y2']:
                _btn_click_action[0] = btn['action']
                break

def _pop_click() -> str | None:
    """Lấy action từ lần click gần nhất (None nếu chưa click)."""
    act = _btn_click_action[0]
    _btn_click_action[0] = None
    return act

def _draw_btn(canvas, x1, y1, x2, y2, label, color, action: str,
              font_scale=0.65, thickness=2, icon: str = ""):
    """
    Vẽ nút lên canvas VÀ đăng ký vào _btn_registry.
    Nút có bo góc nhẹ (rectangle), text căn giữa.
    """
    # Nền nút
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
    # Viền sáng hơn
    lighter = tuple(min(255, int(c * 1.5)) for c in color)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), lighter, 2)

    # Text căn giữa
    full_label = f"{icon} {label}".strip() if icon else label
    (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, thickness)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(canvas, full_label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    # Đăng ký vùng hit
    _btn_registry.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                          'action': action})


def log_and_save(raw_bgr_square: np.ndarray,
                 ai_class: str, final_class: str,
                 conf: float, obj: float, gamma: float) -> str:
    """
    Lưu ảnh RAW hình vuông vào DataRealTest/<final_class>/ và ghi log CSV.
    Trả về đường dẫn file đã lưu.

    QUAN TRỌNG: raw_bgr_square là ảnh BGRchưa qua AGC.
    Training pipeline đã có RandomAdaptiveGammaJitter — không cần AGC ở đây.
    """
    ts       = int(time.time() * 1000)
    filename = f"{final_class}_{ts}.jpg"
    filepath = os.path.join(DATA_DIR, final_class, filename)

    cv2.imwrite(filepath, raw_bgr_square,
                [cv2.IMWRITE_JPEG_QUALITY, SAVE_JPEG_QUALITY])

    correct_flag = 1 if (ai_class == final_class) else 0
    _log_writer.writerow([ts, filename, ai_class, final_class,
                          correct_flag, f"{conf:.4f}", f"{obj:.4f}", f"{gamma:.3f}"])
    _log_fh.flush()

    session_stats['total_reviewed'] += 1
    if correct_flag:
        session_stats['ai_correct'] += 1
    session_stats['saved_per_class'][final_class] += 1

    print(f"[SAVE] {filepath}  (ai={ai_class} → label={final_class})")
    return filepath

# ============================================================
# 6. LOAD ONNX MODEL
# ============================================================

_providers = (
    ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if ort.get_device() == 'GPU'
    else ['CPUExecutionProvider']
)

ort_session  = ort.InferenceSession(ONNX_PATH, providers=_providers)
_input_name  = ort_session.get_inputs()[0].name
_out_logits  = ort_session.get_outputs()[0].name
_out_obj     = ort_session.get_outputs()[1].name

print(f"[INFO] ONNX model: {ONNX_PATH}")
print(f"[INFO] Provider  : {ort_session.get_providers()}")

# ============================================================
# 7. ADAPTIVE GAMMA CORRECTION  (FastAdaptiveGamma)
# ============================================================

class FastAdaptiveGamma:
    def __init__(self, target=128, g_min=0.4, g_max=3.0):
        self.target = float(np.clip(target, 8, 247))
        self.g_min  = g_min
        self.g_max  = g_max
        self._idx   = np.arange(256, dtype=np.float64) / 255.0
        self._last_gamma = -1.0
        self._lut        = None

    def _compute_gamma(self, mean_v):
        mean_v     = float(np.clip(mean_v, 8.0, 247.0))
        log_mean   = math.log(mean_v / 255.0)
        log_target = math.log(self.target / 255.0)
        if abs(log_mean - log_target) < 0.03:
            return 1.0
        return float(np.clip(log_target / log_mean, self.g_min, self.g_max))

    def apply(self, img_rgb):
        mean_v = float(img_rgb.max(axis=2).mean())
        gamma  = self._compute_gamma(mean_v)
        if abs(gamma - 1.0) < 0.02:
            return img_rgb
        if abs(gamma - self._last_gamma) > 0.005:
            lut = (np.power(self._idx, gamma) * 255.0)
            self._lut = lut.clip(0, 255).astype(np.uint8)
            self._last_gamma = gamma
        return cv2.LUT(img_rgb, self._lut)

    def get_last_gamma(self):
        return self._last_gamma


agc = FastAdaptiveGamma(target=AGC_TARGET, g_min=AGC_MIN, g_max=AGC_MAX)
print(f"[INIT] FastAdaptiveGamma ready  (target={AGC_TARGET})")

# ============================================================
# 8. SQUARE CROP
# ============================================================

def get_square_crop(bbox, frame_h, frame_w, pad=CROP_PAD):
    if bbox is None:
        size = min(frame_h, frame_w)
        cx, cy = frame_w // 2, frame_h // 2
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(frame_w, x1 + size)
        y2 = min(frame_h, y1 + size)
        real_size = min(x2 - x1, y2 - y1)
        return (x1, y1, x1 + real_size, y1 + real_size)

    bx1, by1, bx2, by2 = bbox
    cx = (bx1 + bx2) // 2
    cy = (by1 + by2) // 2
    size = max(bx2 - bx1, by2 - by1) + pad * 2

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(frame_w, x1 + size)
    y2 = min(frame_h, y1 + size)
    real_size = min(x2 - x1, y2 - y1)
    x2 = x1 + real_size
    y2 = y1 + real_size

    if real_size < 32:
        return None
    return (x1, y1, x2, y2)

# ============================================================
# 9. TRANSFORMS
# ============================================================

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

tf_base = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

_tta_size = int(img_size * 1.12)
tf_tta_list = [
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(_tta_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(_tta_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
]

# ============================================================
# 10. ONNX INFERENCE
# ============================================================

def _run_inference(img_rgb_square):
    tensors = [tf_base(img_rgb_square).unsqueeze(0).numpy()]
    if N_TTA > 1:
        for tf in tf_tta_list[:N_TTA - 1]:
            tensors.append(tf(img_rgb_square).unsqueeze(0).numpy())

    probs_acc = np.zeros(len(classes), dtype=np.float32)
    obj_acc   = 0.0

    for t in tensors:
        outputs   = ort_session.run([_out_logits, _out_obj], {_input_name: t})
        logits_np = outputs[0][0]
        obj_np    = outputs[1][0]

        e = np.exp(logits_np - logits_np.max())
        probs_acc += (e / e.sum())
        obj_acc   += float(1.0 / (1.0 + np.exp(-obj_np[0])))

    probs_acc /= len(tensors)
    obj_acc   /= len(tensors)

    idx  = int(probs_acc.argmax())
    conf = float(probs_acc[idx])
    return classes[idx], conf, obj_acc, probs_acc


def get_bin(class_name):
    for bin_name, group in BIN_GROUPS.items():
        if class_name in group:
            return bin_name
    return "OTHER"

# ============================================================
# 11. INFERENCE WORKER THREAD
# ============================================================

_infer_lock    = threading.Lock()
_infer_request = None
_infer_result  = None
_infer_busy    = False


def _inference_loop():
    global _infer_request, _infer_result, _infer_busy
    while True:
        req = None
        while req is None:
            with _infer_lock:
                req = _infer_request
                if req is not None:
                    _infer_request = None
                    _infer_busy    = True
            if req is None:
                time.sleep(0.002)

        if req == "STOP":
            break

        frame_rgb, crop_roi = req
        h, w = frame_rgb.shape[:2]

        frame_agc = agc.apply(frame_rgb)
        sq = get_square_crop(crop_roi, h, w)
        if sq is not None:
            x1, y1, x2, y2 = sq
            img_square = frame_agc[y1:y2, x1:x2]
            if img_square.size == 0:
                img_square = frame_agc
        else:
            img_square = frame_agc

        result = _run_inference(img_square)
        with _infer_lock:
            _infer_result = result
            _infer_busy   = False


def submit_inference(frame_rgb, crop_roi=None):
    global _infer_request
    with _infer_lock:
        if not _infer_busy:
            _infer_request = (frame_rgb.copy(), crop_roi)


def pop_inference_result():
    global _infer_result
    with _infer_lock:
        r = _infer_result
        _infer_result = None
        return r


_worker_thread = threading.Thread(target=_inference_loop, daemon=True)
_worker_thread.start()
print("[INIT] Inference worker thread started.")

# ============================================================
# 12. OCCUPANCY DETECTION
# ============================================================

def compute_occupancy(mask_mog2, roi, prev_gray, curr_gray, bg_snapshot_gray=None):
    y1, y2, x1, x2 = roi
    roi_mask = mask_mog2[y1:y2, x1:x2]
    mog2_pix = cv2.countNonZero(roi_mask)

    diff_pix = 0
    if prev_gray is not None and curr_gray is not None:
        diff      = cv2.absdiff(prev_gray[y1:y2, x1:x2], curr_gray[y1:y2, x1:x2])
        _, diff_m = cv2.threshold(diff, DIFF_GRAY_THRESH, 255, cv2.THRESH_BINARY)
        diff_pix  = cv2.countNonZero(diff_m)

    mog2_weak     = mog2_pix <= MOG2_PIXEL_THRESH
    diff_weak     = diff_pix <= DIFF_PIXEL_THRESH
    mog_diff_weak = mog2_weak and diff_weak

    contour_occupied = False
    best_bbox        = None
    contours, _      = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        total_area       = sum(cv2.contourArea(c) for c in contours)
        contour_occupied = total_area > CONTOUR_AREA_THRESH
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            bx, by, bw, bh = cv2.boundingRect(largest)
            best_bbox = (x1+bx, y1+by, x1+bx+bw, y1+by+bh)

    frozen_occupied = False
    frozen_diff_val = 0.0
    if mog_diff_weak and bg_snapshot_gray is not None and curr_gray is not None:
        roi_snap        = bg_snapshot_gray[y1:y2, x1:x2].astype(np.float32)
        roi_curr        = curr_gray[y1:y2, x1:x2].astype(np.float32)
        frozen_diff_val = float(np.mean(np.abs(roi_snap - roi_curr)))
        frozen_occupied = frozen_diff_val > FROZEN_DIFF_THRESH

    is_occupied = (
        (mog2_pix > MOG2_PIXEL_THRESH)
        or (diff_pix > DIFF_PIXEL_THRESH)
        or frozen_occupied
        or contour_occupied
    )
    return is_occupied, mog2_pix, diff_pix, frozen_diff_val, mog_diff_weak, best_bbox

# ============================================================
# 13. DISPLAY HELPERS
# ============================================================

def draw_prob_bars(canvas, probs, class_names, y_start=115, bar_max_w=190):
    top_idx = int(np.argmax(probs))
    for i, (cls, p) in enumerate(zip(class_names, probs)):
        y     = y_start + i * 22
        bw    = int(p * bar_max_w)
        color = (0, 210, 80) if i == top_idx else (70, 70, 70)
        cv2.rectangle(canvas, (10, y), (10 + bw, y + 15), color, -1)
        cv2.putText(canvas, f"{cls[:14]:<14} {p*100:4.1f}%",
                    (10, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (210, 210, 210), 1, cv2.LINE_AA)


def draw_locked_banner(canvas, locked_class, locked_bin, bin_colors):
    h, w  = canvas.shape[:2]
    color = bin_colors.get(locked_bin, (100, 100, 100))
    cv2.rectangle(canvas, (0, 0), (w, 95), color, -1)
    cv2.putText(canvas, f"  BIN: {locked_bin}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"  CLASS: {locked_class}",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (230, 230, 230), 2, cv2.LINE_AA)


def draw_session_stats(canvas, stats, classes_list):
    """Vẽ bảng thống kê session góc phải màn hình."""
    h, w    = canvas.shape[:2]
    total   = stats['total_reviewed']
    correct = stats['ai_correct']
    acc     = (correct / total * 100) if total > 0 else 0.0
    panel_x = w - 100

    cv2.rectangle(canvas, (panel_x - 8, h - 100), (w - 2, h - 2),
                  (20, 20, 20), -1)
    cv2.rectangle(canvas, (panel_x - 8, h - 100), (w - 2, h - 2),
                  (60, 60, 60), 1)

    cv2.putText(canvas, "SESSION STATS",
                (panel_x, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Reviewed : {total}",
                (panel_x, h - 185), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (200, 200, 200), 1, cv2.LINE_AA)
    acc_color = (0, 220, 60) if acc >= 80 else (0, 180, 255) if acc >= 60 else (60, 60, 255)
    cv2.putText(canvas, f"AI Acc   : {acc:.1f}%",
                (panel_x, h - 168), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                acc_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Skipped  : {stats['skipped']}",
                (panel_x, h - 152), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (120, 120, 120), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Saved per class:",
                (panel_x, h - 132), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (160, 160, 160), 1, cv2.LINE_AA)
    for i, cls in enumerate(classes_list):
        cnt = stats['saved_per_class'].get(cls, 0)
        bar = min(int(cnt / max(1, 5) * 80), 80)
        cv2.rectangle(canvas,
                      (panel_x, h - 120 + i * 16),
                      (panel_x + bar, h - 110 + i * 16),
                      (40, 140, 255) if cnt > 0 else (40, 40, 40), -1)
        cv2.putText(canvas, f"{cls[:12]:<12} {cnt}",
                    (panel_x, h - 111 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    (200, 200, 200) if cnt > 0 else (80, 80, 80),
                    1, cv2.LINE_AA)


def build_review_overlay(frozen_bgr: np.ndarray,
                         raw_crop_bgr: np.ndarray,
                         locked_class: str, locked_bin: str,
                         conf: float, obj: float, gamma: float,
                         review_phase: str,  # "YN" | "SELECT"
                         highlight_key: int = -1) -> np.ndarray:
    """
    Tạo ảnh overlay cho STATE_REVIEW với ON-SCREEN BUTTONS (touch/mouse).
    - Bên trái: ảnh crop vuông phóng to
    - Bên phải: thông tin + NÚT BẤM LỚN (click/touch)
    - Vẫn hỗ trợ phím bàn phím song song

    review_phase:
      "YN"     — 3 nút: ĐÚNG / SAI / SKIP
      "SELECT" — N nút class để chọn lại nhãn

    highlight_key: -1 = none; 0 = Y (xanh); 1 = N (đỏ); 2 = S (vàng)
    """
    # Xóa registry nút cũ mỗi frame
    _btn_registry.clear()

    canvas = frozen_bgr.copy()
    h, w   = canvas.shape[:2]

    # ── Làm tối background ─────────────────────────────────────────────
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    # ── Crop thumb (bên trái) ───────────────────────────────────────────
    thumb_size = min(h - 20, 320)
    if raw_crop_bgr.size > 0:
        thumb = cv2.resize(raw_crop_bgr, (thumb_size, thumb_size))
        tx1, ty1 = 10, (h - thumb_size) // 2
        canvas[ty1:ty1 + thumb_size, tx1:tx1 + thumb_size] = thumb
        bin_color = BIN_COLORS.get(locked_bin, (100, 100, 100))
        cv2.rectangle(canvas, (tx1, ty1), (tx1 + thumb_size, ty1 + thumb_size),
                      bin_color, 3)
        cv2.putText(canvas, "RAW CROP (no AGC)",
                    (tx1, ty1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (150, 150, 150), 1, cv2.LINE_AA)

    # ── Panel thông tin (bên phải) ──────────────────────────────────────
    px = thumb_size + 30
    bin_color = BIN_COLORS.get(locked_bin, (100, 100, 100))
    cv2.rectangle(canvas, (px, 10), (w - 10, 100), bin_color, -1)
    cv2.putText(canvas, f"BIN: {locked_bin}",
                (px + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"CLASS: {locked_class}",
                (px + 10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.80,
                (230, 230, 230), 2, cv2.LINE_AA)

    # Confidence / Obj / Gamma
    conf_color = (0, int(255 * conf), int(255 * (1 - conf)))
    cv2.putText(canvas, f"Conf: {conf*100:.1f}%   Obj: {obj*100:.0f}%   y={gamma:.2f}",
                (px + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                conf_color, 1, cv2.LINE_AA)

    # ── ON-SCREEN BUTTONS ──────────────────────────────────────────────
    if review_phase == "YN":
        # Tiêu đề
        cv2.putText(canvas, "[ AI DUNG HAY SAI? ]",
                    (px + 10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 200, 255), 2, cv2.LINE_AA)

        btn_w  = w - px - 20          # chiều rộng nút = toàn bộ panel
        btn_h  = max(54, (h - 175) // 3 - 10)   # chiều cao co giãn
        gap    = 12
        bx1    = px + 10
        bx2    = bx1 + btn_w

        specs = [
            # (label_top, label_bot, color_BGR,    action,  highlight_idx)
            ("DUNG",  "[Y]  Luu nhan AI",     (30, 160, 50),  "Y",  highlight_key == 0),
            ("SAI",   "[N]  Chon lai nhan",   (50,  50, 200), "N",  highlight_key == 1),
            ("SKIP",  "[S]  Bo qua",          (0,  140, 210), "S",  highlight_key == 2),
        ]
        gy = 160
        for main_lbl, sub_lbl, color, action, active in specs:
            top_color = tuple(min(255, int(c * 1.3)) for c in color) if active else color
            by2 = gy + btn_h
            # Nền nút
            cv2.rectangle(canvas, (bx1, gy), (bx2, by2), top_color, -1)
            lighter = tuple(min(255, int(c * 1.6)) for c in top_color)
            cv2.rectangle(canvas, (bx1, gy), (bx2, by2), lighter, 2)

            # Label chính (lớn)
            (tw, th), _ = cv2.getTextSize(main_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.putText(canvas, main_lbl,
                        (bx1 + (btn_w - tw) // 2, gy + btn_h // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            # Sub-label (nhỏ, phím tắt)
            (sw, _), _ = cv2.getTextSize(sub_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.putText(canvas, sub_lbl,
                        (bx1 + (btn_w - sw) // 2, gy + btn_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 240, 200), 1, cv2.LINE_AA)

            # Đăng ký vùng click
            _btn_registry.append({'x1': bx1, 'y1': gy, 'x2': bx2, 'y2': by2,
                                   'action': action})
            gy += btn_h + gap

    elif review_phase == "SELECT":
        cv2.putText(canvas, "CHON NHAN DUNG:",
                    (px + 10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 200, 255), 2, cv2.LINE_AA)

        cols   = 2
        gap    = 8
        avail_w = (w - px - 20 - gap) // cols
        avail_h = max(44, (h - 165 - 10) // ((NUM_CLASSES + 1) // cols) - gap)

        for i, cls in enumerate(classes):
            col = i % cols
            row = i // cols
            bx1 = px + 10 + col * (avail_w + gap)
            bx2 = bx1 + avail_w
            by1 = 162 + row * (avail_h + gap)
            by2 = by1 + avail_h

            bg_color = (40, 90, 180) if col == 0 else (60, 60, 80)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg_color, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (110, 110, 140), 2)

            label = f"{i+1}. {cls}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            tx = bx1 + (avail_w - tw) // 2
            ty = by1 + (avail_h + th) // 2
            cv2.putText(canvas, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

            # Đăng ký: action = "C<index>" (vd "C0", "C1", ...)
            _btn_registry.append({'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2,
                                   'action': f"C{i}"})

        # Nút SKIP bên dưới
        skip_y1 = 162 + ((NUM_CLASSES + cols - 1) // cols) * (avail_h + gap)
        skip_y2 = skip_y1 + avail_h
        _draw_btn(canvas, px + 10, skip_y1, w - 10, skip_y2,
                  "SKIP (bo qua)", (0, 120, 180), "S",
                  font_scale=0.55, thickness=1)

    # ── Tiêu đề REVIEW (footer) ─────────────────────────────────────────
    title_color = (0, 200, 255) if review_phase == "YN" else (0, 165, 255)
    title_text  = ("Nhan nut hoac bam phim  Y / N / S"
                   if review_phase == "YN"
                   else f"Nhan nut hoac bam phim  1-{NUM_CLASSES} / S")
    cv2.putText(canvas, title_text,
                (px + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                title_color, 1, cv2.LINE_AA)

    return canvas

# ============================================================
# 14. STATE MACHINE CONSTANTS
# ============================================================

STATE_WARMUP    = "WARMUP"
STATE_WAITING   = "WAITING"
STATE_DETECTING = "DETECTING"
STATE_LOCKED    = "LOCKED"
STATE_REVIEW    = "REVIEW"      # ← NEW: Human-in-the-loop
STATE_COOLDOWN  = "COOLDOWN"

# ============================================================
# 15. CAMERA & BACKGROUND SUBTRACTOR
# ============================================================

cap = cv2.VideoCapture("http://192.168.1.16:4747/video")
if not cap.isOpened():
    raise RuntimeError(f"[FATAL] Không mở được camera ID={CAMERA_ID}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Tạo window trước để đăng ký mouse callback (hỗ trợ touch/click)
cv2.namedWindow("SmartTrashBin [TrainReal]", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("SmartTrashBin [TrainReal]", _mouse_callback)
print("[INIT] Mouse/Touch callback registered on window.")

backSub    = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=30, detectShadows=True)
mog_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ============================================================
# 16. STATE VARIABLES
# ============================================================

state         = STATE_WARMUP
warmup_start  = time.time()

object_count      = 0
empty_count       = 0
cooldown_count    = 0
lock_frame_count  = 0

vote_history      = deque(maxlen=VOTE_WINDOW)
locked_class      = None
locked_bin        = None
locked_conf       = 0.0       # ← Lưu confidence khi lock để dùng ở REVIEW
locked_obj        = 0.0       # ← Lưu objectness khi lock
locked_gamma      = 1.0       # ← Lưu gamma khi lock
locked_raw_crop   = None      # ← Ảnh BGR RAW hình vuông khi lock (chưa AGC)
locked_frozen_bgr = None      # ← Frame đóng băng để hiển thị REVIEW overlay

detect_start_time  = None
stable_frame_count = 0

smoothed_probs = np.zeros(len(classes), dtype=np.float32)
fps_history    = deque(maxlen=30)

prev_gray    = None
ema_bg_gray  = None

frame_counter = 0
last_infer    = None

print(f"[INIT] WARMUP {WARMUP_SEC:.0f}s | TTA={N_TTA} | img_size={img_size}")
print(f"[INIT] INFER_EVERY_N={INFER_EVERY_N}")
print(f"[INIT] CONF={CONF_THRESH} OBJ={OBJ_THRESH} "
      f"VOTE={VOTE_MIN}/{VOTE_WINDOW} DOMINANT={DOMINANT_RATIO:.0%}")
print(f"\n[HITL] Phím REVIEW: Y=Đúng  N=Sai(chọn lại)  S=Skip")
print(f"[HITL] Data sẽ lưu vào: {os.path.abspath(DATA_DIR)}/\n")

# ============================================================
# 17. MAIN LOOP
# ============================================================

try:
    while True:
        t_start = time.time()
        frame_counter += 1

        ret, frame_bgr = cap.read()
        if not ret:
            print("[ERROR] Không đọc được frame — thử lại...")
            time.sleep(0.1)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        annotated = frame_bgr.copy()
        h, w      = frame_bgr.shape[:2]

        # ROI tuyệt đối
        y1 = int(h * ROI_Y1_RATIO);  y2 = int(h * ROI_Y2_RATIO)
        x1 = int(w * ROI_X1_RATIO);  x2 = int(w * ROI_X2_RATIO)
        roi_coords = (y1, y2, x1, x2)

        # ── Background subtraction ──────────────────────────────────────
        lr = 0.007 if state in (STATE_WARMUP, STATE_WAITING,
                                STATE_COOLDOWN, STATE_REVIEW) else 0.0
        raw_mask = backSub.apply(frame_rgb, learningRate=lr)
        fg_mask  = np.where(raw_mask == 255, 255, 0).astype(np.uint8)
        fg_mask  = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  mog_kernel)
        fg_mask  = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, mog_kernel)

        # ── Occupancy ──────────────────────────────────────────────────
        is_occupied, mog2_pix, diff_pix, frozen_diff, mog_diff_weak, obj_bbox = \
            compute_occupancy(fg_mask, roi_coords, prev_gray, curr_gray,
                              bg_snapshot_gray=ema_bg_gray)
        prev_gray = curr_gray.copy()

        if mog_diff_weak:
            if ema_bg_gray is None:
                ema_bg_gray = curr_gray.astype(np.float32)
            else:
                ema_bg_gray = ((1 - SNAPSHOT_EMA_ALPHA) * ema_bg_gray
                               + SNAPSHOT_EMA_ALPHA * curr_gray.astype(np.float32))

        sq_crop = get_square_crop(obj_bbox, h, w)

        # Vẽ ROI + crop box (ẩn khi REVIEW để không che overlay)
        if state != STATE_REVIEW:
            roi_color = (0, 200, 255) if is_occupied else (60, 60, 60)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), roi_color, 1)
            if sq_crop is not None and state == STATE_DETECTING:
                sx1, sy1, sx2, sy2 = sq_crop
                cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (255, 200, 0), 1)
                g_val = agc.get_last_gamma()
                cv2.putText(annotated, f"γ={g_val:.2f}",
                            (sx1, sy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (255, 200, 0), 1, cv2.LINE_AA)

            cv2.putText(annotated,
                        f"mog2={mog2_pix} diff={diff_pix} frz={frozen_diff:.1f}",
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (120, 120, 120), 1, cv2.LINE_AA)

        # =================================================================
        # STATE: WARMUP
        # =================================================================
        if state == STATE_WARMUP:
            elapsed_warmup = time.time() - warmup_start
            remaining      = max(0.0, WARMUP_SEC - elapsed_warmup)
            cv2.putText(annotated, f"WARMING UP... {remaining:.1f}s",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 200, 255), 2, cv2.LINE_AA)
            if elapsed_warmup >= WARMUP_SEC:
                state        = STATE_WAITING
                object_count = 0
                print("[STATE] WARMUP → WAITING")

        # =================================================================
        # STATE: WAITING
        # =================================================================
        elif state == STATE_WAITING:
            smoothed_probs[:] = 0.0
            if is_occupied:
                object_count += 1
                if object_count >= OBJECT_CONFIRM_FRAMES:
                    state              = STATE_DETECTING
                    object_count       = 0
                    empty_count        = 0
                    vote_history       = deque(maxlen=VOTE_WINDOW)
                    detect_start_time  = time.time()
                    stable_frame_count = 0
                    print("[STATE] WAITING → DETECTING")
            else:
                object_count = 0

            cv2.putText(annotated,
                        f"WAITING  (obj_cnt={object_count}/{OBJECT_CONFIRM_FRAMES})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (160, 160, 160), 2, cv2.LINE_AA)
            draw_session_stats(annotated, session_stats, classes)

        # =================================================================
        # STATE: DETECTING
        # =================================================================
        elif state == STATE_DETECTING:
            if not is_occupied:
                empty_count += 1
                if empty_count >= EMPTY_DETECT_FRAMES:
                    state              = STATE_WAITING
                    empty_count        = 0
                    object_count       = 0
                    vote_history       = deque(maxlen=VOTE_WINDOW)
                    detect_start_time  = None
                    stable_frame_count = 0
                    print("[STATE] DETECTING → WAITING (object gone)")
            else:
                empty_count = 0

                if prev_gray is not None:
                    _roi_diff  = cv2.absdiff(prev_gray[y1:y2, x1:x2],
                                             curr_gray[y1:y2, x1:x2])
                    roi_motion = float(np.mean(_roi_diff))
                else:
                    roi_motion = 999.0

                is_stable = roi_motion < STABILITY_DIFF_THRESH
                if is_stable:
                    stable_frame_count = min(stable_frame_count + 1,
                                             STABLE_FRAMES_REQUIRED + 10)
                else:
                    stable_frame_count = 0

                elapsed_detect    = time.time() - detect_start_time if detect_start_time else 0.0
                vote_delay_passed = elapsed_detect >= VOTE_DELAY_SEC
                gate_open         = vote_delay_passed and (stable_frame_count >= STABLE_FRAMES_REQUIRED)

                if gate_open and frame_counter % INFER_EVERY_N == 0:
                    submit_inference(frame_rgb, crop_roi=obj_bbox)

                result = pop_inference_result()
                if result is not None:
                    last_infer = result

                if last_infer is not None:
                    top_class, conf, obj_score, probs = last_infer

                    if gate_open:
                        smoothed_probs = EMA_ALPHA * probs + (1 - EMA_ALPHA) * smoothed_probs
                    else:
                        smoothed_probs[:] = 0.0
                        last_infer        = None
                        top_class, conf, obj_score = "...", 0.0, 0.0

                    if gate_open:
                        if conf > CONF_THRESH and obj_score > OBJ_THRESH:
                            vote_history.append(top_class)
                        else:
                            vote_history.append("_uncertain")

                    valid_votes = [c for c in vote_history if c != "_uncertain"]
                    n_valid     = len(valid_votes)
                    lock_ready  = False
                    lock_reason = ""

                    if n_valid >= VOTE_MIN:
                        candidate  = max(set(valid_votes), key=valid_votes.count)
                        cand_count = valid_votes.count(candidate)
                        if (cand_count / n_valid) >= DOMINANT_RATIO:
                            lock_ready   = True
                            locked_class = candidate
                            locked_bin   = get_bin(locked_class)
                            lock_reason  = f"vote {cand_count}/{n_valid}"

                    if not lock_ready and detect_start_time is not None:
                        elapsed_detect = time.time() - detect_start_time
                        if elapsed_detect >= VOTE_TIMEOUT_SEC:
                            if valid_votes:
                                candidate    = max(set(valid_votes), key=valid_votes.count)
                                cand_count   = valid_votes.count(candidate)
                                lock_reason  = f"TIMEOUT {elapsed_detect:.1f}s vote({cand_count}/{n_valid})"
                            else:
                                best_idx     = int(np.argmax(smoothed_probs))
                                candidate    = classes[best_idx]
                                cand_count   = 0
                                lock_reason  = f"TIMEOUT {elapsed_detect:.1f}s prob({smoothed_probs[best_idx]:.2f})"
                            lock_ready   = True
                            locked_class = candidate
                            locked_bin   = get_bin(locked_class)

                    if lock_ready:
                        # ── Chuyển sang LOCKED, lưu thông tin cần cho REVIEW ──
                        locked_conf  = conf
                        locked_obj   = obj_score
                        locked_gamma = agc.get_last_gamma()

                        # Lấy ảnh RAW hình vuông (CHƯA qua AGC) để lưu
                        if sq_crop is not None:
                            sx1, sy1, sx2, sy2 = sq_crop
                            raw_sq = frame_bgr[sy1:sy2, sx1:sx2].copy()
                            if raw_sq.size == 0:
                                raw_sq = frame_bgr.copy()
                        else:
                            size = min(h, w)
                            cy_, cx_ = h // 2, w // 2
                            raw_sq = frame_bgr[cy_ - size//2:cy_ + size//2,
                                               cx_ - size//2:cx_ + size//2].copy()
                        locked_raw_crop   = raw_sq
                        locked_frozen_bgr = annotated.copy()   # đóng băng frame hiện tại

                        state            = STATE_LOCKED
                        empty_count      = 0
                        lock_frame_count = 0
                        detect_start_time = None
                        print(f"[STATE] DETECTING → LOCKED: "
                              f"{locked_class} → {locked_bin}  [{lock_reason}]")

                    if detect_start_time is not None:
                        elapsed_detect = time.time() - detect_start_time
                        remaining_vote = max(0.0, VOTE_TIMEOUT_SEC - elapsed_detect)
                        timer_ratio = remaining_vote / VOTE_TIMEOUT_SEC
                        timer_color = (0, int(255 * timer_ratio), int(255 * (1 - timer_ratio)))
                        tw = int((1 - timer_ratio) * 200)
                        cv2.rectangle(annotated, (10, 72), (210, 82), (40, 40, 40), -1)
                        cv2.rectangle(annotated, (10, 72), (10 + tw, 82), timer_color, -1)

                    conf_color = (0, int(255 * conf), int(255 * (1 - conf)))
                    gate_txt = ""
                    if not vote_delay_passed and detect_start_time:
                        delay_remain = max(0.0, VOTE_DELAY_SEC - elapsed_detect)
                        gate_txt = f"  [DELAY {delay_remain:.1f}s]"
                    elif not gate_open:
                        gate_txt = f"  [WAIT {stable_frame_count}/{STABLE_FRAMES_REQUIRED}]"

                    cv2.putText(annotated,
                                f"DETECTING: {top_class}  C={conf:.2f}{gate_txt}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                conf_color, 2, cv2.LINE_AA)

                    n_v   = len(valid_votes)
                    bar_w = min(int((n_v / max(VOTE_MIN, 1)) * 200), 200)
                    cv2.rectangle(annotated, (10, 56), (210, 68), (40, 40, 40), -1)
                    bar_color = (0, 255, 100) if n_v >= VOTE_MIN else (0, 180, 255)
                    cv2.rectangle(annotated, (10, 56), (10 + bar_w, 68), bar_color, -1)
                    cv2.putText(annotated, f"votes {n_v}/{VOTE_MIN}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (180, 180, 180), 1, cv2.LINE_AA)
                else:
                    cv2.putText(annotated, "DETECTING: waiting inference...",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (100, 180, 255), 2, cv2.LINE_AA)

            draw_prob_bars(annotated, smoothed_probs, classes)
            draw_session_stats(annotated, session_stats, classes)

        # =================================================================
        # STATE: LOCKED  →  Chuẩn bị chuyển sang REVIEW
        # =================================================================
        elif state == STATE_LOCKED:
            lock_frame_count += 1
            draw_locked_banner(annotated, locked_class, locked_bin, BIN_COLORS)

            # Sau MIN_LOCK_HOLD_FRAMES, tự động chuyển REVIEW
            if lock_frame_count >= MIN_LOCK_HOLD_FRAMES:
                state = STATE_REVIEW
                print(f"\n[STATE] LOCKED → REVIEW  ({locked_class} / {locked_bin})")
                print(f"        Conf={locked_conf*100:.1f}%  Obj={locked_obj*100:.0f}%  γ={locked_gamma:.2f}")
                print(f"        Phím: Y=Đúng  N=Sai  S=Skip")
            else:
                remaining_hold = MIN_LOCK_HOLD_FRAMES - lock_frame_count
                cv2.putText(annotated, f"Confirming... {remaining_hold} frames",
                            (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (200, 200, 0), 1, cv2.LINE_AA)

        # =================================================================
        # STATE: REVIEW  —  Human-in-the-Loop
        # =================================================================
        elif state == STATE_REVIEW:
            # Hiển thị overlay đóng băng, không update frame mới
            review_canvas = build_review_overlay(
                locked_frozen_bgr, locked_raw_crop,
                locked_class, locked_bin,
                locked_conf, locked_obj, locked_gamma,
                review_phase="YN"
            )
            draw_session_stats(review_canvas, session_stats, classes)
            cv2.imshow("SmartTrashBin [TrainReal]", review_canvas)

            # Chờ input — NON-BLOCKING (1ms) để vẫn phản hồi click/touch
            key    = cv2.waitKey(1) & 0xFF
            action = _pop_click()   # click chuột / touch

            # Ánh xạ phím bàn phím → action (giữ tương thích)
            if action is None:
                if key == ord('y') or key == ord('Y'):
                    action = "Y"
                elif key == ord('n') or key == ord('N'):
                    action = "N"
                elif key == ord('s') or key == ord('S'):
                    action = "S"
                elif key == ord('q'):
                    break

            if action == "Y":
                # ── ĐÚNG: lưu với nhãn AI ──────────────────────────────
                log_and_save(locked_raw_crop, locked_class, locked_class,
                             locked_conf, locked_obj, locked_gamma)
                print(f"[REVIEW] ✓ ĐÚNG — đã lưu: {locked_class}")
                state          = STATE_COOLDOWN
                cooldown_count = 0
                print("[STATE] REVIEW → COOLDOWN")

            elif action == "N":
                # ── SAI: vào vòng chờ chọn class ───────────────────────
                print(f"[REVIEW] ✗ SAI — Chọn nhãn đúng (1-{NUM_CLASSES}) "
                      f"hoặc nhấn nút trên màn hình:")
                for i, c in enumerate(classes):
                    print(f"    [{i+1}] {c}")

                selected = False
                t_wait   = time.time()
                while time.time() - t_wait < 30.0:
                    # Render overlay SELECT
                    sel_canvas = build_review_overlay(
                        locked_frozen_bgr, locked_raw_crop,
                        locked_class, locked_bin,
                        locked_conf, locked_obj, locked_gamma,
                        review_phase="SELECT"
                    )
                    draw_session_stats(sel_canvas, session_stats, classes)
                    cv2.imshow("SmartTrashBin [TrainReal]", sel_canvas)

                    k2     = cv2.waitKey(50) & 0xFF
                    act2   = _pop_click()

                    # Phím bàn phím 1-N
                    if act2 is None:
                        if ord('1') <= k2 <= ord('0') + NUM_CLASSES:
                            act2 = f"C{k2 - ord('1')}"
                        elif k2 == ord('s') or k2 == ord('S'):
                            act2 = "S"
                        elif k2 == ord('q'):
                            raise KeyboardInterrupt

                    if act2 is not None and act2.startswith("C"):
                        idx         = int(act2[1:])
                        final_class = classes[idx]
                        log_and_save(locked_raw_crop, locked_class, final_class,
                                     locked_conf, locked_obj, locked_gamma)
                        print(f"[REVIEW] → Đã sửa nhãn: {locked_class} → {final_class}")
                        selected = True
                        break
                    elif act2 == "S":
                        session_stats['skipped'] += 1
                        print("[REVIEW] → Skip (không lưu)")
                        selected = True
                        break

                if not selected:
                    print("[REVIEW] ⚠ Timeout 30s — Skip tự động")
                    session_stats['skipped'] += 1

                state          = STATE_COOLDOWN
                cooldown_count = 0
                print("[STATE] REVIEW → COOLDOWN")

            elif action == "S":
                # ── SKIP: không lưu ─────────────────────────────────────
                session_stats['skipped'] += 1
                print("[REVIEW] → Skip (không lưu)")
                state          = STATE_COOLDOWN
                cooldown_count = 0
                print("[STATE] REVIEW → COOLDOWN")

            # Không render frame camera khi REVIEW — tiếp tục vòng lặp
            continue

        # =================================================================
        # STATE: COOLDOWN
        # =================================================================
        elif state == STATE_COOLDOWN:
            cooldown_count += 1
            remaining_cd    = max(0, COOLDOWN_FRAMES - cooldown_count)
            cv2.putText(annotated, f"COOLDOWN... {remaining_cd} frames",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 165, 255), 2, cv2.LINE_AA)
            if cooldown_count >= COOLDOWN_FRAMES:
                state              = STATE_WAITING
                locked_class       = None
                locked_bin         = None
                locked_conf        = 0.0
                locked_obj         = 0.0
                locked_gamma       = 1.0
                locked_raw_crop    = None
                locked_frozen_bgr  = None
                lock_frame_count   = 0
                empty_count        = 0
                object_count       = 0
                last_infer         = None
                detect_start_time  = None
                stable_frame_count = 0
                smoothed_probs[:]  = 0.0
                vote_history       = deque(maxlen=VOTE_WINDOW)
                print("[STATE] COOLDOWN → WAITING (cycle reset)")

            draw_session_stats(annotated, session_stats, classes)

        # ── FPS + tên window ─────────────────────────────────────────────
        elapsed = time.time() - t_start
        fps_history.append(1.0 / (elapsed + 1e-9))
        avg_fps = int(np.mean(fps_history))

        state_color_map = {
            STATE_WARMUP:    (0, 200, 255),
            STATE_WAITING:   (160, 160, 160),
            STATE_DETECTING: (0, 180, 255),
            STATE_LOCKED:    BIN_COLORS.get(locked_bin, (100, 100, 100)) if locked_bin else (0, 220, 0),
            STATE_COOLDOWN:  (0, 165, 255),
        }
        fps_color = state_color_map.get(state, (0, 255, 255))
        busy_txt  = "*" if _infer_busy else " "
        cv2.putText(annotated,
                    f"FPS:{avg_fps}  TTA:{N_TTA}x  [{state}]{busy_txt}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    fps_color, 1, cv2.LINE_AA)

        total_saved = sum(session_stats['saved_per_class'].values())
        cv2.putText(annotated, f"Saved: {total_saved}",
                    (w - 110, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow("SmartTrashBin [TrainReal]", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ============================================================
# 18. CLEANUP
# ============================================================

finally:
    # Dừng inference worker
    with _infer_lock:
        _infer_request = "STOP"
    _worker_thread.join(timeout=3.0)

    # Đóng log file
    _log_fh.close()

    cap.release()
    cv2.destroyAllWindows()

    # In tổng kết session
    total   = session_stats['total_reviewed']
    correct = session_stats['ai_correct']
    acc     = (correct / total * 100) if total > 0 else 0.0
    total_saved = sum(session_stats['saved_per_class'].values())

    print("\n" + "=" * 55)
    print("  SESSION SUMMARY")
    print("=" * 55)
    print(f"  Tổng đã review : {total}")
    print(f"  AI đoán đúng   : {correct}  ({acc:.1f}%)")
    print(f"  Đã skip        : {session_stats['skipped']}")
    print(f"  Tổng ảnh lưu   : {total_saved}")
    print()
    for cls in classes:
        cnt = session_stats['saved_per_class'].get(cls, 0)
        bar = "█" * min(cnt, 30)
        print(f"  {cls:<18}: {cnt:>4}  {bar}")
    print(f"\n  Log file       : {os.path.abspath(LOG_FILE)}")
    print(f"  Data folder    : {os.path.abspath(DATA_DIR)}/")
    print("=" * 55)
    print("[EXIT] Đã thoát.")
