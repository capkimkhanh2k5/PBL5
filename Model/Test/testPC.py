"""
smart_trash_bin.py  —  v3.0 (ONNX + Adaptive Gamma + Square Crop)
==================================================================
THAY ĐỔI SO VỚI v2.1
────────────────────────────────────────────────────────────────────────────
[MODEL]   Chuyển từ PyTorch (.pth) sang ONNX Runtime.
          Loại bỏ hoàn toàn WasteDetector, GeM, torch.nn — không cần
          định nghĩa lại kiến trúc mạng khi inference.
          Tốc độ inference tăng ~15-25% trên CPU nhờ ONNX graph optimization.

[AGC-1]   Thêm FastAdaptiveGamma — mirror đúng pipeline v7:
          Đo mean brightness kênh V (max RGB) trên TOÀN BỘ frame_rgb,
          tính gamma = log(target/255) / log(mean_V/255), build LUT O(1).
          Áp dụng TRƯỚC khi crop để đo sáng đại diện cho toàn cảnh.

[AGC-2]   AGC áp dụng đúng thứ tự: full frame → AGC → crop vuông → resize.
          Nếu crop trước rồi mới AGC, một tờ giấy trắng nhỏ sẽ bị dìm
          thành xám đen (mean_V cao → gamma > 1 → darkening sai).

[CROP]    Thay get_dynamic_crop (hình chữ nhật) bằng get_square_crop:
          Lấy cạnh DÀI NHẤT của bbox làm cạnh hình vuông + padding.
          Khi resize hình vuông về (img_size, img_size), aspect ratio
          của rác được bảo toàn 100% — không còn chai nhựa bị "mập" ra.

[TF]      tf_base đơn giản hơn: crop đã là hình vuông nên Resize tuple
          hoàn toàn an toàn. TTA dùng Resize(int) + RandomCrop đúng v7.

[META]    Đọc classes, img_size, agc_target từ model_meta.json (xuất
          cùng ONNX trong notebook) thay vì hard-code.
────────────────────────────────────────────────────────────────────────────
"""

import cv2
import onnxruntime as ort
import numpy as np
from torchvision import transforms
import torch
import json
import math
import os
from collections import deque
import time
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.dirname(SCRIPT_DIR)

# ============================================================
# 1. CẤU HÌNH
# ============================================================

CAMERA_ID  = 0
ONNX_PATH  = os.path.join(MODEL_DIR, "Train", "outputs_v2", "waste_detector.onnx")
META_PATH  = os.path.join(MODEL_DIR, "Train", "outputs_v2", "model_meta.json")
IMG_SIZE   = 384                          # fallback nếu không đọc được meta

# --- [PERF-2] TTA ---
# N_TTA=1: tắt TTA, tối đa FPS.  N_TTA=3: bật TTA, accuracy cao hơn.
N_TTA = 1

# --- [PERF-3] Frame skip ---
INFER_EVERY_N = 2

# --- Ngưỡng inference ---
CONF_THRESH    = 0.82
OBJ_THRESH     = 0.35

# --- Voting ---
VOTE_WINDOW    = 10
VOTE_MIN       = 5
DOMINANT_RATIO = 0.65
VOTE_TIMEOUT_SEC = 10.0

# --- State timing ---
WARMUP_SEC            = 3.0
OBJECT_CONFIRM_FRAMES = 2
EMPTY_DETECT_FRAMES   = 12
EMPTY_LOCKED_FRAMES   = 15
MIN_LOCK_HOLD_FRAMES  = 15
COOLDOWN_FRAMES       = 20

# --- ROI ---
ROI_X1_RATIO = 0.18
ROI_X2_RATIO = 0.82
ROI_Y1_RATIO = 0.18
ROI_Y2_RATIO = 0.82

# --- Occupancy ---
MOG2_PIXEL_THRESH   = 1000
DIFF_PIXEL_THRESH   = 800
DIFF_GRAY_THRESH    = 18
CROP_PAD            = 40

# --- Stability gate ---
STABILITY_DIFF_THRESH   = 8.0
STABLE_FRAMES_REQUIRED  = 4
VOTE_DELAY_SEC          = 2.0

# --- Contour area fallback ---
CONTOUR_AREA_THRESH = 3000

# --- Frozen foreground ---
FROZEN_DIFF_THRESH  = 12.0

# --- EMA ---
EMA_ALPHA           = 0.35
SNAPSHOT_EMA_ALPHA  = 0.05

# --- Bin config ---
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
# 2. ĐỌC MODEL META  (classes, img_size, AGC config)
# ============================================================

_meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        _meta = json.load(f)
    print(f"[INFO] Loaded meta: {META_PATH}")
else:
    print(f"[WARN] {META_PATH} not found — dùng giá trị mặc định")

classes    = _meta.get('classes',    ['Battery','Biological','General_Waste',
                                      'Glass','Metal','Paper_Cardboard','Plastic'])
img_size   = _meta.get('img_size',   IMG_SIZE)
AGC_TARGET = _meta.get('agc_target', 128)
AGC_MIN    = _meta.get('agc_gamma_min', 0.4)
AGC_MAX    = _meta.get('agc_gamma_max', 3.0)

print(f"[INFO] Classes   : {classes}")
print(f"[INFO] img_size  : {img_size}")
print(f"[INFO] AGC       : target={AGC_TARGET}  clip=[{AGC_MIN}, {AGC_MAX}]")

# ============================================================
# 3. LOAD ONNX MODEL
# ============================================================
# onnxruntime tự chọn provider tốt nhất: CUDAExecutionProvider nếu có GPU,
# fallback về CPUExecutionProvider. Không cần torch.device.
# ============================================================

_providers = (
    ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if ort.get_device() == 'GPU'
    else ['CPUExecutionProvider']
)

ort_session = ort.InferenceSession(ONNX_PATH, providers=_providers)
_input_name   = ort_session.get_inputs()[0].name     # 'image'
_out_logits   = ort_session.get_outputs()[0].name    # 'logits'
_out_obj      = ort_session.get_outputs()[1].name    # 'objectness'

print(f"[INFO] ONNX model: {ONNX_PATH}")
print(f"[INFO] Provider  : {ort_session.get_providers()}")
print(f"[INFO] Input     : '{_input_name}'  shape={ort_session.get_inputs()[0].shape}")

# ============================================================
# 4. ADAPTIVE GAMMA CORRECTION  (numpy + OpenCV LUT, không PIL)
# ============================================================
# Thứ tự bắt buộc theo triết lý v7:
#   (a) Đo mean_V trên TOÀN BỘ frame gốc (trước khi crop)
#   (b) Tính gamma = log(target/255) / log(mean_V/255)
#   (c) Build LUT 256 entry, apply qua cv2.LUT — O(1)
#   (d) Sau đó mới crop hình vuông
#
# Lý do KHÔNG đo sau crop:
#   Vùng crop nhỏ có thể tình cờ sáng/tối bất thường (tờ giấy trắng,
#   bóng đổ cục bộ) → gamma sai → ảnh crop bị normalize sai hướng.
# ============================================================

class FastAdaptiveGamma:
    """
    Adaptive Gamma Correction thuần numpy + cv2.LUT.
    Tốc độ: ~0.3–0.8 ms cho frame 720×720 trên Pi 4.

    Công thức: γ = log(target/255) / log(mean_V/255)
      mean_V < target → γ < 1 → kéo sáng ảnh tối
      mean_V > target → γ > 1 → dìm ảnh chói
      mean_V ≈ target → γ ≈ 1 → pass-through
    """
    def __init__(self, target: int = 128,
                 g_min: float = 0.4, g_max: float = 3.0):
        self.target = float(np.clip(target, 8, 247))
        self.g_min  = g_min
        self.g_max  = g_max
        self._idx   = np.arange(256, dtype=np.float64) / 255.0
        # LUT cache: tránh tính lại nếu gamma giống frame trước
        self._last_gamma = -1.0
        self._lut        = None

    def _compute_gamma(self, mean_v: float) -> float:
        mean_v     = float(np.clip(mean_v, 8.0, 247.0))
        log_mean   = math.log(mean_v   / 255.0)
        log_target = math.log(self.target / 255.0)
        if abs(log_mean - log_target) < 0.03:
            return 1.0
        return float(np.clip(log_target / log_mean, self.g_min, self.g_max))

    def apply(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Args:
            img_rgb: numpy array (H, W, 3) uint8, định dạng RGB
        Returns:
            numpy array (H, W, 3) uint8 đã được gamma correction
        """
        # Kênh V = max(R, G, B) — luminance perceptual chuẩn, không cần
        # convert sang HSV, không bị ảnh hưởng bởi hue/saturation.
        mean_v = float(img_rgb.max(axis=2).mean())
        gamma  = self._compute_gamma(mean_v)

        if abs(gamma - 1.0) < 0.02:
            return img_rgb   # pass-through, không cần xử lý

        # Cache LUT nếu gamma giống frame trước (tiết kiệm 0.1 ms)
        if abs(gamma - self._last_gamma) > 0.005:
            lut             = (np.power(self._idx, gamma) * 255.0)
            self._lut       = lut.clip(0, 255).astype(np.uint8)
            self._last_gamma = gamma

        return cv2.LUT(img_rgb, self._lut)

    def get_last_gamma(self) -> float:
        return self._last_gamma


agc = FastAdaptiveGamma(target=AGC_TARGET, g_min=AGC_MIN, g_max=AGC_MAX)
print(f"[INIT] FastAdaptiveGamma ready  (target={AGC_TARGET})")

# ============================================================
# 5. SQUARE CROP  (thay thế get_dynamic_crop)
# ============================================================
# Luôn trả về hình VUÔNG để resize về (img_size, img_size) không méo.
#
# Logic:
#   - Có bbox: lấy cạnh DÀI NHẤT làm kích thước vuông + padding.
#     Căn giữa theo tâm của bbox.
#   - Không có bbox: CenterCrop hình vuông ở giữa frame
#     (mirror đúng val transform của v7 — Pi camera cố định).
# ============================================================

def get_square_crop(bbox, frame_h: int, frame_w: int, pad: int = CROP_PAD):
    """
    Returns: (x1, y1, x2, y2) sao cho (x2-x1) == (y2-y1) — hình vuông.
             None nếu kết quả quá nhỏ (< 32px).
    """
    if bbox is None:
        # Camera cố định → rác ở trung tâm → CenterCrop
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

    # Cạnh dài nhất + padding → kích thước hình vuông
    size = max(bx2 - bx1, by2 - by1) + pad * 2

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(frame_w, x1 + size)
    y2 = min(frame_h, y1 + size)

    # Nếu chạm viền → lấy phần thực tế, ép về hình vuông
    real_size = min(x2 - x1, y2 - y1)
    x2 = x1 + real_size
    y2 = y1 + real_size

    if real_size < 32:
        return None

    return (x1, y1, x2, y2)


# ============================================================
# 6. TRANSFORMS  (v3 — crop đã là hình vuông nên Resize tuple an toàn)
# ============================================================
# Lưu ý: AGC đã được apply trên img_rgb trước khi crop ở _inference_loop.
# Các transform ở đây chỉ còn nhiệm vụ: Resize → ToTensor → Normalize.
#
# tf_base: resize hình vuông về (img_size, img_size) → không méo.
# tf_tta:  thêm một số augment nhẹ để TTA (flip, small crop jitter).
# ============================================================

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Base: resize hình vuông → normalize (giống CenterCrop val pipeline v7)
tf_base = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),   # an toàn vì input đã là hình vuông
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# TTA: thêm jitter nhẹ quanh center — giữ aspect ratio
_tta_size = int(img_size * 1.12)   # 12% margin
tf_tta_list = [
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(_tta_size),          # int → giữ aspect ratio
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
# 7. ONNX INFERENCE FUNCTION
# ============================================================

def _run_inference(img_rgb_square: np.ndarray):
    """
    Chạy inference qua ONNX Runtime.
    img_rgb_square: numpy (H, W, 3) uint8, ĐÃ được AGC + đã là hình vuông.

    Returns: (top_class, conf, obj_score, probs_np)
    """
    tensors = [tf_base(img_rgb_square).unsqueeze(0).numpy()]

    if N_TTA > 1:
        for tf in tf_tta_list[:N_TTA - 1]:
            tensors.append(tf(img_rgb_square).unsqueeze(0).numpy())

    n_cls = len(classes)
    probs_acc = np.zeros(n_cls, dtype=np.float32)
    obj_acc   = 0.0

    for t in tensors:
        # ONNX Runtime inference
        outputs  = ort_session.run(
            [_out_logits, _out_obj],
            {_input_name: t}
        )
        logits_np = outputs[0][0]   # shape (n_classes,)
        obj_np    = outputs[1][0]   # shape (1,)

        # Softmax thủ công (nhanh hơn import torch.nn.functional)
        e = np.exp(logits_np - logits_np.max())
        probs_acc += (e / e.sum())

        # Sigmoid objectness
        obj_acc += float(1.0 / (1.0 + np.exp(-obj_np[0])))

    probs_acc /= len(tensors)
    obj_acc   /= len(tensors)

    idx  = int(probs_acc.argmax())
    conf = float(probs_acc[idx])
    return classes[idx], conf, obj_acc, probs_acc


def get_bin(class_name: str) -> str:
    for bin_name, group in BIN_GROUPS.items():
        if class_name in group:
            return bin_name
    return "OTHER"


# ============================================================
# 8. INFERENCE WORKER  (multi-thread, non-blocking)
# ============================================================
# Pipeline trong worker thread (thứ tự bắt buộc):
#   1. Nhận (frame_rgb, crop_roi) từ main thread
#   2. AGC trên TOÀN BỘ frame_rgb (đo mean_V đại diện toàn cảnh)
#   3. Crop hình VUÔNG từ frame đã AGC theo crop_roi
#   4. ONNX inference
# ============================================================

_infer_lock    = threading.Lock()
_infer_request = None   # (frame_rgb_copy, crop_roi) | None | "STOP"
_infer_result  = None   # (top_class, conf, obj_score, probs) | None
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

        # ── Bước 1: AGC trên TOÀN BỘ frame ──────────────────────────
        # PHẢI làm trước crop để đo mean_V đại diện cho toàn cảnh ánh sáng.
        frame_agc = agc.apply(frame_rgb)

        # ── Bước 2: Crop hình VUÔNG từ frame đã AGC ──────────────────
        sq = get_square_crop(crop_roi, h, w)
        if sq is not None:
            x1, y1, x2, y2 = sq
            img_square = frame_agc[y1:y2, x1:x2]
            if img_square.size == 0:
                img_square = frame_agc
        else:
            img_square = frame_agc

        # ── Bước 3: Inference ────────────────────────────────────────
        result = _run_inference(img_square)

        with _infer_lock:
            _infer_result = result
            _infer_busy   = False


def submit_inference(frame_rgb: np.ndarray, crop_roi=None):
    """Non-blocking. Bỏ qua nếu worker đang bận."""
    global _infer_request
    with _infer_lock:
        if not _infer_busy:
            _infer_request = (frame_rgb.copy(), crop_roi)


def pop_inference_result():
    """Non-blocking. None nếu chưa có kết quả mới."""
    global _infer_result
    with _infer_lock:
        r = _infer_result
        _infer_result = None
        return r


_worker_thread = threading.Thread(target=_inference_loop, daemon=True)
_worker_thread.start()
print("[INIT] Inference worker thread started.")

# ============================================================
# 9. OCCUPANCY DETECTION  (không đổi so với v2.1)
# ============================================================

def compute_occupancy(mask_mog2, roi, prev_gray, curr_gray,
                      bg_snapshot_gray=None):
    y1, y2, x1, x2 = roi

    roi_mask = mask_mog2[y1:y2, x1:x2]
    mog2_pix = cv2.countNonZero(roi_mask)

    diff_pix = 0
    if prev_gray is not None and curr_gray is not None:
        diff      = cv2.absdiff(prev_gray[y1:y2, x1:x2],
                                curr_gray[y1:y2, x1:x2])
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
            best_bbox = (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)

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
# 10. DISPLAY HELPERS
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


# ============================================================
# 11. STATE MACHINE CONSTANTS
# ============================================================

STATE_WARMUP    = "WARMUP"
STATE_WAITING   = "WAITING"
STATE_DETECTING = "DETECTING"
STATE_LOCKED    = "LOCKED"
STATE_COOLDOWN  = "COOLDOWN"

# ============================================================
# 12. CAMERA & BACKGROUND SUBTRACTOR
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"[FATAL] Không mở được camera ID={CAMERA_ID}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

backSub    = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=30, detectShadows=True
)
mog_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ============================================================
# 13. STATE VARIABLES
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
detect_start_time = None
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
print(f"[INIT] FROZEN_DIFF_THRESH={FROZEN_DIFF_THRESH}  "
      f"CONTOUR_AREA_THRESH={CONTOUR_AREA_THRESH}")

# ============================================================
# 14. MAIN LOOP
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

        # ── Background subtraction ──────────────────────────────────
        lr = 0.007 if state in (STATE_WARMUP, STATE_WAITING, STATE_COOLDOWN) else 0.0
        raw_mask = backSub.apply(frame_rgb, learningRate=lr)
        fg_mask  = np.where(raw_mask == 255, 255, 0).astype(np.uint8)
        fg_mask  = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  mog_kernel)
        fg_mask  = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, mog_kernel)

        # ── Occupancy ────────────────────────────────────────────────
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

        # Square crop (thay thế get_dynamic_crop cũ)
        sq_crop = get_square_crop(obj_bbox, h, w)

        # Vẽ ROI + crop box
        roi_color = (0, 200, 255) if is_occupied else (60, 60, 60)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), roi_color, 1)
        if sq_crop is not None and state == STATE_DETECTING:
            sx1, sy1, sx2, sy2 = sq_crop
            cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (255, 200, 0), 1)
            # Hiển thị gamma AGC để debug ánh sáng
            g_val = agc.get_last_gamma()
            cv2.putText(annotated, f"γ={g_val:.2f}",
                        (sx1, sy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 200, 0), 1, cv2.LINE_AA)

        # Debug occupancy
        cv2.putText(annotated,
                    f"mog2={mog2_pix} diff={diff_pix} frz={frozen_diff:.1f}",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (120, 120, 120), 1, cv2.LINE_AA)

        # ============================================================
        # STATE: WARMUP
        # ============================================================
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

        # ============================================================
        # STATE: WAITING
        # ============================================================
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

        # ============================================================
        # STATE: DETECTING
        # ============================================================
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

                roi_area   = max(1, (y2 - y1) * (x2 - x1))
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

                # Submit: truyền frame_rgb gốc (chưa AGC) + sq_crop
                # Worker thread sẽ tự apply AGC trên full frame rồi mới crop
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
                        state            = STATE_LOCKED
                        empty_count      = 0
                        lock_frame_count = 0
                        detect_start_time = None
                        print(f"[STATE] DETECTING → LOCKED: "
                              f"{locked_class} → {locked_bin}  [{lock_reason}]")

                    time_left = ""
                    if detect_start_time is not None:
                        elapsed_detect = time.time() - detect_start_time
                        remaining_vote = max(0.0, VOTE_TIMEOUT_SEC - elapsed_detect)
                        time_left = f"  T-{remaining_vote:.1f}s"
                        timer_ratio = remaining_vote / VOTE_TIMEOUT_SEC
                        timer_color = (0, int(255 * timer_ratio), int(255 * (1 - timer_ratio)))
                        tw = int((1 - timer_ratio) * 200)
                        cv2.rectangle(annotated, (10, 72), (210, 82), (40, 40, 40), -1)
                        cv2.rectangle(annotated, (10, 72), (10 + tw, 82), timer_color, -1)

                    conf_color = (0, int(255 * conf), int(255 * (1 - conf)))
                    if not vote_delay_passed:
                        delay_remain = max(0.0, VOTE_DELAY_SEC - elapsed_detect)
                        gate_txt = f"  [DELAY {delay_remain:.1f}s]"
                    elif not gate_open:
                        gate_txt = f"  [WAIT {stable_frame_count}/{STABLE_FRAMES_REQUIRED}]"
                    else:
                        gate_txt = ""

                    cv2.putText(annotated,
                                f"DETECTING: {top_class}  C={conf:.2f}{time_left}{gate_txt}",
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

        # ============================================================
        # STATE: LOCKED
        # ============================================================
        elif state == STATE_LOCKED:
            lock_frame_count += 1
            min_hold_passed   = (lock_frame_count >= MIN_LOCK_HOLD_FRAMES)

            if not is_occupied and min_hold_passed:
                empty_count += 1
                if empty_count >= EMPTY_LOCKED_FRAMES:
                    state          = STATE_COOLDOWN
                    cooldown_count = 0
                    print(f"[STATE] LOCKED → COOLDOWN  "
                          f"(held {lock_frame_count} frames, empty {empty_count} frames)")
            elif is_occupied:
                empty_count = 0

            draw_locked_banner(annotated, locked_class, locked_bin, BIN_COLORS)
            hold_txt  = f"hold={lock_frame_count}/{MIN_LOCK_HOLD_FRAMES}"
            empty_txt = (f"empty={empty_count}/{EMPTY_LOCKED_FRAMES}"
                         if min_hold_passed else "min_hold...")
            cv2.putText(annotated, f"LOCKED  {hold_txt}  {empty_txt}",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (200, 200, 0), 1, cv2.LINE_AA)

        # ============================================================
        # STATE: COOLDOWN
        # ============================================================
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
                lock_frame_count   = 0
                empty_count        = 0
                object_count       = 0
                last_infer         = None
                detect_start_time  = None
                stable_frame_count = 0
                smoothed_probs[:]  = 0.0
                vote_history       = deque(maxlen=VOTE_WINDOW)
                print("[STATE] COOLDOWN → WAITING (cycle reset)")

        # ── FPS ──────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        fps_history.append(1.0 / (elapsed + 1e-9))
        avg_fps = int(np.mean(fps_history))

        state_color_map = {
            STATE_WARMUP:    (0, 200, 255),
            STATE_WAITING:   (160, 160, 160),
            STATE_DETECTING: (0, 180, 255),
            STATE_LOCKED:    BIN_COLORS.get(locked_bin, (100, 100, 100))
                             if locked_bin else (0, 220, 0),
            STATE_COOLDOWN:  (0, 165, 255),
        }
        fps_color = state_color_map.get(state, (0, 255, 255))
        busy_txt  = "*" if _infer_busy else " "
        cv2.putText(annotated,
                    f"FPS:{avg_fps}  TTA:{N_TTA}x  [{state}]{busy_txt}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    fps_color, 1, cv2.LINE_AA)

        cv2.imshow("SmartTrashBin", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    with _infer_lock:
        _infer_request = "STOP"
    _worker_thread.join(timeout=3.0)
    cap.release()
    cv2.destroyAllWindows()
    print("[EXIT] Đã thoát.")
