"""
smart_trash_bin.py  —  v5.3 (ONNX + Arduino + Firebase Firestore + Cloudinary)
=====================================================================
THAY ĐỔI SO VỚI v5.2
────────────────────────────────────────────────────────────────────
[REFACTOR] Firebase: bỏ Realtime Database, chỉ dùng Firestore.
           - firebase_update_bin() → firebase_log_sensor()
           - Dữ liệu cảm biến ghi vào:
               bin_raw_sensor_logs / {bin_id} / logs / {auto_id}
             với các field: fillOrganic, fillRecycle, fillNonRecycle,
             fillHazardous, recordedAt (SERVER_TIMESTAMP)
           - Bỏ firebase_set_online() (không còn trường status/RTDB)
           - get_firestore_client() chuyển sang firebaseUtil

[NOTE]     classification_logs vẫn được ghi qua cloudinaryUtil.upload_and_log()
           (không thay đổi)
────────────────────────────────────────────────────────────────────
YÊU CẦU:
    pip install onnxruntime opencv-python torchvision numpy
    Firebase + Cloudinary luôn cần: firebase-admin cloudinary
    Nếu FULL_PIPELINE=True: thêm pyserial
    File firebase_credit.json + cloudinary_credit.json cùng thư mục.
────────────────────────────────────────────────────────────────────
"""

import cv2
import onnxruntime as ort
import numpy as np
from torchvision import transforms
import json
import math
import os
import time
import threading
import sys
from collections import deque

# ============================================================
# 1. CẤU HÌNH
# ============================================================

FULL_PIPELINE    = True   # ← Đặt True để bật Arduino (servo + Serial)
USE_ULTRASONIC   = True   # ← Đặt True để gửi lệnh 'F' đọc 4 cảm biến siêu âm
                           #   (chỉ có hiệu lực khi FULL_PIPELINE=True)
                           #   False → dùng mock data cho độ đầy

CAMERA_ID  = 0
ONNX_PATH  = "waste_detector_v2.onnx"
META_PATH  = "model_meta.json"
IMG_SIZE   = 384

from dotenv import load_dotenv
load_dotenv()  # Load biến môi trường từ file .env (nếu có)
BIN_ID = os.getenv("BIN_ID", "").strip()
if not BIN_ID:
    raise EnvironmentError("[FATAL] BIN_ID chưa được set trong .env")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def env_int(name: str, default: int) -> int:
    """Đọc số nguyên từ env; base=0 cho phép LCD_ADDR dạng '0x27' hoặc '39'."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip(), 0)
    except ValueError:
        print(f"[WARN] {name}={value!r} không hợp lệ — dùng {default}")
        return default


# Bật khi debug trên màn hình; tắt khi deploy headless trên Pi để giảm tải.
DISPLAY_ENABLED = env_bool("DISPLAY_ENABLED", True)

# LCD I2C 16x2. Có thể tắt bằng LCD_ENABLED=false trong .env khi chạy không có LCD.
LCD_ENABLED = env_bool("LCD_ENABLED", True)
LCD_ADDR    = env_int("LCD_ADDR", 0x27)
LCD_WIDTH   = env_int("LCD_WIDTH", 16)
LCD_BUS_ID  = env_int("LCD_BUS_ID", 1)

# --- TTA & Performance ---
N_TTA          = 1
INFER_EVERY_N  = 2

# --- Ngưỡng inference ---
CONF_THRESH    = 0.82
OBJ_THRESH     = 0.35

# --- Voting ---
VOTE_WINDOW    = 10
VOTE_MIN       = 5
DOMINANT_RATIO = 0.65
VOTE_TIMEOUT_SEC = 10.0

# --- State timing ---
WARMUP_SEC            = 8.0
WARMUP_MAX_SEC        = 20.0
WARMUP_STABLE_FRAMES  = 30
OBJECT_CONFIRM_FRAMES = 2
EMPTY_DETECT_FRAMES   = 12
EMPTY_LOCKED_FRAMES   = 15
MIN_LOCK_HOLD_FRAMES  = 15
COOLDOWN_SEC          = 1.0
DETECT_LOST_GRACE_SEC = 1.25
DETECT_MIN_HOLD_SEC   = 3.0
MAX_DISPENSE_TIMEOUT_SEC = 25.0

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
WARMUP_DIFF_THRESH      = 5.0
STABILITY_DIFF_THRESH   = 8.0
STABLE_FRAMES_REQUIRED  = 4
VOTE_DELAY_SEC          = 2.0

# --- Contour area fallback ---
CONTOUR_AREA_THRESH = 3000

# --- Frozen foreground ---
FROZEN_DIFF_THRESH       = 12.0
FROZEN_PIXEL_GRAY_THRESH = 20
FROZEN_PIXEL_THRESH      = 900

# --- EMA ---
EMA_ALPHA           = 0.35
BG_WARMUP_ALPHA     = 0.25
BG_IDLE_ALPHA       = 0.03
BG_COOLDOWN_ALPHA   = 0.08

# --- MOG2 learning rate ---
MOG2_LR_WARMUP   = 0.05
MOG2_LR_WAITING  = 0.003
MOG2_LR_COOLDOWN = 0.01

# --- Idle periodic update ---
# Nếu không có rác trong khoảng thời gian này (giây), tự động update Firebase.
IDLE_UPDATE_INTERVAL_SEC = 300.0   # 5 phút

# Chờ thêm sau khi Arduino báo hoàn tất servo trước khi đọc siêu âm/upload.
# Khoảng này giúp rác rơi ổn định, servo dừng hẳn và cảm biến bớt nhiễu.
POST_DISPENSE_SETTLE_SEC = 1.0

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
# 2. IMPORT UTILS (có điều kiện theo FULL_PIPELINE)
# ============================================================

_UTILS_DIR = os.path.join(os.path.dirname(__file__), "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

# ── Firebase + Cloudinary: luôn import (dùng cho cả 2 chế độ) ──────────────
from firebaseUtil import (
    init_firebase,
    firebase_log_sensor,
    get_firestore_client as firebase_get_firestore_client,
)
from cloudinaryUtil import (
    init_cloudinary,
    capture_snapshot,
    upload_and_log,
)

# ── Arduino: chỉ import khi FULL_PIPELINE=True ──────────────────────────────
if FULL_PIPELINE:
    from arduinoUtil import (
        init_arduino,
        arduino_send_command,
        read_fill_levels,
        read_fill_levels_simulated,
        BIN_TO_ARDUINO_CMD,
    )
    _ultrasonic_status = "THỰC (lệnh F)" if USE_ULTRASONIC else "MOCK (simulated)"
    print(f"[INFO] FULL_PIPELINE=True  — Arduino + Firebase + Cloudinary bật.")
    print(f"[INFO]   → Cảm biến siêu âm: {_ultrasonic_status}")
else:
    print("[INFO] FULL_PIPELINE=False — Arduino bị tắt; Firebase + Cloudinary vẫn hoạt động.")
    print("[INFO]   → Servo: DISABLED  |  Cảm biến siêu âm: MOCK  |  Firebase/Cloudinary: ENABLED")

    # Stub Arduino — không kết nối cổng Serial, không điều khiển servo
    def init_arduino(port=None):
        return None

    def arduino_send_command(arduino_serial, bin_type, on_done_callback=None):
        """Giả lập Arduino: delay 2s rồi gọi callback (không gửi serial)."""
        def _mock():
            print(f"[MOCK ARDUINO] Bỏ qua servo (FULL_PIPELINE=False) — bin_type={bin_type}")
            time.sleep(2.0)   # mô phỏng thời gian servo quay
            if on_done_callback:
                on_done_callback()
        threading.Thread(target=_mock, daemon=True).start()

    def read_fill_levels(arduino_serial, timeout=3.0):
        return None   # không có cảm biến thực

    _SIM_EMPTY_DISTANCE_CM = {
        "ORGANIC":    41.0,
        "RECYCLABLE": 41.0,
        "HAZARDOUS":  41.0,
        "OTHER":      41.0,
    }
    _SIM_TRASH_HEIGHT_CM = 27.0
    _SIM_DROP_MIN_RATIO = 0.05
    _SIM_DROP_MAX_RATIO = 0.10
    _sim_distance_cm = dict(_SIM_EMPTY_DISTANCE_CM)
    _sim_lock = threading.Lock()

    def _sim_distance_to_fill_pct(bin_name, distance_cm):
        filled_height = _SIM_EMPTY_DISTANCE_CM[bin_name] - distance_cm
        fill_pct = (filled_height / _SIM_TRASH_HEIGHT_CM) * 100.0
        return round(max(0.0, min(100.0, fill_pct)), 1)

    def read_fill_levels_simulated(added_bin=None):
        """Trả mock fill_levels có trạng thái giống arduinoUtil.read_fill_levels_simulated()."""
        import random
        with _sim_lock:
            if added_bin in _sim_distance_cm:
                delta = random.uniform(_SIM_DROP_MIN_RATIO, _SIM_DROP_MAX_RATIO) * _SIM_TRASH_HEIGHT_CM
                min_dist = _SIM_EMPTY_DISTANCE_CM[added_bin] - _SIM_TRASH_HEIGHT_CM
                old_dist = _sim_distance_cm[added_bin]
                new_dist = max(min_dist, old_dist - delta)
                _sim_distance_cm[added_bin] = new_dist
                print(
                    f"[MOCK ARDUINO] {added_bin}: "
                    f"distance {old_dist:.1f}cm → {new_dist:.1f}cm "
                    f"(+{delta:.1f}cm rác)"
                )

            result = {}
            for bin_name, dist in _sim_distance_cm.items():
                result[bin_name] = {
                    "distance_cm": round(dist, 1),
                    "fill_pct":    _sim_distance_to_fill_pct(bin_name, dist),
                }
        print(f"[MOCK ARDUINO] fill_levels (simulated): {result}")
        return result


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
# 4. KHỞI TẠO FIREBASE, ARDUINO, CLOUDINARY
# ============================================================

firebase_ok      = init_firebase()
cloudinary_ok    = init_cloudinary()
firestore_client = firebase_get_firestore_client()
arduino_serial   = init_arduino() if FULL_PIPELINE else None


# ============================================================
# 5. LOAD ONNX MODEL
# ============================================================

_providers = (
    ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if ort.get_device() == 'GPU'
    else ['CPUExecutionProvider']
)

ort_session   = ort.InferenceSession(ONNX_PATH, providers=_providers)
_input_name   = ort_session.get_inputs()[0].name
_out_logits   = ort_session.get_outputs()[0].name
_out_obj      = ort_session.get_outputs()[1].name

print(f"[INFO] ONNX model: {ONNX_PATH}")
print(f"[INFO] Provider  : {ort_session.get_providers()}")


# ============================================================
# 6. ADAPTIVE GAMMA CORRECTION
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
        log_mean   = math.log(mean_v   / 255.0)
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
            lut             = (np.power(self._idx, gamma) * 255.0)
            self._lut       = lut.clip(0, 255).astype(np.uint8)
            self._last_gamma = gamma
        return cv2.LUT(img_rgb, self._lut)

    def get_last_gamma(self):
        return self._last_gamma


agc = FastAdaptiveGamma(target=AGC_TARGET, g_min=AGC_MIN, g_max=AGC_MAX)


# ============================================================
# 7. SQUARE CROP
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
# 8. TRANSFORMS
# ============================================================

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

tf_base = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

_tta_size    = int(img_size * 1.12)
tf_tta_list  = [
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
# 9. ONNX INFERENCE
# ============================================================

def _run_inference(img_rgb_square):
    tensors = [tf_base(img_rgb_square).unsqueeze(0).numpy()]
    if N_TTA > 1:
        for tf in tf_tta_list[:N_TTA - 1]:
            tensors.append(tf(img_rgb_square).unsqueeze(0).numpy())

    n_cls      = len(classes)
    probs_acc  = np.zeros(n_cls, dtype=np.float32)
    obj_acc    = 0.0

    for t in tensors:
        outputs   = ort_session.run([_out_logits, _out_obj], {_input_name: t})
        logits_np = outputs[0][0]
        obj_np    = outputs[1][0]
        e          = np.exp(logits_np - logits_np.max())
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
# 10. INFERENCE WORKER (non-blocking thread)
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
        h, w  = frame_rgb.shape[:2]
        frame_agc  = agc.apply(frame_rgb)
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
        if not _infer_busy and _infer_request is None:
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
# 11. OCCUPANCY DETECTION
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
            best_bbox = (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)

    frozen_diff_val  = 0.0
    frozen_occupied  = False
    frozen_pix       = 0
    if bg_snapshot_gray is not None and curr_gray is not None:
        fz_diff     = cv2.absdiff(curr_gray[y1:y2, x1:x2],
                                  bg_snapshot_gray[y1:y2, x1:x2].astype(np.uint8))
        frozen_diff_val = float(fz_diff.mean())
        _, fz_mask = cv2.threshold(
            fz_diff, FROZEN_PIXEL_GRAY_THRESH, 255, cv2.THRESH_BINARY
        )
        frozen_pix = cv2.countNonZero(fz_mask)
        frozen_occupied = (
            frozen_diff_val > FROZEN_DIFF_THRESH
            or frozen_pix > FROZEN_PIXEL_THRESH
        )

    is_occupied = (
        (mog2_pix > MOG2_PIXEL_THRESH)
        or (diff_pix > DIFF_PIXEL_THRESH)
        or frozen_occupied
        or contour_occupied
    )
    return is_occupied, mog2_pix, diff_pix, frozen_diff_val, frozen_pix, mog_diff_weak, best_bbox


def update_background_snapshot(curr_gray, alpha):
    global ema_bg_gray
    curr_float = curr_gray.astype(np.float32)
    if ema_bg_gray is None:
        ema_bg_gray = curr_float
    else:
        ema_bg_gray = (1 - alpha) * ema_bg_gray + alpha * curr_float


# ============================================================
# 12. DISPLAY HELPERS
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


def draw_action_status(canvas, text, color=(0, 255, 200)):
    h = canvas.shape[0]
    cv2.putText(canvas, text,
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 2, cv2.LINE_AA)


# ============================================================
# 13. LCD I2C HELPERS
# ============================================================

class LcdDisplay:
    LCD_CHR = 1
    LCD_CMD = 0
    LCD_LINE_1 = 0x80
    LCD_LINE_2 = 0xC0
    ENABLE = 0b00000100
    BACKLIGHT = 0x08

    def __init__(self, enabled=True, address=0x27, width=16, bus_id=1):
        self.enabled = enabled
        self.address = address
        self.width = width
        self.bus = None
        self._last_lines = None
        self._last_write_at = 0.0
        self._lock = threading.Lock()

        if not enabled:
            print("[LCD] Disabled (LCD_ENABLED=false).")
            return

        try:
            import smbus
            self.bus = smbus.SMBus(bus_id)
            self._init_lcd()
            self.write("Smart Bin", "Starting...", force=True)
            print(f"[LCD] OK address=0x{address:02X}, width={width}, bus={bus_id}")
        except Exception as e:
            self.enabled = False
            self.bus = None
            print(f"[LCD] OFFLINE: {e}")

    def _toggle_enable(self, bits):
        time.sleep(0.0005)
        self.bus.write_byte(self.address, bits | self.ENABLE)
        time.sleep(0.0005)
        self.bus.write_byte(self.address, bits & ~self.ENABLE)
        time.sleep(0.0005)

    def _byte(self, bits, mode):
        high_bits = mode | (bits & 0xF0) | self.BACKLIGHT
        low_bits = mode | ((bits << 4) & 0xF0) | self.BACKLIGHT
        self.bus.write_byte(self.address, high_bits)
        self._toggle_enable(high_bits)
        self.bus.write_byte(self.address, low_bits)
        self._toggle_enable(low_bits)

    def _init_lcd(self):
        for cmd in (0x33, 0x32, 0x06, 0x0C, 0x28, 0x01):
            self._byte(cmd, self.LCD_CMD)
        time.sleep(0.005)

    def _write_line(self, message, line):
        message = str(message).encode("ascii", errors="replace").decode("ascii")
        message = message[:self.width].ljust(self.width, " ")
        self._byte(line, self.LCD_CMD)
        for char in message:
            self._byte(ord(char), self.LCD_CHR)

    def write(self, line1, line2="", force=False):
        if not self.enabled or self.bus is None:
            return
        lines = (str(line1)[:self.width], str(line2)[:self.width])
        now = time.time()
        if not force and lines == self._last_lines:
            return
        if not force and (now - self._last_write_at) < 0.35:
            return

        with self._lock:
            try:
                self._write_line(lines[0], self.LCD_LINE_1)
                self._write_line(lines[1], self.LCD_LINE_2)
                self._last_lines = lines
                self._last_write_at = now
            except OSError as e:
                self.enabled = False
                print(f"[LCD] Lỗi giao tiếp, tắt LCD: {e}")

    def clear(self):
        if not self.enabled or self.bus is None:
            return
        with self._lock:
            try:
                self._byte(0x01, self.LCD_CMD)
                self._last_lines = None
                time.sleep(0.005)
            except OSError:
                self.enabled = False


LCD_BIN_LABELS = {
    "ORGANIC":    "Organic bin",
    "RECYCLABLE": "Recycle bin",
    "HAZARDOUS":  "Hazard bin",
    "OTHER":      "Other bin",
}


lcd = LcdDisplay(
    enabled=LCD_ENABLED,
    address=LCD_ADDR,
    width=LCD_WIDTH,
    bus_id=LCD_BUS_ID,
)


def lcd_show(line1, line2="", force=False):
    lcd.write(line1, line2, force=force)


# ============================================================
# 14. STATE MACHINE CONSTANTS
# ============================================================

STATE_WARMUP     = "WARMUP"
STATE_WAITING    = "WAITING"
STATE_DETECTING  = "DETECTING"
STATE_LOCKED     = "LOCKED"
STATE_DISPENSING = "DISPENSING"
STATE_COOLDOWN   = "COOLDOWN"


# ============================================================
# 15. CAMERA & BACKGROUND SUBTRACTOR
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
# 16. STATE VARIABLES
# ============================================================

state         = STATE_WARMUP
warmup_start  = time.time()

object_count       = 0
empty_count        = 0
cooldown_empty_since = None
lock_frame_count   = 0
stable_frame_count = 0
warmup_stable_count = 0

vote_history      = deque(maxlen=VOTE_WINDOW)
locked_class      = None
locked_bin        = None
locked_conf       = 0.0    # confidence tại thời điểm lock
detect_start_time = None
last_object_seen_time = None
last_obj_bbox = None
dispense_event = threading.Event()
dispense_lock = threading.Lock()
dispense_cycle_id = 0
dispense_start_time = None
idle_update_event = threading.Event()
timer_lock = threading.Lock()

# Đường dẫn ảnh snapshot hiện tại (chụp lúc LOCKED, xoá sau upload)
current_snapshot_path = None

smoothed_probs = np.zeros(len(classes), dtype=np.float32)
fps_history    = deque(maxlen=30)

prev_gray    = None
ema_bg_gray  = None

frame_counter = 0
last_infer    = None

# --- Idle periodic update tracking ---
last_classification_time = time.monotonic()   # cập nhật mỗi lần _on_arduino_done() xong
last_idle_update_time    = time.monotonic()   # cập nhật mỗi lần idle update chạy


def mark_activity_timers(now=None):
    global last_classification_time, last_idle_update_time
    ts = time.monotonic() if now is None else now
    with timer_lock:
        last_classification_time = ts
        last_idle_update_time = ts


def mark_idle_update_time(now=None):
    global last_idle_update_time
    ts = time.monotonic() if now is None else now
    with timer_lock:
        last_idle_update_time = ts


def get_idle_timer_snapshot(now=None):
    ts = time.monotonic() if now is None else now
    with timer_lock:
        return ts - last_classification_time, ts - last_idle_update_time


def start_idle_update_thread():
    if idle_update_event.is_set():
        return False

    idle_update_event.set()

    def _idle_update():
        try:
            _idle_fill = (
                read_fill_levels(arduino_serial)
                if (FULL_PIPELINE and USE_ULTRASONIC)
                else read_fill_levels_simulated()
            )
            if _idle_fill is None:
                _idle_fill = read_fill_levels_simulated()

            firebase_log_sensor(
                firebase_ok  = firebase_ok,
                bin_id       = BIN_ID,
                fill_levels  = _idle_fill,
            )
        finally:
            idle_update_event.clear()

    threading.Thread(target=_idle_update, daemon=True).start()
    return True


print(f"[INIT] FULL_PIPELINE  : {FULL_PIPELINE}")
print(f"[INIT] WARMUP {WARMUP_SEC:.0f}-{WARMUP_MAX_SEC:.0f}s | TTA={N_TTA} | img_size={img_size}")
print(f"[INIT] BIN_ID={BIN_ID} | Firebase={'OK' if firebase_ok else 'OFFLINE'}")
print(f"[INIT] Cloudinary={'OK' if cloudinary_ok else 'OFFLINE'}")
print(f"[INIT] Display={'ON' if DISPLAY_ENABLED else 'OFF'}")
print(f"[INIT] LCD={'OK' if lcd.enabled else 'OFF'}")
if FULL_PIPELINE:
    print(f"[INIT] Arduino={'OK' if arduino_serial else 'OFFLINE'} (servo + siêu âm)")
else:
    print(f"[INIT] Arduino=DISABLED (FULL_PIPELINE=False) — fill_levels dùng MOCK")


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

        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        curr_gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        annotated  = frame_bgr.copy() if DISPLAY_ENABLED else None
        h, w       = frame_bgr.shape[:2]

        # ROI tuyệt đối
        y1_roi = int(h * ROI_Y1_RATIO); y2_roi = int(h * ROI_Y2_RATIO)
        x1_roi = int(w * ROI_X1_RATIO); x2_roi = int(w * ROI_X2_RATIO)
        roi_coords = (y1_roi, y2_roi, x1_roi, x2_roi)

        # ── Background subtraction ───────────────────────────────────
        if state == STATE_WARMUP:
            lr = MOG2_LR_WARMUP
        elif state == STATE_WAITING:
            lr = MOG2_LR_WAITING
        elif state == STATE_COOLDOWN:
            lr = MOG2_LR_COOLDOWN
        else:
            lr = 0.0
        raw_mask = backSub.apply(frame_rgb, learningRate=lr)
        fg_mask  = np.where(raw_mask == 255, 255, 0).astype(np.uint8)
        fg_mask  = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  mog_kernel)
        fg_mask  = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, mog_kernel)

        # ── Occupancy ────────────────────────────────────────────────
        is_occupied, mog2_pix, diff_pix, frozen_diff, frozen_pix, mog_diff_weak, obj_bbox = \
            compute_occupancy(fg_mask, roi_coords, prev_gray, curr_gray,
                              bg_snapshot_gray=ema_bg_gray)

        # EMA snapshot là mốc phát hiện vật đứng yên. Học nhanh trong warmup
        # khi thùng đang trống; sau đó chỉ học chậm ở các pha idle/cooldown.
        if state == STATE_WARMUP:
            update_background_snapshot(curr_gray, BG_WARMUP_ALPHA)
        elif state == STATE_WAITING and mog_diff_weak and object_count == 0:
            update_background_snapshot(curr_gray, BG_IDLE_ALPHA)
        elif state == STATE_COOLDOWN and mog_diff_weak and not is_occupied:
            update_background_snapshot(curr_gray, BG_COOLDOWN_ALPHA)

        if DISPLAY_ENABLED:
            sq_crop = get_square_crop(obj_bbox, h, w)

            # Vẽ ROI + crop box
            roi_color = (0, 200, 255) if is_occupied else (60, 60, 60)
            cv2.rectangle(annotated, (x1_roi, y1_roi), (x2_roi, y2_roi), roi_color, 2)
            if sq_crop:
                sx1, sy1, sx2, sy2 = sq_crop
                cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (255, 200, 0), 1)

            occ_txt = f"occ={mog2_pix} diff={diff_pix} frz={frozen_diff:.1f}/{frozen_pix}"
            cv2.putText(annotated, occ_txt,
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (150, 150, 150), 1, cv2.LINE_AA)

        # ============================================================
        # STATE: WARMUP
        # ============================================================
        if state == STATE_WARMUP:
            elapsed_w = time.time() - warmup_start
            remain_w  = max(0.0, WARMUP_SEC - elapsed_w)
            warmup_diff = None
            if prev_gray is not None:
                py1, py2, px1, px2 = roi_coords
                warmup_diff = float(np.mean(cv2.absdiff(
                    prev_gray[py1:py2, px1:px2],
                    curr_gray[py1:py2, px1:px2],
                )))
                if warmup_diff < WARMUP_DIFF_THRESH:
                    warmup_stable_count = min(
                        warmup_stable_count + 1,
                        WARMUP_STABLE_FRAMES,
                    )
                else:
                    warmup_stable_count = 0

            bg_ready = (
                ema_bg_gray is not None
                and (
                    (
                        elapsed_w >= WARMUP_SEC
                        and warmup_stable_count >= WARMUP_STABLE_FRAMES
                    )
                    or elapsed_w >= WARMUP_MAX_SEC
                )
            )
            lcd_show("Learning BG", f"{remain_w:4.1f}s {warmup_stable_count:02d}/{WARMUP_STABLE_FRAMES}")
            if DISPLAY_ENABLED:
                diff_txt = "--" if warmup_diff is None else f"{warmup_diff:.1f}"
                cv2.putText(annotated,
                            f"WARMUP BG... {remain_w:.1f}s stable {warmup_stable_count}/{WARMUP_STABLE_FRAMES} diff={diff_txt}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 200, 255), 2, cv2.LINE_AA)
            if bg_ready:
                state = STATE_WAITING
                lcd_show("Ready", "Waiting trash", force=True)
                print(
                    "[STATE] WARMUP → WAITING "
                    f"(elapsed={elapsed_w:.1f}s, stable={warmup_stable_count}/{WARMUP_STABLE_FRAMES})"
                )

        # ============================================================
        # STATE: WAITING
        # ============================================================
        elif state == STATE_WAITING:
            if DISPLAY_ENABLED:
                cv2.putText(annotated, "WAITING for object...",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (160, 160, 160), 2, cv2.LINE_AA)
            if is_occupied:
                object_count += 1
                if object_count >= OBJECT_CONFIRM_FRAMES:
                    state             = STATE_DETECTING
                    object_count      = 0
                    vote_history      = deque(maxlen=VOTE_WINDOW)
                    smoothed_probs[:] = 0.0
                    last_infer        = None
                    detect_start_time = time.time()
                    last_object_seen_time = detect_start_time
                    last_obj_bbox = obj_bbox
                    stable_frame_count = 0
                    lcd_show("Identifying", "Hold still...", force=True)
                    print("[STATE] WAITING → DETECTING")
            else:
                object_count = 0

        # ============================================================
        # STATE: DETECTING
        # ============================================================
        elif state == STATE_DETECTING:
            if is_occupied:
                last_object_seen_time = time.time()
                if obj_bbox is not None:
                    last_obj_bbox = obj_bbox
                empty_count = 0
            else:
                empty_count += 1
                elapsed_detect = time.time() - detect_start_time if detect_start_time else 0.0
                lost_for_sec = (
                    time.time() - last_object_seen_time
                    if last_object_seen_time is not None
                    else 0.0
                )
                if (empty_count >= EMPTY_DETECT_FRAMES
                        and lost_for_sec >= DETECT_LOST_GRACE_SEC
                        and elapsed_detect >= DETECT_MIN_HOLD_SEC):
                    state        = STATE_WAITING
                    empty_count  = 0
                    object_count = 0
                    last_infer   = None
                    detect_start_time  = None
                    last_object_seen_time = None
                    last_obj_bbox = None
                    stable_frame_count = 0
                    smoothed_probs[:]  = 0.0
                    vote_history       = deque(maxlen=VOTE_WINDOW)
                    lcd_show("Ready", "Waiting trash", force=True)
                    print("[STATE] DETECTING → WAITING (lost object)")

            # Stability gate
            if prev_gray is not None and curr_gray is not None:
                diff_val = float(np.mean(cv2.absdiff(prev_gray, curr_gray)))
                if diff_val < STABILITY_DIFF_THRESH:
                    stable_frame_count = min(stable_frame_count + 1,
                                             STABLE_FRAMES_REQUIRED + 10)
                else:
                    stable_frame_count = 0

            elapsed_detect    = time.time() - detect_start_time if detect_start_time else 0.0
            vote_delay_passed = elapsed_detect >= VOTE_DELAY_SEC
            gate_open         = vote_delay_passed and (stable_frame_count >= STABLE_FRAMES_REQUIRED)

            if gate_open and frame_counter % INFER_EVERY_N == 0:
                submit_inference(frame_rgb, crop_roi=obj_bbox or last_obj_bbox)

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
                        locked_conf  = conf
                        lock_reason  = f"vote {cand_count}/{n_valid}"

                if not lock_ready and detect_start_time is not None:
                    if elapsed_detect >= VOTE_TIMEOUT_SEC:
                        if valid_votes:
                            candidate   = max(set(valid_votes), key=valid_votes.count)
                            cand_count  = valid_votes.count(candidate)
                            locked_conf = conf
                            lock_reason = f"TIMEOUT {elapsed_detect:.1f}s vote({cand_count}/{n_valid})"
                        else:
                            best_idx    = int(np.argmax(smoothed_probs))
                            candidate   = classes[best_idx]
                            locked_conf = float(smoothed_probs[best_idx])
                            lock_reason = f"TIMEOUT {elapsed_detect:.1f}s prob({smoothed_probs[best_idx]:.2f})"
                        lock_ready   = True
                        locked_class = candidate
                        locked_bin   = get_bin(locked_class)

                if lock_ready:
                    state             = STATE_LOCKED
                    empty_count       = 0
                    lock_frame_count  = 0
                    detect_start_time = None
                    last_object_seen_time = None
                    last_obj_bbox = None
                    lcd_show("Detected", LCD_BIN_LABELS.get(locked_bin, locked_bin), force=True)
                    print(f"[STATE] DETECTING → LOCKED: "
                          f"{locked_class} → {locked_bin}  [{lock_reason}]")

                if DISPLAY_ENABLED:
                    # HUD
                    time_left = ""
                    if detect_start_time is not None:
                        remaining_vote = max(0.0, VOTE_TIMEOUT_SEC - elapsed_detect)
                        time_left = f"  T-{remaining_vote:.1f}s"
                        timer_ratio = remaining_vote / VOTE_TIMEOUT_SEC
                        timer_color = (0, int(255 * timer_ratio), int(255 * (1 - timer_ratio)))
                        tw = int((1 - timer_ratio) * 200)
                        cv2.rectangle(annotated, (10, 72), (210, 82), (40, 40, 40), -1)
                        cv2.rectangle(annotated, (10, 72), (10 + tw, 82), timer_color, -1)

                    conf_color = (0, int(255 * conf), int(255 * (1 - conf)))
                    gate_txt   = "" if gate_open else f"  [WAIT {stable_frame_count}/{STABLE_FRAMES_REQUIRED}]"
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
                if DISPLAY_ENABLED:
                    cv2.putText(annotated, "DETECTING: waiting inference...",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (100, 180, 255), 2, cv2.LINE_AA)

            if DISPLAY_ENABLED:
                draw_prob_bars(annotated, smoothed_probs, classes)

        # ============================================================
        # STATE: LOCKED  →  Chụp ảnh + kích hoạt DISPENSING
        # ============================================================
        elif state == STATE_LOCKED:
            lock_frame_count += 1
            if DISPLAY_ENABLED:
                draw_locked_banner(annotated, locked_class, locked_bin, BIN_COLORS)
                draw_action_status(annotated, "LOCKED — chuẩn bị đổ rác...", (0, 255, 200))

            if lock_frame_count >= MIN_LOCK_HOLD_FRAMES:
                # Chụp ảnh tại thời điểm lock (luôn thực hiện)
                current_snapshot_path = capture_snapshot(cap, BIN_ID, locked_bin)

                state = STATE_DISPENSING
                with dispense_lock:
                    dispense_cycle_id += 1
                    _cycle_id = dispense_cycle_id
                    dispense_event.clear()
                    dispense_start_time = time.monotonic()
                lcd_show("Dropping trash", LCD_BIN_LABELS.get(locked_bin, locked_bin), force=True)
                print(f"[STATE] LOCKED → DISPENSING: {locked_bin}"
                      f"  (Arduino={'ON' if FULL_PIPELINE else 'MOCK'})")

                # Capture các giá trị cần dùng trong closure
                _snap = current_snapshot_path
                _cls  = locked_class
                _bin  = locked_bin
                _conf = locked_conf

                def _on_arduino_done():
                    print("[DISPENSE] Hoàn tất. Cập nhật Firebase + Cloudinary...")
                    lcd_show("Drop complete", "Updating data", force=True)
                    time.sleep(POST_DISPENSE_SETTLE_SEC)

                    # fill_levels: đọc cảm biến siêu âm thực khi FULL_PIPELINE=True,
                    #              dùng mock data khi FULL_PIPELINE=False
                    if FULL_PIPELINE and USE_ULTRASONIC:
                        fill_levels = read_fill_levels(arduino_serial)
                        if fill_levels is None:
                            print("[DISPENSE] Cảm biến siêu âm chưa phản hồi — dùng simulated.")
                            fill_levels = read_fill_levels_simulated(_bin)
                    else:
                        fill_levels = read_fill_levels_simulated(_bin)

                    # Ghi log cảm biến vào bin_raw_sensor_logs
                    firebase_log_sensor(
                        firebase_ok  = firebase_ok,
                        bin_id       = BIN_ID,
                        fill_levels  = fill_levels,
                    )

                    # Upload Cloudinary + ghi classification_logs Firestore
                    upload_and_log(
                        cloudinary_ok    = cloudinary_ok,
                        firebase_ok      = firebase_ok,
                        firestore_client = firestore_client,
                        bin_id           = BIN_ID,
                        bin_type         = _bin,
                        locked_class     = _cls,
                        confidence_score = _conf,
                        local_image_path = _snap,
                    )

                    # Reset timer idle: tính từ lúc pipeline xử lý xong
                    mark_activity_timers()
                    with dispense_lock:
                        if _cycle_id == dispense_cycle_id:
                            dispense_event.set()
                        else:
                            print(f"[DISPENSE] Bỏ qua callback cũ cycle={_cycle_id}.")

                # arduino_send_command dùng stub 2s-delay khi FULL_PIPELINE=False
                arduino_send_command(arduino_serial, locked_bin,
                                     on_done_callback=_on_arduino_done)

        # ============================================================
        # STATE: DISPENSING  →  chờ Arduino ACK xong
        # ============================================================
        elif state == STATE_DISPENSING:
            if DISPLAY_ENABLED:
                draw_locked_banner(annotated, locked_class, locked_bin, BIN_COLORS)

            with dispense_lock:
                _dispense_start = dispense_start_time
                _dispense_cycle = dispense_cycle_id
            dispense_done = dispense_event.is_set()
            dispense_elapsed = (
                time.monotonic() - _dispense_start
                if _dispense_start is not None
                else 0.0
            )
            dispense_timeout = dispense_elapsed >= MAX_DISPENSE_TIMEOUT_SEC
            arduino_status = "DONE ✓" if dispense_done else f"đang xử lý... {dispense_elapsed:.1f}s"
            if DISPLAY_ENABLED:
                draw_action_status(annotated,
                                   f"DISPENSING → {locked_bin}  [{arduino_status}]",
                                   (0, 220, 255))

            if dispense_done or dispense_timeout:
                if dispense_timeout and not dispense_done:
                    with dispense_lock:
                        dispense_cycle_id += 1
                    print(
                        "[WARN] DISPENSING timeout — chuyển sang COOLDOWN "
                        f"sau {dispense_elapsed:.1f}s"
                    )
                    lcd_show("Dispense timeout", "Cooling down", force=True)
                state          = STATE_COOLDOWN
                cooldown_empty_since = None
                with dispense_lock:
                    if _dispense_cycle == dispense_cycle_id:
                        dispense_start_time = None
                if dispense_done:
                    lcd_show("Done", "Cooling down", force=True)
                print("[STATE] DISPENSING → COOLDOWN")

        # ============================================================
        # STATE: COOLDOWN
        # ============================================================
        elif state == STATE_COOLDOWN:
            now_mono = time.monotonic()
            if not is_occupied:
                if cooldown_empty_since is None:
                    cooldown_empty_since = now_mono
            else:
                cooldown_empty_since = None

            cooldown_elapsed = (
                now_mono - cooldown_empty_since
                if cooldown_empty_since is not None
                else 0.0
            )
            remaining_cd = max(0.0, COOLDOWN_SEC - cooldown_elapsed)
            if DISPLAY_ENABLED:
                cv2.putText(annotated, f"COOLDOWN... {remaining_cd:.1f}s, waiting for empty bin",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 165, 255), 2, cv2.LINE_AA)
            if cooldown_elapsed >= COOLDOWN_SEC:
                state                 = STATE_WAITING
                locked_class          = None
                locked_bin            = None
                locked_conf           = 0.0
                lock_frame_count      = 0
                empty_count           = 0
                object_count          = 0
                last_infer            = None
                detect_start_time     = None
                last_object_seen_time = None
                last_obj_bbox         = None
                stable_frame_count    = 0
                smoothed_probs[:]     = 0.0
                vote_history          = deque(maxlen=VOTE_WINDOW)
                dispense_event.clear()
                with dispense_lock:
                    dispense_start_time = None
                cooldown_empty_since = None
                current_snapshot_path = None
                lcd_show("Ready", "Waiting trash", force=True)
                print("[STATE] COOLDOWN → WAITING (cycle reset)")

        # ============================================================
        # IDLE PERIODIC UPDATE
        # Nếu state == WAITING và không có rác >= IDLE_UPDATE_INTERVAL_SEC
        # kể từ lần phân loại cuối → update Firebase định kỳ để giữ bin ONLINE.
        # ============================================================
        if state == STATE_WAITING:
            _now = time.monotonic()
            _idle_since, _since_last_idle_push = get_idle_timer_snapshot(_now)

            if (_idle_since          >= IDLE_UPDATE_INTERVAL_SEC
                    and _since_last_idle_push >= IDLE_UPDATE_INTERVAL_SEC
                    and not idle_update_event.is_set()):

                print(f"[IDLE] {_idle_since:.0f}s không có rác — "
                      f"cập nhật Firebase định kỳ...")
                mark_idle_update_time(_now)
                start_idle_update_thread()

        prev_gray = curr_gray.copy()

        # ── FPS ──────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        fps_history.append(1.0 / (elapsed + 1e-9))
        avg_fps = int(np.mean(fps_history))

        if DISPLAY_ENABLED:
            state_color_map = {
                STATE_WARMUP:     (0, 200, 255),
                STATE_WAITING:    (160, 160, 160),
                STATE_DETECTING:  (0, 180, 255),
                STATE_LOCKED:     BIN_COLORS.get(locked_bin, (100, 100, 100))
                                  if locked_bin else (0, 220, 0),
                STATE_DISPENSING: (0, 220, 255),
                STATE_COOLDOWN:   (0, 165, 255),
            }
            fps_color = state_color_map.get(state, (0, 255, 255))
            busy_txt  = "*" if _infer_busy else " "
            cv2.putText(annotated,
                        f"FPS:{avg_fps}  TTA:{N_TTA}x  [{state}]{busy_txt}",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        fps_color, 1, cv2.LINE_AA)

            cv2.imshow("SmartTrashBin v5.0", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    lcd_show("Smart Bin", "Shutting down", force=True)
    with _infer_lock:
        _infer_request = "STOP"
    _worker_thread.join(timeout=3.0)
    if arduino_serial:
        arduino_serial.close()
        print("[EXIT] Đóng cổng Arduino.")
    cap.release()
    if DISPLAY_ENABLED:
        cv2.destroyAllWindows()
    lcd.clear()
    print("[EXIT] Đã thoát.")
