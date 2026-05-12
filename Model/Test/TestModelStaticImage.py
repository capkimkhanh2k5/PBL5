"""
TestModel.py  —  v3.0  (ONNX + Adaptive Gamma + Square-aware Pipeline)
=======================================================================
Cập nhật để khớp với flow TrainModel v7 + testPC v3.0:

[MODEL]   Chuyển từ PyTorch (.pth) sang ONNX Runtime.
          Không cần định nghĩa lại WasteDetectorModel / GeM / WasteDetector.
          Đọc classes, img_size, agc_target từ model_meta.json (xuất
          cùng ONNX trong notebook) thay vì hard-code.

[AGC]     Thêm FastAdaptiveGamma — mirror đúng pipeline v7:
          Đo mean brightness kênh V (max RGB) trên TOÀN BỘ ảnh,
          tính gamma = log(target/255) / log(mean_V/255), build LUT O(1).
          Áp dụng TRƯỚC Resize/CenterCrop để đo sáng đại diện.

[TF]      Val transform khớp đúng v7:
            Resize(int) → AGC → CenterCrop(img_size) → ToTensor → Normalize
          (Resize int giữ aspect ratio, CenterCrop lấy vùng trung tâm)

[ONNX]    Inference qua ort.InferenceSession, outputs: ['logits', 'objectness']
          Softmax + Sigmoid thủ công bằng numpy — không cần torch.nn.

[GRADCAM] GradCAM giữ lại nhưng dùng PyTorch model load từ .pth (optional)
          nếu file .pth tồn tại bên cạnh ONNX. Nếu không có .pth, bounding
          box được tạo từ contour của foreground mask đơn giản.
=======================================================================
"""

import os
import base64
import time
import math
import json
import io

import numpy as np
import cv2
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template_string

# ============================================================
# 1. ĐƯỜNG DẪN FILE
# ============================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.dirname(SCRIPT_DIR)

TEST_DIR    = os.path.join(SCRIPT_DIR, "TestImage")
ONNX_PATH   = os.path.join(MODEL_DIR, "Train", "outputs", "waste_detector.onnx")
META_PATH   = os.path.join(MODEL_DIR, "Train", "outputs", "model_meta.json")
PTH_PATH    = os.path.join(MODEL_DIR, "Train", "outputs", "best_model.pth")

# ============================================================
# 2. ĐỌC MODEL META  (classes, img_size, AGC config)
# ============================================================

IMG_SIZE   = 384   # fallback

_meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        _meta = json.load(f)
    print(f"[INFO] Loaded meta: {META_PATH}")
else:
    print(f"[WARN] {META_PATH} not found — dùng giá trị mặc định")

CLASS_NAMES = _meta.get('classes', [
    'Battery', 'Biological', 'General_Waste',
    'Glass', 'Metal', 'Paper_Cardboard', 'Plastic'
])
IMG_SIZE    = _meta.get('img_size',        IMG_SIZE)
AGC_TARGET  = _meta.get('agc_target',      128)
AGC_MIN     = _meta.get('agc_gamma_min',   0.4)
AGC_MAX     = _meta.get('agc_gamma_max',   3.0)

print(f"[INFO] Classes   : {CLASS_NAMES}")
print(f"[INFO] img_size  : {IMG_SIZE}")
print(f"[INFO] AGC       : target={AGC_TARGET}  clip=[{AGC_MIN}, {AGC_MAX}]")

# ============================================================
# 3. CLASS ALIAS & WASTE CATEGORIES
# ============================================================

CLASS_ALIASES = {
    'battery':           'Battery',
    'biological':        'Biological',
    'general_waste':     'General_Waste',
    'trash':             'General_Waste',
    'glass':             'Glass',
    'brown-glass':       'Glass',
    'white-glass':       'Glass',
    'green-glass':       'Glass',
    'metal':             'Metal',
    'paper_cardboard':   'Paper_Cardboard',
    'paper-cardboard':   'Paper_Cardboard',
    'paper':             'Paper_Cardboard',
    'cardboard':         'Paper_Cardboard',
    'plastic':           'Plastic',
    'textiles':          'General_Waste',
    'clothes':           'General_Waste',
    'shoes':             'General_Waste',
}

WASTE_CATEGORIES = {
    'Battery':         {'category': 'Hazardous',      'icon': 'alert-triangle', 'color': '#FF003C', 'bg': 'rgba(255,0,60,0.15)',    'border': 'rgba(255,0,60,0.4)'},
    'Biological':      {'category': 'Organic',         'icon': 'leaf',           'color': '#0AFF00', 'bg': 'rgba(10,255,0,0.15)',    'border': 'rgba(10,255,0,0.4)'},
    'General_Waste':   {'category': 'Non-Recyclable',  'icon': 'trash-2',        'color': '#94A3B8', 'bg': 'rgba(148,163,184,0.15)', 'border': 'rgba(148,163,184,0.4)'},
    'Glass':           {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)',  'border': 'rgba(59,130,246,0.4)'},
    'Metal':           {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)',  'border': 'rgba(59,130,246,0.4)'},
    'Paper_Cardboard': {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)',  'border': 'rgba(59,130,246,0.4)'},
    'Plastic':         {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)',  'border': 'rgba(59,130,246,0.4)'},
}


def normalize_class_name(class_name: str) -> str:
    key = str(class_name).strip().lower().replace(' ', '_').replace('-', '_')
    return CLASS_ALIASES.get(key, class_name)


def get_waste_category(class_name: str) -> dict:
    normalized = normalize_class_name(class_name)
    return WASTE_CATEGORIES.get(normalized, {
        'category': 'Unknown', 'icon': 'help-circle',
        'color': '#64748B', 'bg': 'rgba(100,116,139,0.15)',
        'border': 'rgba(100,116,139,0.4)'
    })


def get_actual_class(filename: str) -> str:
    """Trích class thực tế từ tên file."""
    name = filename.lower()
    OLD_TO_NEW = {
        'battery':        'Battery',
        'biological':     'Biological',
        'cardboard':      'Paper_Cardboard',
        'paper':          'Paper_Cardboard',
        'paper_cardboard':'Paper_Cardboard',
        'paper-cardboard':'Paper_Cardboard',
        'clothes':        'General_Waste',
        'shoes':          'General_Waste',
        'trash':          'General_Waste',
        'glass':          'Glass',
        'brown-glass':    'Glass',
        'white-glass':    'Glass',
        'green-glass':    'Glass',
        'metal':          'Metal',
        'plastic':        'Plastic',
        'textiles':       'General_Waste',
        'general_waste':  'General_Waste',
    }
    for old, new in OLD_TO_NEW.items():
        if old in name:
            return new
    return "unknown"


# ============================================================
# 4. ADAPTIVE GAMMA CORRECTION  (FastAdaptiveGamma — numpy + cv2.LUT)
# ============================================================
# Mirror đúng testPC v3.0: đo mean_V trên TOÀN BỘ ảnh, build LUT O(1).
# ============================================================

class FastAdaptiveGamma:
    """
    Adaptive Gamma Correction thuần numpy + cv2.LUT.
    Dùng như torchvision Transform (callable nhận PIL Image, trả PIL Image).

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
        self._last_gamma = -1.0
        self._lut        = None

    def _compute_gamma(self, mean_v: float) -> float:
        mean_v     = float(np.clip(mean_v, 8.0, 247.0))
        log_mean   = math.log(mean_v / 255.0)
        log_target = math.log(self.target / 255.0)
        if abs(log_mean - log_target) < 0.03:
            return 1.0
        return float(np.clip(log_target / log_mean, self.g_min, self.g_max))

    def apply_numpy(self, img_rgb: np.ndarray) -> np.ndarray:
        """img_rgb: (H, W, 3) uint8 RGB → (H, W, 3) uint8 RGB"""
        mean_v = float(img_rgb.max(axis=2).mean())
        gamma  = self._compute_gamma(mean_v)

        if abs(gamma - 1.0) < 0.02:
            return img_rgb

        if abs(gamma - self._last_gamma) > 0.005:
            lut             = (np.power(self._idx, gamma) * 255.0)
            self._lut       = lut.clip(0, 255).astype(np.uint8)
            self._last_gamma = gamma

        return cv2.LUT(img_rgb, self._lut)

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        """Dùng như torchvision transform — nhận PIL, trả PIL."""
        arr     = np.array(pil_img, dtype=np.uint8)
        arr_agc = self.apply_numpy(arr)
        return Image.fromarray(arr_agc)

    def get_last_gamma(self) -> float:
        return self._last_gamma


agc = FastAdaptiveGamma(target=AGC_TARGET, g_min=AGC_MIN, g_max=AGC_MAX)
print(f"[INIT] FastAdaptiveGamma ready  (target={AGC_TARGET})")

# ============================================================
# 5. LOAD ONNX MODEL
# ============================================================

_providers = (
    ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if ort.get_device() == 'GPU'
    else ['CPUExecutionProvider']
)

ort_session   = ort.InferenceSession(ONNX_PATH, providers=_providers)
_input_name   = ort_session.get_inputs()[0].name      # 'image'
_out_logits   = ort_session.get_outputs()[0].name     # 'logits'
_out_obj      = ort_session.get_outputs()[1].name     # 'objectness'

print(f"[INFO] ONNX model  : {ONNX_PATH}")
print(f"[INFO] Provider    : {ort_session.get_providers()}")
print(f"[INFO] Input       : '{_input_name}'  shape={ort_session.get_inputs()[0].shape}")

# ============================================================
# 6. TRANSFORM PIPELINE  (khớp đúng val pipeline v7)
# ============================================================
# Thứ tự bắt buộc:
#   Resize(int) → AGC → CenterCrop(img_size) → ToTensor → Normalize
#
# - Resize(int): giữ aspect ratio, resize cạnh ngắn
# - AGC trước CenterCrop: đo sáng toàn bộ ảnh (không bị bias bởi crop)
# - CenterCrop: lấy vùng trung tâm (giống val transform train)
# ============================================================

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_val_resize = int(IMG_SIZE * 1.15)   # resize nhẹ trước CenterCrop

tf_val = transforms.Compose([
    transforms.Resize(_val_resize),        # int → giữ aspect ratio
    agc,                                   # AGC trước crop
    transforms.CenterCrop(IMG_SIZE),       # crop trung tâm
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# TTA: thêm margin rộng hơn + random crop + flip
_tta_resize = int(IMG_SIZE * 1.25)
tf_tta_list = [
    transforms.Compose([
        transforms.Resize(_tta_resize),
        agc,
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    transforms.Compose([
        transforms.Resize(_tta_resize),
        agc,
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
]

# ============================================================
# 7. ONNX INFERENCE
# ============================================================

def run_inference(pil_img: Image.Image, n_tta: int = 1):
    """
    Chạy inference qua ONNX Runtime với tùy chọn TTA.

    Args:
        pil_img : PIL Image (RGB)
        n_tta   : 1 = tắt TTA (nhanh nhất); 3 = bật 3 passes

    Returns:
        (predicted_class, confidence, obj_score, probs_np, gamma_used)
    """
    # Pass gốc
    tensors = [tf_val(pil_img).unsqueeze(0).numpy()]

    # TTA passes (nếu có)
    if n_tta > 1:
        for tf in tf_tta_list[:n_tta - 1]:
            tensors.append(tf(pil_img).unsqueeze(0).numpy())

    n_cls     = len(CLASS_NAMES)
    probs_acc = np.zeros(n_cls, dtype=np.float32)
    obj_acc   = 0.0

    for t in tensors:
        outputs   = ort_session.run([_out_logits, _out_obj], {_input_name: t})
        logits_np = outputs[0][0]   # (n_classes,)
        obj_np    = outputs[1][0]   # (1,)

        # Softmax thủ công
        e = np.exp(logits_np - logits_np.max())
        probs_acc += (e / e.sum())

        # Sigmoid objectness
        obj_acc += float(1.0 / (1.0 + np.exp(-obj_np[0])))

    probs_acc /= len(tensors)
    obj_acc   /= len(tensors)

    idx          = int(probs_acc.argmax())
    conf         = float(probs_acc[idx])
    predicted    = normalize_class_name(CLASS_NAMES[idx])
    gamma_used   = agc.get_last_gamma()

    return predicted, conf, obj_acc, probs_acc, gamma_used


# ============================================================
# 8. BOUNDING BOX  (contour-based từ foreground, không cần GradCAM)
# ============================================================
# GradCAM yêu cầu backward pass PyTorch — không khả dụng với ONNX.
# Thay thế: dùng saliency đơn giản từ threshold luminance để xác định
# vùng đối tượng, đủ dùng cho visualization trong TestModel.
# ============================================================

def get_bbox_from_saliency(img_rgb: np.ndarray, threshold: float = 0.25):
    """
    Tìm bounding box đối tượng bằng cách threshold độ sáng.
    Trả về (x, y, w, h) hoặc None nếu không tìm được.
    """
    gray = img_rgb.max(axis=2).astype(np.uint8)   # kênh V = max(R,G,B)

    # Đảo: vùng tối nền → trắng, đối tượng → đen (hoặc ngược lại tùy ảnh)
    # Dùng Otsu để tự động threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology để loại noise
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Lọc contour nhỏ
    h_img, w_img = gray.shape
    min_area     = (h_img * w_img) * threshold * 0.1
    valid        = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        return None

    largest        = max(valid, key=cv2.contourArea)
    x, y, bw, bh   = cv2.boundingRect(largest)

    # Padding 8%
    pad_x = int(bw * 0.08)
    pad_y = int(bh * 0.08)
    x  = max(0, x - pad_x)
    y  = max(0, y - pad_y)
    bw = min(w_img - x, bw + 2 * pad_x)
    bh = min(h_img - y, bh + 2 * pad_y)

    return (x, y, bw, bh)


def draw_detection_on_image(img_cv2, bbox, class_name, confidence, obj_score):
    """Vẽ bounding box và label lên ảnh (BGR)."""
    result = img_cv2.copy()

    if bbox is not None:
        x, y, w, h = bbox
        color = (0, 255, 100) if confidence > 0.8 else (
                 (0, 200, 255) if confidence > 0.5 else (0, 100, 255))

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        label        = f"{class_name}: {confidence*100:.1f}%"
        font         = cv2.FONT_HERSHEY_SIMPLEX
        font_scale   = 0.6
        thickness    = 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(result, (x, y - text_h - 10), (x + text_w + 8, y), color, -1)
        cv2.putText(result, label, (x + 4, y - 5), font, font_scale, (0, 0, 0), thickness)

        obj_label = f"Obj: {obj_score*100:.0f}%  γ={agc.get_last_gamma():.2f}"
        cv2.putText(result, obj_label, (x + 4, y + h + 18),
                    font, 0.40, color, 1, cv2.LINE_AA)

    return result


# ============================================================
# 9. CLASSIFY ALL IMAGES IN TestImage FOLDER
# ============================================================

N_TTA = 1   # 1 = tắt TTA (tối đa tốc độ); 3 = bật TTA

def classify_images():
    """Classify tất cả ảnh trong TestImage/ bằng ONNX + AGC pipeline."""
    if not os.path.exists(TEST_DIR):
        print(f"[WARN] '{TEST_DIR}' không tồn tại — đang tạo...")
        os.makedirs(TEST_DIR)
        return [], 0

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images    = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(valid_ext)]

    if not images:
        print(f"[WARN] Không có ảnh trong '{TEST_DIR}'")
        return [], 0

    print(f"[INFO] Tìm thấy {len(images)} ảnh để test")
    results    = []
    start_time = time.time()

    for idx, img_name in enumerate(images):
        img_path = os.path.join(TEST_DIR, img_name)
        try:
            pil_img       = Image.open(img_path).convert('RGB')
            original_size = pil_img.size   # (W, H)

            # ── INFERENCE (ONNX + AGC + correct transform) ──────────
            predicted_class, conf, obj_score, probs, gamma_used = \
                run_inference(pil_img, n_tta=N_TTA)

            # ── BOUNDING BOX (saliency-based) ────────────────────────
            img_rgb  = np.array(pil_img, dtype=np.uint8)
            bbox     = get_bbox_from_saliency(img_rgb) if obj_score > 0.3 else None

            # ── DRAW ─────────────────────────────────────────────────
            img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            if bbox is not None:
                result_img = draw_detection_on_image(img_bgr, bbox, predicted_class, conf, obj_score)
            else:
                result_img = img_bgr.copy()
                label = f"{predicted_class}: {conf*100:.1f}%  γ={gamma_used:.2f}"
                cv2.putText(result_img, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

            # ── ENCODE BASE64 ─────────────────────────────────────────
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(result_rgb)
            pil_result.thumbnail((400, 400))
            buf        = io.BytesIO()
            pil_result.save(buf, format='JPEG', quality=85)
            b64_img    = base64.b64encode(buf.getvalue()).decode()

            # ── METADATA ─────────────────────────────────────────────
            actual_class = get_actual_class(img_name)
            is_correct   = (predicted_class == actual_class) if actual_class != "unknown" else False
            cat_info     = get_waste_category(predicted_class)

            results.append({
                'image_b64':  b64_img,
                'actual':     actual_class,
                'predicted':  predicted_class,
                'confidence': conf,
                'obj_score':  obj_score,
                'gamma':      gamma_used,
                'correct':    is_correct,
                'has_bbox':   bbox is not None,
                'category':   cat_info['category'],
                'cat_icon':   cat_info['icon'],
                'cat_color':  cat_info['color'],
                'cat_bg':     cat_info['bg'],
                'cat_border': cat_info['border'],
            })

            status = "✓" if is_correct else ("?" if actual_class == "unknown" else "✗")
            print(f"  [{idx+1}/{len(images)}] {img_name}: "
                  f"{predicted_class} ({conf*100:.1f}%) "
                  f"obj={obj_score:.2f} γ={gamma_used:.2f} {status}")

        except Exception as e:
            print(f"[ERROR] {img_name}: {e}")
            import traceback; traceback.print_exc()
            try:
                fallback = Image.open(img_path).convert('RGB')
                fallback.thumbnail((400, 400))
                buf = io.BytesIO()
                fallback.save(buf, format='JPEG', quality=85)
                b64_img = base64.b64encode(buf.getvalue()).decode()
            except:
                b64_img = ""

            results.append({
                'image_b64': b64_img,
                'actual':    get_actual_class(img_name),
                'predicted': "Error",
                'confidence': None,
                'obj_score':  None,
                'gamma':      None,
                'correct':    False,
                'has_bbox':   False,
                'category':   'Unknown',
                'cat_icon':   'help-circle',
                'cat_color':  '#64748B',
                'cat_bg':     'rgba(100,116,139,0.15)',
                'cat_border': 'rgba(100,116,139,0.4)',
            })

    avg_time = (time.time() - start_time) / max(len(images), 1)
    return results, avg_time


# ============================================================
# 10. FLASK APP
# ============================================================

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S.I.G.M.A. Evaluation Console | Garbage Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: { sans: ['Space Grotesk', 'sans-serif'] },
                    colors: {
                        cyber: { 400: '#2dd4bf', 500: '#14b8a6' },
                        neon:  { blue: '#00f3ff', green: '#0aff00', red: '#ff003c' }
                    },
                    animation: { 'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite' }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #050505;
            background-image:
                radial-gradient(circle at 50% 0%, #1a1a2e 0%, transparent 60%),
                linear-gradient(rgba(0,243,255,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,243,255,0.03) 1px, transparent 1px);
            background-size: 100% 100%, 40px 40px, 40px 40px;
        }
        .glass-panel {
            background: rgba(10,10,15,0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
        }
        .text-glow { text-shadow: 0 0 10px rgba(0,243,255,0.5); }
        .card-hover:hover {
            transform: translateY(-4px);
            box-shadow: 0 0 20px rgba(0,243,255,0.15);
            border-color: rgba(0,243,255,0.4);
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0a0a0f; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body class="text-slate-300 min-h-screen p-6">

    <!-- Header -->
    <header class="max-w-7xl mx-auto mb-10 flex flex-col md:flex-row justify-between items-center gap-6">
        <div class="flex items-center gap-4">
            <div class="relative w-12 h-12 flex items-center justify-center rounded-xl bg-slate-900 border border-cyber-500/30 shadow-[0_0_15px_rgba(45,212,191,0.2)]">
                <i data-lucide="cpu" class="w-6 h-6 text-cyber-400"></i>
                <div class="absolute inset-0 rounded-xl bg-cyber-500/10 animate-pulse-slow"></div>
            </div>
            <div>
                <h1 class="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-cyan-200 to-cyan-500 tracking-tight">S.I.G.M.A.</h1>
                <p class="text-xs font-mono text-cyan-500/70 tracking-widest uppercase">ONNX Runtime + Adaptive Gamma v3.0</p>
            </div>
        </div>
        <div class="flex gap-4">
            <div class="glass-panel px-6 py-3 rounded-lg flex flex-col items-center min-w-[140px]">
                <span class="text-xs text-slate-500 font-mono uppercase tracking-wider mb-1">Total Images</span>
                <span class="text-2xl font-bold text-white">{{ total }}</span>
            </div>
            <div class="glass-panel px-6 py-3 rounded-lg flex flex-col items-center min-w-[140px] relative overflow-hidden">
                <span class="text-xs text-slate-500 font-mono uppercase tracking-wider mb-1">Accuracy</span>
                <span class="text-2xl font-bold {{ 'text-green-400' if accuracy >= 90 else ('text-yellow-400' if accuracy >= 70 else 'text-red-400') }} text-glow">
                    {{ "%.1f"|format(accuracy) }}%
                </span>
                <div class="absolute bottom-0 left-0 h-1 bg-slate-800 w-full">
                    <div class="h-full {{ 'bg-green-400' if accuracy >= 90 else ('bg-yellow-400' if accuracy >= 70 else 'bg-red-400') }}" style="width: {{ accuracy }}%"></div>
                </div>
            </div>
            <div class="glass-panel px-6 py-3 rounded-lg flex flex-col items-center min-w-[140px]">
                <span class="text-xs text-slate-500 font-mono uppercase tracking-wider mb-1">Latency</span>
                <span class="text-2xl font-bold text-cyan-300">{{ "%.0f"|format(avg_time * 1000) }}<span class="text-sm font-normal text-slate-500 ml-1">ms</span></span>
            </div>
        </div>
    </header>

    <!-- Controls -->
    <div class="max-w-7xl mx-auto mb-8 flex flex-col md:flex-row justify-between items-center gap-4">
        <div class="glass-panel p-1.5 rounded-lg flex gap-1">
            <button onclick="filterCards('all')"     id="btn-all"     class="px-4 py-2 rounded-md text-sm font-medium transition-all bg-white/10 text-white shadow-sm border border-white/5">All</button>
            <button onclick="filterCards('correct')" id="btn-correct" class="px-4 py-2 rounded-md text-sm font-medium transition-all text-slate-400 hover:text-white hover:bg-white/5">
                <span class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-green-400 shadow-[0_0_8px_#4ade80]"></span> Correct</span>
            </button>
            <button onclick="filterCards('wrong')"   id="btn-wrong"   class="px-4 py-2 rounded-md text-sm font-medium transition-all text-slate-400 hover:text-white hover:bg-white/5">
                <span class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-red-400 shadow-[0_0_8px_#f87171]"></span> Errors</span>
            </button>
        </div>
        <div class="flex items-center gap-6 text-xs text-slate-500 font-mono">
            <div class="flex items-center gap-2"><div class="w-2 h-2 bg-green-400 rounded-full"></div> Match</div>
            <div class="flex items-center gap-2"><div class="w-2 h-2 bg-red-400 rounded-full"></div> Mismatch</div>
            <div class="flex items-center gap-2"><div class="w-2 h-2 bg-yellow-400 rounded-full"></div> Unknown</div>
            <div class="flex items-center gap-2"><span class="text-cyan-400 font-bold">γ</span> AGC gamma</div>
        </div>
    </div>

    <!-- Grid -->
    <div class="max-w-7xl mx-auto grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-5 pb-20">
        {% for r in results %}
        <div class="card-hover glass-panel rounded-xl overflow-hidden transition-all duration-300 group relative result-card"
             data-status="{{ 'correct' if r.correct else ('unknown' if r.actual == 'unknown' else 'wrong') }}">

            <!-- Image -->
            <div class="relative aspect-square overflow-hidden bg-slate-900">
                <img src="data:image/jpeg;base64,{{ r.image_b64 }}" alt="{{ r.actual }}"
                     class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110">
                <div class="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-transparent to-transparent opacity-60"></div>

                <!-- Status badge -->
                <div class="absolute top-3 right-3">
                    {% if r.correct %}
                        <div class="bg-green-500/20 backdrop-blur-md border border-green-500/50 text-green-400 p-1.5 rounded-lg">
                            <i data-lucide="check" class="w-3.5 h-3.5"></i>
                        </div>
                    {% elif r.actual == 'unknown' %}
                        <div class="bg-yellow-500/20 backdrop-blur-md border border-yellow-500/50 text-yellow-400 p-1.5 rounded-lg">
                            <i data-lucide="help-circle" class="w-3.5 h-3.5"></i>
                        </div>
                    {% else %}
                        <div class="bg-red-500/20 backdrop-blur-md border border-red-500/50 text-red-400 p-1.5 rounded-lg">
                            <i data-lucide="x" class="w-3.5 h-3.5"></i>
                        </div>
                    {% endif %}
                </div>

                <!-- Confidence -->
                {% if r.confidence %}
                <div class="absolute top-3 left-3">
                    <div class="px-2 py-1 rounded bg-black/60 backdrop-blur-sm text-[10px] font-mono text-cyan-300 border border-cyan-500/30">
                        {{ "%.1f"|format(r.confidence * 100) }}%
                    </div>
                </div>
                {% endif %}

                <!-- BBox / Gamma badges -->
                <div class="absolute bottom-3 left-3 flex flex-col gap-1">
                    {% if r.has_bbox %}
                    <div class="px-2 py-1 rounded bg-cyan-500/20 text-[9px] font-mono text-cyan-300 border border-cyan-400/40 flex items-center gap-1">
                        <i data-lucide="scan" class="w-2.5 h-2.5"></i> BBOX
                    </div>
                    {% endif %}
                    {% if r.gamma %}
                    <div class="px-2 py-1 rounded bg-yellow-500/20 text-[9px] font-mono text-yellow-300 border border-yellow-400/30">
                        γ={{ "%.2f"|format(r.gamma) }}
                    </div>
                    {% endif %}
                </div>
            </div>

            <!-- Content -->
            <div class="p-4 relative">
                <div class="absolute top-0 left-0 w-full h-[1px] bg-cyan-400/50 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500"></div>
                <div class="mb-3">
                    <p class="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-0.5">Prediction</p>
                    <div class="text-base font-bold text-white truncate">{{ r.predicted if r.predicted else 'No Detection' }}</div>
                </div>
                {% if r.category %}
                <div class="mb-3 flex items-center gap-1.5">
                    <div class="px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wider flex items-center gap-1"
                         style="background: {{ r.cat_bg }}; border: 1px solid {{ r.cat_border }}; color: {{ r.cat_color }}">
                        <i data-lucide="{{ r.cat_icon }}" class="w-2.5 h-2.5"></i>
                        {{ r.category }}
                    </div>
                </div>
                {% endif %}
                <div class="flex items-center justify-between pt-3 border-t border-white/5">
                    <div class="flex flex-col">
                        <span class="text-[9px] text-slate-500 uppercase tracking-wider">Actual</span>
                        <span class="text-xs font-medium {{ 'text-slate-300' if r.correct else ('text-yellow-400' if r.actual == 'unknown' else 'text-red-400') }} truncate max-w-[100px]" title="{{ r.actual }}">
                            {{ r.actual }}
                        </span>
                    </div>
                    <div class="text-[9px] font-mono text-slate-600">#{{ loop.index }}</div>
                </div>
            </div>

            {% if not r.correct and r.actual != 'unknown' %}
            <div class="absolute inset-0 border border-red-500/30 pointer-events-none rounded-xl"></div>
            {% endif %}
        </div>
        {% endfor %}
    </div>

    {% if not results %}
    <div class="max-w-2xl mx-auto text-center py-20">
        <div class="w-20 h-20 mx-auto bg-slate-800/50 rounded-full flex items-center justify-center mb-6">
            <i data-lucide="image-off" class="w-10 h-10 text-slate-600"></i>
        </div>
        <h3 class="text-xl font-bold text-white mb-2">No Images Found</h3>
        <p class="text-slate-400">Add images to the <code class="bg-slate-800 px-2 py-1 rounded text-cyan-400">TestImage</code> folder to begin analysis.</p>
    </div>
    {% endif %}

    <footer class="max-w-7xl mx-auto text-center text-slate-600 text-xs py-8 border-t border-slate-800/50">
        <p>POWERED BY ONNX Runtime + Adaptive Gamma Correction + MobileNetV3 | WASTE DETECTOR v3.0</p>
    </footer>

    <script>
        lucide.createIcons();

        function filterCards(status) {
            const cards   = document.querySelectorAll('.result-card');
            const buttons = ['all', 'correct', 'wrong'];
            buttons.forEach(btn => {
                const el = document.getElementById('btn-' + btn);
                if (btn === status) {
                    el.classList.add('bg-white/10', 'text-white', 'shadow-sm', 'border', 'border-white/5');
                    el.classList.remove('text-slate-400', 'hover:text-white', 'hover:bg-white/5');
                } else {
                    el.classList.remove('bg-white/10', 'text-white', 'shadow-sm', 'border', 'border-white/5');
                    el.classList.add('text-slate-400', 'hover:text-white', 'hover:bg-white/5');
                }
            });
            cards.forEach(card => {
                const s = card.getAttribute('data-status');
                card.style.display = (
                    status === 'all'     ? 'block' :
                    status === 'correct' ? (s === 'correct' ? 'block' : 'none') :
                    (s === 'wrong' || s === 'unknown') ? 'block' : 'none'
                );
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    results, avg_time = classify_images()

    known     = [r for r in results if r['actual'] != 'unknown']
    correct   = sum(1 for r in known if r['correct'])
    accuracy  = (correct / len(known) * 100) if known else 0

    return render_template_string(
        HTML_TEMPLATE,
        results=results,
        correct=correct,
        total=len(results),
        total_known=len(known),
        accuracy=accuracy,
        avg_time=avg_time,
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 S.I.G.M.A. EVALUATION CONSOLE v3.0")
    print("   Backend : ONNX Runtime (không dùng PyTorch inference)")
    print("   Pipeline: AGC → Resize(int) → CenterCrop → ToTensor → Normalize")
    print(f"  Model   : {ONNX_PATH}")
    print(f"  Meta    : {META_PATH}")
    print(f"  img_size: {IMG_SIZE}  |  AGC target: {AGC_TARGET}")
    print(f"  Classes : {CLASS_NAMES}")
    print("=" * 60)

    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        print(f"\n📁 TestImage/ đã tạo tại: {TEST_DIR}")
        print("   → Thêm ảnh vào đó rồi reload trang.")
    else:
        imgs = [f for f in os.listdir(TEST_DIR)
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))]
        print(f"\n📁 TestImage/ sẵn sàng: {len(imgs)} ảnh")

    print("\n🌐 http://localhost:8888\n")
    app.run(debug=True, port=8888, use_reloader=False)
