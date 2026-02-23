import os
import base64
import json
import time
import threading
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "trainModel/model.pth")

CLASS_NAMES = []
IMG_SIZE = 224          # ← Giảm từ 720 → 224 cho realtime inference
REALTIME_IMG_SIZE = 224

# Waste category mapping
WASTE_CATEGORIES = {
    'battery':    {'category': 'Hazardous',      'icon': 'alert-triangle', 'color': '#FF003C', 'bg': 'rgba(255,0,60,0.15)',   'border': 'rgba(255,0,60,0.4)'},
    'biological': {'category': 'Organic',         'icon': 'leaf',           'color': '#0AFF00', 'bg': 'rgba(10,255,0,0.15)',   'border': 'rgba(10,255,0,0.4)'},
    'cardboard':  {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)', 'border': 'rgba(59,130,246,0.4)'},
    'clothes':    {'category': 'Non-Recyclable',  'icon': 'trash-2',        'color': '#94A3B8', 'bg': 'rgba(148,163,184,0.15)','border': 'rgba(148,163,184,0.4)'},
    'glass':      {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)', 'border': 'rgba(59,130,246,0.4)'},
    'metal':      {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)', 'border': 'rgba(59,130,246,0.4)'},
    'paper':      {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)', 'border': 'rgba(59,130,246,0.4)'},
    'plastic':    {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)', 'border': 'rgba(59,130,246,0.4)'},
    'shoes':      {'category': 'Non-Recyclable',  'icon': 'trash-2',        'color': '#94A3B8', 'bg': 'rgba(148,163,184,0.15)','border': 'rgba(148,163,184,0.4)'},
    'trash':      {'category': 'Non-Recyclable',  'icon': 'trash-2',        'color': '#94A3B8', 'bg': 'rgba(148,163,184,0.15)','border': 'rgba(148,163,184,0.4)'},
}

def get_waste_category(class_name):
    return WASTE_CATEGORIES.get(class_name, {
        'category': 'Unknown', 'icon': 'help-circle',
        'color': '#64748B', 'bg': 'rgba(100,116,139,0.15)', 'border': 'rgba(100,116,139,0.4)'
    })

app = Flask(__name__)

# ─── Thread pool + lock cho AI inference ──────────────────────────────────────
ai_executor = ThreadPoolExecutor(max_workers=2)
model_lock  = threading.Lock()


class WasteDetectorModel(nn.Module):
    """MobileNetV3-based waste classifier with objectness detection"""

    def __init__(self, num_classes=10):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(960, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

        self.objectness = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.features(x)
        pooled   = self.avgpool(features)
        pooled   = pooled.flatten(1)
        class_logits = self.classifier(pooled)
        obj_score    = self.objectness(pooled)
        return class_logits, obj_score, features


def load_model(model_path):
    """Load model + TorchScript compilation để tối đa tốc độ"""
    global CLASS_NAMES, IMG_SIZE, REALTIME_IMG_SIZE

    print(f"  Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    CLASS_NAMES = checkpoint.get('classes',
        ['battery', 'biological', 'cardboard', 'clothes', 'glass',
         'metal', 'paper', 'plastic', 'shoes', 'trash'])
    IMG_SIZE          = checkpoint.get('img_size', 720)
    REALTIME_IMG_SIZE = 224   # Cố định 224 cho realtime

    num_classes = len(CLASS_NAMES)
    base_model  = WasteDetectorModel(num_classes=num_classes)

    state_dict = checkpoint.get('model_state',
                 checkpoint.get('model_state_dict', checkpoint))

    try:
        base_model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"  Warning: Strict loading failed, loose loading: {e}")
        base_model.load_state_dict(state_dict, strict=False)

    base_model.eval()

    # ─── TorchScript trace để tăng tốc inference ~20-30% ──────────────────
    try:
        example = torch.randn(1, 3, REALTIME_IMG_SIZE, REALTIME_IMG_SIZE)
        with torch.inference_mode():
            scripted = torch.jit.trace(base_model, example)
        scripted.eval()
        val_acc = checkpoint.get('val_acc', 'N/A')
        print(f"  TorchScript OK! Classes: {num_classes}, Val acc: {val_acc}")
        print(f"  Inference size: {REALTIME_IMG_SIZE}x{REALTIME_IMG_SIZE} (original: {IMG_SIZE})")
        return scripted
    except Exception as e:
        print(f"  TorchScript failed ({e}), dùng model gốc")
        return base_model


# ─── Transform cache ──────────────────────────────────────────────────────────
_transform_cache = {}

def get_transform(img_size):
    if img_size not in _transform_cache:
        _transform_cache[img_size] = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return _transform_cache[img_size]


def generate_gradcam(model, input_tensor, class_idx):
    """Generate GradCAM — chỉ dùng khi model không phải TorchScript"""
    if isinstance(model, torch.jit.ScriptModule):
        return None   # TorchScript không hỗ trợ hooks

    model.eval()
    feature_maps = []
    gradients    = []

    def forward_hook(module, input, output):
        feature_maps.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    last_conv  = model.features[-1]
    fwd_handle = last_conv.register_forward_hook(forward_hook)
    bwd_handle = last_conv.register_full_backward_hook(backward_hook)

    input_tensor.requires_grad_(True)
    class_logits, _, _ = model(input_tensor)
    model.zero_grad()
    one_hot = torch.zeros_like(class_logits)
    one_hot[0, class_idx] = 1
    class_logits.backward(gradient=one_hot, retain_graph=True)

    fwd_handle.remove()
    bwd_handle.remove()

    if not gradients or not feature_maps:
        return None

    grads   = gradients[0]
    fmaps   = feature_maps[0]
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)
    cam     = torch.sum(weights * fmaps, dim=1, keepdim=True)
    cam     = torch.relu(cam)
    cam     = cam.squeeze().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def get_bounding_box_from_cam(cam, original_size, threshold=0.3):
    cam_resized = cv2.resize(cam, (original_size[0], original_size[1]))
    binary      = (cam_resized > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest    = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    pad_x = int(w * 0.1); pad_y = int(h * 0.1)
    x = max(0, x - pad_x); y = max(0, y - pad_y)
    w = min(original_size[0] - x, w + 2*pad_x)
    h = min(original_size[1] - y, h + 2*pad_y)
    return (x, y, w, h)


def draw_detection(img_cv2, bbox, class_name, confidence, obj_score):
    result = img_cv2.copy()
    h_img, w_img = result.shape[:2]
    color = (0,255,100) if confidence > 0.8 else ((0,200,255) if confidence > 0.5 else (0,100,255))
    font  = cv2.FONT_HERSHEY_SIMPLEX

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(result, (x,y), (x+w,y+h), color, 2)
        label = f"{class_name}: {confidence*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(result, (x, y-th-10), (x+tw+8, y), color, -1)
        cv2.putText(result, label, (x+4, y-5), font, 0.7, (0,0,0), 2)

    label_top = f"{class_name}: {confidence*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label_top, font, 1.0, 2)
    cv2.rectangle(result, (8,8), (tw+20, th+20), (0,0,0), -1)
    cv2.rectangle(result, (8,8), (tw+20, th+20), color, 2)
    cv2.putText(result, label_top, (14, th+14), font, 1.0, color, 2)

    bar_text = f"Object: {obj_score*100:.0f}%"
    cv2.putText(result, bar_text, (12, h_img-12), font, 0.6, (200,200,200), 1)
    return result


class BackgroundStateMachine:
    """MOG2 State Machine — REALTIME OPTIMIZED
    
    Thay đổi so với bản gốc:
      warmup_required      : 50  → 20  frames (~2.5× nhanh hơn)
      early_stable_frames  : 4   → 2   frames
      full_stable_frames   : 7   → 4   frames
      idle_frames_required : 10  → 5   frames
      early_conf_threshold : 75% → 55% (lock sớm hơn)
      stable_threshold     : 15  → 20  px (chịu rung tay tốt hơn)
      history              : 300 → 200 (học nền nhanh hơn)
    """
    STATE_IDLE   = 'idle'
    STATE_MOTION = 'motion'
    STATE_STABLE = 'stable'
    STATE_LOCKED = 'locked'

    def __init__(self, history=200, var_threshold=40, min_area=2000,
                 stable_threshold=20, early_stable_frames=2,
                 full_stable_frames=4, idle_frames_required=5,
                 early_confidence_threshold=55.0):

        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True)
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))

        self.min_area                   = min_area
        self.stable_threshold           = stable_threshold
        self.early_stable_frames        = early_stable_frames
        self.full_stable_frames         = full_stable_frames
        self.idle_frames_required       = idle_frames_required
        self.early_confidence_threshold = early_confidence_threshold

        self.state           = self.STATE_IDLE
        self.prev_bbox       = None
        self.stable_count    = 0
        self.idle_count      = 0
        self.locked_result   = None
        self.locked_bbox     = None
        self.warmup_frames   = 0
        self.warmup_required = 20   # ← 50 → 20
        self.early_triggered = False
        self.early_result    = None

        print(f"[MOG2] REALTIME OPTIMIZED: warmup={self.warmup_required}f "
              f"early={early_stable_frames}f full={full_stable_frames}f "
              f"conf>={early_confidence_threshold}%")

    def reset(self):
        self.state = self.STATE_IDLE; self.prev_bbox = None
        self.stable_count = 0; self.idle_count = 0
        self.locked_result = None; self.locked_bbox = None
        self.early_triggered = False; self.early_result = None
        print("[MOG2] Reset → IDLE")

    def process_frame(self, frame_bgr):
        if self.state == self.STATE_LOCKED:
            lr = 0
        elif self.warmup_frames < self.warmup_required:
            lr = -1
        else:
            lr = 0.005

        fg_mask = self.mog2.apply(frame_bgr, learningRate=lr)

        if self.warmup_frames < self.warmup_required:
            self.warmup_frames += 1
            pct = int((self.warmup_frames / self.warmup_required) * 100)
            return {'state':'warmup','contour_area':0,'stable_count':0,'bbox':None,
                    'should_early_classify':False,'should_full_classify':False,
                    'warmup_progress':pct,'tier':None}

        fg_mask[fg_mask == 127] = 0
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > self.min_area]

        current_bbox = None; total_area = 0
        if valid:
            largest = max(valid, key=cv2.contourArea)
            total_area = cv2.contourArea(largest)
            current_bbox = cv2.boundingRect(largest)

        should_early = False; should_full = False
        prev_state   = self.state

        if self.state == self.STATE_IDLE:
            if current_bbox is not None and total_area > self.min_area:
                self.state = self.STATE_MOTION; self.stable_count = 0
                self.idle_count = 0; self.early_triggered = False
                self.early_result = None; self.prev_bbox = current_bbox
                print(f"[MOG2] IDLE→MOTION (area={total_area:.0f})")

        elif self.state == self.STATE_MOTION:
            if current_bbox is None or total_area < self.min_area:
                self.idle_count += 1
                if self.idle_count >= self.idle_frames_required:
                    self.state = self.STATE_IDLE; self.prev_bbox = None
                    self.stable_count = 0; self.early_triggered = False
                    self.early_result = None
                    print("[MOG2] MOTION→IDLE")
            else:
                self.idle_count = 0
                shift = self._shift(self.prev_bbox, current_bbox)
                if shift < self.stable_threshold:
                    self.stable_count += 1
                    if self.stable_count >= self.early_stable_frames and not self.early_triggered:
                        self.state = self.STATE_STABLE
                        print(f"[MOG2] MOTION→STABLE (shift={shift:.1f} count={self.stable_count})")
                else:
                    self.stable_count = 0; self.early_triggered = False; self.early_result = None
                self.prev_bbox = current_bbox

        elif self.state == self.STATE_STABLE:
            if current_bbox is None or total_area < self.min_area:
                self.idle_count += 1
                if self.idle_count >= self.idle_frames_required:
                    self.state = self.STATE_IDLE; self.prev_bbox = None
                    self.stable_count = 0; self.early_triggered = False
                    self.early_result = None
                    print("[MOG2] STABLE→IDLE (removed)")
            else:
                self.idle_count = 0
                shift = self._shift(self.prev_bbox, current_bbox)
                if shift < self.stable_threshold:
                    self.stable_count += 1
                    if not self.early_triggered and self.stable_count >= self.early_stable_frames:
                        should_early = True; self.early_triggered = True
                        print(f"[MOG2] ★ Tier1 @ frame {self.stable_count}")
                    if self.stable_count >= self.full_stable_frames and self.state != self.STATE_LOCKED:
                        should_full = True; self.state = self.STATE_LOCKED
                        self.locked_bbox = current_bbox
                        print(f"[MOG2] STABLE→LOCKED ★ Tier2")
                else:
                    self.state = self.STATE_MOTION; self.stable_count = 0
                    self.early_triggered = False; self.early_result = None
                    print(f"[MOG2] STABLE→MOTION (shift={shift:.1f})")
                self.prev_bbox = current_bbox

        elif self.state == self.STATE_LOCKED:
            if current_bbox is None or total_area < self.min_area:
                self.idle_count += 1
                if self.idle_count >= self.idle_frames_required:
                    self.state = self.STATE_IDLE; self.prev_bbox = None
                    self.stable_count = 0; self.locked_result = None
                    self.locked_bbox = None; self.early_triggered = False
                    self.early_result = None
                    print("[MOG2] LOCKED→IDLE (removed)")
            else:
                self.idle_count = 0
                if self.locked_bbox:
                    shift = self._shift(self.locked_bbox, current_bbox)
                    if shift > self.stable_threshold * 3:
                        self.state = self.STATE_MOTION; self.stable_count = 0
                        self.locked_result = None; self.locked_bbox = None
                        self.early_triggered = False; self.early_result = None
                        print(f"[MOG2] LOCKED→MOTION (shift={shift:.1f})")

        tier = 'early' if should_early else ('full' if should_full else None)
        return {
            'state': self.state, 'contour_area': int(total_area),
            'stable_count': self.stable_count, 'bbox': current_bbox,
            'should_early_classify': should_early, 'should_full_classify': should_full,
            'warmup_progress': 100, 'tier': tier
        }

    def _shift(self, b1, b2):
        if b1 is None or b2 is None: return float('inf')
        cx1=b1[0]+b1[2]/2; cy1=b1[1]+b1[3]/2
        cx2=b2[0]+b2[2]/2; cy2=b2[1]+b2[3]/2
        return ((cx2-cx1)**2+(cy2-cy1)**2)**0.5

    def lock_early(self, result):
        if result['confidence'] >= self.early_confidence_threshold:
            self.state = self.STATE_LOCKED; self.locked_result = result
            self.locked_bbox = self.prev_bbox
            print(f"[MOG2] ★★ EARLY LOCK {result['predicted_class']} {result['confidence']}%")
            return True
        self.early_result = result
        print(f"[MOG2] Tier1 low conf {result['confidence']}% < {self.early_confidence_threshold}%, wait Tier2")
        return False

    def set_locked_result(self, r): self.locked_result = r
    def get_locked_result(self):    return self.locked_result


# ─── Global state machine ────────────────────────────────────────────────────
def make_state_machine():
    return BackgroundStateMachine(
        history=200, var_threshold=40, min_area=2000,
        stable_threshold=20, early_stable_frames=2,
        full_stable_frames=4, idle_frames_required=5,
        early_confidence_threshold=55.0,
    )

state_machine = make_state_machine()


def classify_frame(model, pil_img, skip_gradcam=False, mog2_bbox=None):
    """Classify frame — dùng torch.inference_mode() + REALTIME_IMG_SIZE=224"""
    transform     = get_transform(REALTIME_IMG_SIZE)
    original_size = pil_img.size
    input_tensor  = transform(pil_img).unsqueeze(0)

    # torch.inference_mode nhanh hơn no_grad ~15%
    with torch.inference_mode():
        class_logits, obj_score_raw, _ = model(input_tensor)

    probabilities   = torch.softmax(class_logits, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_val  = confidence.item()
    obj_score       = torch.sigmoid(obj_score_raw).item()

    top5_probs, top5_idx = torch.topk(probabilities, min(5, len(CLASS_NAMES)), dim=1)
    top5 = [{'class': CLASS_NAMES[top5_idx[0][i].item()],
              'confidence': round(top5_probs[0][i].item() * 100, 1)}
            for i in range(top5_probs.shape[1])]

    bbox = mog2_bbox
    if bbox is None and not skip_gradcam:
        t_c = time.time()
        cam = generate_gradcam(model, transform(pil_img).unsqueeze(0), predicted_idx.item())
        print(f"[CAM] {(time.time()-t_c)*1000:.1f}ms")
        if cam is not None and obj_score > 0.05:
            bbox = get_bounding_box_from_cam(cam, original_size)

    img_cv2    = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result_img = draw_detection(img_cv2, bbox, predicted_class, confidence_val, obj_score)

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)).save(buf, format='JPEG', quality=85)
    result_b64 = base64.b64encode(buf.getvalue()).decode()

    cat = get_waste_category(predicted_class)
    return {
        'predicted_class': predicted_class,
        'confidence':      round(confidence_val * 100, 1),
        'obj_score':       round(obj_score * 100, 1),
        'top5':            top5,
        'result_image':    result_b64,
        'has_bbox':        bbox is not None,
        'category':        cat['category'],
        'cat_icon':        cat['icon'],
        'cat_color':       cat['color'],
        'cat_bg':          cat['bg'],
        'cat_border':      cat['border'],
    }


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>W.A.S.T.E. Scanner | AI Waste Detection</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
            --bg-primary: #0B0B10; --bg-secondary: #12121A;
            --bg-card: rgba(18,18,26,0.7); --border: rgba(255,255,255,0.06);
            --text-primary: #F8FAFC; --text-secondary: #94A3B8; --text-muted: #64748B;
            --neon-cyan: #00F3FF; --neon-green: #0AFF00; --neon-red: #FF003C;
            --neon-yellow: #FFD600; --accent-blue: #3B82F6;
            --font-display: 'Orbitron', sans-serif; --font-body: 'Exo 2', sans-serif;
        }
        body {
            background-color: var(--bg-primary); color: var(--text-primary);
            font-family: var(--font-body); min-height: 100vh; overflow-x: hidden;
            background-image:
                radial-gradient(circle at 30% 20%, rgba(0,243,255,0.04) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(59,130,246,0.03) 0%, transparent 50%),
                linear-gradient(rgba(0,243,255,0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,243,255,0.015) 1px, transparent 1px);
            background-size: 100%, 100%, 60px 60px, 60px 60px;
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
        .header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 16px 0; margin-bottom: 32px; border-bottom: 1px solid var(--border);
        }
        .logo-section { display: flex; align-items: center; gap: 16px; }
        .logo-icon {
            width: 48px; height: 48px; border-radius: 12px;
            background: linear-gradient(135deg, rgba(0,243,255,0.15), rgba(59,130,246,0.1));
            border: 1px solid rgba(0,243,255,0.3); display: flex;
            align-items: center; justify-content: center;
            box-shadow: 0 0 20px rgba(0,243,255,0.15); position: relative;
        }
        .logo-icon::after {
            content:''; position:absolute; inset:0; border-radius:12px;
            background:rgba(0,243,255,0.08); animation:pulse-glow 3s ease-in-out infinite;
        }
        @keyframes pulse-glow { 0%,100%{opacity:.3}50%{opacity:1} }
        .logo-title {
            font-family: var(--font-display); font-size: 1.5rem; font-weight: 700;
            letter-spacing: 3px; background: linear-gradient(90deg,#fff,var(--neon-cyan));
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        }
        .logo-subtitle { font-size:.65rem; letter-spacing:4px; text-transform:uppercase; color:var(--neon-cyan); opacity:.6; }
        .status-badges { display:flex; gap:12px; }
        .badge {
            padding:8px 16px; border-radius:8px; background:var(--bg-card);
            border:1px solid var(--border); backdrop-filter:blur(12px);
            font-size:.75rem; display:flex; align-items:center; gap:8px;
        }
        .badge-dot { width:6px; height:6px; border-radius:50%; animation:pulse-dot 2s ease-in-out infinite; }
        .badge-dot.active { background:var(--neon-green); box-shadow:0 0 8px var(--neon-green); }
        @keyframes pulse-dot { 0%,100%{opacity:1}50%{opacity:.4} }
        .main-content { display:grid; grid-template-columns:1fr 420px; gap:24px; min-height:calc(100vh - 160px); }
        .camera-panel {
            background:var(--bg-card); border-radius:16px; border:1px solid var(--border);
            backdrop-filter:blur(12px); overflow:hidden; display:flex; flex-direction:column;
        }
        .panel-header {
            padding:16px 20px; border-bottom:1px solid var(--border);
            display:flex; align-items:center; justify-content:space-between;
        }
        .panel-title {
            font-family:var(--font-display); font-size:.8rem; letter-spacing:2px;
            text-transform:uppercase; color:var(--text-secondary); display:flex; align-items:center; gap:8px;
        }
        .camera-viewport {
            flex:1; position:relative; background:#000;
            display:flex; align-items:center; justify-content:center; min-height:480px;
        }
        .camera-viewport video { width:100%; height:100%; object-fit:contain; }
        .camera-viewport img {
            position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain; z-index:2;
        }
        .scan-line {
            position:absolute; top:0; left:0; right:0; height:2px;
            background:linear-gradient(90deg,transparent,var(--neon-cyan),transparent);
            opacity:0; pointer-events:none; z-index:10;
        }
        .scan-line.active { opacity:1; animation:scan-sweep 1.5s ease-in-out; }
        @keyframes scan-sweep { 0%{top:0;opacity:0}10%{opacity:1}90%{opacity:1}100%{top:100%;opacity:0} }
        .corner-brackets { position:absolute; inset:20px; pointer-events:none; z-index:5; }
        .corner-brackets::before,.corner-brackets::after {
            content:''; position:absolute; width:30px; height:30px; border-color:var(--neon-cyan); opacity:.4;
        }
        .corner-brackets::before { top:0;left:0;border-top:2px solid;border-left:2px solid; }
        .corner-brackets::after  { top:0;right:0;border-top:2px solid;border-right:2px solid; }
        .corner-brackets-bottom  { position:absolute; inset:20px; pointer-events:none; z-index:5; }
        .corner-brackets-bottom::before {
            content:''; position:absolute; width:30px; height:30px;
            bottom:0; left:0; border-bottom:2px solid var(--neon-cyan); border-left:2px solid var(--neon-cyan); opacity:.4;
        }
        .corner-brackets-bottom::after {
            content:''; position:absolute; width:30px; height:30px;
            bottom:0; right:0; border-bottom:2px solid var(--neon-cyan); border-right:2px solid var(--neon-cyan); opacity:.4;
        }
        .camera-placeholder { text-align:center; color:var(--text-muted); }
        .camera-controls {
            padding:16px 20px; border-top:1px solid var(--border);
            display:flex; align-items:center; justify-content:center; gap:16px;
        }
        .result-panel { display:flex; flex-direction:column; gap:16px; }
        .result-card {
            background:var(--bg-card); border-radius:16px; border:1px solid var(--border);
            backdrop-filter:blur(12px); overflow:hidden; transition:border-color .5s ease;
        }
        .result-card.detected { border-color:rgba(0,243,255,0.3); box-shadow:0 0 30px rgba(0,243,255,0.08); }
        .result-main { padding:24px; text-align:center; }
        .result-idle { padding:60px 24px; text-align:center; color:var(--text-muted); }
        .result-idle i { margin-bottom:16px; opacity:.3; }
        .result-idle p { font-size:.85rem; line-height:1.6; }
        .result-class {
            font-family:var(--font-display); font-size:1.8rem; font-weight:700;
            letter-spacing:2px; text-transform:uppercase; margin-bottom:8px;
            text-shadow:0 0 20px rgba(0,243,255,0.4);
        }
        .result-confidence { font-size:3rem; font-weight:700; font-family:var(--font-display); margin:8px 0; }
        .result-confidence.high   { color:var(--neon-green);  text-shadow:0 0 20px rgba(10,255,0,0.3); }
        .result-confidence.medium { color:var(--neon-yellow); text-shadow:0 0 20px rgba(255,214,0,0.3); }
        .result-confidence.low    { color:var(--neon-red);    text-shadow:0 0 20px rgba(255,0,60,0.3); }
        .result-label { font-size:.7rem; color:var(--text-muted); letter-spacing:2px; text-transform:uppercase; }
        .category-badge {
            display:none; align-items:center; justify-content:center; gap:6px;
            padding:8px 16px; border-radius:8px; font-size:.75rem; font-weight:600;
            letter-spacing:1.5px; text-transform:uppercase; margin:12px auto 0; width:fit-content;
        }
        .category-badge.visible { display:flex; }
        .top5-card { background:var(--bg-card); border-radius:16px; border:1px solid var(--border); backdrop-filter:blur(12px); }
        .top5-list { padding:4px 20px 16px; }
        .top5-item { display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.03); }
        .top5-item:last-child { border-bottom:none; }
        .top5-rank { font-family:var(--font-display); font-size:.65rem; color:var(--text-muted); width:20px; text-align:center; }
        .top5-name { flex:1; font-size:.85rem; font-weight:500; text-transform:capitalize; }
        .top5-bar-container { width:100px; height:4px; background:rgba(255,255,255,0.05); border-radius:2px; overflow:hidden; }
        .top5-bar { height:100%; border-radius:2px; background:linear-gradient(90deg,var(--neon-cyan),var(--accent-blue)); transition:width .8s ease; }
        .top5-percent { font-family:var(--font-display); font-size:.75rem; color:var(--text-secondary); width:50px; text-align:right; }
        .history-card {
            background:var(--bg-card); border-radius:16px; border:1px solid var(--border);
            backdrop-filter:blur(12px); flex:1; overflow:hidden; display:flex; flex-direction:column;
        }
        .history-list { padding:8px 12px; overflow-y:auto; max-height:280px; flex:1; }
        .history-item { display:flex; align-items:center; gap:10px; padding:8px; border-radius:8px; transition:background .2s; cursor:pointer; }
        .history-item:hover { background:rgba(255,255,255,0.03); }
        .history-thumb { width:40px; height:40px; border-radius:6px; object-fit:cover; border:1px solid var(--border); }
        .history-info { flex:1; min-width:0; }
        .history-class { font-size:.8rem; font-weight:600; text-transform:capitalize; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .history-time { font-size:.65rem; color:var(--text-muted); }
        .history-conf { font-family:var(--font-display); font-size:.7rem; font-weight:600; }
        .history-empty { padding:32px; text-align:center; color:var(--text-muted); font-size:.8rem; }
        .processing-overlay {
            position:absolute; inset:0; background:rgba(11,11,16,0.85);
            display:none; align-items:center; justify-content:center; flex-direction:column; gap:16px; z-index:20;
        }
        .processing-overlay.active { display:flex; }
        .spinner {
            width:48px; height:48px; border:3px solid rgba(0,243,255,0.1);
            border-top-color:var(--neon-cyan); border-radius:50%; animation:spin .8s linear infinite;
        }
        @keyframes spin { to{transform:rotate(360deg)} }
        .mog2-state-bar {
            display:flex; align-items:center; gap:8px; padding:6px 12px; border-radius:8px;
            background:var(--bg-card); border:1px solid var(--border);
            font-family:var(--font-display); font-size:.65rem; letter-spacing:1.5px;
            text-transform:uppercase; transition:all .3s ease;
        }
        .mog2-state-dot { width:8px; height:8px; border-radius:50%; transition:all .3s ease; }
        .mog2-state-bar[data-state='idle']   { color:var(--text-muted); border-color:rgba(100,116,139,0.3); }
        .mog2-state-bar[data-state='idle'] .mog2-state-dot { background:var(--text-muted); }
        .mog2-state-bar[data-state='warmup'] { color:var(--accent-blue); border-color:rgba(59,130,246,0.3); }
        .mog2-state-bar[data-state='warmup'] .mog2-state-dot { background:var(--accent-blue); box-shadow:0 0 6px var(--accent-blue); animation:pulse-dot .8s ease-in-out infinite; }
        .mog2-state-bar[data-state='motion'] { color:var(--neon-yellow); border-color:rgba(255,214,0,0.3); }
        .mog2-state-bar[data-state='motion'] .mog2-state-dot { background:var(--neon-yellow); box-shadow:0 0 6px var(--neon-yellow); animation:pulse-dot .6s ease-in-out infinite; }
        .mog2-state-bar[data-state='stable'] { color:#FF9800; border-color:rgba(255,152,0,0.3); }
        .mog2-state-bar[data-state='stable'] .mog2-state-dot { background:#FF9800; box-shadow:0 0 6px #FF9800; animation:pulse-dot .4s ease-in-out infinite; }
        .mog2-state-bar[data-state='locked'] { color:var(--neon-green); border-color:rgba(10,255,0,0.3); box-shadow:0 0 12px rgba(10,255,0,0.1); }
        .mog2-state-bar[data-state='locked'] .mog2-state-dot { background:var(--neon-green); box-shadow:0 0 8px var(--neon-green); }
        .mog2-progress { width:60px; height:4px; background:rgba(255,255,255,0.08); border-radius:2px; overflow:hidden; }
        .mog2-progress-fill { height:100%; border-radius:2px; background:linear-gradient(90deg,var(--neon-cyan),var(--neon-green)); transition:width .3s ease; width:0%; }
        .scan-toggle-btn {
            padding:10px 24px; border-radius:10px; border:2px solid var(--neon-cyan);
            background:rgba(0,243,255,0.08); color:var(--neon-cyan);
            font-family:var(--font-display); font-size:.75rem; letter-spacing:2px;
            cursor:pointer; transition:all .3s ease; display:flex; align-items:center; gap:8px;
        }
        .scan-toggle-btn:hover { background:rgba(0,243,255,0.15); box-shadow:0 0 20px rgba(0,243,255,0.2); }
        .scan-toggle-btn.active { border-color:var(--neon-red); color:var(--neon-red); background:rgba(255,0,60,0.08); }
        .scan-toggle-btn.active:hover { background:rgba(255,0,60,0.15); box-shadow:0 0 20px rgba(255,0,60,0.2); }
        .fps-counter { font-family:var(--font-display); font-size:.65rem; color:var(--text-muted); letter-spacing:1px; }
        .camera-select {
            background:var(--bg-secondary); color:var(--text-secondary); border:1px solid var(--border);
            border-radius:6px; padding:4px 8px; font-family:var(--font-body); font-size:.7rem;
            cursor:pointer; outline:none; max-width:220px;
        }
        .camera-select:focus { border-color:var(--neon-cyan); }
        .camera-select option { background:var(--bg-secondary); color:var(--text-primary); }
        /* Tier badge */
        .tier-badge {
            display:inline-block; padding:2px 8px; border-radius:4px;
            font-size:.55rem; font-family:var(--font-display); letter-spacing:1px;
        }
        .tier-early { background:rgba(255,214,0,0.15); color:var(--neon-yellow); border:1px solid rgba(255,214,0,0.3); }
        .tier-full  { background:rgba(10,255,0,0.15);  color:var(--neon-green);  border:1px solid rgba(10,255,0,0.3); }
        #debugLog {
            position:fixed; bottom:12px; left:12px; background:rgba(0,0,0,0.85);
            color:#0AFF00; font-family:monospace; font-size:.65rem; padding:8px 12px;
            border-radius:6px; border:1px solid rgba(10,255,0,0.2); max-width:440px;
            z-index:9999; pointer-events:none; display:none;
        }
        @media (max-width:900px) {
            .main-content { grid-template-columns:1fr; }
            .header { flex-direction:column; gap:16px; }
            .status-badges { width:100%; justify-content:center; }
        }
    </style>
</head>
<body>
<div class="container">
    <header class="header">
        <div class="logo-section">
            <div class="logo-icon">
                <i data-lucide="scan-line" style="width:24px;height:24px;color:var(--neon-cyan);position:relative;z-index:1"></i>
            </div>
            <div>
                <div class="logo-title">W.A.S.T.E.</div>
                <div class="logo-subtitle">Waste Analysis Scanner &amp; Type Engine</div>
            </div>
        </div>
        <div class="status-badges">
            <div class="badge">
                <span class="badge-dot active" id="cameraDot"></span>
                <span id="cameraStatus">Initializing...</span>
            </div>
            <div class="badge">
                <span class="badge-dot active" style="background:var(--accent-blue);box-shadow:0 0 8px var(--accent-blue)"></span>
                <span>MobileNetV3 224px</span>
            </div>
            <div class="badge">
                <i data-lucide="images" style="width:14px;height:14px;color:var(--text-muted)"></i>
                <span id="scanCount">0</span> scans
            </div>
        </div>
    </header>

    <div class="main-content">
        <!-- Camera -->
        <div class="camera-panel">
            <div class="panel-header">
                <div class="panel-title">
                    <i data-lucide="video" style="width:16px;height:16px;color:var(--neon-cyan)"></i>
                    LIVE FEED
                </div>
                <div style="display:flex;align-items:center;gap:10px">
                    <input type="color" id="bgColorPicker" value="#000000"
                        style="width:24px;height:24px;border:none;border-radius:4px;cursor:pointer;background:transparent;padding:0;">
                    <select class="camera-select" id="cameraSelect" onchange="switchCamera(this.value)">
                        <option>Loading...</option>
                    </select>
                    <div class="panel-title" style="color:var(--text-muted);font-family:var(--font-body);letter-spacing:0;font-size:.75rem" id="resolution">--</div>
                </div>
            </div>
            <div class="camera-viewport" id="viewport">
                <video id="cameraFeed" autoplay playsinline muted style="display:none"></video>
                <img id="resultImage" style="display:none" alt="">
                <canvas id="captureCanvas" style="display:none"></canvas>
                <canvas id="smallCanvas"   style="display:none"></canvas>
                <div class="corner-brackets"></div>
                <div class="corner-brackets-bottom"></div>
                <div class="scan-line" id="scanLine"></div>
                <div class="processing-overlay" id="processingOverlay">
                    <div class="spinner"></div>
                    <div style="font-family:var(--font-display);font-size:.75rem;letter-spacing:3px;color:var(--neon-cyan)">ANALYZING...</div>
                </div>
                <div class="camera-placeholder" id="cameraPlaceholder">
                    <i data-lucide="camera-off" style="width:48px;height:48px"></i>
                    <p style="font-size:.9rem;font-weight:500;margin-bottom:4px">Camera Initializing</p>
                    <p style="font-size:.75rem">Please allow camera access</p>
                </div>
            </div>
            <div class="camera-controls">
                <div class="mog2-state-bar" id="mog2StateBar" data-state="idle">
                    <span class="mog2-state-dot"></span>
                    <span id="mog2StateText">STANDBY</span>
                    <div class="mog2-progress" id="mog2Progress" style="display:none">
                        <div class="mog2-progress-fill" id="mog2ProgressFill"></div>
                    </div>
                </div>
                <button class="scan-toggle-btn" id="scanToggleBtn" onclick="toggleScanning()">
                    <i data-lucide="scan" style="width:16px;height:16px"></i>
                    <span id="scanBtnText">START SCAN</span>
                </button>
                <span class="fps-counter" id="fpsCounter">-- fps</span>
            </div>
        </div>

        <!-- Results -->
        <div class="result-panel">
            <div class="result-card" id="resultCard">
                <div class="panel-header">
                    <div class="panel-title">
                        <i data-lucide="brain" style="width:16px;height:16px;color:var(--neon-cyan)"></i>
                        DETECTION RESULT
                    </div>
                    <span id="tierBadgeWrap" style="display:none"></span>
                </div>
                <div class="result-idle" id="resultIdle">
                    <i data-lucide="scan" style="width:40px;height:40px"></i>
                    <p>Point camera at waste items<br>
                    <kbd style="padding:2px 6px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:3px;font-family:var(--font-display);font-size:.65rem">SPACE</kbd>
                    to toggle scan</p>
                </div>
                <div class="result-main" id="resultMain" style="display:none">
                    <div class="result-label">CLASSIFIED AS</div>
                    <div class="result-class" id="resultClass">--</div>
                    <div class="result-confidence" id="resultConfidence">--</div>
                    <div class="result-label">CONFIDENCE</div>
                    <div class="category-badge" id="categoryBadge">
                        <i id="categoryIcon" style="width:14px;height:14px"></i>
                        <span id="categoryText">--</span>
                    </div>
                </div>
            </div>

            <div class="top5-card" id="top5Card" style="display:none">
                <div class="panel-header">
                    <div class="panel-title">
                        <i data-lucide="bar-chart-3" style="width:16px;height:16px;color:var(--neon-cyan)"></i>
                        PROBABILITY DISTRIBUTION
                    </div>
                </div>
                <div class="top5-list" id="top5List"></div>
            </div>

            <div class="history-card">
                <div class="panel-header">
                    <div class="panel-title">
                        <i data-lucide="history" style="width:16px;height:16px;color:var(--neon-cyan)"></i>
                        SCAN HISTORY
                    </div>
                </div>
                <div class="history-list" id="historyList">
                    <div class="history-empty" id="historyEmpty">No scans yet</div>
                </div>
            </div>
        </div>
    </div>
</div>

<div id="debugLog"></div>

<script>
    lucide.createIcons();

    // ── State ─────────────────────────────────────────────────────────────────
    let stream = null, isScanning = false, scanInterval = null;
    let scanCount = 0, frameCount = 0, lastFpsTime = Date.now();
    let debugVisible = false, hasLockedResult = false, isBusy = false;

    const video       = document.getElementById('cameraFeed');
    const canvas      = document.getElementById('captureCanvas');
    const smallCanvas = document.getElementById('smallCanvas');
    const resultImage = document.getElementById('resultImage');
    const placeholder = document.getElementById('cameraPlaceholder');
    const scanLine    = document.getElementById('scanLine');
    const debugLog    = document.getElementById('debugLog');

    function dbg(msg) {
        if (debugVisible) console.log('[WASTE]', msg);
        debugLog.textContent = msg;
    }

    function isVideoReady() {
        return video.readyState >= 2 && video.videoWidth > 0
            && video.videoHeight > 0 && !video.paused && !video.ended;
    }

    // ── Camera ────────────────────────────────────────────────────────────────
    async function initCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode:'environment', width:{ideal:720}, height:{ideal:720} }
            });
            video.srcObject = stream;
            await new Promise((res, rej) => {
                const t = setTimeout(() => rej(new Error('timeout')), 10000);
                video.onloadedmetadata = () => { clearTimeout(t); video.play().then(res).catch(rej); };
            });
            video.style.display = 'block';
            placeholder.style.display = 'none';
            document.getElementById('resolution').textContent = `${video.videoWidth}x${video.videoHeight}`;
            document.getElementById('cameraStatus').textContent = 'Camera Active';
            await populateCameraList();
            // Tự động bắt đầu sau 800ms (warmup chỉ 20 frames ≈ 600ms)
            setTimeout(() => { if (!isScanning) startScanning(); }, 800);
        } catch (err) {
            document.getElementById('cameraStatus').textContent = 'Camera Error';
            document.getElementById('cameraDot').style.background = 'var(--neon-red)';
            dbg('Camera error: ' + err.message);
        }
    }

    async function populateCameraList() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const sel = document.getElementById('cameraSelect');
            sel.innerHTML = '';
            devices.filter(d => d.kind === 'videoinput').forEach((d, i) => {
                const o = document.createElement('option');
                o.value = d.deviceId;
                o.textContent = d.label || `Camera ${i+1}`;
                if (stream) {
                    const t = stream.getVideoTracks()[0];
                    if (t && t.getSettings().deviceId === d.deviceId) o.selected = true;
                }
                sel.appendChild(o);
            });
        } catch (e) { console.error(e); }
    }

    async function switchCamera(deviceId) {
        if (!deviceId) return;
        const was = isScanning; if (was) stopScanning();
        if (stream) stream.getTracks().forEach(t => t.stop());
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId:{exact:deviceId}, width:{ideal:720}, height:{ideal:720} }
            });
            video.srcObject = stream;
            await new Promise((res, rej) => {
                const t = setTimeout(() => rej(new Error('timeout')), 10000);
                video.onloadedmetadata = () => { clearTimeout(t); video.play().then(res).catch(rej); };
            });
            video.style.display = 'block';
            placeholder.style.display = 'none';
            document.getElementById('resolution').textContent = `${video.videoWidth}x${video.videoHeight}`;
            if (was) startScanning();
        } catch (e) { document.getElementById('cameraStatus').textContent = 'Switch Failed'; }
    }

    // ── Scanning ──────────────────────────────────────────────────────────────
    function toggleScanning() { isScanning ? stopScanning() : startScanning(); }

    function startScanning() {
        if (!stream) return;
        if (!isVideoReady()) { setTimeout(startScanning, 200); return; }
        isScanning = true; isBusy = false; hasLockedResult = false;
        const btn = document.getElementById('scanToggleBtn');
        btn.classList.add('active');
        document.getElementById('scanBtnText').textContent = 'STOP SCAN';
        setIcon(btn, 'scan-line');
        updateMog2UI('idle', 0);
        document.getElementById('cameraStatus').textContent = 'MOG2 Active';
        scheduleNext();
    }

    function scheduleNext() {
        if (!isScanning) return;
        // 0ms delay → gửi liên tục, isBusy guard tránh overlap
        scanInterval = setTimeout(() => {
            processFrame().finally(() => { if (isScanning) scheduleNext(); });
        }, 0);
    }

    function stopScanning() {
        isScanning = false; isBusy = false;
        if (scanInterval) { clearTimeout(scanInterval); scanInterval = null; }
        const btn = document.getElementById('scanToggleBtn');
        btn.classList.remove('active');
        document.getElementById('scanBtnText').textContent = 'START SCAN';
        setIcon(btn, 'scan');
        updateMog2UI('idle', 0);
        document.getElementById('cameraStatus').textContent = 'Camera Active';
        document.getElementById('fpsCounter').textContent = '-- fps';
        video.style.display = 'block';
        resultImage.style.display = 'none';
    }

    function setIcon(btn, icon) {
        const old = btn.querySelector('.lucide'); if (old) old.remove();
        const i = document.createElement('i');
        i.setAttribute('data-lucide', icon); i.style.width='16px'; i.style.height='16px';
        btn.insertBefore(i, btn.firstChild); lucide.createIcons();
    }

    function updateMog2UI(state, stableCount) {
        const bar  = document.getElementById('mog2StateBar');
        const txt  = document.getElementById('mog2StateText');
        const prog = document.getElementById('mog2Progress');
        const fill = document.getElementById('mog2ProgressFill');
        bar.setAttribute('data-state', state);
        const labels = {idle:'STANDBY',warmup:'LEARNING...',motion:'MOTION',stable:'STABILIZING...',locked:'LOCKED ★'};
        txt.textContent = labels[state] || state.toUpperCase();
        if (state === 'stable') {
            prog.style.display = 'block';
            fill.style.width = Math.min(100, (stableCount/4)*100) + '%'; // /4 = full_stable_frames
        } else if (state === 'warmup') {
            prog.style.display = 'block';
        } else {
            prog.style.display = 'none'; fill.style.width = '0%';
        }
    }

    // ── Frame processing ──────────────────────────────────────────────────────
    async function processFrame() {
        if (isBusy || !isScanning || !stream || !isVideoReady()) return;
        isBusy = true;
        try {
            // Frame NHỎ 360px cho MOG2 (giảm ~4× data so với 720px)
            const sw = Math.min(video.videoWidth, 360);
            const sh = Math.min(video.videoHeight, 360);
            smallCanvas.width = sw; smallCanvas.height = sh;
            smallCanvas.getContext('2d').drawImage(video, 0, 0, sw, sh);

            const t0  = performance.now();
            const res = await fetch('/api/process_frame', {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({ image: smallCanvas.toDataURL('image/jpeg', 0.45) })
            });
            if (!res.ok) { dbg(`Error ${res.status}`); return; }

            const data    = await res.json();
            const elapsed = Math.round(performance.now() - t0);
            if (data.error) { dbg('API: ' + data.error); return; }

            // FPS
            frameCount++;
            const now = Date.now();
            if (now - lastFpsTime >= 2000) {
                document.getElementById('fpsCounter').textContent =
                    (frameCount / ((now-lastFpsTime)/1000)).toFixed(1) + ' fps';
                frameCount = 0; lastFpsTime = now;
            }

            if (data.state === 'warmup') {
                document.getElementById('mog2ProgressFill').style.width = (data.warmup_progress||0) + '%';
                updateMog2UI('warmup', 0);
                dbg(`Learning... ${data.warmup_progress}%`);
                return;
            }

            updateMog2UI(data.state, data.stable_count || 0);
            dbg(`[${data.state.toUpperCase()}] area=${data.contour_area} stable=${data.stable_count} tier=${data.tier||'-'} ${elapsed}ms`);

            if (data.state === 'locked' && data.result && !hasLockedResult) {
                hasLockedResult = true;

                // Tier badge
                const tw = document.getElementById('tierBadgeWrap');
                if (data.tier === 'early') {
                    tw.innerHTML = '<span class="tier-badge tier-early">⚡ EARLY LOCK</span>';
                    tw.style.display = 'inline';
                } else if (data.tier === 'full') {
                    tw.innerHTML = '<span class="tier-badge tier-full">★ FULL LOCK</span>';
                    tw.style.display = 'inline';
                }

                displayResult(data.result);
                scanLine.classList.remove('active');
                void scanLine.offsetWidth;
                scanLine.classList.add('active');
            } else if (data.state === 'idle' && hasLockedResult) {
                hasLockedResult = false;
                document.getElementById('tierBadgeWrap').style.display = 'none';
                clearResult();
            }
        } catch (err) {
            dbg('Fetch: ' + err.message);
        } finally {
            isBusy = false;
        }
    }

    function clearResult() {
        document.getElementById('resultCard').classList.remove('detected');
        document.getElementById('resultIdle').style.display = 'block';
        document.getElementById('resultMain').style.display = 'none';
        document.getElementById('top5Card').style.display = 'none';
        document.getElementById('categoryBadge').classList.remove('visible');
        const p = document.getElementById('resultImgPreview'); if (p) p.remove();
    }

    function displayResult(result) {
        scanCount++;
        document.getElementById('scanCount').textContent = scanCount;

        document.getElementById('resultCard').classList.add('detected');
        document.getElementById('resultIdle').style.display = 'none';
        document.getElementById('resultMain').style.display = 'block';

        let p = document.getElementById('resultImgPreview');
        if (!p) {
            p = document.createElement('img');
            p.id='resultImgPreview'; p.style.width='100%';
            p.style.borderRadius='8px'; p.style.marginBottom='12px';
            document.getElementById('resultMain').insertBefore(p, document.getElementById('resultMain').firstChild);
        }
        p.src = 'data:image/jpeg;base64,' + result.result_image;

        document.getElementById('resultClass').textContent = result.predicted_class;
        const c = document.getElementById('resultConfidence');
        c.textContent = result.confidence.toFixed(1) + '%';
        c.className = 'result-confidence ' + (result.confidence>=80?'high':result.confidence>=50?'medium':'low');

        const cb = document.getElementById('categoryBadge');
        cb.classList.add('visible');
        document.getElementById('categoryText').textContent = result.category;
        cb.style.background = result.cat_bg;
        cb.style.border     = '1px solid ' + result.cat_border;
        cb.style.color      = result.cat_color;
        const os = cb.querySelector('.lucide'); if (os) os.remove();
        const ni = document.createElement('i');
        ni.setAttribute('data-lucide', result.cat_icon);
        ni.style.width='14px'; ni.style.height='14px'; ni.style.color=result.cat_color;
        cb.insertBefore(ni, cb.firstChild); lucide.createIcons();

        document.getElementById('top5Card').style.display = 'block';
        const list = document.getElementById('top5List');
        list.innerHTML = '';
        result.top5.forEach((item, i) => {
            const d = document.createElement('div'); d.className='top5-item';
            d.innerHTML = `<span class="top5-rank">${i+1}</span>
                <span class="top5-name">${item.class}</span>
                <div class="top5-bar-container"><div class="top5-bar" style="width:${item.confidence}%"></div></div>
                <span class="top5-percent">${item.confidence}%</span>`;
            list.appendChild(d);
        });

        addHistory(result);
    }

    function addHistory(result) {
        document.getElementById('historyEmpty').style.display = 'none';
        const hl = document.getElementById('historyList');
        while (hl.children.length > 20) hl.removeChild(hl.lastChild);

        const t = new Date().toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
        const item = document.createElement('div'); item.className='history-item';
        item.innerHTML = `
            <img class="history-thumb" src="data:image/jpeg;base64,${result.result_image}">
            <div class="history-info">
                <div class="history-class">${result.predicted_class}</div>
                <div class="history-time" style="display:flex;align-items:center;gap:4px">${t}
                    <span style="font-size:.55rem;padding:1px 4px;border-radius:3px;background:${result.cat_bg};border:1px solid ${result.cat_border};color:${result.cat_color}">${result.category}</span>
                </div>
            </div>
            <span class="history-conf" style="color:${result.confidence>=80?'var(--neon-green)':result.confidence>=50?'var(--neon-yellow)':'var(--neon-red)'}">${result.confidence}%</span>`;
        item.addEventListener('click', () => displayResult(result));
        hl.insertBefore(item, hl.firstChild);
    }

    // ── Keyboard ──────────────────────────────────────────────────────────────
    document.addEventListener('keydown', e => {
        if (e.repeat) return;
        if (e.code==='Space') { e.preventDefault(); toggleScanning(); }
        if (e.code==='KeyD') {
            debugVisible = !debugVisible;
            debugLog.style.display = debugVisible ? 'block' : 'none';
        }
        if (e.code==='KeyR') {
            fetch('/api/reset_mog2', {method:'POST'}).then(() => {
                hasLockedResult = false; clearResult(); updateMog2UI('idle',0);
                document.getElementById('tierBadgeWrap').style.display='none';
                dbg('MOG2 reset');
            });
        }
    });

    document.getElementById('bgColorPicker').addEventListener('input', e => {
        document.getElementById('viewport').style.background = e.target.value;
    });

    initCamera();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """Manual classify endpoint (không dùng MOG2)"""
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'No body'}), 400
        image_data = data.get('image', '')
        if not image_data: return jsonify({'error': 'No image'}), 400
        if ',' in image_data: image_data = image_data.split(',')[1]

        pil_img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        if pil_img.width < 10 or pil_img.height < 10:
            return jsonify({'error': f'Too small: {pil_img.size}'}), 400

        with model_lock:
            result = classify_frame(model, pil_img, skip_gradcam=True)
        print(f"[/classify] {result['predicted_class']} {result['confidence']}%")
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    """MOG2 + AI inference — REALTIME OPTIMIZED"""
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'No body'}), 400
        image_data = data.get('image', '')
        if not image_data: return jsonify({'error': 'No image'}), 400
        if ',' in image_data: image_data = image_data.split(',')[1]

        nparr     = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None or frame_bgr.shape[0] < 10:
            return jsonify({'error': 'Invalid image'}), 400

        # MOG2 state machine
        sm = state_machine.process_frame(frame_bgr)

        resp = {
            'state':           sm['state'],
            'contour_area':    sm['contour_area'],
            'stable_count':    sm['stable_count'],
            'warmup_progress': sm['warmup_progress'],
            'tier':            sm.get('tier'),
        }

        # ── Tier 1: early classify ────────────────────────────────────────────
        if sm['should_early_classify']:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            t0  = time.time()
            with model_lock:
                ai = classify_frame(model, pil, skip_gradcam=True, mog2_bbox=sm['bbox'])
            print(f"[API] Tier1 {(time.time()-t0)*1000:.0f}ms → {ai['predicted_class']} {ai['confidence']}%")
            if state_machine.lock_early(ai):
                resp['state']  = 'locked'
                resp['result'] = ai
                resp['tier']   = 'early'
            else:
                resp['tier'] = 'early_pending'

        # ── Tier 2: full classify ─────────────────────────────────────────────
        elif sm['should_full_classify']:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            t0  = time.time()
            with model_lock:
                ai = classify_frame(model, pil, skip_gradcam=True, mog2_bbox=sm['bbox'])
            print(f"[API] Tier2 {(time.time()-t0)*1000:.0f}ms → {ai['predicted_class']} {ai['confidence']}%")
            state_machine.set_locked_result(ai)
            resp['result'] = ai
            resp['tier']   = 'full'

        # ── Trả về kết quả cached khi locked ─────────────────────────────────
        elif sm['state'] == 'locked' and state_machine.get_locked_result():
            resp['result'] = state_machine.get_locked_result()

        return jsonify(resp)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset_mog2', methods=['POST'])
def api_reset_mog2():
    global state_machine
    state_machine = make_state_machine()
    print("[API] MOG2 reset by user")
    return jsonify({'status': 'ok'})


if __name__ == "__main__":
    print("=" * 60)
    print("  W.A.S.T.E. SCANNER — REALTIME OPTIMIZED v2")
    print("  ✓ TorchScript compilation")
    print("  ✓ torch.inference_mode (vs no_grad)")
    print("  ✓ Inference size: 224px (vs 720px)")
    print("  ✓ MOG2 warmup: 20f | early: 2f | full: 4f")
    print("  ✓ Small frame (360px, q=0.45) cho MOG2")
    print("  ✓ 0ms polling delay + isBusy guard")
    print("  ✓ Confidence threshold: 55% (vs 75%)")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"  Model not found: {MODEL_PATH}"); exit(1)

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        import traceback; traceback.print_exc(); exit(1)

    print()
    print("=" * 60)
    print("  ONLINE → http://localhost:9999")
    print("  SPACE → toggle scan | D → debug | R → reset MOG2")
    print("=" * 60)
    print()

    flask_host = os.environ.get('FLASK_HOST', '127.0.0.1')
    app.run(host=flask_host, debug=True, port=9999, use_reloader=False)