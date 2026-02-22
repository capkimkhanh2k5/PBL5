import os
import base64
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "trainModel/model.pth")

CLASS_NAMES = []
IMG_SIZE = 720

# Waste category mapping
WASTE_CATEGORIES = {
    'battery':    {'category': 'Hazardous',      'icon': 'alert-triangle', 'color': '#FF003C', 'bg': 'rgba(255,0,60,0.15)',  'border': 'rgba(255,0,60,0.4)'},
    'biological': {'category': 'Organic',         'icon': 'leaf',           'color': '#0AFF00', 'bg': 'rgba(10,255,0,0.15)',  'border': 'rgba(10,255,0,0.4)'},
    'cardboard':  {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)','border': 'rgba(59,130,246,0.4)'},
    'clothes':    {'category': 'Non-Recyclable',  'icon': 'trash-2',        'color': '#94A3B8', 'bg': 'rgba(148,163,184,0.15)','border': 'rgba(148,163,184,0.4)'},
    'glass':      {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)','border': 'rgba(59,130,246,0.4)'},
    'metal':      {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)','border': 'rgba(59,130,246,0.4)'},
    'paper':      {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)','border': 'rgba(59,130,246,0.4)'},
    'plastic':    {'category': 'Recyclable',      'icon': 'recycle',        'color': '#3B82F6', 'bg': 'rgba(59,130,246,0.15)','border': 'rgba(59,130,246,0.4)'},
    'shoes':      {'category': 'Non-Recyclable',  'icon': 'trash-2',        'color': '#94A3B8', 'bg': 'rgba(148,163,184,0.15)','border': 'rgba(148,163,184,0.4)'},
    'trash':      {'category': 'Non-Recyclable',  'icon': 'trash-2',        'color': '#94A3B8', 'bg': 'rgba(148,163,184,0.15)','border': 'rgba(148,163,184,0.4)'},
}

def get_waste_category(class_name):
    return WASTE_CATEGORIES.get(class_name, {'category': 'Unknown', 'icon': 'help-circle', 'color': '#64748B', 'bg': 'rgba(100,116,139,0.15)', 'border': 'rgba(100,116,139,0.4)'})

app = Flask(__name__)


class WasteDetectorModel(nn.Module):
    """MobileNetV3-based waste classifier with objectness detection"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
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
        pooled = self.avgpool(features)
        pooled = pooled.flatten(1)
        class_logits = self.classifier(pooled)
        obj_score = self.objectness(pooled)
        return class_logits, obj_score, features


def load_model(model_path):
    """Load the trained MobileNetV3 model"""
    global CLASS_NAMES, IMG_SIZE
    
    print(f"  Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    CLASS_NAMES = checkpoint.get('classes',
        ['battery', 'biological', 'cardboard', 'clothes', 'glass',
         'metal', 'paper', 'plastic', 'shoes', 'trash'])
    IMG_SIZE = checkpoint.get('img_size', 720)
    
    num_classes = len(CLASS_NAMES)
    model = WasteDetectorModel(num_classes=num_classes)
    
    state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint))
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"  Warning: Strict loading failed, using loose loading: {e}")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    val_acc = checkpoint.get('val_acc', 'N/A')
    print(f"  Model loaded! Classes: {num_classes}, Val acc: {val_acc}")
    
    return model


_transform_cache = {}

def get_transform(img_size):
    if img_size not in _transform_cache:
        _transform_cache[img_size] = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return _transform_cache[img_size]


def generate_gradcam(model, input_tensor, class_idx):
    """Generate GradCAM heatmap"""
    model.eval()
    feature_maps = []
    gradients = []
    
    def forward_hook(module, input, output):
        feature_maps.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    last_conv = model.features[-1]
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
    
    grads = gradients[0]
    fmaps = feature_maps[0]
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * fmaps, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def get_bounding_box_from_cam(cam, original_size, threshold=0.3):
    """Extract bounding box from GradCAM heatmap"""
    cam_resized = cv2.resize(cam, (original_size[0], original_size[1]))
    binary = (cam_resized > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(original_size[0] - x, w + 2 * pad_x)
    h = min(original_size[1] - y, h + 2 * pad_y)
    
    return (x, y, w, h)


def draw_detection(img_cv2, bbox, class_name, confidence, obj_score):
    """Draw bounding box and labels on image"""
    result = img_cv2.copy()
    h_img, w_img = result.shape[:2]
    
    color = (0, 255, 100) if confidence > 0.8 else ((0, 200, 255) if confidence > 0.5 else (0, 100, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        label = f"{class_name}: {confidence*100:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(result, (x, y - text_h - 10), (x + text_w + 8, y), color, -1)
        cv2.putText(result, label, (x + 4, y - 5), font, 0.7, (0, 0, 0), 2)
    
    label_top = f"{class_name}: {confidence*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label_top, font, 1.0, 2)
    cv2.rectangle(result, (8, 8), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.rectangle(result, (8, 8), (tw + 20, th + 20), color, 2)
    cv2.putText(result, label_top, (14, th + 14), font, 1.0, color, 2)
    
    bar_text = f"Object: {obj_score*100:.0f}%"
    (btw, bth), _ = cv2.getTextSize(bar_text, font, 0.6, 1)
    cv2.putText(result, bar_text, (12, h_img - 12), font, 0.6, (200, 200, 200), 1)
    
    return result


def classify_frame(model, pil_img, skip_gradcam=False):
    """Classify a single image frame from camera"""
    import time
    transform = get_transform(IMG_SIZE)
    original_size = pil_img.size
    
    input_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        class_logits, obj_score_raw, _ = model(input_tensor)
    
    probabilities = torch.softmax(class_logits, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_val = confidence.item()
    obj_score = torch.sigmoid(obj_score_raw).item()
    
    top5_probs, top5_indices = torch.topk(probabilities, min(5, len(CLASS_NAMES)), dim=1)
    top5 = []
    for i in range(top5_probs.shape[1]):
        top5.append({
            'class': CLASS_NAMES[top5_indices[0][i].item()],
            'confidence': round(top5_probs[0][i].item() * 100, 1)
        })
    
    bbox = None
    if not skip_gradcam:
        input_for_cam = transform(pil_img).unsqueeze(0)
        
        start_cam = time.time()
        cam = generate_gradcam(model, input_for_cam, predicted_idx.item())
        cam_taken = (time.time() - start_cam) * 1000
        print(f"[CAM] GradCAM hoàn thành sau {cam_taken:.1f}ms (skip_gradcam={skip_gradcam})")
        
        if cam is not None and obj_score > 0.05:
            bbox = get_bounding_box_from_cam(cam, original_size)
    
    img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result_img = draw_detection(img_cv2, bbox, predicted_class, confidence_val, obj_score)
    
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)
    buffer = io.BytesIO()
    result_pil.save(buffer, format='JPEG', quality=90)
    result_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    cat_info = get_waste_category(predicted_class)
    
    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence_val * 100, 1),
        'obj_score': round(obj_score * 100, 1),
        'top5': top5,
        'result_image': result_b64,
        'has_bbox': bbox is not None,
        'category': cat_info['category'],
        'cat_icon': cat_info['icon'],
        'cat_color': cat_info['color'],
        'cat_bg': cat_info['bg'],
        'cat_border': cat_info['border'],
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
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        :root {
            --bg-primary: #0B0B10;
            --bg-secondary: #12121A;
            --bg-card: rgba(18, 18, 26, 0.7);
            --border: rgba(255, 255, 255, 0.06);
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
            --neon-cyan: #00F3FF;
            --neon-green: #0AFF00;
            --neon-red: #FF003C;
            --neon-yellow: #FFD600;
            --accent-blue: #3B82F6;
            --font-display: 'Orbitron', sans-serif;
            --font-body: 'Exo 2', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-body);
            min-height: 100vh;
            overflow-x: hidden;
            background-image:
                radial-gradient(circle at 30% 20%, rgba(0, 243, 255, 0.04) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(59, 130, 246, 0.03) 0%, transparent 50%),
                linear-gradient(rgba(0, 243, 255, 0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 243, 255, 0.015) 1px, transparent 1px);
            background-size: 100%, 100%, 60px 60px, 60px 60px;
        }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }

        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 0;
            margin-bottom: 32px;
            border-bottom: 1px solid var(--border);
        }

        .logo-section {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(0, 243, 255, 0.15), rgba(59, 130, 246, 0.1));
            border: 1px solid rgba(0, 243, 255, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.15);
            position: relative;
        }

        .logo-icon::after {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 12px;
            background: rgba(0, 243, 255, 0.08);
            animation: pulse-glow 3s ease-in-out infinite;
        }

        @keyframes pulse-glow {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }

        .logo-title {
            font-family: var(--font-display);
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: 3px;
            background: linear-gradient(90deg, #fff, var(--neon-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-subtitle {
            font-size: 0.65rem;
            letter-spacing: 4px;
            text-transform: uppercase;
            color: var(--neon-cyan);
            opacity: 0.6;
        }

        .status-badges {
            display: flex;
            gap: 12px;
        }

        .badge {
            padding: 8px 16px;
            border-radius: 8px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            backdrop-filter: blur(12px);
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .badge-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            animation: pulse-dot 2s ease-in-out infinite;
        }

        .badge-dot.active { background: var(--neon-green); box-shadow: 0 0 8px var(--neon-green); }
        .badge-dot.idle { background: var(--text-muted); }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 420px;
            gap: 24px;
            min-height: calc(100vh - 160px);
        }

        .camera-panel {
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border);
            backdrop-filter: blur(12px);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .panel-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .panel-title {
            font-family: var(--font-display);
            font-size: 0.8rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .camera-viewport {
            flex: 1;
            position: relative;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 480px;
        }

        .camera-viewport video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .camera-viewport img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            z-index: 2;
        }

        .scan-line {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
            opacity: 0;
            pointer-events: none;
            z-index: 10;
        }

        .scan-line.active {
            opacity: 1;
            animation: scan-sweep 1.5s ease-in-out;
        }

        @keyframes scan-sweep {
            0% { top: 0; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }

        .corner-brackets {
            position: absolute;
            inset: 20px;
            pointer-events: none;
            z-index: 5;
        }

        .corner-brackets::before,
        .corner-brackets::after {
            content: '';
            position: absolute;
            width: 30px;
            height: 30px;
            border-color: var(--neon-cyan);
            opacity: 0.4;
        }

        .corner-brackets::before {
            top: 0;
            left: 0;
            border-top: 2px solid;
            border-left: 2px solid;
        }

        .corner-brackets::after {
            top: 0;
            right: 0;
            border-top: 2px solid;
            border-right: 2px solid;
        }

        .corner-brackets-bottom {
            position: absolute;
            inset: 20px;
            pointer-events: none;
            z-index: 5;
        }

        .corner-brackets-bottom::before {
            content: '';
            position: absolute;
            width: 30px;
            height: 30px;
            bottom: 0;
            left: 0;
            border-bottom: 2px solid var(--neon-cyan);
            border-left: 2px solid var(--neon-cyan);
            opacity: 0.4;
        }

        .corner-brackets-bottom::after {
            content: '';
            position: absolute;
            width: 30px;
            height: 30px;
            bottom: 0;
            right: 0;
            border-bottom: 2px solid var(--neon-cyan);
            border-right: 2px solid var(--neon-cyan);
            opacity: 0.4;
        }

        .camera-placeholder {
            text-align: center;
            color: var(--text-muted);
        }

        .camera-placeholder i {
            margin-bottom: 16px;
        }

        .camera-controls {
            padding: 16px 20px;
            border-top: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
        }

        .result-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .result-card {
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border);
            backdrop-filter: blur(12px);
            overflow: hidden;
            transition: border-color 0.5s ease;
        }

        .result-card.detected {
            border-color: rgba(0, 243, 255, 0.3);
            box-shadow: 0 0 30px rgba(0, 243, 255, 0.08);
        }

        .result-main {
            padding: 24px;
            text-align: center;
        }

        .result-idle {
            padding: 60px 24px;
            text-align: center;
            color: var(--text-muted);
        }

        .result-idle i {
            margin-bottom: 16px;
            opacity: 0.3;
        }

        .result-idle p {
            font-size: 0.85rem;
            line-height: 1.6;
        }

        .result-class {
            font-family: var(--font-display);
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 8px;
            text-shadow: 0 0 20px rgba(0, 243, 255, 0.4);
        }

        .result-confidence {
            font-size: 3rem;
            font-weight: 700;
            font-family: var(--font-display);
            margin: 8px 0;
        }

        .result-confidence.high { color: var(--neon-green); text-shadow: 0 0 20px rgba(10, 255, 0, 0.3); }
        .result-confidence.medium { color: var(--neon-yellow); text-shadow: 0 0 20px rgba(255, 214, 0, 0.3); }
        .result-confidence.low { color: var(--neon-red); text-shadow: 0 0 20px rgba(255, 0, 60, 0.3); }

        .result-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        .category-badge {
            display: none;
            align-items: center;
            justify-content: center;
            gap: 6px;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin: 12px auto 0;
            width: fit-content;
        }

        .category-badge.visible {
            display: flex;
        }

        .top5-card {
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border);
            backdrop-filter: blur(12px);
        }

        .top5-list {
            padding: 4px 20px 16px;
        }

        .top5-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }

        .top5-item:last-child {
            border-bottom: none;
        }

        .top5-rank {
            font-family: var(--font-display);
            font-size: 0.65rem;
            color: var(--text-muted);
            width: 20px;
            text-align: center;
        }

        .top5-name {
            flex: 1;
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: capitalize;
        }

        .top5-bar-container {
            width: 100px;
            height: 4px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 2px;
            overflow: hidden;
        }

        .top5-bar {
            height: 100%;
            border-radius: 2px;
            background: linear-gradient(90deg, var(--neon-cyan), var(--accent-blue));
            transition: width 0.8s ease;
        }

        .top5-percent {
            font-family: var(--font-display);
            font-size: 0.75rem;
            color: var(--text-secondary);
            width: 50px;
            text-align: right;
        }

        .history-card {
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border);
            backdrop-filter: blur(12px);
            flex: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .history-list {
            padding: 8px 12px;
            overflow-y: auto;
            max-height: 280px;
            flex: 1;
        }

        .history-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px;
            border-radius: 8px;
            transition: background 0.2s;
            cursor: pointer;
        }

        .history-item:hover {
            background: rgba(255, 255, 255, 0.03);
        }

        .history-thumb {
            width: 40px;
            height: 40px;
            border-radius: 6px;
            object-fit: cover;
            border: 1px solid var(--border);
        }

        .history-info {
            flex: 1;
            min-width: 0;
        }

        .history-class {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: capitalize;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .history-time {
            font-size: 0.65rem;
            color: var(--text-muted);
        }

        .history-conf {
            font-family: var(--font-display);
            font-size: 0.7rem;
            font-weight: 600;
        }

        .history-empty {
            padding: 32px;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
        }

        .processing-overlay {
            position: absolute;
            inset: 0;
            background: rgba(11, 11, 16, 0.85);
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 16px;
            z-index: 20;
        }

        .processing-overlay.active {
            display: flex;
        }

        .spinner {
            width: 48px;
            height: 48px;
            border: 3px solid rgba(0, 243, 255, 0.1);
            border-top-color: var(--neon-cyan);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .processing-text {
            font-family: var(--font-display);
            font-size: 0.75rem;
            letter-spacing: 3px;
            color: var(--neon-cyan);
        }

        .realtime-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.7rem;
            font-family: var(--font-display);
            letter-spacing: 1px;
            color: var(--text-muted);
        }

        .realtime-indicator.scanning {
            color: var(--neon-green);
        }

        .realtime-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
        }

        .realtime-indicator.scanning .realtime-dot {
            background: var(--neon-red);
            box-shadow: 0 0 8px var(--neon-red);
            animation: pulse-dot 1s ease-in-out infinite;
        }

        .scan-toggle-btn {
            padding: 10px 24px;
            border-radius: 10px;
            border: 2px solid var(--neon-cyan);
            background: rgba(0, 243, 255, 0.08);
            color: var(--neon-cyan);
            font-family: var(--font-display);
            font-size: 0.75rem;
            letter-spacing: 2px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .scan-toggle-btn:hover {
            background: rgba(0, 243, 255, 0.15);
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.2);
        }

        .scan-toggle-btn.active {
            border-color: var(--neon-red);
            color: var(--neon-red);
            background: rgba(255, 0, 60, 0.08);
        }

        .scan-toggle-btn.active:hover {
            background: rgba(255, 0, 60, 0.15);
            box-shadow: 0 0 20px rgba(255, 0, 60, 0.2);
        }

        .fps-counter {
            font-family: var(--font-display);
            font-size: 0.65rem;
            color: var(--text-muted);
            letter-spacing: 1px;
        }

        .camera-select {
            background: var(--bg-secondary);
            color: var(--text-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 4px 8px;
            font-family: var(--font-body);
            font-size: 0.7rem;
            cursor: pointer;
            outline: none;
            max-width: 220px;
        }

        .camera-select:focus {
            border-color: var(--neon-cyan);
        }

        .camera-select option {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }

        /* Debug log */
        #debugLog {
            position: fixed;
            bottom: 12px;
            left: 12px;
            background: rgba(0,0,0,0.85);
            color: #0AFF00;
            font-family: monospace;
            font-size: 0.65rem;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid rgba(10,255,0,0.2);
            max-width: 380px;
            z-index: 9999;
            pointer-events: none;
            display: none;
        }

        @media (max-width: 900px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            .header {
                flex-direction: column;
                gap: 16px;
            }
            .status-badges {
                width: 100%;
                justify-content: center;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }
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
                    <div class="logo-subtitle">Waste Analysis Scanner & Type Engine</div>
                </div>
            </div>
            <div class="status-badges">
                <div class="badge">
                    <span class="badge-dot active" id="cameraDot"></span>
                    <span id="cameraStatus">Initializing...</span>
                </div>
                <div class="badge">
                    <span class="badge-dot active" style="background:var(--accent-blue);box-shadow:0 0 8px var(--accent-blue)"></span>
                    <span>MobileNetV3</span>
                </div>
                <div class="badge">
                    <i data-lucide="images" style="width:14px;height:14px;color:var(--text-muted)"></i>
                    <span id="scanCount">0</span> scans
                </div>
            </div>
        </header>

        <div class="main-content">
            <div class="camera-panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <i data-lucide="video" style="width:16px;height:16px;color:var(--neon-cyan)"></i>
                        LIVE FEED
                    </div>
                    <div style="display:flex;align-items:center;gap:10px">
                        <input type="color" id="bgColorPicker" value="#000000" title="Chỉnh màu nền" style="width:24px;height:24px;border:none;border-radius:4px;cursor:pointer;background:transparent;padding:0;">
                        <select class="camera-select" id="cameraSelect" onchange="switchCamera(this.value)">
                            <option value="">Loading cameras...</option>
                        </select>
                        <div class="panel-title" style="color:var(--text-muted);font-family:var(--font-body);letter-spacing:0;font-size:0.75rem" id="resolution">--</div>
                    </div>
                </div>
                <div class="camera-viewport" id="viewport">
                    <video id="cameraFeed" autoplay playsinline muted style="display:none"></video>
                    <img id="resultImage" style="display:none" alt="Detection result">
                    <canvas id="captureCanvas" style="display:none"></canvas>
                    
                    <div id="liveOverlay" style="position:absolute;top:12px;left:12px;z-index:15;display:none">
                        <div id="liveLabel" style="background:rgba(0,0,0,0.8);padding:8px 16px;border-radius:8px;border:2px solid var(--neon-green);font-family:var(--font-display);font-size:1rem;letter-spacing:1px;color:var(--neon-green);text-transform:uppercase">--</div>
                        <div id="liveConfidence" style="background:rgba(0,0,0,0.8);padding:4px 16px;border-radius:0 0 8px 8px;font-family:var(--font-display);font-size:0.75rem;color:var(--text-secondary);text-align:center">--</div>
                    </div>
                    
                    <div class="corner-brackets"></div>
                    <div class="corner-brackets-bottom"></div>
                    <div class="scan-line" id="scanLine"></div>
                    
                    <div class="processing-overlay" id="processingOverlay">
                        <div class="spinner"></div>
                        <div class="processing-text">ANALYZING...</div>
                    </div>
                    
                    <div class="camera-placeholder" id="cameraPlaceholder">
                        <i data-lucide="camera-off" style="width:48px;height:48px"></i>
                        <p style="font-size:0.9rem;font-weight:500;margin-bottom:4px">Camera Initializing</p>
                        <p style="font-size:0.75rem">Please allow camera access</p>
                    </div>
                </div>
                <div class="camera-controls">
                    <div class="realtime-indicator" id="realtimeIndicator">
                        <span class="realtime-dot"></span>
                        <span id="realtimeStatus">STANDBY</span>
                    </div>
                    <button class="scan-toggle-btn" id="scanToggleBtn" onclick="toggleScanning()">
                        <i data-lucide="scan" style="width:16px;height:16px"></i>
                        <span id="scanBtnText">START SCAN</span>
                    </button>
                    <span class="fps-counter" id="fpsCounter">-- fps</span>
                </div>
            </div>

            <div class="result-panel">
                <div class="result-card" id="resultCard">
                    <div class="panel-header">
                        <div class="panel-title">
                            <i data-lucide="brain" style="width:16px;height:16px;color:var(--neon-cyan)"></i>
                            DETECTION RESULT
                        </div>
                    </div>
                    <div class="result-idle" id="resultIdle">
                        <i data-lucide="scan" style="width:40px;height:40px"></i>
                        <p>Point your camera at waste items<br>and press <kbd style="padding:2px 6px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:3px;font-family:var(--font-display);font-size:0.65rem">SPACE</kbd> to scan</p>
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
                        <div class="history-empty" id="historyEmpty">
                            No scans yet
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Debug log (toggle with D key) -->
    <div id="debugLog"></div>

    <script>
        lucide.createIcons();

        // ─── State ───────────────────────────────────────────────────────────
        let stream = null;
        let isProcessing = false;
        let scanInterval = null;
        let isScanning = false;
        let scanCount = 0;
        let lastFpsTime = Date.now();
        let frameCount = 0;
        let debugVisible = false;

        // ─── DOM ─────────────────────────────────────────────────────────────
        const video      = document.getElementById('cameraFeed');
        const canvas     = document.getElementById('captureCanvas');
        const resultImage = document.getElementById('resultImage');
        const placeholder = document.getElementById('cameraPlaceholder');
        const scanLine   = document.getElementById('scanLine');
        const debugLog   = document.getElementById('debugLog');

        // ─── Debug helper ────────────────────────────────────────────────────
        function dbg(msg) {
            console.log('[WASTE]', msg);
            debugLog.textContent = msg;
        }

        // ─── Camera ready check ──────────────────────────────────────────────
        // BUG FIX #1: Don't rely on `video.onplaying` event which can fire
        // before the handler is assigned. Poll readyState instead.
        function isVideoReady() {
            return (
                video.readyState >= 2          // HAVE_CURRENT_DATA or better
                && video.videoWidth > 0
                && video.videoHeight > 0
                && !video.paused
                && !video.ended
            );
        }

        // ─── Camera Init ─────────────────────────────────────────────────────
        async function initCamera() {
            try {
                dbg('Requesting camera access...');
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: 'environment',
                        width:  { ideal: 720 },
                        height: { ideal: 720 }
                    }
                });

                video.srcObject = stream;

                // BUG FIX #1 cont'd: Wait for metadata + a short play delay
                // so videoWidth/Height are guaranteed non-zero.
                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => reject(new Error('Video load timeout')), 10000);
                    video.onloadedmetadata = () => {
                        clearTimeout(timeout);
                        video.play().then(resolve).catch(reject);
                    };
                });

                video.style.display = 'block';
                placeholder.style.display = 'none';

                document.getElementById('resolution').textContent =
                    `${video.videoWidth} x ${video.videoHeight}`;
                document.getElementById('cameraStatus').textContent = 'Camera Active';
                document.getElementById('cameraDot').classList.add('active');

                dbg(`Camera ready: ${video.videoWidth}x${video.videoHeight}`);
                await populateCameraList();
                
                // Tự động bật nhận diện thời gian thực sau 1.5 giây
                setTimeout(() => {
                    if (!isScanning) startScanning();
                }, 1500);

            } catch (err) {
                console.error('Camera error:', err);
                document.getElementById('cameraStatus').textContent = 'Camera Error';
                document.getElementById('cameraDot').style.background = 'var(--neon-red)';
                placeholder.innerHTML = `
                    <i data-lucide="camera-off" style="width:48px;height:48px;margin-bottom:16px"></i>
                    <p style="font-size:0.9rem;font-weight:500;margin-bottom:4px">Camera Access Denied</p>
                    <p style="font-size:0.75rem">Please allow camera permissions and reload</p>
                `;
                lucide.createIcons();
                dbg('Camera error: ' + err.message);
            }
        }

        async function populateCameraList() {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(d => d.kind === 'videoinput');
                const select = document.getElementById('cameraSelect');
                select.innerHTML = '';
                videoDevices.forEach((device, idx) => {
                    const option = document.createElement('option');
                    option.value = device.deviceId;
                    option.textContent = device.label || ('Camera ' + (idx + 1));
                    if (stream) {
                        const currentTrack = stream.getVideoTracks()[0];
                        if (currentTrack && currentTrack.getSettings().deviceId === device.deviceId) {
                            option.selected = true;
                        }
                    }
                    select.appendChild(option);
                });
            } catch (err) {
                console.error('Failed to enumerate devices:', err);
            }
        }

        async function switchCamera(deviceId) {
            if (!deviceId) return;
            const wasScanning = isScanning;
            if (wasScanning) stopScanning();
            if (stream) stream.getTracks().forEach(t => t.stop());

            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: { exact: deviceId }, width: { ideal: 720 }, height: { ideal: 720 } }
                });
                video.srcObject = stream;

                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => reject(new Error('Switch timeout')), 10000);
                    video.onloadedmetadata = () => {
                        clearTimeout(timeout);
                        video.play().then(resolve).catch(reject);
                    };
                });

                video.style.display = 'block';
                placeholder.style.display = 'none';
                document.getElementById('resolution').textContent =
                    `${video.videoWidth} x ${video.videoHeight}`;
                document.getElementById('cameraStatus').textContent = 'Camera Active';

                if (wasScanning) startScanning();
            } catch (err) {
                console.error('Failed to switch camera:', err);
                document.getElementById('cameraStatus').textContent = 'Switch Failed';
            }
        }

        // ─── Scanning Loop ───────────────────────────────────────────────────
        function toggleScanning() {
            if (isScanning) stopScanning(); else startScanning();
        }

        function startScanning() {
            if (!stream) {
                dbg('Cannot start: no stream');
                return;
            }

            // BUG FIX #2: Ensure video is truly ready before we start;
            // if not, wait with a small poll instead of a blind setTimeout.
            if (!isVideoReady()) {
                dbg('Video not ready yet, retrying in 300ms...');
                setTimeout(startScanning, 300);
                return;
            }

            isScanning = true;

            const btn = document.getElementById('scanToggleBtn');
            btn.classList.add('active');
            document.getElementById('scanBtnText').textContent = 'STOP SCAN';
            const oldSvg = btn.querySelector('.lucide');
            if (oldSvg) oldSvg.remove();
            const newI = document.createElement('i');
            newI.setAttribute('data-lucide', 'scan-line');
            newI.style.width = '16px'; newI.style.height = '16px';
            btn.insertBefore(newI, btn.firstChild);
            lucide.createIcons();

            const indicator = document.getElementById('realtimeIndicator');
            indicator.classList.add('scanning');
            document.getElementById('realtimeStatus').textContent = 'SCANNING';
            document.getElementById('cameraStatus').textContent = 'Scanning...';

            dbg('Scanning started');
            scheduleNextDetect();
        }

        function scheduleNextDetect() {
            if (!isScanning) return;
            autoDetect().finally(() => {
                // BUG FIX #3: Always use .finally() so next frame is
                // scheduled even if autoDetect threw an unhandled error.
                if (isScanning) {
                    scanInterval = setTimeout(scheduleNextDetect, 800);
                }
            });
        }

        function stopScanning() {
            isScanning = false;
            isProcessing = false;   // BUG FIX #4: reset flag on stop
            if (scanInterval) { clearTimeout(scanInterval); scanInterval = null; }

            const btn = document.getElementById('scanToggleBtn');
            btn.classList.remove('active');
            document.getElementById('scanBtnText').textContent = 'START SCAN';
            const oldSvg = btn.querySelector('.lucide');
            if (oldSvg) oldSvg.remove();
            const newI = document.createElement('i');
            newI.setAttribute('data-lucide', 'scan');
            newI.style.width = '16px'; newI.style.height = '16px';
            btn.insertBefore(newI, btn.firstChild);
            lucide.createIcons();

            document.getElementById('realtimeIndicator').classList.remove('scanning');
            document.getElementById('realtimeStatus').textContent = 'STANDBY';
            document.getElementById('cameraStatus').textContent = 'Camera Active';
            document.getElementById('fpsCounter').textContent = '-- fps';

            video.style.display = 'block';
            resultImage.style.display = 'none';
            document.getElementById('liveOverlay').style.display = 'none';

            dbg('Scanning stopped');
        }

        // ─── Frame Capture & API Call ─────────────────────────────────────────
        async function autoDetect() {
            if (isProcessing) {
                dbg('Skipped: already processing');
                return;
            }
            if (!isScanning || !stream) return;

            // BUG FIX #5: Always verify readyState right before capture,
            // not just at scan-start, because the camera may have changed.
            if (!isVideoReady()) {
                dbg(`Skipped: video not ready (readyState=${video.readyState} ${video.videoWidth}x${video.videoHeight})`);
                return;
            }

            isProcessing = true;
            try {
                // Capture frame
                canvas.width  = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);

                // BUG FIX #6: Validate the captured frame is not blank
                // by checking that the canvas has valid dimensions.
                if (canvas.width === 0 || canvas.height === 0) {
                    dbg('Canvas is empty, skipping');
                    return;
                }

                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                dbg(`Sending ${Math.round(imageData.length / 1024)}KB frame...`);

                const t0 = performance.now();
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData, realtime: true })
                });

                if (!response.ok) {
                    const txt = await response.text();
                    dbg(`Server error ${response.status}: ${txt.slice(0, 80)}`);
                    return;
                }

                const result = await response.json();
                const elapsed = Math.round(performance.now() - t0);

                if (result.error) {
                    dbg('API error: ' + result.error);
                    return;
                }

                dbg(`${result.predicted_class} ${result.confidence}% (${elapsed}ms)`);

                // FPS
                frameCount++;
                const now = Date.now();
                if (now - lastFpsTime >= 2000) {
                    const fps = (frameCount / ((now - lastFpsTime) / 1000)).toFixed(1);
                    document.getElementById('fpsCounter').textContent = fps + ' fps';
                    frameCount = 0;
                    lastFpsTime = now;
                }

                displayRealtimeResult(result);

            } catch (err) {
                dbg('Fetch error: ' + err.message);
                console.error('[DETECT]', err);
            } finally {
                // BUG FIX #4 cont'd: always release the lock
                isProcessing = false;
            }
        }

        // ─── UI Updates ───────────────────────────────────────────────────────
        function displayRealtimeResult(result) {
            scanCount++;
            document.getElementById('scanCount').textContent = scanCount;

            // TẮT HIỂN THỊ ẢNH TĨNH CÓ BOUNDING BOX ĐÈ LÊN VIDEO (GIÚP VIDEO MƯỢT HƠN)
            // resultImage.src = 'data:image/jpeg;base64,' + result.result_image;
            // resultImage.style.display = 'block';

            // CHỈ CẬP NHẬT LABEL REALTIME (NẾU CẦN, HOẶC CŨNG CÓ THỂ TẮT)
            const liveOverlay = document.getElementById('liveOverlay');
            // const liveLabel   = document.getElementById('liveLabel');
            // const liveConf    = document.getElementById('liveConfidence');
            // liveOverlay.style.display = 'block';
            liveOverlay.style.display = 'none'; // Người dùng không cần hiển thị trên màn hình loại gì

            // Tự động cuộn xuống History (Tùy chọn)
            document.getElementById('resultCard').classList.add('detected');
            document.getElementById('resultIdle').style.display  = 'none';
            document.getElementById('resultMain').style.display  = 'block';
            
            // Cập nhật ảnh tĩnh bên RESULT CARD thay vì đè lên video
            let resultImgPreview = document.getElementById('resultImgPreview');
            if(!resultImgPreview) {
               resultImgPreview = document.createElement('img');
               resultImgPreview.id = 'resultImgPreview';
               resultImgPreview.style.width = '100%';
               resultImgPreview.style.borderRadius = '8px';
               resultImgPreview.style.marginBottom = '12px';
               document.getElementById('resultMain').insertBefore(resultImgPreview, document.getElementById('resultMain').firstChild);
            }
            resultImgPreview.src = 'data:image/jpeg;base64,' + result.result_image;
            
            document.getElementById('resultClass').textContent   = result.predicted_class;

            const confEl = document.getElementById('resultConfidence');
            confEl.textContent = result.confidence.toFixed(1) + '%';
            confEl.className = 'result-confidence '
                + (result.confidence >= 80 ? 'high' : result.confidence >= 50 ? 'medium' : 'low');

            const catBadge  = document.getElementById('categoryBadge');
            const catIconEl = document.getElementById('categoryIcon');
            catBadge.classList.add('visible');
            document.getElementById('categoryText').textContent = result.category;
            catBadge.style.background = result.cat_bg;
            catBadge.style.border     = '1px solid ' + result.cat_border;
            catBadge.style.color      = result.cat_color;
            const oldSvg = catBadge.querySelector('.lucide');
            if (oldSvg) oldSvg.remove();
            const newI = document.createElement('i');
            newI.id = 'categoryIcon';
            newI.setAttribute('data-lucide', result.cat_icon);
            newI.style.width = '14px'; newI.style.height = '14px';
            newI.style.color = result.cat_color;
            catBadge.insertBefore(newI, catBadge.firstChild);
            lucide.createIcons();

            const top5Card = document.getElementById('top5Card');
            top5Card.style.display = 'block';
            const top5List = document.getElementById('top5List');
            top5List.innerHTML = '';
            result.top5.forEach((item, idx) => {
                const div = document.createElement('div');
                div.className = 'top5-item';
                div.innerHTML = `
                    <span class="top5-rank">${idx + 1}</span>
                    <span class="top5-name">${item.class}</span>
                    <div class="top5-bar-container">
                        <div class="top5-bar" style="width:${item.confidence}%"></div>
                    </div>
                    <span class="top5-percent">${item.confidence}%</span>`;
                top5List.appendChild(div);
            });

            addToHistory(result);
        }

        function addToHistory(result) {
            document.getElementById('historyEmpty').style.display = 'none';
            const historyList = document.getElementById('historyList');
            while (historyList.children.length > 20) {
                historyList.removeChild(historyList.lastChild);
            }

            const timeStr = new Date().toLocaleTimeString('vi-VN',
                { hour: '2-digit', minute: '2-digit', second: '2-digit' });

            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <img class="history-thumb" src="data:image/jpeg;base64,${result.result_image}" alt="${result.predicted_class}">
                <div class="history-info">
                    <div class="history-class">${result.predicted_class}</div>
                    <div class="history-time" style="display:flex;align-items:center;gap:4px">
                        ${timeStr}
                        <span style="font-size:0.55rem;padding:1px 4px;border-radius:3px;background:${result.cat_bg};border:1px solid ${result.cat_border};color:${result.cat_color}">${result.category}</span>
                    </div>
                </div>
                <span class="history-conf" style="color:${result.confidence>=80?'var(--neon-green)':result.confidence>=50?'var(--neon-yellow)':'var(--neon-red)'}">${result.confidence}%</span>`;

            item.addEventListener('click', () => {
                let resultImgPreview = document.getElementById('resultImgPreview');
                if(!resultImgPreview) {
                   resultImgPreview = document.createElement('img');
                   resultImgPreview.id = 'resultImgPreview';
                   resultImgPreview.style.width = '100%';
                   resultImgPreview.style.borderRadius = '8px';
                   resultImgPreview.style.marginBottom = '12px';
                   document.getElementById('resultMain').insertBefore(resultImgPreview, document.getElementById('resultMain').firstChild);
                }
                resultImgPreview.src = 'data:image/jpeg;base64,' + result.result_image;
                
                // Không hiển thị đè lên luồng livestream
                // resultImage.src = 'data:image/jpeg;base64,' + result.result_image;
                // resultImage.style.display = 'block';
                // video.style.display = 'none';
                
                // Đảm bảo video vẫn tiếp tục phát
                video.style.display = 'block';
                resultImage.style.display = 'none';
                
                document.getElementById('resultClass').textContent = result.predicted_class;
                const c = document.getElementById('resultConfidence');
                c.textContent = result.confidence.toFixed(1) + '%';
                c.className = 'result-confidence '
                    + (result.confidence>=80?'high':result.confidence>=50?'medium':'low');
                const catBadge = document.getElementById('categoryBadge');
                catBadge.classList.add('visible');
                document.getElementById('categoryText').textContent = result.category;
                catBadge.style.background = result.cat_bg;
                catBadge.style.border     = '1px solid ' + result.cat_border;
                catBadge.style.color      = result.cat_color;
                const oldSvg = catBadge.querySelector('.lucide');
                if (oldSvg) oldSvg.remove();
                const newI = document.createElement('i');
                newI.id = 'categoryIcon';
                newI.setAttribute('data-lucide', result.cat_icon);
                newI.style.width = '14px'; newI.style.height = '14px';
                newI.style.color = result.cat_color;
                catBadge.insertBefore(newI, catBadge.firstChild);
                lucide.createIcons();
            });

            historyList.insertBefore(item, historyList.firstChild);
        }

        // ─── Keyboard shortcuts ───────────────────────────────────────────────
        document.addEventListener('keydown', (e) => {
            if (e.repeat) return;
            if (e.code === 'Space') {
                e.preventDefault();
                toggleScanning();
            }
            // Press D to toggle debug overlay
            if (e.code === 'KeyD') {
                debugVisible = !debugVisible;
                debugLog.style.display = debugVisible ? 'block' : 'none';
            }
        });

        // Đổi màu nền camera 
        document.getElementById('bgColorPicker').addEventListener('input', (e) => {
            document.getElementById('viewport').style.background = e.target.value;
        });

        // ─── Boot ─────────────────────────────────────────────────────────────
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
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body received'}), 400

        image_data = data.get('image', '')
        realtime   = data.get('realtime', False)

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode and open image
        img_bytes = base64.b64decode(image_data)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Validate image isn't tiny/blank
        if pil_img.width < 10 or pil_img.height < 10:
            return jsonify({'error': f'Image too small: {pil_img.size}'}), 400

        # Buộc bật GradCAM để luôn lấy tọa độ bounding box hiển thị
        result = classify_frame(model, pil_img, skip_gradcam=False)
        
        # IN LOG KIỂM TRA RA TERMINAL
        print(f"[API] Nhận diện: {result['predicted_class'].upper()} "
              f"| Độ tự tin: {result['confidence']}% "
              f"| Điểm vật thể: {result['obj_score']}% "
              f"| Vẽ khung: {'Có' if result['has_bbox'] else 'Không'}")
              
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  W.A.S.T.E. SCANNER - AI Waste Detection Camera")
    print("  Waste Analysis Scanner & Type Engine")
    print("=" * 60)

    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            print(f"  Model load failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    else:
        print(f"  Model not found: {MODEL_PATH}")
        exit(1)

    print()
    print("=" * 60)
    print("  SCANNER ONLINE")
    print("  Open: http://localhost:9999")
    print("  Press SPACE to toggle scanning")
    print("  Press D    to toggle debug overlay")
    print("=" * 60)
    print()

    app.run(debug=True, port=9999, use_reloader=False)