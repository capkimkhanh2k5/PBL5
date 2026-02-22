
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
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
        self.objectness = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
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
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    val_acc = checkpoint.get('val_acc', 'N/A')
    print(f"  Model loaded! Classes: {num_classes}, Val acc: {val_acc}")
    
    return model


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


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
    
    if bbox is not None:
        x, y, w, h = bbox
        color = (0, 255, 100) if confidence > 0.8 else ((0, 200, 255) if confidence > 0.5 else (0, 100, 255))
        
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        label = f"{class_name}: {confidence*100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(result, (x, y - text_h - 10), (x + text_w + 8, y), color, -1)
        cv2.putText(result, label, (x + 4, y - 5), font, 0.7, (0, 0, 0), 2)
    
    return result


def classify_frame(model, pil_img):
    """Classify a single image frame from camera"""
    transform = get_transform(IMG_SIZE)
    original_size = pil_img.size
    
    # Inference
    input_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        class_logits, obj_score_raw, _ = model(input_tensor)
    
    probabilities = torch.softmax(class_logits, dim=1)
    confidence, predicted_idx = torch.max(probabilities, dim=1)
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_val = confidence.item()
    obj_score = torch.sigmoid(obj_score_raw).item()
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probabilities, min(5, len(CLASS_NAMES)), dim=1)
    top5 = []
    for i in range(top5_probs.shape[1]):
        top5.append({
            'class': CLASS_NAMES[top5_indices[0][i].item()],
            'confidence': round(top5_probs[0][i].item() * 100, 1)
        })
    
    # GradCAM + Bounding Box
    input_for_cam = transform(pil_img).unsqueeze(0)
    cam = generate_gradcam(model, input_for_cam, predicted_idx.item())
    
    bbox = None
    if cam is not None and obj_score > 0.3:
        bbox = get_bounding_box_from_cam(cam, original_size)
    
    # Draw detection on image
    img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result_img = draw_detection(img_cv2, bbox, predicted_class, confidence_val, obj_score)
    
    # Convert result to base64
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)
    buffer = io.BytesIO()
    result_pil.save(buffer, format='JPEG', quality=90)
    result_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence_val * 100, 1),
        'obj_score': round(obj_score * 100, 1),
        'top5': top5,
        'result_image': result_b64,
        'has_bbox': bbox is not None
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

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }

        /* Header */
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

        /* Main Layout */
        .main-content {
            display: grid;
            grid-template-columns: 1fr 420px;
            gap: 24px;
            min-height: calc(100vh - 160px);
        }

        /* Camera Panel */
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

        .camera-viewport video,
        .camera-viewport img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Scan overlay animation */
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

        /* Corner brackets overlay */
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

        /* Camera Controls */
        .camera-controls {
            padding: 16px 20px;
            border-top: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
        }

        .capture-btn {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            border: 3px solid var(--neon-cyan);
            background: rgba(0, 243, 255, 0.08);
            cursor: pointer;
            position: relative;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .capture-btn:hover {
            background: rgba(0, 243, 255, 0.2);
            box-shadow: 0 0 30px rgba(0, 243, 255, 0.3);
            transform: scale(1.05);
        }

        .capture-btn:active {
            transform: scale(0.95);
        }

        .capture-btn::after {
            content: '';
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--neon-cyan), var(--accent-blue));
            transition: all 0.2s ease;
        }

        .capture-btn.capturing::after {
            animation: capture-flash 0.3s ease;
        }

        @keyframes capture-flash {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.8); }
            100% { opacity: 1; transform: scale(1); }
        }

        .key-hint {
            font-size: 0.7rem;
            color: var(--text-muted);
            letter-spacing: 1px;
        }

        .key-hint kbd {
            display: inline-block;
            padding: 2px 8px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 4px;
            font-family: var(--font-display);
            font-size: 0.65rem;
            margin: 0 2px;
        }

        /* Result Panel */
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

        /* Top 5 Predictions */
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

        /* History */
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

        /* Processing overlay */
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

        /* Responsive */
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
        <!-- Header -->
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

        <!-- Main Content -->
        <div class="main-content">
            <!-- Camera Feed -->
            <div class="camera-panel">
                <div class="panel-header">
                    <div class="panel-title">
                        <i data-lucide="video" style="width:16px;height:16px;color:var(--neon-cyan)"></i>
                        LIVE FEED
                    </div>
                    <div class="panel-title" style="color:var(--text-muted);font-family:var(--font-body);letter-spacing:0;font-size:0.75rem" id="resolution">--</div>
                </div>
                <div class="camera-viewport" id="viewport">
                    <video id="cameraFeed" autoplay playsinline muted style="display:none"></video>
                    <img id="resultImage" style="display:none" alt="Detection result">
                    <canvas id="captureCanvas" style="display:none"></canvas>
                    
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
                    <div class="key-hint">Press <kbd>SPACE</kbd> to capture</div>
                    <button class="capture-btn" id="captureBtn" onclick="captureFrame()" title="Capture Frame">
                    </button>
                    <button class="badge" id="toggleViewBtn" onclick="toggleView()" style="cursor:pointer;display:none">
                        <i data-lucide="eye" style="width:14px;height:14px;color:var(--neon-cyan)"></i>
                        <span>Live View</span>
                    </button>
                </div>
            </div>

            <!-- Results Sidebar -->
            <div class="result-panel">
                <!-- Main Result -->
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
                    </div>
                </div>

                <!-- Top 5 -->
                <div class="top5-card" id="top5Card" style="display:none">
                    <div class="panel-header">
                        <div class="panel-title">
                            <i data-lucide="bar-chart-3" style="width:16px;height:16px;color:var(--neon-cyan)"></i>
                            PROBABILITY DISTRIBUTION
                        </div>
                    </div>
                    <div class="top5-list" id="top5List"></div>
                </div>

                <!-- History -->
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

    <script>
        // Initialize Lucide icons
        lucide.createIcons();

        // State
        let stream = null;
        let isProcessing = false;
        let showingResult = false;
        let scanHistory = [];
        let scanCount = 0;

        // DOM Elements
        const video = document.getElementById('cameraFeed');
        const canvas = document.getElementById('captureCanvas');
        const resultImage = document.getElementById('resultImage');
        const placeholder = document.getElementById('cameraPlaceholder');
        const processingOverlay = document.getElementById('processingOverlay');
        const scanLine = document.getElementById('scanLine');
        const captureBtn = document.getElementById('captureBtn');
        const toggleViewBtn = document.getElementById('toggleViewBtn');

        // Initialize Camera
        async function initCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: 'environment',
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                });
                
                video.srcObject = stream;
                video.style.display = 'block';
                placeholder.style.display = 'none';
                
                video.onloadedmetadata = () => {
                    document.getElementById('resolution').textContent = 
                        `${video.videoWidth} x ${video.videoHeight}`;
                };
                
                document.getElementById('cameraStatus').textContent = 'Camera Active';
                document.getElementById('cameraDot').classList.add('active');
                
            } catch (err) {
                console.error('Camera error:', err);
                document.getElementById('cameraStatus').textContent = 'Camera Error';
                document.getElementById('cameraDot').classList.remove('active');
                document.getElementById('cameraDot').style.background = 'var(--neon-red)';
                placeholder.innerHTML = `
                    <i data-lucide="camera-off" style="width:48px;height:48px;margin-bottom:16px"></i>
                    <p style="font-size:0.9rem;font-weight:500;margin-bottom:4px">Camera Access Denied</p>
                    <p style="font-size:0.75rem">Please allow camera permissions and reload</p>
                `;
                lucide.createIcons();
            }
        }

        // Capture & Analyze
        async function captureFrame() {
            if (isProcessing || !stream) return;
            
            isProcessing = true;
            captureBtn.classList.add('capturing');
            processingOverlay.classList.add('active');
            
            // Show scan animation
            scanLine.classList.remove('active');
            void scanLine.offsetWidth;
            scanLine.classList.add('active');
            
            // Capture frame from video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            // Convert to base64
            const imageData = canvas.toDataURL('image/jpeg', 0.9);
            
            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                
                const result = await response.json();
                
                if (result.error) {
                    console.error('Classification error:', result.error);
                    return;
                }
                
                // Show result
                displayResult(result);
                
            } catch (err) {
                console.error('Network error:', err);
            } finally {
                isProcessing = false;
                captureBtn.classList.remove('capturing');
                processingOverlay.classList.remove('active');
            }
        }

        function displayResult(result) {
            scanCount++;
            document.getElementById('scanCount').textContent = scanCount;
            
            // Switch to result image
            showingResult = true;
            video.style.display = 'none';
            resultImage.src = 'data:image/jpeg;base64,' + result.result_image;
            resultImage.style.display = 'block';
            toggleViewBtn.style.display = 'flex';
            
            // Update result card
            const resultCard = document.getElementById('resultCard');
            resultCard.classList.add('detected');
            document.getElementById('resultIdle').style.display = 'none';
            document.getElementById('resultMain').style.display = 'block';
            
            document.getElementById('resultClass').textContent = result.predicted_class;
            
            const confEl = document.getElementById('resultConfidence');
            confEl.textContent = result.confidence.toFixed(1) + '%';
            confEl.className = 'result-confidence ' + 
                (result.confidence >= 80 ? 'high' : (result.confidence >= 50 ? 'medium' : 'low'));
            
            // Update top 5
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
                        <div class="top5-bar" style="width: ${item.confidence}%"></div>
                    </div>
                    <span class="top5-percent">${item.confidence}%</span>
                `;
                top5List.appendChild(div);
            });
            
            // Add to history
            addToHistory(result);
        }

        function addToHistory(result) {
            document.getElementById('historyEmpty').style.display = 'none';
            
            const historyList = document.getElementById('historyList');
            const now = new Date();
            const timeStr = now.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <img class="history-thumb" src="data:image/jpeg;base64,${result.result_image}" alt="${result.predicted_class}">
                <div class="history-info">
                    <div class="history-class">${result.predicted_class}</div>
                    <div class="history-time">${timeStr}</div>
                </div>
                <span class="history-conf" style="color: ${result.confidence >= 80 ? 'var(--neon-green)' : (result.confidence >= 50 ? 'var(--neon-yellow)' : 'var(--neon-red)')}">${result.confidence}%</span>
            `;
            
            // Click to show this result image
            item.addEventListener('click', () => {
                resultImage.src = 'data:image/jpeg;base64,' + result.result_image;
                resultImage.style.display = 'block';
                video.style.display = 'none';
                showingResult = true;
                toggleViewBtn.style.display = 'flex';
                
                // Update result display
                document.getElementById('resultClass').textContent = result.predicted_class;
                const confEl = document.getElementById('resultConfidence');
                confEl.textContent = result.confidence.toFixed(1) + '%';
                confEl.className = 'result-confidence ' + 
                    (result.confidence >= 80 ? 'high' : (result.confidence >= 50 ? 'medium' : 'low'));
            });
            
            historyList.insertBefore(item, historyList.firstChild);
        }

        function toggleView() {
            if (showingResult) {
                // Switch back to live view
                video.style.display = 'block';
                resultImage.style.display = 'none';
                showingResult = false;
                toggleViewBtn.querySelector('span').textContent = 'Result';
            } else {
                // Switch to result
                video.style.display = 'none';
                resultImage.style.display = 'block';
                showingResult = true;
                toggleViewBtn.querySelector('span').textContent = 'Live View';
            }
        }

        // Keyboard shortcut - Space to capture
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !e.repeat) {
                e.preventDefault();
                captureFrame();
            }
        });

        // Start
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
        image_data = data.get('image', '')
        
        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to PIL image
        img_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Classify
        result = classify_frame(model, pil_img)
        
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
    
    # Load Model
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
    print("  Press SPACE to capture & analyze")
    print("=" * 60)
    print()
    
    app.run(debug=True, port=9999, use_reloader=False)
