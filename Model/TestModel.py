
import os
import base64
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from flask import Flask, render_template_string
from PIL import Image
import io

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
TEST_DIR = os.path.join(SCRIPT_DIR, "TestImage")
MODEL_PATH = os.path.join(SCRIPT_DIR, "trainModel/model.pth")

# Will be loaded from checkpoint
CLASS_NAMES = []
IMG_SIZE = 720

app = Flask(__name__)

class WasteDetectorModel(nn.Module):
    """MobileNetV3-based waste classifier with objectness detection"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # Load MobileNetV3-Large backbone
        backbone = models.mobilenet_v3_large(weights=None)
        
        # Extract features (convolutional layers)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Classifier head (same structure as saved model)
        self.classifier = nn.Sequential(
            nn.Linear(960, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
        # Objectness head - predicts if there's a waste object in the image
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
    """Load the trained MobileNetV3 model from .pth checkpoint"""
    global CLASS_NAMES, IMG_SIZE
    
    print(f"📦 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    CLASS_NAMES = checkpoint.get('classes', 
        ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
         'metal', 'paper', 'plastic', 'shoes', 'trash'])
    IMG_SIZE = checkpoint.get('img_size', 720)
    
    num_classes = len(CLASS_NAMES)
    model = WasteDetectorModel(num_classes=num_classes)
    
    # Load state dict
    state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    val_acc = checkpoint.get('val_acc', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    
    print(f"✅ Model loaded successfully!")
    print(f"   Architecture: MobileNetV3-Large + Objectness Head")
    print(f"   Classes ({num_classes}): {CLASS_NAMES}")
    print(f"   Image size: {IMG_SIZE}")
    print(f"   Val accuracy: {val_acc}")
    print(f"   Epoch: {epoch}")
    
    return model


# ============================================================
# IMAGE PREPROCESSING & INFERENCE
# ============================================================

def get_transform(img_size):
    """Get the image transform pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])


def generate_gradcam(model, input_tensor, class_idx):
    """Generate GradCAM heatmap to visualize where the model is looking"""
    model.eval()
    
    # Hook to capture feature maps and gradients
    feature_maps = []
    gradients = []
    
    def forward_hook(module, input, output):
        feature_maps.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks on the last conv layer
    last_conv = model.features[-1]
    fwd_handle = last_conv.register_forward_hook(forward_hook)
    bwd_handle = last_conv.register_full_backward_hook(backward_hook)
    
    # Forward pass
    input_tensor.requires_grad_(True)
    class_logits, _, _ = model(input_tensor)
    
    # Backward pass for target class
    model.zero_grad()
    one_hot = torch.zeros_like(class_logits)
    one_hot[0, class_idx] = 1
    class_logits.backward(gradient=one_hot, retain_graph=True)
    
    # Remove hooks
    fwd_handle.remove()
    bwd_handle.remove()
    
    if not gradients or not feature_maps:
        return None
    
    # Compute GradCAM
    grads = gradients[0]
    fmaps = feature_maps[0]
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * fmaps, dim=1, keepdim=True)
    cam = torch.relu(cam)
    
    # Normalize
    cam = cam.squeeze().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam


def get_bounding_box_from_cam(cam, original_size, threshold=0.3):
    """Extract bounding box from GradCAM heatmap"""
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (original_size[0], original_size[1]))
    
    # Threshold
    binary = (cam_resized > threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Add some padding (10%)
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(original_size[0] - x, w + 2 * pad_x)
    h = min(original_size[1] - y, h + 2 * pad_y)
    
    return (x, y, w, h)


def draw_detection_on_image(img_cv2, bbox, class_name, confidence, obj_score):
    """Draw bounding box and labels on image"""
    result = img_cv2.copy()
    
    if bbox is not None:
        x, y, w, h = bbox
        
        # Color based on confidence
        if confidence > 0.8:
            color = (0, 255, 100)  # Green
        elif confidence > 0.5:
            color = (0, 200, 255)  # Yellow
        else:
            color = (0, 100, 255)  # Orange
        
        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # Label background
        label = f"{class_name}: {confidence*100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(result, (x, y - text_h - 10), (x + text_w + 8, y), color, -1)
        cv2.putText(result, label, (x + 4, y - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Draw objectness score
        obj_label = f"Obj: {obj_score*100:.0f}%"
        cv2.putText(result, obj_label, (x + 4, y + h + 18), font, 0.45, color, 1)
    
    return result



def classify_images(model):
    """Classify all images in TestImage folder using MobileNetV3"""
    if not os.path.exists(TEST_DIR):
        print(f"⚠️ Test directory '{TEST_DIR}' not found!")
        print(f"   Creating '{TEST_DIR}' folder...")
        os.makedirs(TEST_DIR)
        print(f"   Please add test images to '{TEST_DIR}' folder and refresh.")
        return [], 0
    
    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"⚠️ No images found in '{TEST_DIR}' folder!")
        return [], 0
    
    print(f"📷 Found {len(images)} images to test")
    results = []
    transform = get_transform(IMG_SIZE)
    
    start_time = time.time()
    
    for idx, img_name in enumerate(images):
        img_path = os.path.join(TEST_DIR, img_name)
        
        try:
            # Load original image
            pil_img = Image.open(img_path).convert('RGB')
            original_size = pil_img.size  # (width, height)
            
            # Transform for model input
            input_tensor = transform(pil_img).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                class_logits, obj_score_raw, _ = model(input_tensor)
            
            # Get prediction
            probabilities = torch.softmax(class_logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence_val = confidence.item()
            
            # Objectness score (sigmoid for probability)
            obj_score = torch.sigmoid(obj_score_raw).item()
            
            # Generate GradCAM and bounding box
            input_for_cam = transform(pil_img).unsqueeze(0)
            cam = generate_gradcam(model, input_for_cam, predicted_idx.item())
            
            bbox = None
            if cam is not None and obj_score > 0.3:
                bbox = get_bounding_box_from_cam(cam, original_size)
            
            # Draw detection on image
            img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if bbox is not None:
                result_img = draw_detection_on_image(img_cv2, bbox, predicted_class, confidence_val, obj_score)
            else:
                # No bounding box available, add label on top
                result_img = img_cv2.copy()
                label = f"{predicted_class}: {confidence_val*100:.1f}%"
                cv2.putText(result_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            
            # Convert to base64
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            img_pil_result = Image.fromarray(result_img_rgb)
            img_pil_result.thumbnail((400, 400))
            buffer = io.BytesIO()
            img_pil_result.save(buffer, format='JPEG', quality=85)
            base64_img = base64.b64encode(buffer.getvalue()).decode()
            
            # Determine actual class from filename
            actual_class = get_actual_class(img_name)
            is_correct = predicted_class == actual_class if actual_class != "unknown" else False
            
            results.append({
                'image_b64': base64_img,
                'actual': actual_class,
                'predicted': predicted_class,
                'confidence': confidence_val,
                'obj_score': obj_score,
                'correct': is_correct,
                'has_bbox': bbox is not None
            })
            
            status_icon = "✓" if is_correct else ("?" if actual_class == "unknown" else "✗")
            print(f"  [{idx+1}/{len(images)}] {img_name}: {predicted_class} ({confidence_val*100:.1f}%) obj={obj_score:.2f} {status_icon}")
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: show original image
            try:
                fallback_img = Image.open(img_path).convert('RGB')
                fallback_img.thumbnail((400, 400))
                buffer = io.BytesIO()
                fallback_img.save(buffer, format='JPEG', quality=85)
                base64_img = base64.b64encode(buffer.getvalue()).decode()
            except:
                base64_img = ""
            
            results.append({
                'image_b64': base64_img,
                'actual': get_actual_class(img_name),
                'predicted': "Error",
                'confidence': None,
                'obj_score': None,
                'correct': False,
                'has_bbox': False
            })
            
    total_time = time.time() - start_time
    avg_time = total_time / len(images) if images else 0
    
    return results, avg_time


def get_actual_class(filename):
    """Extract actual class from filename"""
    filename_lower = filename.lower()
    for cls in CLASS_NAMES:
        if cls in filename_lower:
            return cls
    return "unknown"


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
                    fontFamily: {
                        sans: ['Space Grotesk', 'sans-serif'],
                    },
                    colors: {
                        cyber: {
                            50: '#f0fdfa',
                            100: '#ccfbf1',
                            200: '#99f6e4',
                            300: '#5eead4',
                            400: '#2dd4bf',
                            500: '#14b8a6',
                            600: '#0d9488',
                            700: '#0f766e',
                            800: '#115e59',
                            900: '#134e4a',
                            950: '#042f2e',
                        },
                        neon: {
                            blue: '#00f3ff',
                            purple: '#bc13fe',
                            green: '#0aff00',
                            red: '#ff003c',
                        }
                    },
                    animation: {
                        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        'scan': 'scan 2s linear infinite',
                    },
                    keyframes: {
                        scan: {
                            '0%': { transform: 'translateY(-100%)' },
                            '100%': { transform: 'translateY(100%)' },
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #050505;
            background-image: 
                radial-gradient(circle at 50% 0%, #1a1a2e 0%, transparent 60%),
                linear-gradient(rgba(0, 243, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 243, 255, 0.03) 1px, transparent 1px);
            background-size: 100% 100%, 40px 40px, 40px 40px;
        }
        .glass-panel {
            background: rgba(10, 10, 15, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        .text-glow {
            text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
        }
        .card-hover:hover {
            transform: translateY(-4px);
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.15);
            border-color: rgba(0, 243, 255, 0.4);
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0a0a0f; 
        }
        ::-webkit-scrollbar-thumb {
            background: #334155; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569; 
        }
    </style>
</head>
<body class="text-slate-300 min-h-screen p-6">

    <!-- Header -->
    <header class="max-w-7xl mx-auto mb-10 flex flex-col md:flex-row justify-between items-center gap-6 animate-fade-in-down">
        <div class="flex items-center gap-4">
            <div class="relative w-12 h-12 flex items-center justify-center rounded-xl bg-slate-900 border border-cyber-500/30 shadow-[0_0_15px_rgba(45,212,191,0.2)]">
                <i data-lucide="cpu" class="w-6 h-6 text-cyber-400"></i>
                <div class="absolute inset-0 rounded-xl bg-cyber-500/10 animate-pulse-slow"></div>
            </div>
            <div>
                <h1 class="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-cyan-200 to-cyan-500 tracking-tight">S.I.G.M.A.</h1>
                <p class="text-xs font-mono text-cyan-500/70 tracking-widest uppercase">MobileNetV3 + GradCAM Detection v2.0</p>
            </div>
        </div>
        
        <div class="flex gap-4">
            <div class="glass-panel px-6 py-3 rounded-lg flex flex-col items-center min-w-[140px]">
                <span class="text-xs text-slate-500 font-mono uppercase tracking-wider mb-1">Total Images</span>
                <span class="text-2xl font-bold text-white">{{ total }}</span>
            </div>
            <div class="glass-panel px-6 py-3 rounded-lg flex flex-col items-center min-w-[140px] relative overflow-hidden">
                <span class="text-xs text-slate-500 font-mono uppercase tracking-wider mb-1">Accuracy</span>
                <span class="text-2xl font-bold {{ 'text-neon-green' if accuracy >= 90 else ('text-yellow-400' if accuracy >= 70 else 'text-neon-red') }} text-glow">
                    {{ "%.1f"|format(accuracy) }}%
                </span>
                
                <!-- Progress bar background -->
                <div class="absolute bottom-0 left-0 h-1 bg-slate-800 w-full">
                    <div class="h-full {{ 'bg-neon-green' if accuracy >= 90 else ('bg-yellow-400' if accuracy >= 70 else 'bg-neon-red') }}" style="width: {{ accuracy }}%"></div>
                </div>
            </div>
             <div class="glass-panel px-6 py-3 rounded-lg flex flex-col items-center min-w-[140px]">
                <span class="text-xs text-slate-500 font-mono uppercase tracking-wider mb-1">Latency</span>
                <span class="text-2xl font-bold text-cyan-300">{{ "%.0f"|format(avg_time * 1000) }}<span class="text-sm font-normal text-slate-500 ml-1">ms</span></span>
            </div>
        </div>
    </header>

    <!-- Controls & Legend -->
    <div class="max-w-7xl mx-auto mb-8 flex flex-col md:flex-row justify-between items-center gap-4">
        
        <!-- Filter Tabs -->
        <div class="glass-panel p-1.5 rounded-lg flex gap-1" x-data="{ filter: 'all' }">
            <button onclick="filterAnalysis('all')" id="btn-all" class="px-4 py-2 rounded-md text-sm font-medium transition-all bg-white/10 text-white shadow-sm border border-white/5">
                All Results
            </button>
            <button onclick="filterAnalysis('correct')" id="btn-correct" class="px-4 py-2 rounded-md text-sm font-medium transition-all text-slate-400 hover:text-white hover:bg-white/5">
                <span class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-neon-green shadow-[0_0_8px_#0aff00]"></span> Correct
                </span>
            </button>
            <button onclick="filterAnalysis('wrong')" id="btn-wrong" class="px-4 py-2 rounded-md text-sm font-medium transition-all text-slate-400 hover:text-white hover:bg-white/5">
                <span class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-neon-red shadow-[0_0_8px_#ff003c]"></span> Errors
                </span>
            </button>
        </div>

        <!-- Legend -->
        <div class="flex items-center gap-6 text-xs text-slate-500 font-mono">
            <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-neon-green rounded-full shadow-[0_0_5px_#0aff00]"></div> Match
            </div>
            <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-neon-red rounded-full shadow-[0_0_5px_#ff003c]"></div> Mismatch
            </div>
            <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-yellow-500 rounded-full shadow-[0_0_5px_#eab308]"></div> Unknown Label
            </div>
            <div class="flex items-center gap-2">
                <div class="w-3 h-3 border border-cyan-400 rounded-sm"></div> BBox (GradCAM)
            </div>
        </div>
    </div>

    <!-- Grid -->
    <div class="max-w-7xl mx-auto grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-5 pb-20">
        {% for r in results %}
        <div class="card-hover glass-panel rounded-xl overflow-hidden transition-all duration-300 group relative result-card" data-status="{{ 'correct' if r.correct else ('unknown' if r.actual == 'unknown' else 'wrong') }}">
            
            <!-- Image Container -->
            <div class="relative aspect-square overflow-hidden bg-slate-900">
                <img src="data:image/jpeg;base64,{{ r.image_b64 }}" alt="{{ r.actual }}" 
                     class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110">
                
                <!-- Overlay Gradient -->
                <div class="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-transparent to-transparent opacity-60"></div>
                
                <!-- Status Badge -->
                <div class="absolute top-3 right-3">
                    {% if r.correct %}
                        <div class="bg-neon-green/20 backdrop-blur-md border border-neon-green/50 text-neon-green p-1.5 rounded-lg shadow-[0_0_10px_rgba(10,255,0,0.2)]">
                            <i data-lucide="check" class="w-3.5 h-3.5"></i>
                        </div>
                    {% elif r.actual == 'unknown' %}
                        <div class="bg-yellow-500/20 backdrop-blur-md border border-yellow-500/50 text-yellow-500 p-1.5 rounded-lg">
                            <i data-lucide="help-circle" class="w-3.5 h-3.5"></i>
                        </div>
                    {% else %}
                        <div class="bg-neon-red/20 backdrop-blur-md border border-neon-red/50 text-neon-red p-1.5 rounded-lg shadow-[0_0_10px_rgba(255,0,60,0.2)]">
                            <i data-lucide="x" class="w-3.5 h-3.5"></i>
                        </div>
                    {% endif %}
                </div>

                <!-- Confidence Badge -->
                {% if r.confidence %}
                <div class="absolute top-3 left-3">
                    <div class="px-2 py-1 rounded bg-black/60 backdrop-blur-sm text-[10px] font-mono text-cyan-300 border border-cyan-500/30">
                        {{ "%.1f"|format(r.confidence * 100) }}%
                    </div>
                </div>
                {% endif %}

                <!-- BBox indicator -->
                {% if r.has_bbox %}
                <div class="absolute bottom-3 left-3">
                    <div class="px-2 py-1 rounded bg-cyan-500/20 backdrop-blur-sm text-[9px] font-mono text-cyan-300 border border-cyan-400/40 flex items-center gap-1">
                        <i data-lucide="scan" class="w-2.5 h-2.5"></i> DETECTED
                    </div>
                </div>
                {% endif %}
            </div>

            <!-- Content -->
            <div class="p-4 relative">
                <!-- Scan Line Effect (Hover) -->
                <div class="absolute top-0 left-0 w-full h-[1px] bg-cyan-400/50 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500"></div>

                <div class="mb-3">
                    <p class="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-0.5">Prediction</p>
                    <div class="text-base font-bold text-white truncate flex items-center gap-2">
                        {{ r.predicted if r.predicted else 'No Detection' }}
                    </div>
                </div>

                <div class="flex items-center justify-between pt-3 border-t border-white/5">
                    <div class="flex flex-col">
                        <span class="text-[9px] text-slate-500 uppercase tracking-wider">Actual</span>
                        <span class="text-xs font-medium {{ 'text-slate-300' if r.correct else ('text-yellow-400' if r.actual == 'unknown' else 'text-red-400') }} truncate max-w-[100px]" title="{{ r.actual }}">
                            {{ r.actual }}
                        </span>
                    </div>
                    <div class="text-[9px] font-mono text-slate-600">
                        #{{ loop.index }}
                    </div>
                </div>
            </div>
            
            <!-- Glow Effect on wrong predictions -->
            {% if not r.correct and r.actual != 'unknown' %}
            <div class="absolute inset-0 border border-neon-red/30 pointer-events-none rounded-xl"></div>
            {% endif %}
        </div>
        {% endfor %}
    </div>

    <!-- Empty State -->
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
        <p>POWERED BY MobileNetV3 + GradCAM &bull; WASTE DETECTOR MODEL</p>
    </footer>

    <script>
        // Init Icons
        lucide.createIcons();

        // Filter Logic
        function filterAnalysis(status) {
            const cards = document.querySelectorAll('.result-card');
            const buttons = ['all', 'correct', 'wrong'];
            
            // Update Buttons
            buttons.forEach(btn => {
                const el = document.getElementById('btn-' + btn);
                if (btn === status) {
                    el.classList.remove('text-slate-400', 'hover:text-white', 'hover:bg-white/5', 'bg-transparent');
                    el.classList.add('bg-white/10', 'text-white', 'shadow-sm', 'border', 'border-white/5');
                } else {
                    el.classList.add('text-slate-400', 'hover:text-white', 'hover:bg-white/5', 'bg-transparent');
                    el.classList.remove('bg-white/10', 'text-white', 'shadow-sm', 'border', 'border-white/5');
                }
            });

            // Filter Grid
            cards.forEach(card => {
                const cardStatus = card.getAttribute('data-status');
                if (status === 'all') {
                    card.style.display = 'block';
                } else if (status === 'correct') {
                    card.style.display = cardStatus === 'correct' ? 'block' : 'none';
                } else if (status === 'wrong') {
                    card.style.display = (cardStatus === 'wrong' || cardStatus === 'unknown') ? 'block' : 'none';
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    results, avg_time = classify_images(model)
    
    # Calculate accuracy (excluding unknown labels)
    known_results = [r for r in results if r['actual'] != 'unknown']
    correct = sum(1 for r in known_results if r['correct'])
    total_known = len(known_results)
    accuracy = (correct / total_known * 100) if total_known > 0 else 0
    
    return render_template_string(
        HTML_TEMPLATE,
        results=results,
        correct=correct,
        total=len(results),
        total_known=total_known,
        accuracy=accuracy,
        avg_time=avg_time
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 S.I.G.M.A. EVALUATION CONSOLE STARTING...")
    print("   Model: MobileNetV3 + GradCAM BBox Detection")
    print("=" * 60)
    
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            print(f"❌ Core Initialization Failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    else:
        print(f"❌ Critical Error: Model file missing at {MODEL_PATH}")
        print("   Please ensure 'model.pth' is in the trainModel directory.")
        exit(1)
    
    # Check test directory
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        print(f"\n📁 Initialized Data Buffer: {TEST_DIR}")
        print(f"   -> Waiting for input data...")
    else:
        images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        print(f"\n📁 Data Buffer Ready: {TEST_DIR}")
        print(f"   -> {len(images)} samples detected")
    
    # Start Server
    print("\n" + "=" * 60)
    print("🌐 INTERFACE ONLINE")
    print("👉 Access Console: http://localhost:8888")
    print("=" * 60 + "\n")
    app.run(debug=True, port=8888, use_reloader=False)
