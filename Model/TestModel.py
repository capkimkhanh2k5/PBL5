import os
import base64
from flask import Flask, render_template_string
from ultralytics import YOLO
from PIL import Image
import io

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
TEST_DIR = os.path.join(SCRIPT_DIR, "TestImage")
MODEL_PATH = os.path.join(SCRIPT_DIR, "Result_Train/yolo26n_garbage_best.pt")

# Class names từ data.yaml
CLASS_NAMES = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 
               'metal', 'paper', 'plastic', 'shoes', 'trash']

app = Flask(__name__)

# HTML Template with modern UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Model</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            padding: 30px;
            color: #fff;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
        }
        
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 12px;
            margin-bottom: 40px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 10px;
            text-align: center;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .card.correct {
            border-color: #00ff88;
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
        }
        
        .card.wrong {
            border-color: #ff4757;
            box-shadow: 0 0 15px rgba(255, 71, 87, 0.2);
        }
        
        .card.unknown {
            border-color: #ffa502;
            box-shadow: 0 0 15px rgba(255, 165, 2, 0.2);
        }
        
        .card img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        
        .card .index {
            position: absolute;
            top: 5px;
            left: 5px;
            background: rgba(0,0,0,0.6);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
        }
        
        .card .prediction {
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 4px;
            color: #00d9ff;
        }
        
        .card .confidence {
            font-size: 10px;
            color: #00ff88;
            margin-bottom: 4px;
        }
        
        .card .actual {
            font-size: 10px;
            color: #888;
        }
        
        .card .status {
            font-size: 18px;
            margin-top: 5px;
        }
        
        .accuracy-box {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            max-width: 500px;
            margin: 0 auto;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .accuracy-box h2 {
            font-size: 1.2rem;
            color: #aaa;
            margin-bottom: 10px;
            letter-spacing: 3px;
        }
        
        .accuracy-score {
            font-size: 5rem;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .accuracy-score.high { color: #00ff88; }
        .accuracy-score.medium { color: #ffa502; }
        .accuracy-score.low { color: #ff4757; }
        
        .accuracy-detail {
            color: #888;
            font-size: 1.1rem;
        }
        
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #aaa;
        }
        
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .legend-dot.correct { background: #00ff88; }
        .legend-dot.wrong { background: #ff4757; }
        .legend-dot.unknown { background: #ffa502; }
        
        @media (max-width: 1200px) {
            .grid { grid-template-columns: repeat(5, 1fr); }
        }
        
        @media (max-width: 768px) {
            .grid { grid-template-columns: repeat(4, 1fr); }
            .accuracy-score { font-size: 3rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🗑️ YOLO Garbage Classification Results</h1>
        <p class="subtitle">YOLOv8 Model (yolo26n_garbage_best.pt) Testing on {{ total }} Images</p>
        
        <div class="grid">
            {% for r in results %}
            <div class="card {{ 'correct' if r.correct else ('unknown' if r.actual == 'unknown' else 'wrong') }}">
                <img src="data:image/jpeg;base64,{{ r.image_b64 }}" alt="{{ r.actual }}">
                <div class="prediction">{{ r.predicted if r.predicted else 'No Detection' }}</div>
                {% if r.confidence %}
                <div class="confidence">Conf: {{ "%.1f"|format(r.confidence * 100) }}%</div>
                {% endif %}
                <div class="actual">Actual: {{ r.actual }}</div>
                <div class="status">{{ '✓' if r.correct else ('?' if r.actual == 'unknown' else '✗') }}</div>
            </div>
            {% endfor %}
        </div>
        
        <div class="accuracy-box">
            <h2>MODEL ACCURACY</h2>
            <div class="accuracy-score {{ 'high' if accuracy >= 90 else ('medium' if accuracy >= 70 else 'low') }}">
                {{ "%.1f"|format(accuracy) }}%
            </div>
            <p class="accuracy-detail">Correct: {{ correct }}/{{ total_known }} images (excluding unknown)</p>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot correct"></div>
                    <span>Correct</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot wrong"></div>
                    <span>Wrong</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot unknown"></div>
                    <span>Unknown Label</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

def image_to_base64(img_path):
    """Convert image to base64 for embedding in HTML"""
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        img = img.resize((200, 200))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

def get_actual_class(filename):
    """Extract actual class from filename"""
    filename_lower = filename.lower()
    for cls in CLASS_NAMES:
        if cls in filename_lower:
            return cls
    return "unknown"

def classify_images(model):
    """Classify all images in TestImage folder using YOLO"""
    if not os.path.exists(TEST_DIR):
        print(f"⚠️ Test directory '{TEST_DIR}' not found!")
        print(f"   Creating '{TEST_DIR}' folder...")
        os.makedirs(TEST_DIR)
        print(f"   Please add test images to '{TEST_DIR}' folder and refresh.")
        return []
    
    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"⚠️ No images found in '{TEST_DIR}' folder!")
        return []
    
    print(f"📷 Found {len(images)} images to test")
    results = []
    
    for idx, img_name in enumerate(images):
        img_path = os.path.join(TEST_DIR, img_name)
        
        try:
            # YOLO prediction
            prediction = model.predict(img_path, verbose=False)
            
            # Get the best prediction
            predicted_class = None
            confidence = None
            
            if len(prediction) > 0 and prediction[0].boxes is not None and len(prediction[0].boxes) > 0:
                # Get the box with highest confidence
                boxes = prediction[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                if len(confidences) > 0:
                    best_idx = confidences.argmax()
                    predicted_class = CLASS_NAMES[int(classes[best_idx])]
                    confidence = float(confidences[best_idx])
            
            # For classification models (no boxes, just probs)
            if predicted_class is None and hasattr(prediction[0], 'probs') and prediction[0].probs is not None:
                probs = prediction[0].probs
                top_class_idx = probs.top1
                predicted_class = CLASS_NAMES[top_class_idx]
                confidence = float(probs.top1conf)
            
            actual_class = get_actual_class(img_name)
            is_correct = predicted_class == actual_class if actual_class != "unknown" else False
            
            results.append({
                'image_b64': image_to_base64(img_path),
                'actual': actual_class,
                'predicted': predicted_class if predicted_class else "No Detection",
                'confidence': confidence,
                'correct': is_correct
            })
            
            status = "✓" if is_correct else ("?" if actual_class == "unknown" else "✗")
            if confidence:
                print(f"  [{idx+1}/{len(images)}] {img_name}: {predicted_class} ({confidence*100:.1f}% conf) {status}")
            else:
                print(f"  [{idx+1}/{len(images)}] {img_name}: No detection")
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            results.append({
                'image_b64': image_to_base64(img_path),
                'actual': get_actual_class(img_name),
                'predicted': "Error",
                'confidence': None,
                'correct': False
            })
    
    return results

@app.route('/')
def index():
    results = classify_images(model)
    
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
        accuracy=accuracy
    )

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 YOLO GARBAGE CLASSIFICATION TEST SERVER")
    print("=" * 50)
    
    # Load YOLO Model
    if os.path.exists(MODEL_PATH):
        print(f"\n📦 Loading model: {MODEL_PATH}")
        try:
            model = YOLO(MODEL_PATH)
            print("✅ Model loaded successfully!")
            print(f"   Model type: {model.task}")
            print(f"   Classes: {CLASS_NAMES}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(1)
    else:
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print("   Please make sure 'yolo26n.pt' exists in the Model folder.")
        exit(1)
    
    # Check test directory
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        print(f"\n📁 Created '{TEST_DIR}' folder")
        print(f"   Please add test images to this folder!")
    else:
        images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n📁 Test folder: {TEST_DIR} ({len(images)} images)")
    
    # Start Server
    print("\n" + "=" * 50)
    print("🌐 SERVER STARTED")
    print("👉 Open browser: http://localhost:8888")
    print("=" * 50 + "\n")
    app.run(debug=False, port=8888)
