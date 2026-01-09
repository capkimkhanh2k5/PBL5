import os
import json
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.utils import load_img, img_to_array
# New import for loading full model
from tensorflow.keras.models import load_model 
from PIL import Image
import io

# Configuration
TEST_DIR = "TestImage"
IMG_SIZE = (288, 288)
BATCH_SIZE = 64
NUM_CLASSES = 12
L2_WEIGHT_DECAY = 1e-5
# MODEL_PATH = "mobilenetv3.weights.h5" 
MODEL_PATH = "mobilenetv3_full.keras"
CLASS_NAMES_PATH = "class_names.json"

# Load class names from JSON (fallback to default if not found)
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            return json.load(f)
    # Fallback default
    return ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
            'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

CLASS_NAMES = load_class_names()

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
            font-size: 11px;
            font-weight: 600;
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
        <h1>🗑️ Garbage Classification Results</h1>
        <p class="subtitle">MobileNetV3-Large Model Testing on {{ total }} Random Images</p>
        
        <div class="grid">
            {% for r in results %}
            <div class="card {{ 'correct' if r.correct else 'wrong' }}">
                <img src="data:image/jpeg;base64,{{ r.image_b64 }}" alt="{{ r.actual }}">
                <div class="prediction">{{ r.predicted }}</div>
                <div class="actual">Actual: {{ r.actual }}</div>
                <div class="status">{{ '✓' if r.correct else '✗' }}</div>
            </div>
            {% endfor %}
        </div>
        
        <div class="accuracy-box">
            <h2>MODEL ACCURACY</h2>
            <div class="accuracy-score {{ 'high' if accuracy >= 90 else ('medium' if accuracy >= 70 else 'low') }}">
                {{ "%.1f"|format(accuracy) }}%
            </div>
            <p class="accuracy-detail">Correct: {{ correct }}/{{ total }} images</p>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot correct"></div>
                    <span>Correct Prediction</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot wrong"></div>
                    <span>Wrong Prediction</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# Build + load model by weigh
# def build_model():
#     """Rebuild the model architecture to load weights."""
#     print("Building model architecture...")
#     base_model = MobileNetV3Large(
#         include_top=False,
#         weights=None,  # No need to load Imagenet weights, we load our own
#         input_shape=IMG_SIZE + (3,),
#         include_preprocessing=True
#     )
#     
#     # Reconstruct the exact architecture
#     inputs = layers.Input(shape=IMG_SIZE + (3,))
#     x = base_model(inputs, training=False)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(0.2)(x)
#     outputs = layers.Dense(
#         NUM_CLASSES, 
#         activation="softmax",
#         dtype="float32",
#         kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
#     )(x)
#     
#     model = Model(inputs, outputs)
#     return model

def image_to_base64(img_path):
    """Convert image to base64 for embedding in HTML"""
    with Image.open(img_path) as img:
        img = img.resize((150, 150))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

def classify_images(model):
    """Classify all images in TestImage folder"""
    if not os.path.exists(TEST_DIR):
        return []
        
    images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:40]
    results = []
    
    for img_name in images:
        img_path = os.path.join(TEST_DIR, img_name)
        
        try:
            # Load and predict
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            
            actual_class = "unknown"
            for cls in CLASS_NAMES:
                if img_name.lower().startswith(cls):
                    actual_class = cls
                    break
            
            results.append({
                'image_b64': image_to_base64(img_path),
                'actual': actual_class,
                'predicted': predicted_class,
                'correct': predicted_class == actual_class
            })
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    return results

@app.route('/')
def index():
    results = classify_images(model)
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    return render_template_string(
        HTML_TEMPLATE,
        results=results,
        correct=correct,
        total=total,
        accuracy=accuracy
    )

if __name__ == "__main__":
    print("=" * 50)
    print("INITIALIZING TEST SERVER")
    print("=" * 50)
    
    # 1. Load Model (Keras format)
    if os.path.exists(MODEL_PATH):
        print(f"Loading full model from: {MODEL_PATH}")
        try:
            model = load_model(MODEL_PATH)
            print("✅ Model loaded successfully (Architecture + Weights)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(1)
    else:
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print("Please run ExportModel.py first to generate the .keras file.")
        exit(1)

    # 2. Build + Load Model
    # model = build_model()
    # if os.path.exists("mobilenetv3.weights.h5"):
    #    model.load_weights("mobilenetv3.weights.h5")
        
    # 3. Start Server
    print("\n" + "=" * 50)
    print("🚀 SERVER STARTED")
    print("👉 Open browser: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)
