import os
import base64
import numpy as np
from flask import Flask, render_template_string
from keras.models import load_model
from keras.utils import load_img, img_to_array
from PIL import Image
import io

# Configuration
TEST_DIR = "TestImage"
IMAGE_SIZE = (224, 224)

# Class names
CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

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
        <p class="subtitle">ResNet50V2 Model Testing on {{ total }} Random Images</p>
        
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

def image_to_base64(img_path):
    """Convert image to base64 for embedding in HTML"""
    with Image.open(img_path) as img:
        img = img.resize((150, 150))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

def classify_images(model):
    """Classify all images in TestImage folder"""
    images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:40]
    results = []
    
    for img_name in images:
        img_path = os.path.join(TEST_DIR, img_name)
        
        # Load and predict
        img = load_img(img_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        
        # Get actual class from filename
        actual_class = img_name.split('_')[0]
        
        results.append({
            'image_b64': image_to_base64(img_path),
            'actual': actual_class,
            'predicted': predicted_class,
            'correct': predicted_class == actual_class
        })
    
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
    print("Loading model...")
    model = load_model('resnet50v2_garbage_classifier.keras')
    print("Model loaded!")
    print("\n" + "=" * 50)
    print("Starting web server...")
    print("Open browser: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)
