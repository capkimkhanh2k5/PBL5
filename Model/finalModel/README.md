# Garbage Detection Model - Final Ultimate

## Model Information
- **Version**: Final Ultimate (IMPROVE)
- **Image Size**: 800px
- **mAP50**: 0.9826
- **mAP50-95**: 0.9824

## Performance
- Precision: 0.9465
- Recall: 0.9414

## Classes
battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash

## Usage
```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict('image.jpg')
```

## Files
- `best.pt`: PyTorch model
- `data.yaml`: Dataset configuration
- `metadata.json`: Model metadata
