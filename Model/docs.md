# 🗑️ Garbage Detection - YOLO Nano Optimal Training

## 📋 Mô tả

Script training tối ưu cho việc nhận diện rác thải sử dụng YOLO11 Nano với **chiến lược Progressive Training 3 stages**.

## ✨ Tính năng chính

### 🎯 Chiến lược Progressive Training

**Stage 1: Warm-up (50 epochs)**
- Image size: 416x416
- Learning rate thấp (0.001)
- Augmentation conservative
- Focus: Học features cơ bản

**Stage 2: Main Training (150 epochs)**
- Image size: 512x512
- Augmentation aggressive (mosaic, mixup, copy-paste)
- Focus: Generalization và robustness

**Stage 3: Fine-tuning (100 epochs)**
- Image size: 640x640
- Learning rate rất thấp (0.0001)
- Augmentation balanced
- Focus: Perfect fine-tuning

### 🚀 Tối ưu hóa

- ✅ **Mixed Precision Training (AMP)**: Tăng tốc 2-3x
- ✅ **Cache images**: Tăng tốc data loading
- ✅ **AdamW Optimizer**: Convergence tốt hơn
- ✅ **Advanced Augmentation**: Mosaic, Mixup, Copy-Paste
- ✅ **Early Stopping**: Tự động dừng khi không cải thiện
- ✅ **Progressive Image Size**: Tăng dần resolution

### 📊 10 Classes Garbage Detection

1. **plastic_bottle** - Chai nhựa
2. **plastic_bag** - Túi nilon
3. **can** - Lon, hộp kim loại
4. **paper** - Giấy
5. **cardboard** - Bìa carton
6. **glass** - Thủy tinh
7. **organic_waste** - Rác hữu cơ
8. **styrofoam** - Xốp
9. **electronic_waste** - Rác điện tử
10. **other_waste** - Rác khác

## 📁 Cấu trúc thư mục Dataset

```
data/
├── images/
│   ├── train/          # Ảnh training
│   ├── val/            # Ảnh validation
│   └── test/           # Ảnh test
└── labels/
    ├── train/          # Labels training (YOLO format)
    ├── val/            # Labels validation
    └── test/           # Labels test
```

## 🔧 Cài đặt

### Requirements

```bash
pip install -r requirements.txt
```

### Kiểm tra GPU (khuyến nghị)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 🚀 Cách sử dụng

### 1. Training cơ bản

```bash
python train_garbage_detection.py
```

### 2. Tùy chỉnh data path

```python
from train_garbage_detection import GarbageDetectionTrainer

trainer = GarbageDetectionTrainer(
    data_path='./path/to/your/data',
    project_name='my_garbage_model'
)

final_model, results = trainer.train_progressive()
```

### 3. Chỉ train một stage

```python
model = YOLO('yolo11n.pt')
trainer = GarbageDetectionTrainer()

# Chỉ stage 1
config = trainer.get_training_config_stage1()
results = model.train(
    data='data.yaml',
    **config
)
```

## 📈 Kết quả Training

Script tự động lưu:
- ✅ Best weights mỗi stage
- ✅ Training logs
- ✅ Validation metrics
- ✅ Confusion matrix
- ✅ Training curves
- ✅ Predictions visualization

Thư mục output:
```
runs/garbage_detection_optimal_YYYYMMDD_HHMMSS/
├── stage1/
│   └── warmup/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       └── results.png
├── stage2/
│   └── main/
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── stage3/
│   └── finetune/
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── final_best.pt           # Model cuối cùng
└── data.yaml
```

## 🎯 Performance Tips

### Tăng tốc Training

1. **Sử dụng GPU**: Training nhanh hơn 10-50x
2. **Tăng batch size**: Nếu có đủ VRAM
3. **Cache images**: Đã được enable mặc định
4. **Giảm workers**: Nếu CPU yếu, giảm từ 8 xuống 4

### Cải thiện Accuracy

1. **Tăng epochs**: Stage 2 từ 150 lên 200-300
2. **Data augmentation**: Điều chỉnh trong config
3. **Ensemble**: Kết hợp nhiều models
4. **Hard negative mining**: Focus vào ảnh khó

### Giảm Overfitting

1. **Tăng augmentation**: Đã được optimize
2. **Weight decay**: Tăng từ 0.0005 lên 0.001
3. **Dropout**: Có thể thêm vào model
4. **More data**: Thu thập thêm dữ liệu

## 🔄 Export Models

Script tự động export sang nhiều format:

- **ONNX**: Universal format, tốc độ tốt
- **TensorRT**: Nvidia GPU, siêu nhanh
- **TFLite**: Mobile deployment

```python
trainer.export_model('final_best.pt')
```

## 📊 Validation & Testing

```python
# Validate model
results = trainer.validate_model('final_best.pt')

# Inference
model = YOLO('final_best.pt')
results = model.predict(
    source='test_images/',
    save=True,
    conf=0.25
)
```

## ⚙️ Hyperparameter Tuning

Để tìm hyperparameters tốt nhất:

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.tune(
    data='data.yaml',
    epochs=30,
    iterations=300,
    optimizer='AdamW',
    plots=True,
    save=True,
    val=True
)
```

## 🐛 Troubleshooting

### Out of Memory

```python
# Giảm batch size
config['batch'] = 8

# Giảm image size
config['imgsz'] = 320

# Tắt cache
config['cache'] = False
```

### Training quá chậm

```python
# Giảm workers
config['workers'] = 4

# Tắt augmentation phức tạp
config['mosaic'] = 0
config['mixup'] = 0
```

### Model không converge

```python
# Giảm learning rate
config['lr0'] = 0.0001

# Tăng warmup epochs
config['warmup_epochs'] = 10
```

## 📚 Tài liệu tham khảo

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLO11 Paper](https://arxiv.org/abs/2304.00501)
- [Data Augmentation Guide](https://docs.ultralytics.com/modes/train/#augmentation)

## 📝 License

MIT License

## 🤝 Contributing

Mọi đóng góp đều được chào đón! Hãy tạo Pull Request hoặc Issue.

## ⭐ Acknowledgments

- Ultralytics team cho YOLO implementation tuyệt vời
- Garbage detection dataset contributors

---

**Happy Training! 🚀**


# 🚀 HƯỚNG DẪN SỬ DỤNG NHANH

## 📋 Tổng quan

Bộ công cụ training AI nhận diện rác với YOLO11 Nano - được tối ưu hóa để đạt hiệu quả cao nhất.

## 🔧 Cài đặt

### Bước 1: Clone/Download project
```bash
# Nếu từ git
git clone <repository-url>
cd garbage-detection

# Hoặc giải nén file zip
unzip garbage-detection.zip
cd garbage-detection
```

### Bước 2: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 3: Kiểm tra GPU (khuyến nghị)
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📁 Chuẩn bị Dataset

### Cấu trúc thư mục

Đặt dữ liệu của bạn theo cấu trúc sau:

```
data/
├── images/
│   ├── train/          # Ảnh training (70%)
│   ├── val/            # Ảnh validation (20%)
│   └── test/           # Ảnh test (10%)
└── labels/
    ├── train/          # Labels YOLO format
    ├── val/
    └── test/
```

### Format Label (YOLO format)

Mỗi ảnh có 1 file .txt tương ứng, mỗi dòng định dạng:
```
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

### Tự động chia dataset

Nếu bạn có tất cả ảnh trong 1 thư mục, dùng script này:

```bash
python prepare_dataset.py
```

Script sẽ:
- ✅ Tự động chia train/val/test (70/20/10)
- ✅ Kiểm tra chất lượng data
- ✅ Tạo biểu đồ phân tích
- ✅ Báo cáo các vấn đề (nếu có)

## 🎯 Training

### Training cơ bản (khuyến nghị)

```bash
python train_garbage_detection.py
```

Script này sẽ tự động chạy **Progressive Training 3 stages**:
- Stage 1: Warm-up (50 epochs)
- Stage 2: Main training (150 epochs)  
- Stage 3: Fine-tuning (100 epochs)

**Tổng thời gian:** ~8-12 giờ trên GPU V100

### Tùy chỉnh training

```python
from train_garbage_detection import GarbageDetectionTrainer

trainer = GarbageDetectionTrainer(
    data_path='./data',
    project_name='my_model'
)

# Chạy progressive training
final_model, results = trainer.train_progressive()
```

### Chỉ train 1 stage

```python
from ultralytics import YOLO
from train_garbage_detection import GarbageDetectionTrainer

model = YOLO('yolo11n.pt')
trainer = GarbageDetectionTrainer()

# Lấy config stage 2
config = trainer.get_training_config_stage2()

# Train
results = model.train(
    data='data.yaml',
    epochs=150,
    **config
)
```

## 📊 Theo dõi Training

### TensorBoard

```bash
tensorboard --logdir runs/
```

Mở trình duyệt: http://localhost:6006

### Xem logs

```bash
tail -f training_garbage.log
```

## 🎬 Inference (Sử dụng Model)

### Trên ảnh

```bash
python inference.py \
    --model runs/garbage_detection_optimal_*/final_best.pt \
    --source test_image.jpg \
    --output result.jpg \
    --show
```

### Trên video

```bash
python inference.py \
    --model runs/garbage_detection_optimal_*/final_best.pt \
    --source test_video.mp4 \
    --output result.mp4 \
    --conf 0.3
```

### Webcam real-time

```bash
python inference.py \
    --model runs/garbage_detection_optimal_*/final_best.pt \
    --source webcam \
    --show
```

### Parameters

- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IOU threshold cho NMS (default: 0.45)
- `--show`: Hiển thị kết quả
- `--output`: Lưu kết quả

## 📈 Kết quả Expected

### Hiệu suất Training

| Stage | Epochs | Image Size | Time | mAP50 |
|-------|--------|------------|------|-------|
| 1     | 50     | 416        | 2h   | ~0.60 |
| 2     | 150    | 512        | 6h   | ~0.75 |
| 3     | 100    | 640        | 4h   | ~0.80 |

**Total: 300 epochs, ~12h, mAP50 ~0.80**

### Inference Speed

- GPU (V100): ~100 FPS
- GPU (RTX 3080): ~150 FPS  
- CPU: ~5 FPS

## 🐛 Troubleshooting

### Out of Memory

```python
# Giảm batch size
config['batch'] = 8

# Giảm image size
config['imgsz'] = 320

# Tắt cache
config['cache'] = False
```

### Model không converge

```python
# Giảm learning rate
config['lr0'] = 0.0001

# Tăng warmup
config['warmup_epochs'] = 10
```

### Training quá chậm

- Kiểm tra GPU: `nvidia-smi`
- Giảm workers: `config['workers'] = 4`
- Giảm augmentation

## 📝 Tips & Tricks

### Tăng Accuracy

1. **Thu thập thêm data**: Càng nhiều càng tốt
2. **Data augmentation**: Đã được optimize
3. **Ensemble**: Kết hợp nhiều models
4. **Tăng epochs**: Stage 2 lên 200-300 epochs

### Giảm Overfitting

1. **Tăng augmentation**: Đã có sẵn
2. **Weight decay**: Tăng lên 0.001
3. **Dropout**: Modify model architecture
4. **Early stopping**: Đã enable

### Tăng tốc Training

1. **Mixed precision**: Đã enable
2. **Cache images**: Đã enable
3. **Tăng batch size**: Nếu có đủ VRAM
4. **Multi-GPU**: Dùng DDP

## 🎓 Best Practices

### Training

- ✅ Luôn dùng GPU
- ✅ Cache images nếu có đủ RAM
- ✅ Monitor training curves
- ✅ Save checkpoints thường xuyên
- ✅ Validate trên test set

### Data

- ✅ Cân bằng classes (tránh imbalance)
- ✅ Diverse data (nhiều điều kiện khác nhau)
- ✅ High quality annotations
- ✅ Remove duplicates
- ✅ Clean corrupted files

### Deployment

- ✅ Export sang ONNX/TensorRT
- ✅ Optimize inference
- ✅ Test trên real data
- ✅ Monitor performance
- ✅ Update model regularly

## 📚 Tài liệu tham khảo

- [YOLO11 Docs](https://docs.ultralytics.com/)
- [Training Tips](https://docs.ultralytics.com/modes/train/)
- [Model Export](https://docs.ultralytics.com/modes/export/)

## 💡 Ví dụ sử dụng

### Ví dụ 1: Train từ đầu

```bash
# Chuẩn bị data
python prepare_dataset.py

# Train
python train_garbage_detection.py

# Inference
python inference.py --model runs/*/final_best.pt --source test.jpg --show
```

### Ví dụ 2: Fine-tune model có sẵn

```python
from ultralytics import YOLO

model = YOLO('path/to/pretrained.pt')
results = model.train(
    data='data.yaml',
    epochs=50,
    lr0=0.0001,  # Learning rate thấp
    freeze=10    # Freeze 10 layers đầu
)
```

### Ví dụ 3: Hyperparameter tuning

```python
model = YOLO('yolo11n.pt')
results = model.tune(
    data='data.yaml',
    epochs=30,
    iterations=300
)
```

---

## 🆘 Support

Nếu gặp vấn đề:
1. Đọc kỹ error message
2. Check GPU/RAM
3. Xem logs: `training_garbage.log`
4. Tham khảo documentation

**Good luck with your training! 🚀**