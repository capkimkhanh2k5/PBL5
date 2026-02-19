from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='ncnn', imgsz=320)