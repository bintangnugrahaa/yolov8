from ultralytics import YOLO

modul = YOLO('weights/yolov8n-seg.pt')

result = modul(source=0, conf=0.3, show=True)