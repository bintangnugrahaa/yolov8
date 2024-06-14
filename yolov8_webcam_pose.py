from ultralytics import YOLO

model = YOLO("weights\yolov8n-pose.pt")

result = model(source=0, conf=0.3, show=True)
