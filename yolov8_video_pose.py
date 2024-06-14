from ultralytics import YOLO

# Load a pretrained YOLOv8x-pose model
model = YOLO("weights/yolov8n-pose.pt")

# Predict pose on a video
result = model.predict(source="inference/videos/afriq0.MP4", show=True)