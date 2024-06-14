from ultralytics import YOLO

# Load a pretrained YOLOv8x-pose model
model = YOLO("weights/yolov8x-pose.pt")

# Predict pose on a video
result = model.predict(source="inference\images\person.jpg", show=True, save=True)