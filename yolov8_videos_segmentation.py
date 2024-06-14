from ultralytics import YOLO

# load a pretrained YOLOv8x-seg model
model = YOLO("weights/yolov8n-seg.pt")

# predict on a video
result = model.predict(source="inference/videos/afriq0.MP4", show=True)