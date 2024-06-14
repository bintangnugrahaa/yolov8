from ultralytics import YOLO
# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8x.pt")  

# predict on an image
result = model.predict(source="inference/videos/afriq1.MP4", conf=0.25, save=True, show=True)