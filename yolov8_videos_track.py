from ultralytics import YOLO

# load a pretrained YOLOv8n model 
model = YOLO('weights/yolov8m.pt')

# run inference on the source
results = model.track(source='inference/videos/mobil_di_tol.mp4', show=True, tracker='bytetrack.yaml')