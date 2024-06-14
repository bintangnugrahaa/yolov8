from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-obb.pt')  # load an official model

# Predict with the model
results = model('inference/videos/mobil_di_tol.mp4', show=True, save=True) # predict on an image