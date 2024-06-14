from ultralytics import YOLO

# Load a model
model = YOLO('weights/yolov8x-obb.pt')  # load an official model

# Predict with the model
results = model('inference/images/kapal.jpg', show=True, save=True) # predict on an image