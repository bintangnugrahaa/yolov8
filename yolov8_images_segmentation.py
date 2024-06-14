from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8x-seg model
model = YOLO("weights/yolov8x-seg.pt")

# predict on a image
result = model.predict(source="dataset_giraffe\giraffe animal jungle\Image_10.jpg", save=True)
