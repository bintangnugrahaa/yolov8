from ultralytics import YOLO

model = YOLO("weights/yolov8x-cls.pt")  # load a pretrained YOLOv8n classification model
model.train(data="dataset/animals", epochs=20)  # train the model
test_train = model("inference/images/giraffe.jpg")  # predict on an image

# Display tensor array
print(test_train)

# Display numpy array
print(test_train[0].numpy())
