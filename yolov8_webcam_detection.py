from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("weights/yolov8n.pt")

# Load object classes
classNames = open("utils/coco.txt", "r").read().strip().split("\n")

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Process detection results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Extract confidence and class ID
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Display class name and confidence with percentage symbol
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img,
                f"{classNames[cls]}: {confidence}%",
                (int(x1), int(y1) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display webcam feed
    cv2.imshow("Webcam", img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
