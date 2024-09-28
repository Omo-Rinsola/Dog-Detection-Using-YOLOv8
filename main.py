# Importing necessary libraries
import cv2
from ultralytics import YOLO
import numpy as np 
import torch

# Check if Metal Performance Shaders (MPS) is available for Mac M1/M2
# MPS is Apple's GPU acceleration framework, used here for faster processing.
print(torch.backends.mps.is_available())

# Load video and YOLOv8 model
cap = cv2.VideoCapture("dogs.mp4")
model = YOLO("yolov8m.pt")

# Process video frames
while True:
    ret, frame = cap.read()

    if not ret:
        break 

    # Perform object detection
    results = model(frame, device="mps")
    result = results[0]

    # Extract bounding boxes and class labels
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls,bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox

        
        cv2.rectangle(frame,(x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1, 
                    (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Img", frame)

    # Exit on 'Esc' key
    key = cv2.waitKey(0)
    if key == 27:
        break
cap.realease()
cv2.destroyAllWindows()
