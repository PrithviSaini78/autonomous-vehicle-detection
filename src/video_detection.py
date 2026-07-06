from ultralytics import YOLO
import cv2
from collections import Counter
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("../videos/road.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()

prev_time = time.time()
frame_count=0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count +=1
    if frame_count %2 !=0:
        continue

    frame = cv2.resize(frame,(640,480))
    # Run object detection
    results = model(frame,imgsz=320,conf=0.5,verbose=False)

    # Count detected objects

    detections = results[0].boxes.cls.tolist()
    class_names = results[0].names
    labels = [class_names[int(i)] for i in detections]
    object_count = Counter(labels)



    # Draw bounding boxes
    annotated_frame = results[0].plot()
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (500, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    y=35
    for label, count in object_count.items():
        text = f"{label}: {count}"
        cv2.putText(
            annotated_frame,
            text,
            (10, y),
         cv2.FONT_HERSHEY_SIMPLEX,
         0.7,
         (0, 255, 0),
         2
        )
        y += 35

    # Show video
    cv2.imshow("Autonomous Vehicle Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()