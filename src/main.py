from ultralytics import YOLO
import cv2

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Load image
image = cv2.imread("../videos/test.jpg.webp")

if image is None:
    print("Image not found!")
    exit()


# Detect objects
results = model(image)

# Draw detection results
annotated = results[0].plot()

# Show image
cv2.imshow("Object Detection", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()