import cv2
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results=model.predict(mode="predict", model="yolov8m.pt", conf=0.8, classes=[0], source=frame)

    detections = results[0].boxes

    class_counts = {}

    for detection in detections:
        class_id = int(detection.cls)
        if class_id in class_counts:
            class_counts[class_id] += 1
        else:
            class_counts[class_id] = 1
    totalcount = 0
    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} instances")
        totalcount = count
    strcount = str(totalcount)
    annotated_image = results[0].plot()
    cv2.putText(frame, f'Person Count: {strcount}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    annotated_image = results[0].plot()
    cv2.imshow('Webcam Feed', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
