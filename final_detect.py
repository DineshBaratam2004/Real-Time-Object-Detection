from ultralytics import YOLO
import cv2
import time
import os

# Load trained YOLOv8 model
model = YOLO("runs/detect/train7/weights/best.pt")

# Output directory for detected objects
output_dir = "all_detections"
os.makedirs(output_dir, exist_ok=True)

# Track saved object classes
saved_classes = set()

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)
    total_objects = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            total_objects += 1

            # Draw bounding box and label with confidence in %
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf*100:.0f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save only one image per object class
            if label not in saved_classes:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    class_dir = os.path.join(output_dir, label)
                    os.makedirs(class_dir, exist_ok=True)
                    filename = os.path.join(class_dir, f"{label}.jpg")
                    cv2.imwrite(filename, crop)
                    saved_classes.add(label)
                    print(f"[SAVED] {label} -> {filename}")

    # Show total object count (in red)
    cv2.putText(frame, f"Total Objects: {total_objects}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Live Detection", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection stopped.")




