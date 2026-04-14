import cv2
import backend.main as main
from ultralytics import YOLO

model = YOLO("weights/best.pt")

# def process_frame(frame):
#     results = model(frame)

#     count = len(results[0].boxes)

#     annotated = results[0].plot()

#     return annotated, count

def process_frame(frame):
    results = model(frame)

    valid_boxes = []

    frame_area = frame.shape[0] * frame.shape[1]

    for box in results[0].boxes:
        conf = float(box.conf)

        # 🔥 1. Confidence filter
        if conf < 0.7:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 🔥 2. Area filter (ignore big objects like face)
        box_area = (x2 - x1) * (y2 - y1)

        if box_area > 0.5 * frame_area:
            continue

        valid_boxes.append(box)

    # 🔥 Replace boxes with filtered ones
    results[0].boxes = valid_boxes

    count = len(valid_boxes)

    annotated = results[0].plot()

    return annotated, count

# def run_webcam():
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

    

#         results = model(frame)

#         # 🔥 COUNT OBJECTS
#         count = len(results[0].boxes)

#         # 🔥 UPDATE GLOBAL COUNT
#         main.live_total = count

#         # OPTIONAL: keep annotated frame (for later streaming)
#         annotated = results[0].plot()

    

#     cap.release()
#     cv2.destroyAllWindows()

def run_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, count = process_frame(frame)

        main.live_total = count

        # Optional display (if needed)
        cv2.imshow("Live", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()