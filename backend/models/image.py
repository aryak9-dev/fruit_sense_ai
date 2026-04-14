from ultralytics import YOLO
import cv2
from collections import Counter
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", "weights", "best.pt")

model = YOLO(MODEL_PATH)

CONF_THRESHOLD = 0.5

def detect_image(image_path):
    frame = cv2.imread(image_path)
    results = model(frame, conf=CONF_THRESHOLD)

    counts = Counter()

    for box in results[0].boxes:
        conf = float(box.conf)

        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls)
        label = model.names[cls_id]

        counts[label] += 1

    # 🔥 DRAW BOUNDING BOXES
    annotated = results[0].plot()

    # 🔥 SAVE IMAGE
    output_path = "output_detected.jpg"
    cv2.imwrite(output_path, annotated)

    return {
        "total_count": sum(counts.values()),
        "counts": dict(counts),
        "image_path": output_path   # 👈 IMPORTANT
    }

#     return {
#         "total_count": sum(counts.values()),
#         "counts": dict(counts)
#     }

# def detect_image(image_path):
#     frame = cv2.imread(image_path)
#     frame = cv2.resize(frame, (800, 800))

#     results = model(frame, conf=CONF_THRESHOLD, iou=0.5,augment=True)

#     counts = Counter()
#     filtered_boxes = []

#     def iou(box1, box2):
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])

#         inter = max(0, x2-x1) * max(0, y2-y1)
#         area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
#         area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

#         return inter / (area1 + area2 - inter + 1e-6)

#     frame_area = frame.shape[0] * frame.shape[1]

#     for box in results[0].boxes:
#         conf = float(box.conf)
#         if conf < 0.65:
#             continue

#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         box_area = (x2 - x1) * (y2 - y1)

#         # if box_area < 0.001 * frame_area:
#         #     continue

#         coords = [x1, y1, x2, y2]

#         duplicate = False
#         for fb in filtered_boxes:
#             if iou(coords, fb["coords"]) > 0.5:
#                 duplicate = True
#                 break

#         if not duplicate:
#             cls_id = int(box.cls)
#             label = model.names[cls_id]

#             filtered_boxes.append({
#                 "coords": coords,
#                 "label": label
#             })

#             counts[label] += 1

#     # 🔥 DRAW BOUNDING BOXES
#     annotated = results[0].plot()

#     # 🔥 SAVE IMAGE
#     output_path = "output_detected.jpg"
#     cv2.imwrite(output_path, annotated)

#     return {
#         "total_count": sum(counts.values()),
#         "counts": dict(counts),
#         "image_path": output_path   # 👈 IMPORTANT
#     }

#     # return {
#     #     "total_count": sum(counts.values()),
#     #     "counts": dict(counts)
#     # }