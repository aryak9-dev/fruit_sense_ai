

from ultralytics import YOLO
from collections import Counter
import cv2
import math

model = YOLO("weights/best.pt")
CONF_THRESHOLD = 0.5

# 🔥 Track objects using centroid distance
tracked_objects = {}
next_id = 0
# DIST_THRESHOLD = 50

MAX_MISSING = 10   # frames to keep lost objects
DIST_THRESHOLD = 80  # increase tolerance

tracked_objects = {}
next_id = 0

def get_centroid(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def detect_video(video_path):
    global tracked_objects, next_id

    cap = cv2.VideoCapture(video_path)
        # 🔥 VIDEO OUTPUT SETUP
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 20  # fallback
    fps = int(fps)

    output_path = "output_video.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total_counts = Counter()
    counted_ids = set()

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD)

        # # 🔥 DRAW BOUNDING BOXES
        # annotated = results[0].plot()

        # # 🔥 WRITE FRAME TO VIDEO
        # out.write(annotated)

        annotated = frame.copy()

        



        current_objects = []

        for box in results[0].boxes:
            conf = float(box.conf)
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls)
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centroid = get_centroid(x1, y1, x2, y2)

            current_objects.append((centroid, label,box))

        # 🔥 Match with existing tracked objects
        # new_tracked = {}

        # for centroid, label in current_objects:
        #     matched_id = None

        #     for obj_id, (prev_centroid, prev_label) in tracked_objects.items():
        #         if label == prev_label and distance(centroid, prev_centroid) < DIST_THRESHOLD:
        #             matched_id = obj_id
        #             break

        #     if matched_id is None:
        #         matched_id = next_id
        #         next_id += 1

        #     new_tracked[matched_id] = (centroid, label)

        #     # ✅ Count only once
        #     if matched_id not in counted_ids:
        #         total_counts[label] += 1
        #         counted_ids.add(matched_id)
        
        new_tracked = {}
        used_ids = set()

        for centroid, label, box in current_objects:
            best_match = None
            min_dist = float("inf")

            # 🔍 Find best match instead of first match
            for obj_id, obj in tracked_objects.items():
                prev_centroid, prev_label, miss = obj

                if label != prev_label:
                    continue

                dist = distance(centroid, prev_centroid)

                if dist < DIST_THRESHOLD and dist < min_dist:
                    min_dist = dist
                    best_match = obj_id

            if best_match is not None:
                new_tracked[best_match] = (centroid, label, 0)
                used_ids.add(best_match)

                # ✅ Count only once
                if best_match not in counted_ids:
                    total_counts[label] += 1
                    counted_ids.add(best_match)

            else:
                # 🆕 New object
                # new_tracked[next_id] = (centroid, label, 0)
                # counted_ids.add(next_id)
                # total_counts[label] += 1
                # next_id += 1
                assigned_id = next_id
                new_tracked[assigned_id] = (centroid, label, 0)
                counted_ids.add(assigned_id)
                total_counts[label] += 1
                next_id += 1    
                

            # Get bounding box again
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label + ID
            current_id = best_match if best_match is not None else assigned_id
            text = f"{label} ID:{current_id}"
            # text = f"{label} ID:{best_match if best_match is not None else next_id-1}"
            cv2.putText(
                annotated,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )           

        # out.write(annotated)



            # 🔄 Keep old objects for few frames
        for obj_id, (centroid, label, miss) in tracked_objects.items():
            if obj_id not in used_ids:
                if miss < MAX_MISSING:
                    new_tracked[obj_id] = (centroid, label, miss + 1)   



        tracked_objects = new_tracked
        out.write(annotated)

    # # 🔄 Keep old objects for few frames
    # for obj_id, (centroid, label, miss) in tracked_objects.items():
    #     if obj_id not in used_ids:
    #         if miss < MAX_MISSING:
    #             new_tracked[obj_id] = (centroid, label, miss + 1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # return {
    #     "counts": dict(total_counts)
    #     "video_path": output_path
    # }
    return {
    "counts": dict(total_counts),
    "video_path": output_path   # 🔥 IMPORTANT
}
