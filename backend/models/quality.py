# from ultralytics import YOLO
# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, "..", "weights", "quality.pt")

# model = YOLO(MODEL_PATH)

# def predict_quality(image_path):
#     results = model(image_path)

#     # ❌ Safety check
#     if results is None or len(results) == 0:
#         return "unknown"

#     result = results[0]

#     if result.boxes is None or len(result.boxes) == 0:
#         return "unknown"

#     # ✅ Get highest confidence detection
#     box = result.boxes[0]
#     cls_id = int(box.cls)

#     label = model.names[cls_id]   # e.g., Apple_Bad

#     return label.lower()

from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", "weights", "quality.pt")

model = YOLO(MODEL_PATH)

print(model.task)

def predict_quality(image_path):
    results = model(image_path)

    if not results or len(results) == 0:
        return "unknown"

    result = results[0]

    # ✅ For classification model
    if hasattr(result, "probs") and result.probs is not None:
        probs = result.probs

        cls_id = int(probs.top1)
        label = model.names[cls_id]

        # return label.lower()
    
        label = label.lower()

        return {
            "fruit": label.split("_")[0],
            "quality": label.split("_")[-1]
        }

        # changed code final

        # probs_list = probs.data.tolist()

        # best_label = None
        # best_conf = 0

        # for i, conf in enumerate(probs_list):
        #     label = model.names[i].lower()

        #     # ✅ skip pure fruit labels like "orange"
        #     if "_" not in label:
        #         continue

        #     if conf > best_conf:
        #         best_conf = conf
        #         best_label = label

        # # fallback
        # if best_label is None:
        #     return {
        #     "fruit": "unknown",
        #     "quality": "unknown"
        #     }

        # fruit, quality = best_label.split("_")

        # return {
        #     "fruit": fruit,
        #     "quality": quality
        # }
    
    # end of changed code

        # # extract only quality
        # if "_" in label:
        #     return label.split("_")[-1]

        # return label

        print("RESULT:", results)
    # fallback
    return {
    "fruit": "unknown",
    "quality": "unknown"
}

