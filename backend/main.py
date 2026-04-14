from fastapi import FastAPI, UploadFile, File
import shutil
import os
from fastapi.responses import StreamingResponse
import cv2
from fastapi import Request

from backend.models.image import detect_image
from backend.models.video import detect_video
from backend.models.live import run_webcam
from backend.models.live import process_frame
import backend.main as main
from backend.models.image import detect_image
from backend.models.quality import predict_quality
camera_running = False
cap = None

from fastapi.staticfiles import StaticFiles



app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.mount("/results", StaticFiles(directory="."), name="results")
# @app.post("/detect-image")
# async def detect_image_api(file: UploadFile = File(...)):
#     path = f"{UPLOAD_FOLDER}/{file.filename}"

#     with open(path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     return detect_image(path)

@app.post("/detect-image")
async def detect_image_api(file: UploadFile = File(...)):
    path = f"uploads/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return detect_image(path)

@app.post("/detect-quality")
async def detect_quality_api(file: UploadFile = File(...)):
    path = f"uploads/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    quality = predict_quality(path)

    return {
        "quality": quality
    }

@app.post("/detect-video")
async def detect_video_api(file: UploadFile = File(...)):
    path = f"{UPLOAD_FOLDER}/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return detect_video(path)

live_total = 0

@app.get("/live-count")
def get_live_count():
    return {"count": live_total}


@app.get("/live-stream")
async def live_stream(request: Request):
    return StreamingResponse(
        generate_frames(request),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/stop-camera")
def stop_camera():
    global camera_running

    camera_running = False
    return {"status": "camera stopped"}


async def generate_frames(request):
    global cap, camera_running

    cap = cv2.VideoCapture(0)
    camera_running = True

    try:
        while True:
            if not camera_running:
                break

            if await request.is_disconnected():
                break

            success, frame = cap.read()
            if not success:
                break

            # 🔥 PROCESS FRAME
            annotated_frame, count = process_frame(frame)

            main.live_total = count

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

    finally:
        print("🔥 Releasing camera...")
        if cap:
            cap.release()
        cv2.destroyAllWindows()

# async def generate_frames(request):
#     global cap, camera_running

#     cap = cv2.VideoCapture(0)
#     camera_running = True

#     while camera_running:
#         success, frame = cap.read()
#         if not success:
#             break

#         # 🔥 CALL LIVE MODULE
#         annotated_frame, count = process_frame(frame)

#         # 🔥 UPDATE COUNT
#         main.live_total = count

#         # encode
#         _, buffer = cv2.imencode('.jpg', annotated_frame)
#         frame_bytes = buffer.tobytes()

#         yield (
#             b'--frame\r\n'
#             b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
#         )

#             # 🔥 RELEASE CAMERA WHEN STOPPED
#     if cap:
#         cap.release()
#         # cap = None