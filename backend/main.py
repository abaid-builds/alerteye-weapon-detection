from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import base64
import json
import asyncio
import datetime
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from detector import WeaponDetector

executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="AlertEye API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = WeaponDetector(model_path="best_openvino_model/")

THREATS       = ['pistol', 'rifle', 'knife', 'explosive', 'shotgun']
REPORTS_DIR   = "reports"
SNAPSHOTS_DIR = "reports/snapshots"
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
csv_lock = threading.Lock()


def log_detection(label: str, confidence: float, frame):
    timestamp = datetime.datetime.now()
    ts_str    = timestamp.strftime('%Y%m%d_%H%M%S_%f')
    img_path  = f"{SNAPSHOTS_DIR}/alert_{ts_str}.jpg"
    cv2.imwrite(img_path, frame)

    csv_path = f"{REPORTS_DIR}/history.csv"
    row = {
        'Timestamp':  [timestamp.strftime('%Y-%m-%d %H:%M:%S')],
        'Class':      [label],
        'Confidence': [round(confidence * 100, 1)],
        'Snapshot':   [img_path]
    }
    with csv_lock:
        df = pd.DataFrame(row)
        df.to_csv(csv_path, mode='a', index=False,
                  header=not os.path.exists(csv_path))


# ── Health check ──────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "AlertEye backend is running"}

# ── Image detection ───────────────────────────────────────
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    frame, detections = detector.detect_image(contents)

    _, buffer = cv2.imencode(".jpg", frame)
    encoded = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "detections": detections,
        "annotated_image": encoded,
        "weapon_detected": len(detections) > 0
    })

# ── Live stream via WebSocket ─────────────────────────────
@app.websocket("/detect/stream")
async def stream(websocket: WebSocket, camera: str = "0"):
    await websocket.accept()

    source = int(camera) if camera == "0" else camera
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        await websocket.send_text(json.dumps({
            "error": f"Cannot open camera: {camera}"
        }))
        await websocket.close()
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── Shared state ──────────────────────────────────────────
    latest_detections = []
    latest_weapon_detected = False
    frame_count = 0
    DETECT_EVERY = 5  # Run AI every 5th frame only

    def run_detection(frame):
        nonlocal latest_detections, latest_weapon_detected
        _, dets, confirmed = detector.detect_frame(frame.copy())
        latest_detections = dets
        latest_weapon_detected = confirmed
        if confirmed:
            for det in dets:
                if det["class"] in THREATS:
                    log_detection(det["class"], det["confidence"], frame)

    try:
        loop = asyncio.get_event_loop()

        while True:
            # ── Always grab latest frame (drain buffer) ───────
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                break

            frame_count += 1

            # ── Run detection only every 5th frame ────────────
            if frame_count % DETECT_EVERY == 0:
                await loop.run_in_executor(
                    executor, run_detection, frame.copy()
                )

            # ── Draw latest detections on every frame ─────────
            display_frame = frame.copy()
            for det in latest_detections:
                x1 = det["bbox"]["x1"]
                y1 = det["bbox"]["y1"]
                x2 = det["bbox"]["x2"]
                y2 = det["bbox"]["y2"]
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ── Encode and send every frame ───────────────────
            _, buffer = cv2.imencode(".jpg", display_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 60])
            encoded = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_text(json.dumps({
                "frame": encoded,
                "detections": latest_detections,
                "weapon_detected": latest_weapon_detected
            }))

            # ── Tiny sleep to keep websocket breathing ────────
            await asyncio.sleep(0.001)

    except Exception as e:
        print(f"Stream ended: {e}")
    finally:
        cap.release()

# ── Detection history ─────────────────────────────────────────
@app.get("/api/history")
def get_history(period: str = "daily"):
    csv_path = f"{REPORTS_DIR}/history.csv"
    if not os.path.exists(csv_path):
        return JSONResponse([])
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    now = datetime.datetime.now()
    if period == 'daily':
        df = df[df['Timestamp'].dt.date == now.date()]
    elif period == 'weekly':
        df = df[df['Timestamp'] >= now - datetime.timedelta(days=7)]
    elif period == 'monthly':
        df = df[df['Timestamp'] >= now - datetime.timedelta(days=30)]
    df['Timestamp'] = df['Timestamp'].astype(str)
    return JSONResponse(df.to_dict('records'))


# ── Report download ───────────────────────────────────────────
@app.get("/download_report")
def download_report(period: str = "all"):
    csv_path = f"{REPORTS_DIR}/history.csv"
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "No reports yet"}, status_code=404)
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    now = datetime.datetime.now()
    if period == 'daily':
        df = df[df['Timestamp'].dt.date == now.date()]
    elif period == 'weekly':
        df = df[df['Timestamp'] >= now - datetime.timedelta(days=7)]
    elif period == 'monthly':
        df = df[df['Timestamp'] >= now - datetime.timedelta(days=30)]
    report_path = f"{REPORTS_DIR}/report_{period}_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(report_path, index=False)
    return FileResponse(
        report_path,
        media_type='text/csv',
        filename=os.path.basename(report_path)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)