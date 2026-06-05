import cv2
import threading
import time
import datetime
import os
import pandas as pd
from flask import Flask, Response, render_template_string, jsonify, send_file, request
from detector import WeaponDetector

# ─── CONFIG ──────────────────────────────────────────────────────
CAMERA_LABEL  = 'Main Entrance'
JPEG_QUALITY  = 92          # ↑ raised from 85 — sharper stream
INFER_WIDTH   = 640         # ↑ raised from 320 — inference frame width
INFER_HEIGHT  = 480         # ↑ raised from 180 — inference frame height
IP_WEBCAM_URL = 0
THREATS       = ['pistol', 'rifle', 'knife', 'explosive', 'shotgun']

# Initialize folders
os.makedirs('reports/snapshots', exist_ok=True)

app      = Flask(__name__)
detector = WeaponDetector(model_path="best.pt")

# Shared state
lock           = threading.Lock()
alert_active   = False
last_boxes     = []
current_conf   = 0.25
stats          = {'fps': 0, 'detections': 0, 'total': 0}
alert_log      = []
current_source = IP_WEBCAM_URL


# ─── COMPONENT 1: THREADED VIDEO CAPTURE ───────────────────────
class VideoStream:
    def __init__(self, src=0):
        self.src     = src
        self.cap     = self._open(src)
        self.frame   = None
        self.stopped = False
        ret, self.frame = self.cap.read()
        self._lock   = threading.Lock()

    def _open(self, src):
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Only set these for non-HTTP sources
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 60)
            actual_w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[Camera] Actual: {int(actual_w)}x{int(actual_h)} @ {actual_fps:.0f}fps")
        else:
            print(f"[Camera] Opening stream: {src}")

        return cap

    def change_source(self, src):
        print(f"[Camera] Switching to: {src}")
        with self._lock:
            self.cap.release()
            time.sleep(0.3)  # let the old cap fully release
            self.src = src
            self.cap = self._open(src)
            self.frame = None  # reset frame

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            with self._lock:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame = frame

    def read(self):
        return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# ─── COMPONENT 2: AI DETECTION THREAD ──────────────────────────
def ai_worker(vs):
    global alert_active, last_boxes, stats
    detector.confidence_threshold = current_conf

    while True:
        frame = vs.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # Downscale a COPY for inference only — full-res frame stays untouched
        infer_frame = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))

        # detector returns the frame WITHOUT any drawings on it (drawing removed from detector.py)
        _, dets, confirmed = detector.detect_frame(infer_frame)

        new_boxes = []

        for det in dets:
            x1       = det["bbox"]["x1"]
            y1       = det["bbox"]["y1"]
            x2       = det["bbox"]["x2"]
            y2       = det["bbox"]["y2"]
            label    = det["class"]
            conf_val = round(det["confidence"] * 100, 1)

            # Store inference-resolution coords — generate_frames() scales them up
            new_boxes.append((x1, y1, x2, y2, label, conf_val))

            # Log to CSV/snapshot only for confirmed threats
            if confirmed and label in THREATS:
                log_detection(label, conf_val, frame)

        with lock:
            last_boxes   = new_boxes
            alert_active = confirmed
            if confirmed:
                stats['detections'] += 1
                stats['total']      += 1

        time.sleep(0.01)


def log_detection(label, conf, frame):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    img_path  = f"reports/snapshots/alert_{timestamp}.jpg"
    cv2.imwrite(img_path, frame)

    log_entry = {
        'Timestamp':  [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Class':      [label],
        'Confidence': [conf],
        'Snapshot':   [img_path]
    }
    df = pd.DataFrame(log_entry)
    df.to_csv(
        'reports/history.csv', mode='a', index=False,
        header=not os.path.exists('reports/history.csv')
    )

    with lock:
        alert_log.insert(0, {
            'time':  datetime.datetime.now().strftime('%H:%M:%S'),
            'class': label,
            'conf':  conf
        })
        if len(alert_log) > 50:
            alert_log.pop()


# ─── COMPONENT 3: FRAME GENERATOR ──────────────────────────────
def generate_frames(vs):
    fps_counter = 0
    fps_timer   = time.time()

    while True:
        frame = vs.read()
        if frame is None:
            continue

        display   = frame.copy()
        display_h, display_w = display.shape[:2]

        with lock:
            boxes  = last_boxes.copy()
            active = alert_active

        for (x1, y1, x2, y2, label, conf) in boxes:
            # Scale bounding box from inference resolution → actual display resolution
            # This is the ONLY place boxes are drawn — detector.py draws nothing
            scale_x = display_w / INFER_WIDTH
            scale_y = display_h / INFER_HEIGHT

            rx1 = int(x1 * scale_x)
            ry1 = int(y1 * scale_y)
            rx2 = int(x2 * scale_x)
            ry2 = int(y2 * scale_y)

            # Red for confirmed threats, orange for non-threat detections
            color = (0, 0, 255) if label in THREATS else (0, 165, 255)

            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), color, 2)
            cv2.putText(
                display, f"{label} {conf}%",
                (rx1, max(ry1 - 10, 15)),  # clamp label so it doesn't go off-screen
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        # FPS overlay
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            stats['fps'] = fps_counter
            fps_counter  = 0
            fps_timer    = time.time()

        cv2.putText(
            display, f"FPS: {stats['fps']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        _, jpeg = cv2.imencode(
            '.jpg', display,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + jpeg.tobytes()
            + b'\r\n'
        )


# ─── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(vs),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/status')
def api_status():
    with lock:
        return jsonify({
            'alert':      alert_active,
            'fps':        stats['fps'],
            'total':      stats['total'],
            'detections': stats['detections'],
            'log':        alert_log[:10]
        })

@app.route('/api/change_source', methods=['POST'])
def change_source():
    data = request.get_json()
    src  = data.get('source', '0')
    src  = int(src) if src.isdigit() else src  # ← keeps URL as string ✅
    vs.change_source(src)
    return jsonify(success=True)

@app.route('/api/set_conf/<val>')
def set_conf(val):
    global current_conf
    current_conf = float(val)
    detector.confidence_threshold = current_conf
    return jsonify(success=True)

@app.route('/api/history')
def get_history():
    period = request.args.get('period', 'daily')
    if not os.path.exists('reports/history.csv'):
        return jsonify([])
    df  = pd.read_csv('reports/history.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    now = datetime.datetime.now()

    if period == 'daily':
        df = df[df['Timestamp'].dt.date == now.date()]
    elif period == 'weekly':
        df = df[df['Timestamp'] >= now - datetime.timedelta(days=7)]
    elif period == 'monthly':
        df = df[df['Timestamp'] >= now - datetime.timedelta(days=30)]

    return jsonify(df.to_dict('records'))

@app.route('/download_report')
def download_report():
    period = request.args.get('period', 'all')
    if not os.path.exists('reports/history.csv'):
        return "No reports yet", 404

    df  = pd.read_csv('reports/history.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    now = datetime.datetime.now()

    if period == 'daily':
        df = df[df['Timestamp'].dt.date == now.date()]
    elif period == 'weekly':
        df = df[df['Timestamp'] >= now - datetime.timedelta(days=7)]
    elif period == 'monthly':
        df = df[df['Timestamp'] >= now - datetime.timedelta(days=30)]

    report_path = f'reports/report_{period}_{now.strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(report_path, index=False)
    return send_file(report_path, as_attachment=True)


# ─── HTML DASHBOARD ─────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlertEye | Weapon Detection System</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:       #070c18;
            --bg2:      #0c1424;
            --bg3:      #101b30;
            --bg4:      #142038;
            --border:   #1a2c4a;
            --border2:  #243d66;
            --blue:     #3b82f6;
            --blue-dim: rgba(59,130,246,0.12);
            --blue-glow:rgba(59,130,246,0.25);
            --red:      #ef4444;
            --red-dim:  rgba(239,68,68,0.10);
            --red-b:    rgba(239,68,68,0.30);
            --amber:    #f59e0b;
            --green:    #10b981;
            --text:     #e2e8f0;
            --text2:    #94a3b8;
            --text3:    #475569;
            --ui:       'Plus Jakarta Sans', sans-serif;
            --mono:     'JetBrains Mono', monospace;
            --r:        8px;
            --r-lg:     12px;
        }

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; }

        body {
            background: var(--bg);
            color: var(--text);
            font-family: var(--ui);
            font-size: 14px;
            line-height: 1.5;
            display: flex;
            flex-direction: column;
            -webkit-font-smoothing: antialiased;
        }

        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg2); }
        ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

        /* ── Top bar ── */
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 28px;
            height: 58px;
            background: var(--bg2);
            border-bottom: 1px solid var(--border);
            flex-shrink: 0;
        }

        .brand {
            font-family: var(--ui);
            font-size: 17px;
            font-weight: 800;
            letter-spacing: 3px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .brand-eye { color: var(--red); }
        .brand-accent { color: var(--blue); }

        .srv-badge {
            display: flex;
            align-items: center;
            gap: 7px;
            font-family: var(--mono);
            font-size: 10px;
            letter-spacing: 1px;
            color: var(--text2);
        }
        .dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 6px var(--green);
            flex-shrink: 0;
        }
        .dot.red { background: var(--red); box-shadow: 0 0 6px var(--red); }

        /* ── Tab nav ── */
        nav {
            display: flex;
            padding: 0 28px;
            background: var(--bg2);
            border-bottom: 1px solid var(--border);
            flex-shrink: 0;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 7px;
            padding: 13px 18px;
            color: var(--text3);
            font-family: var(--ui);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            transition: color .15s, border-color .15s;
            user-select: none;
        }
        .nav-item svg { width: 14px; height: 14px; flex-shrink: 0; }
        .nav-item:hover { color: var(--text2); }
        .nav-item.active { color: var(--blue); border-bottom-color: var(--blue); }

        /* ── Page content ── */
        main {
            flex: 1;
            overflow-y: auto;
            padding: 24px 28px;
            display: none;
        }
        main.active { display: block; }

        .page-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .page-title {
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--text);
        }

        /* ── Cards ── */
        .card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: var(--r-lg);
            padding: 18px;
        }
        .card + .card { margin-top: 14px; }
        .card-title {
            font-family: var(--ui);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
            color: var(--text2);
            text-transform: uppercase;
            margin-bottom: 14px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }

        /* ── Live Feed ── */
        .live-grid { display: grid; grid-template-columns: 1fr 400px; gap: 16px; }

        .feed-wrap {
            position: relative;
            border-radius: var(--r);
            overflow: hidden;
            border: 1px solid var(--border);
            background: #000;
            transition: border-color .3s, box-shadow .3s;
        }
        .feed-wrap img { width: 100%; display: block; }
        .feed-wrap.alarmed {
            border-color: var(--red);
            box-shadow: 0 0 24px rgba(239,68,68,0.2);
        }

        .rec-badge {
            position: absolute;
            top: 12px; right: 12px;
            background: rgba(239,68,68,0.88);
            color: #fff;
            font-family: var(--mono);
            font-size: 9px;
            font-weight: 500;
            letter-spacing: 1px;
            padding: 4px 10px;
            border-radius: 20px;
        }

        .res-badge {
            font-family: var(--mono);
            font-size: 10px;
            color: var(--text3);
            letter-spacing: 0.5px;
            margin-top: 8px;
            text-align: right;
        }

        .cam-row {
            display: flex;
            gap: 8px;
            margin-bottom: 14px;
        }
        .cam-row input {
            flex: 1;
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 5px;
            color: var(--text);
            padding: 8px 12px;
            font-family: var(--mono);
            font-size: 11px;
            outline: none;
            transition: border-color .2s;
        }
        .cam-row input:focus { border-color: var(--blue); }
        .cam-row input::placeholder { color: var(--text3); }

        /* ── Stat cards ── */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(3,1fr);
            gap: 8px;
            margin-bottom: 16px;
        }
        .stat {
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: var(--r);
            padding: 12px 10px;
            text-align: center;
        }
        .stat-val {
            font-family: var(--mono);
            font-size: 24px;
            font-weight: 500;
            color: var(--text);
        }
        .stat-val.red { color: var(--red); }
        .stat-label {
            font-family: var(--ui);
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 1.5px;
            color: var(--text3);
            margin-top: 3px;
            text-transform: uppercase;
        }

        /* ── Alert log ── */
        .log-list { display: flex; flex-direction: column; gap: 6px; }
        .log-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            background: var(--bg3);
            border: 1px solid var(--border);
            border-left: 2px solid var(--red);
            border-radius: 5px;
            animation: fadein .2s ease;
        }
        @keyframes fadein {
            from { opacity: 0; transform: translateY(4px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .log-class {
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--red);
            text-transform: uppercase;
        }
        .log-time { font-family: var(--mono); font-size: 10px; color: var(--text3); }
        .log-conf { font-family: var(--mono); font-size: 11px; color: var(--amber); margin-left: auto; }
        .log-empty {
            font-family: var(--mono);
            font-size: 10px;
            color: var(--text3);
            letter-spacing: 3px;
            text-align: center;
            padding: 28px 0;
        }

        /* ── Buttons ── */
        .btn {
            padding: 8px 16px;
            border: 1px solid;
            border-radius: 5px;
            font-family: var(--ui);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            cursor: pointer;
            transition: background .15s, box-shadow .15s;
            white-space: nowrap;
        }
        .btn-blue {
            background: var(--blue-dim);
            border-color: var(--blue);
            color: var(--blue);
        }
        .btn-blue:hover {
            background: rgba(59,130,246,0.2);
            box-shadow: 0 0 12px var(--blue-glow);
        }
        .btn-red {
            background: var(--red-dim);
            border-color: var(--red-b);
            color: var(--red);
        }
        .btn-red:hover { background: rgba(239,68,68,0.18); }

        /* ── Charts ── */
        .chart-wrap { position: relative; height: 190px; }
        canvas { max-height: 190px; }

        /* ── Analytics grid ── */
        .analytics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 14px;
        }

        /* ── Reports ── */
        .report-grid {
            display: grid;
            grid-template-columns: repeat(3,1fr);
            gap: 14px;
            margin-bottom: 14px;
        }
        .report-card {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: var(--r-lg);
            padding: 22px 18px;
            text-align: center;
        }
        .report-period {
            font-family: var(--ui);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
            color: var(--blue);
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .report-count {
            font-family: var(--mono);
            font-size: 40px;
            font-weight: 500;
            color: var(--text);
            line-height: 1;
            margin-bottom: 6px;
        }
        .report-sub {
            font-family: var(--mono);
            font-size: 9px;
            letter-spacing: 1.5px;
            color: var(--text3);
            margin-bottom: 16px;
        }

        .report-table { width: 100%; border-collapse: collapse; }
        .report-table th {
            font-family: var(--ui);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.5px;
            color: var(--text3);
            padding: 8px 14px;
            border-bottom: 1px solid var(--border);
            text-align: left;
            text-transform: uppercase;
        }
        .report-table td {
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            transition: background .12s;
        }
        .report-table tr:hover td { background: var(--bg3); }
        .td-mono { font-family: var(--mono); font-size: 11px; color: var(--text2); }
        .td-class { font-size: 12px; font-weight: 700; color: var(--red); text-transform: uppercase; }
        .td-conf { font-family: var(--mono); font-size: 11px; color: var(--amber); }
        .table-wrap { max-height: 360px; overflow-y: auto; }

        /* ── Settings ── */
        .settings-card { max-width: 520px; }
        .setting-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 0;
            border-bottom: 1px solid var(--border);
        }
        .setting-row:last-child { border-bottom: none; padding-bottom: 0; }
        .setting-label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
        }
        .setting-desc {
            font-size: 11px;
            color: var(--text3);
            margin-top: 2px;
        }
        .setting-control { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
        input[type=range] { width: 140px; accent-color: var(--blue); }
        #conf-val {
            font-family: var(--mono);
            font-size: 13px;
            color: var(--blue);
            min-width: 34px;
            text-align: right;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 11px 0;
            border-bottom: 1px solid var(--border);
            font-size: 12px;
        }
        .info-row:last-child { border-bottom: none; padding-bottom: 0; }
        .info-key { color: var(--text2); font-weight: 500; }
        .info-val { font-family: var(--mono); font-size: 11px; color: var(--blue); }

        [data-theme="light"] {
            --bg:       #f1f5f9;
            --bg2:      #ffffff;
            --bg3:      #f8fafc;
            --bg4:      #e2e8f0;
            --border:   #e2e8f0;
            --border2:  #cbd5e1;
            --blue:     #2563eb;
            --blue-dim: rgba(37,99,235,0.08);
            --blue-glow:rgba(37,99,235,0.2);
            --red:      #dc2626;
            --red-dim:  rgba(220,38,38,0.08);
            --red-b:    rgba(220,38,38,0.25);
            --amber:    #d97706;
            --green:    #059669;
            --text:     #0f172a;
            --text2:    #475569;
            --text3:    #94a3b8;
        }

        body { transition: background-color .2s, color .2s; }
        .card, header, nav, .feed-wrap, .cam-row input,
        .report-card, .stat, .log-item { transition: background-color .2s, border-color .2s; }

        .btn-mute {
            background: transparent;
            border-color: var(--border2);
            color: var(--text2);
        }
        .btn-mute:hover { border-color: var(--text3); color: var(--text); }
        .btn-mute.muted { border-color: var(--amber); color: var(--amber); }

        .btn-save {
            background: transparent;
            border-color: var(--border2);
            color: var(--text2);
            font-size: 12px;
            padding: 8px 20px;
        }
        .btn-save:hover:not(:disabled) { border-color: var(--text3); color: var(--text); }
        .btn-save:disabled { opacity: 0.3; cursor: not-allowed; }

        .feed-footer { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; }
    </style>
</head>
<body>

<header>
    <div class="brand">
        <span class="brand-eye">◉</span>
        ALERT<span class="brand-accent">EYE</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <button class="btn btn-mute" id="theme-btn" onclick="toggleTheme()">☀️ Light</button>
        <button class="btn btn-mute" id="mute-btn" onclick="toggleMute()">🔊 Sound</button>
        <div class="srv-badge">
            <div class="dot" id="srv-dot"></div>
            <span id="srv-label">BACKEND ONLINE</span>
        </div>
    </div>
</header>

<nav>
    <div class="nav-item active" onclick="showTab('live', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="15" rx="2"/><polyline points="17 2 12 7 7 2"/></svg>
        Live Feed
    </div>
    <div class="nav-item" onclick="showTab('analytics', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
        Analytics
    </div>
    <div class="nav-item" onclick="showTab('reports', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
        Reports
    </div>
    <div class="nav-item" onclick="showTab('settings', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        Settings
    </div>
</nav>

<!-- ── Live Feed ── -->
<main id="tab-live" class="active">
    <div class="page-header">
        <div class="page-title">Live Feed</div>
    </div>
    <div class="cam-row">
        <input type="text" id="cam-url" placeholder="IP Camera URL or device index (leave empty for webcam)">
        <button class="btn btn-blue" onclick="connectCamera()">Connect</button>
        <button class="btn btn-red"  onclick="disconnectCamera()">Disconnect</button>
    </div>
    <div class="live-grid">
        <div>
            <div class="feed-wrap" id="feed-wrap">
                <img src="/video_feed" id="feed-img">
                <div class="rec-badge">● REC</div>
            </div>
            <div class="feed-footer">
                <div class="res-badge" id="res-badge">Detecting stream resolution…</div>
                <button class="btn btn-save" id="save-btn" onclick="downloadAlert()" disabled>Save Alert</button>
            </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:14px;">
            <div class="card">
                <div class="card-title">Detection Stats</div>
                <div class="stats-row">
                    <div class="stat"><div class="stat-val" id="fps-val">0</div><div class="stat-label">FPS</div></div>
                    <div class="stat"><div class="stat-val red" id="det-val">0</div><div class="stat-label">Threats</div></div>
                    <div class="stat"><div class="stat-val" id="total-val">0</div><div class="stat-label">Total</div></div>
                </div>
                <div class="chart-wrap"><canvas id="conf-chart"></canvas></div>
            </div>
            <div class="card">
                <div class="card-title">Recent Alerts</div>
                <div class="log-list" id="alert-log"><div class="log-empty">NO DETECTIONS YET</div></div>
            </div>
        </div>
    </div>
</main>

<!-- ── Analytics ── -->
<main id="tab-analytics">
    <div class="page-header"><div class="page-title">Analytics</div></div>
    <div class="analytics-grid">
        <div class="card">
            <div class="card-title">Detections by Class</div>
            <div class="chart-wrap"><canvas id="class-chart"></canvas></div>
        </div>
        <div class="card">
            <div class="card-title">Hourly Activity</div>
            <div class="chart-wrap"><canvas id="hourly-chart"></canvas></div>
        </div>
    </div>
    <div class="card">
        <div class="card-title">Confidence Distribution</div>
        <div class="chart-wrap"><canvas id="conf-dist-chart"></canvas></div>
    </div>
</main>

<!-- ── Reports ── -->
<main id="tab-reports">
    <div class="page-header"><div class="page-title">Reports</div></div>
    <div class="report-grid">
        <div class="report-card">
            <div class="report-period">Today</div>
            <div class="report-count" id="daily-count">0</div>
            <div class="report-sub">DETECTIONS</div>
            <button class="btn btn-blue" onclick="downloadReport('daily')">⬇ Download CSV</button>
        </div>
        <div class="report-card">
            <div class="report-period">This Week</div>
            <div class="report-count" id="weekly-count">0</div>
            <div class="report-sub">DETECTIONS</div>
            <button class="btn btn-blue" onclick="downloadReport('weekly')">⬇ Download CSV</button>
        </div>
        <div class="report-card">
            <div class="report-period">This Month</div>
            <div class="report-count" id="monthly-count">0</div>
            <div class="report-sub">DETECTIONS</div>
            <button class="btn btn-blue" onclick="downloadReport('monthly')">⬇ Download CSV</button>
        </div>
    </div>
    <div class="card">
        <div class="card-title">Detection History</div>
        <div class="table-wrap">
            <table class="report-table">
                <thead><tr><th>Timestamp</th><th>Class</th><th>Confidence</th></tr></thead>
                <tbody id="history-table"></tbody>
            </table>
        </div>
    </div>
</main>

<!-- ── Settings ── -->
<main id="tab-settings">
    <div class="page-header"><div class="page-title">Settings</div></div>
    <div class="card settings-card" style="margin-bottom:14px;">
        <div class="card-title">Detection</div>
        <div class="setting-row">
            <div>
                <div class="setting-label">Confidence Threshold</div>
                <div class="setting-desc">Higher value = fewer false positives, may miss some detections</div>
            </div>
            <div class="setting-control">
                <input type="range" min="0.10" max="0.90" step="0.05" value="0.25" id="conf-slider" oninput="updateConf(this.value)">
                <span id="conf-val">0.25</span>
            </div>
        </div>
    </div>
    <div class="card settings-card">
        <div class="card-title">Stream Info</div>
        <div class="info-row"><span class="info-key">Inference Resolution</span><span class="info-val">640 × 480</span></div>
        <div class="info-row"><span class="info-key">Stream Quality</span><span class="info-val">JPEG Q92</span></div>
        <div class="info-row"><span class="info-key">Target Capture</span><span class="info-val">1920 × 1080 @ 60fps</span></div>
    </div>
</main>

<script>
let muted        = false;
let audioCtx     = null;
let lastAlertFrame = null;
let prevAlert    = false;

function toggleMute() {
    muted = !muted;
    const btn = document.getElementById('mute-btn');
    btn.textContent = muted ? '🔇 Muted' : '🔊 Sound';
    btn.classList.toggle('muted', muted);
}

function playBeep() {
    if (muted) return;
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc  = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain); gain.connect(audioCtx.destination);
        osc.frequency.value = 880; osc.type = 'sine';
        gain.gain.setValueAtTime(0.4, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.4);
        osc.start(audioCtx.currentTime); osc.stop(audioCtx.currentTime + 0.4);
    } catch(e) {}
}

function captureFrame() {
    const img = document.getElementById('feed-img');
    if (!img.src || !img.naturalWidth) return null;
    try {
        const canvas = document.createElement('canvas');
        canvas.width  = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext('2d').drawImage(img, 0, 0);
        return canvas.toDataURL('image/jpeg', 0.92);
    } catch(e) { return null; }
}

function downloadAlert() {
    if (!lastAlertFrame) return;
    const a = document.createElement('a');
    a.href = lastAlertFrame;
    a.download = `alert_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
    a.click();
}

function showTab(id, el) {
    document.querySelectorAll('main').forEach(m => m.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
    el.classList.add('active');
    if (id === 'reports')   loadReports();
    if (id === 'analytics') loadAnalytics();
}

async function connectCamera() {
    const val = document.getElementById('cam-url').value.trim() || '0';
    const res = await fetch('/api/change_source', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source: val})
    });
    const data = await res.json();
    if (data.success) {
        setTimeout(() => {
            document.getElementById('feed-img').src = '/video_feed?' + Date.now();
        }, 800);
    }
}

function disconnectCamera() {
    document.getElementById('feed-img').src = '';
}

const feedImg = document.getElementById('feed-img');
feedImg.addEventListener('load', () => {
    const w = feedImg.naturalWidth;
    const h = feedImg.naturalHeight;
    if (w && h) document.getElementById('res-badge').textContent = `Stream resolution: ${w} × ${h}`;
});

function updateConf(val) {
    document.getElementById('conf-val').textContent = val;
    fetch('/api/set_conf/' + val);
}

let GRID_COLOR = '#1a2c4a';
let TICK_COLOR = '#475569';

function toggleTheme() {
    const html    = document.documentElement;
    const isLight = html.getAttribute('data-theme') === 'light';
    html.setAttribute('data-theme', isLight ? 'dark' : 'light');
    GRID_COLOR = isLight ? '#1a2c4a' : '#e2e8f0';
    TICK_COLOR = isLight ? '#475569' : '#94a3b8';
    document.getElementById('theme-btn').textContent = isLight ? '☀️ Light' : '🌙 Dark';
}
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
        y: { beginAtZero: true, grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, font: { family: 'JetBrains Mono', size: 10 } } },
        x: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 6 } }
    },
    animation: false
};

const confChart = new Chart(document.getElementById('conf-chart').getContext('2d'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [{ borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.08)', data: [], fill: true, tension: 0.4, pointRadius: 0, borderWidth: 1.5 }]
    },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, max: 100 } } }
});

setInterval(async () => {
    try {
        const r    = await fetch('/api/status');
        const data = await r.json();

        document.getElementById('fps-val').textContent   = data.fps;
        document.getElementById('det-val').textContent   = data.detections;
        document.getElementById('total-val').textContent = data.total;
        document.getElementById('srv-dot').classList.remove('red');
        document.getElementById('srv-label').textContent = 'BACKEND ONLINE';
        document.getElementById('feed-wrap').classList.toggle('alarmed', data.alert);

        if (data.alert && !prevAlert) {
            playBeep();
            const frame = captureFrame();
            if (frame) {
                lastAlertFrame = frame;
                document.getElementById('save-btn').disabled = false;
            }
        }
        prevAlert = data.alert;

        if (data.log.length > 0) {
            confChart.data.labels.push(data.log[0].time);
            confChart.data.datasets[0].data.push(data.log[0].conf);
            if (confChart.data.labels.length > 15) {
                confChart.data.labels.shift();
                confChart.data.datasets[0].data.shift();
            }
            confChart.update();
        }

        const logEl = document.getElementById('alert-log');
        logEl.innerHTML = data.log.length === 0
            ? '<div class="log-empty">NO DETECTIONS YET</div>'
            : data.log.slice(0, 8).map(l => `
                <div class="log-item">
                    <span class="log-class">${l.class.toUpperCase()}</span>
                    <span class="log-time">${l.time}</span>
                    <span class="log-conf">${l.conf}%</span>
                </div>`).join('');
    } catch(e) {
        document.getElementById('srv-dot').classList.add('red');
        document.getElementById('srv-label').textContent = 'BACKEND OFFLINE';
    }
}, 1000);

async function loadReports() {
    for (const p of ['daily', 'weekly', 'monthly']) {
        const r    = await fetch(`/api/history?period=${p}`);
        const data = await r.json();
        document.getElementById(`${p}-count`).textContent = data.length;
    }
    const r    = await fetch('/api/history?period=monthly');
    const data = await r.json();
    document.getElementById('history-table').innerHTML = data.slice(0, 50).map(row => `
        <tr>
            <td class="td-mono">${row.Timestamp}</td>
            <td class="td-class">${row.Class?.toUpperCase()}</td>
            <td class="td-conf">${row.Confidence}%</td>
        </tr>`).join('') ||
        '<tr><td colspan="3" style="text-align:center;font-family:var(--mono);font-size:10px;letter-spacing:2px;color:var(--text3);padding:24px">NO DATA YET</td></tr>';
}

function downloadReport(period) {
    window.location.href = `/download_report?period=${period}`;
}

let classChartInst = null, hourlyChartInst = null, confDistInst = null;

async function loadAnalytics() {
    const r    = await fetch('/api/history?period=monthly');
    const data = await r.json();

    if (classChartInst) { classChartInst.destroy(); classChartInst = null; }
    if (hourlyChartInst){ hourlyChartInst.destroy(); hourlyChartInst = null; }
    if (confDistInst)   { confDistInst.destroy();   confDistInst   = null; }

    if (!data.length) return;

    const classCounts = {};
    const hourlyData  = new Array(24).fill(0);
    data.forEach(d => {
        classCounts[d.Class] = (classCounts[d.Class] || 0) + 1;
        const h = new Date(d.Timestamp).getHours();
        if (!isNaN(h)) hourlyData[h]++;
    });

    classChartInst = new Chart(document.getElementById('class-chart').getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: Object.keys(classCounts),
            datasets: [{ data: Object.values(classCounts), backgroundColor: ['#ef4444','#3b82f6','#f59e0b','#10b981','#8b5cf6'], borderWidth: 0 }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 }, boxWidth: 10 } } } }
    });

    hourlyChartInst = new Chart(document.getElementById('hourly-chart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: Array.from({length: 24}, (_, i) => i.toString().padStart(2,'0') + 'h'),
            datasets: [{ data: hourlyData, backgroundColor: 'rgba(59,130,246,0.35)', borderColor: '#3b82f6', borderWidth: 1, borderRadius: 3 }]
        },
        options: { ...chartDefaults }
    });

    const confs = data.map(d => d.Confidence);
    confDistInst = new Chart(document.getElementById('conf-dist-chart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: data.slice(-20).map(d => d.Timestamp?.slice(11,16)),
            datasets: [{ data: confs.slice(-20), backgroundColor: 'rgba(239,68,68,0.35)', borderColor: '#ef4444', borderWidth: 1, borderRadius: 3 }]
        },
        options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, max: 100 } } }
    });
}
</script>
</body>
</html>
"""

if __name__ == '__main__':
    vs = VideoStream(src=IP_WEBCAM_URL).start()
    threading.Thread(target=ai_worker, args=(vs,), daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)