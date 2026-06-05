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

    def _open(self, src):
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Force MJPEG codec — unlocks high-res + high-fps on USB/IP cameras
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Request 1080p @ 60fps
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)

        # Log what the camera actually achieved
        actual_w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[Camera] Actual: {int(actual_w)}x{int(actual_h)} @ {actual_fps:.0f}fps")

        return cap

    def change_source(self, src):
        self.cap.release()
        self.src = src
        self.cap = self._open(src)

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with lock:
                    self.frame = frame

    def read(self):
        with lock:
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
    src  = int(src) if src.isdigit() else src
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
<html>
<head>
    <title>AlertEYE | Weapon Detection System</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:      #080c10;
            --bg2:     #0d1117;
            --bg3:     #111820;
            --border:  #1e2d3d;
            --red:     #e63946;
            --green:   #2cb67d;
            --accent:  #00b4d8;
            --text:    #c9d6e3;
            --dim:     #4a6177;
            --mono:    'Share Tech Mono', monospace;
            --head:    'Rajdhani', sans-serif;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: var(--bg); color: var(--text); font-family: var(--head); display: flex; height: 100vh; overflow: hidden; }

        nav { width: 64px; background: var(--bg2); border-right: 1px solid var(--border); display: flex; flex-direction: column; align-items: center; padding: 16px 0; gap: 4px; z-index: 100; transition: width .25s; }
        nav:hover { width: 180px; }
        .brand { font-family: var(--head); font-size: 18px; font-weight: 700; letter-spacing: 3px; color: var(--text); margin-bottom: 24px; white-space: nowrap; overflow: hidden; }
        .brand span { color: var(--red); }
        .nav-item { width: 100%; padding: 12px 0; display: flex; align-items: center; color: var(--dim); cursor: pointer; transition: .2s; position: relative; border-left: 3px solid transparent; }
        .nav-item svg { width: 20px; height: 20px; min-width: 64px; }
        .nav-item .label { opacity: 0; font-size: 13px; font-weight: 600; letter-spacing: 1.5px; white-space: nowrap; transition: .2s; }
        nav:hover .nav-item .label { opacity: 1; }
        .nav-item:hover { color: var(--text); background: rgba(255,255,255,.03); }
        .nav-item.active { color: var(--accent); border-left-color: var(--accent); background: rgba(0,180,216,.05); }

        main { flex: 1; overflow-y: auto; padding: 20px; display: none; }
        main.active { display: block; }

        .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 16px; }
        .card-title { font-family: var(--mono); font-size: 11px; letter-spacing: 3px; color: var(--accent); margin-bottom: 12px; }

        .live-grid { display: grid; grid-template-columns: 1fr 340px; gap: 16px; }
        .feed-wrap { position: relative; border-radius: 6px; overflow: hidden; border: 2px solid var(--border); background: #000; }
        .feed-wrap img { width: 100%; display: block; }
        .feed-wrap.alarmed { border-color: var(--red); box-shadow: 0 0 30px rgba(230,57,70,.3); }
        .rec-badge { position: absolute; top: 10px; right: 10px; background: rgba(230,57,70,.9); color: #fff; font-family: var(--mono); font-size: 10px; letter-spacing: 2px; padding: 3px 8px; border-radius: 2px; animation: blink 1.2s infinite; }
        @keyframes blink { 50% { opacity: 0; } }

        .stats-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-bottom: 16px; }
        .stat { background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; padding: 12px; text-align: center; }
        .stat-val { font-family: var(--mono); font-size: 26px; color: var(--text); }
        .stat-val.red { color: var(--red); }
        .stat-label { font-family: var(--mono); font-size: 9px; letter-spacing: 2px; color: var(--dim); margin-top: 4px; }

        .log-item { display: flex; align-items: center; gap: 10px; padding: 8px 10px; background: var(--bg3); border-left: 3px solid var(--red); border-radius: 2px; margin-bottom: 6px; animation: slidein .3s ease; }
        @keyframes slidein { from { opacity:0; transform: translateX(10px); } to { opacity:1; transform: translateX(0); } }
        .log-class { font-weight: 700; letter-spacing: 1px; color: var(--red); font-size: 13px; }
        .log-conf { font-family: var(--mono); font-size: 12px; color: #f4a261; margin-left: auto; }
        .log-time { font-family: var(--mono); font-size: 10px; color: var(--dim); }
        .log-empty { font-family: var(--mono); font-size: 11px; color: var(--dim); letter-spacing: 3px; text-align: center; padding: 30px; }

        .topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
        .page-title { font-size: 20px; font-weight: 700; letter-spacing: 3px; }
        .server-badge { display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 11px; color: var(--dim); }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); }
        .dot.red { background: var(--red); box-shadow: 0 0 8px var(--red); animation: blink .8s infinite; }

        .cam-input { display: flex; gap: 8px; margin-bottom: 12px; }
        .cam-input input { flex: 1; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 3px; font-family: var(--mono); font-size: 11px; outline: none; }
        .cam-input input:focus { border-color: var(--accent); }
        .btn { padding: 8px 16px; border: 1px solid; border-radius: 3px; font-family: var(--head); font-size: 12px; font-weight: 600; letter-spacing: 1.5px; cursor: pointer; transition: .15s; }
        .btn-accent { background: rgba(0,180,216,.1); border-color: var(--accent); color: var(--accent); }
        .btn-accent:hover { background: rgba(0,180,216,.2); }
        .btn-red { background: rgba(230,57,70,.1); border-color: var(--red); color: var(--red); }
        .btn-red:hover { background: rgba(230,57,70,.2); }

        .slider-row { display: flex; align-items: center; gap: 12px; margin-top: 10px; }
        .slider-row label { font-family: var(--mono); font-size: 11px; color: var(--dim); letter-spacing: 1px; white-space: nowrap; }
        input[type=range] { flex: 1; accent-color: var(--accent); }
        #conf-val { font-family: var(--mono); font-size: 13px; color: var(--accent); min-width: 36px; }

        .report-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 20px; }
        .report-card { background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; padding: 20px; text-align: center; }
        .report-card h3 { font-family: var(--mono); font-size: 12px; letter-spacing: 2px; color: var(--accent); margin-bottom: 8px; }
        .report-card .report-count { font-family: var(--mono); font-size: 32px; color: var(--text); margin: 8px 0; }
        .report-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .report-table th { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; color: var(--dim); padding: 8px; border-bottom: 1px solid var(--border); text-align: left; }
        .report-table td { padding: 10px 8px; border-bottom: 1px solid var(--bg3); }
        .report-table tr:hover td { background: var(--bg3); }

        .chart-wrap { position: relative; height: 200px; }
        canvas { max-height: 200px; }
        .res-badge { font-family: var(--mono); font-size: 10px; color: var(--dim); letter-spacing: 1px; margin-top: 6px; text-align: right; padding-right: 4px; }
    </style>
</head>
<body>

<nav>
    <div class="brand">◉ ALERT<span>EYE</span></div>
    <div class="nav-item active" onclick="showTab('live', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="15" rx="2"/><polyline points="17 2 12 7 7 2"/></svg>
        <span class="label">LIVE FEED</span>
    </div>
    <div class="nav-item" onclick="showTab('analytics', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
        <span class="label">ANALYTICS</span>
    </div>
    <div class="nav-item" onclick="showTab('reports', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
        <span class="label">REPORTS</span>
    </div>
    <div class="nav-item" onclick="showTab('settings', this)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        <span class="label">SETTINGS</span>
    </div>
</nav>

<!-- ── Live Feed Tab ── -->
<main id="tab-live" class="active">
    <div class="topbar">
        <div class="page-title">LIVE FEED</div>
        <div class="server-badge"><div class="dot" id="srv-dot"></div><span id="srv-label">BACKEND ONLINE</span></div>
    </div>
    <div class="cam-input">
        <input type="text" id="cam-url" placeholder="IP Camera URL or device index (leave empty for webcam)">
        <button class="btn btn-accent" onclick="connectCamera()">CONNECT</button>
        <button class="btn btn-red" onclick="disconnectCamera()">DISCONNECT</button>
    </div>
    <div class="live-grid">
        <div>
            <div class="feed-wrap" id="feed-wrap">
                <img src="/video_feed" id="feed-img">
                <div class="rec-badge">● REC</div>
            </div>
            <div class="res-badge" id="res-badge">STREAM: DETECTING…</div>
        </div>
        <div>
            <div class="card" style="margin-bottom:12px">
                <div class="card-title">DETECTION STATS</div>
                <div class="stats-row">
                    <div class="stat"><div class="stat-val" id="fps-val">0</div><div class="stat-label">FPS</div></div>
                    <div class="stat"><div class="stat-val red" id="det-val">0</div><div class="stat-label">THREATS</div></div>
                    <div class="stat"><div class="stat-val" id="total-val">0</div><div class="stat-label">TOTAL</div></div>
                </div>
                <div class="chart-wrap"><canvas id="conf-chart"></canvas></div>
            </div>
            <div class="card">
                <div class="card-title">RECENT ALERTS</div>
                <div id="alert-log"><div class="log-empty">NO DETECTIONS YET</div></div>
            </div>
        </div>
    </div>
</main>

<!-- ── Analytics Tab ── -->
<main id="tab-analytics">
    <div class="topbar"><div class="page-title">ANALYTICS</div></div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px;">
        <div class="card">
            <div class="card-title">DETECTIONS BY CLASS</div>
            <div class="chart-wrap"><canvas id="class-chart"></canvas></div>
        </div>
        <div class="card">
            <div class="card-title">HOURLY ACTIVITY</div>
            <div class="chart-wrap"><canvas id="hourly-chart"></canvas></div>
        </div>
    </div>
    <div class="card">
        <div class="card-title">CONFIDENCE DISTRIBUTION</div>
        <div class="chart-wrap"><canvas id="conf-dist-chart"></canvas></div>
    </div>
</main>

<!-- ── Reports Tab ── -->
<main id="tab-reports">
    <div class="topbar"><div class="page-title">REPORTS</div></div>
    <div class="report-grid">
        <div class="report-card">
            <h3>TODAY</h3>
            <div class="report-count" id="daily-count">0</div>
            <div style="font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:1px;margin-bottom:12px">DETECTIONS</div>
            <button class="btn btn-accent" onclick="downloadReport('daily')">⬇ DOWNLOAD</button>
        </div>
        <div class="report-card">
            <h3>THIS WEEK</h3>
            <div class="report-count" id="weekly-count">0</div>
            <div style="font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:1px;margin-bottom:12px">DETECTIONS</div>
            <button class="btn btn-accent" onclick="downloadReport('weekly')">⬇ DOWNLOAD</button>
        </div>
        <div class="report-card">
            <h3>THIS MONTH</h3>
            <div class="report-count" id="monthly-count">0</div>
            <div style="font-family:var(--mono);font-size:10px;color:var(--dim);letter-spacing:1px;margin-bottom:12px">DETECTIONS</div>
            <button class="btn btn-accent" onclick="downloadReport('monthly')">⬇ DOWNLOAD</button>
        </div>
    </div>
    <div class="card">
        <div class="card-title">DETECTION HISTORY</div>
        <table class="report-table">
            <thead><tr><th>TIMESTAMP</th><th>CLASS</th><th>CONFIDENCE</th></tr></thead>
            <tbody id="history-table"></tbody>
        </table>
    </div>
</main>

<!-- ── Settings Tab ── -->
<main id="tab-settings">
    <div class="topbar"><div class="page-title">SETTINGS</div></div>
    <div class="card" style="max-width:500px; margin-bottom:16px;">
        <div class="card-title">DETECTION THRESHOLD</div>
        <p style="font-size:13px;color:var(--dim);margin-bottom:12px">Higher threshold = fewer false positives but may miss some weapons</p>
        <div class="slider-row">
            <label>CONFIDENCE</label>
            <input type="range" min="0.10" max="0.90" step="0.05" value="0.25" id="conf-slider" oninput="updateConf(this.value)">
            <span id="conf-val">0.25</span>
        </div>
    </div>
    <div class="card" style="max-width:500px;">
        <div class="card-title">STREAM INFO</div>
        <div style="font-family:var(--mono);font-size:12px;color:var(--dim);line-height:2.2;">
            <div>INFERENCE RES &nbsp;: <span style="color:var(--accent)">640 × 480</span></div>
            <div>STREAM QUALITY : <span style="color:var(--accent)">JPEG Q92</span></div>
            <div>TARGET CAPTURE : <span style="color:var(--accent)">1920 × 1080 @ 60fps</span></div>
            <div style="margin-top:8px;font-size:10px;">Actual resolution depends on camera hardware support.</div>
        </div>
    </div>
</main>

<script>
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
    await fetch('/api/change_source', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source: val})
    });
    document.getElementById('feed-img').src = '/video_feed?' + Date.now();
}

function disconnectCamera() {
    document.getElementById('feed-img').src = '';
}

const feedImg = document.getElementById('feed-img');
feedImg.addEventListener('load', () => {
    const w = feedImg.naturalWidth;
    const h = feedImg.naturalHeight;
    if (w && h) document.getElementById('res-badge').textContent = `STREAM: ${w} × ${h}`;
});

function updateConf(val) {
    document.getElementById('conf-val').textContent = val;
    fetch('/api/set_conf/' + val);
}

const confChart = new Chart(document.getElementById('conf-chart').getContext('2d'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [{ label: 'Confidence %', borderColor: '#e63946', backgroundColor: 'rgba(230,57,70,0.1)', data: [], fill: true, tension: 0.4 }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
            y: { beginAtZero: true, max: 100, grid: { color: '#1e2d3d' }, ticks: { color: '#4a6177' } },
            x: { grid: { color: '#1e2d3d' }, ticks: { color: '#4a6177', maxTicksLimit: 6 } }
        },
        plugins: { legend: { display: false } },
        animation: false
    }
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
            <td style="font-family:var(--mono);font-size:12px">${row.Timestamp}</td>
            <td style="color:var(--red);font-weight:700">${row.Class?.toUpperCase()}</td>
            <td style="font-family:var(--mono);color:#f4a261">${row.Confidence}%</td>
        </tr>`).join('') ||
        '<tr><td colspan="3" style="text-align:center;color:var(--dim);font-family:var(--mono);padding:20px">NO DATA YET</td></tr>';
}

function downloadReport(period) {
    window.location.href = `/download_report?period=${period}`;
}

let classChartInst = null, confDistInst = null;

async function loadAnalytics() {
    const r    = await fetch('/api/history?period=monthly');
    const data = await r.json();
    if (!data.length) return;

    if (classChartInst) { classChartInst.destroy(); classChartInst = null; }
    if (confDistInst)   { confDistInst.destroy();   confDistInst   = null; }

    const classCounts = {};
    data.forEach(d => { classCounts[d.Class] = (classCounts[d.Class] || 0) + 1; });

    classChartInst = new Chart(document.getElementById('class-chart').getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: Object.keys(classCounts),
            datasets: [{ data: Object.values(classCounts), backgroundColor: ['#e63946','#00b4d8','#f4a261','#2cb67d','#9b5de5'] }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#c9d6e3' } } } }
    });

    const confs = data.map(d => d.Confidence);
    confDistInst = new Chart(document.getElementById('conf-dist-chart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: data.slice(-20).map(d => d.Timestamp?.slice(11,16)),
            datasets: [{ label: 'Confidence %', data: confs.slice(-20), backgroundColor: 'rgba(0,180,216,0.4)', borderColor: '#00b4d8', borderWidth: 1 }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 100, grid: { color: '#1e2d3d' }, ticks: { color: '#4a6177' } },
                x: { grid: { color: '#1e2d3d' }, ticks: { color: '#4a6177' } }
            },
            plugins: { legend: { display: false } }
        }
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