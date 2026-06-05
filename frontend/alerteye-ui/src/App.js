import { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";
const WS  = "ws://127.0.0.1:8000";

// ── Utility ──────────────────────────────────────────────────────────────────
const fmt = (n) => (n * 100).toFixed(1) + "%";
const now = () => new Date().toLocaleTimeString("en-US", { hour12: false });

// ── Alert Banner ──────────────────────────────────────────────────────────────
function AlertBanner({ alerts }) {
  if (!alerts.length) return null;
  const latest = alerts[0];
  return (
    <div className="alert-banner">
      <span className="alert-icon">⚠</span>
      <span className="alert-text">
        WEAPON DETECTED — <strong>{latest.class.toUpperCase()}</strong> &nbsp;|&nbsp;
        Confidence: <strong>{fmt(latest.confidence)}</strong> &nbsp;|&nbsp;
        {latest.time}
      </span>
      <span className="alert-pulse" />
    </div>
  );
}

// ── Stat Card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, accent }) {
  return (
    <div className={`stat-card ${accent ? "stat-card--accent" : ""}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

// ── Detection Row ─────────────────────────────────────────────────────────────
function DetectionRow({ det, index }) {
  const bar = Math.round(det.confidence * 100);
  return (
    <div className="det-row" style={{ animationDelay: `${index * 60}ms` }}>
      <div className="det-header">
        <span className="det-class">{det.class.toUpperCase()}</span>
        <span className="det-conf">{fmt(det.confidence)}</span>
      </div>
      <div className="det-bar-track">
        <div className="det-bar-fill" style={{ width: `${bar}%` }} />
      </div>
      <div className="det-meta">{det.time}</div>
    </div>
  );
}

// ── Live Feed Tab ─────────────────────────────────────────────────────────────
function LiveFeed({ onDetection }) {
  const [cameraUrl, setCameraUrl]   = useState("");
  const [status, setStatus]         = useState("idle");
  const [frame, setFrame]           = useState(null);
  const [fps, setFps]               = useState(0);
  const [muted, setMuted]           = useState(false);
  const [alertFrame, setAlertFrame] = useState(null);

  const wsRef    = useRef(null);
  const fpsRef   = useRef({ count: 0, last: Date.now() });
  const audioRef = useRef(null);
  const mutedRef = useRef(false);

  const toggleMute = () => {
    setMuted(m => { mutedRef.current = !m; return !m; });
  };

  const playBeep = useCallback(() => {
    if (mutedRef.current) return;
    try {
      if (!audioRef.current)
        audioRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const ctx  = audioRef.current;
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880;
      osc.type = "sine";
      gain.gain.setValueAtTime(0.4, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.4);
    } catch {}
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current) { wsRef.current.close(); }
    setStatus("connecting");
    fpsRef.current = { count: 0, last: Date.now() };

    const cameraParam = cameraUrl.trim()
      ? `?camera=${encodeURIComponent(cameraUrl.trim())}`
      : "";

    const ws = new WebSocket(`${WS}/detect/stream${cameraParam}`);
    wsRef.current = ws;

    ws.onopen  = () => setStatus("live");
    ws.onerror = () => setStatus("error");
    ws.onclose = () => { setStatus("idle"); setFps(0); };

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.error) { setStatus("error"); return; }

      const frameUrl = "data:image/jpeg;base64," + data.frame;
      setFrame(frameUrl);

      fpsRef.current.count++;
      const ts = Date.now();
      if (ts - fpsRef.current.last >= 1000) {
        setFps(fpsRef.current.count);
        fpsRef.current.count = 0;
        fpsRef.current.last  = ts;
      }

      if (data.weapon_detected) {
        setAlertFrame(frameUrl);
        playBeep();
        data.detections.forEach(d => onDetection({ ...d, time: now() }));
      }
    };
  }, [onDetection, cameraUrl, playBeep]);

  const disconnect = () => {
    wsRef.current?.close();
    setStatus("idle");
    setFrame(null);
    setFps(0);
  };

  const downloadAlert = () => {
    if (!alertFrame) return;
    const a = document.createElement("a");
    a.href = alertFrame;
    a.download = `alert_${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`;
    a.click();
  };

  useEffect(() => () => wsRef.current?.close(), []);

  return (
    <div className="feed-panel">
      <div className="panel-header">
        <span className="panel-title">LIVE CAMERA FEED</span>
        <span className={`status-dot status-dot--${status}`} />
        <span className="status-label">{status.toUpperCase()}</span>
      </div>

      <div className="feed-viewport">
        {frame
          ? <img src={frame} alt="live" className="feed-img" />
          : <div className="feed-placeholder">
              <div className="feed-placeholder-icon">◎</div>
              <div className="feed-placeholder-text">
                {status === "connecting" ? "CONNECTING…" : "NO SIGNAL"}
              </div>
            </div>
        }
        {status === "live" && <div className="feed-rec-badge">● REC</div>}
        {status === "live" && <div className="feed-fps-badge">{fps} FPS</div>}
      </div>

      <div className="camera-input-row">
        <input
          className="camera-input"
          type="text"
          placeholder="IP Camera URL (leave empty for webcam)"
          value={cameraUrl}
          onChange={e => setCameraUrl(e.target.value)}
          disabled={status === "live"}
        />
      </div>

      <div className="feed-controls">
        {status !== "live"
          ? <button className="btn btn--primary" onClick={connect}>CONNECT CAMERA</button>
          : <button className="btn btn--danger"  onClick={disconnect}>DISCONNECT</button>
        }
        <button
          className={`btn btn--secondary${muted ? " btn--muted" : ""}`}
          onClick={toggleMute}
          title={muted ? "Unmute alerts" : "Mute alerts"}
        >
          {muted ? "🔇 MUTED" : "🔊 SOUND"}
        </button>
        <button
          className="btn btn--secondary"
          onClick={downloadAlert}
          disabled={!alertFrame}
          title="Download last alert frame"
        >
          SAVE ALERT
        </button>
      </div>
    </div>
  );
}

// ── Image Upload Tab ──────────────────────────────────────────────────────────
function ImageUpload({ onDetection }) {
  const [preview,  setPreview]  = useState(null);
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const inputRef = useRef();

  const handleFile = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setLoading(true);

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res  = await fetch(`${API}/detect/image`, { method: "POST", body: fd });
      const data = await res.json();
      setResult(data);
      if (data.weapon_detected) {
        data.detections.forEach(d => onDetection({ ...d, time: now() }));
      }
    } catch {
      alert("Backend unreachable. Make sure the server is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-panel">
      <div className="panel-header">
        <span className="panel-title">IMAGE ANALYSIS</span>
      </div>

      <div
        className="upload-drop"
        onClick={() => inputRef.current.click()}
      >
        {preview
          ? <img src={
              result?.annotated_image
                ? "data:image/jpeg;base64," + result.annotated_image
                : preview
            } alt="preview" className="upload-preview" />
          : <>
              <div className="upload-icon">⊕</div>
              <div className="upload-hint">CLICK TO UPLOAD IMAGE</div>
              <div className="upload-sub">JPG, PNG, WEBP supported</div>
            </>
        }
        {loading && <div className="upload-overlay">ANALYZING…</div>}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleFile}
      />

      {result && (
        <div className={`upload-result ${result.weapon_detected ? "upload-result--alert" : "upload-result--clear"}`}>
          {result.weapon_detected
            ? `⚠ ${result.detections.length} WEAPON(S) DETECTED`
            : "✓ NO WEAPONS DETECTED"}
        </div>
      )}

      {preview && (
        <button className="btn btn--secondary" onClick={() => { setPreview(null); setResult(null); }}>
          CLEAR
        </button>
      )}
    </div>
  );
}

// ── Detection Log ─────────────────────────────────────────────────────────────
function DetectionLog({ detections, totalScanned, alerts }) {
  return (
    <div className="log-panel">
      <div className="panel-header">
        <span className="panel-title">DETECTION LOG</span>
        <span className="log-count">{detections.length} EVENTS</span>
      </div>

      <div className="stats-row">
        <StatCard label="TOTAL SCANNED"  value={totalScanned} />
        <StatCard label="THREATS FOUND"  value={alerts}       accent />
        <StatCard label="ACCURACY"       value="≥ 50%"        />
      </div>

      <div className="det-list">
        {detections.length === 0
          ? <div className="det-empty">NO DETECTIONS YET</div>
          : detections.map((d, i) => <DetectionRow key={i} det={d} index={i} />)
        }
      </div>
    </div>
  );
}

// ── Reports Tab ──────────────────────────────────────────────────────────────
function ReportsTab() {
  const [period,  setPeriod]  = useState("daily");
  const [data,    setData]    = useState([]);
  const [counts,  setCounts]  = useState({ daily: 0, weekly: 0, monthly: 0 });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadCounts = async () => {
      try {
        const [d, w, m] = await Promise.all([
          fetch(`${API}/api/history?period=daily`).then(r => r.json()),
          fetch(`${API}/api/history?period=weekly`).then(r => r.json()),
          fetch(`${API}/api/history?period=monthly`).then(r => r.json()),
        ]);
        setCounts({ daily: d.length, weekly: w.length, monthly: m.length });
      } catch {}
    };
    loadCounts();
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const r = await fetch(`${API}/api/history?period=${period}`);
        setData(await r.json());
      } catch { setData([]); }
      finally { setLoading(false); }
    };
    load();
  }, [period]);

  const download = () => window.open(`${API}/download_report?period=${period}`, "_blank");

  return (
    <div className="reports-panel">
      <div className="reports-summary">
        {[["daily", "TODAY"], ["weekly", "THIS WEEK"], ["monthly", "THIS MONTH"]].map(([p, label]) => (
          <div
            key={p}
            className={`summary-card ${period === p ? "summary-card--active" : ""}`}
            onClick={() => setPeriod(p)}
          >
            <div className="summary-count">{counts[p]}</div>
            <div className="summary-label">{label}</div>
            <div className="summary-sub">DETECTIONS</div>
          </div>
        ))}
      </div>

      <div className="reports-table-panel">
        <div className="panel-header">
          <span className="panel-title">DETECTION HISTORY — {period.toUpperCase()}</span>
          <button className="btn btn--primary reports-dl-btn" onClick={download}>
            ⬇ DOWNLOAD CSV
          </button>
        </div>

        {loading ? (
          <div className="reports-empty">LOADING…</div>
        ) : data.length === 0 ? (
          <div className="reports-empty">NO DETECTIONS FOR THIS PERIOD</div>
        ) : (
          <div className="reports-table-wrap">
            <table className="reports-table">
              <thead>
                <tr>
                  <th>TIMESTAMP</th>
                  <th>CLASS</th>
                  <th>CONFIDENCE</th>
                </tr>
              </thead>
              <tbody>
                {data.map((row, i) => (
                  <tr key={i} className="reports-row">
                    <td className="reports-td-mono">{row.Timestamp}</td>
                    <td className="reports-td-class">{row.Class?.toUpperCase()}</td>
                    <td className="reports-td-conf">{row.Confidence}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab,          setTab]          = useState("live");
  const [detections,   setDetections]   = useState([]);
  const [alerts,       setAlerts]       = useState([]);
  const [totalScanned, setTotalScanned] = useState(0);
  const [serverOk,     setServerOk]     = useState(false);

  // Health check
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/health`);
        setServerOk(r.ok);
      } catch { setServerOk(false); }
    };
    check();
    const id = setInterval(check, 5000);
    return () => clearInterval(id);
  }, []);

  const handleDetection = useCallback((det) => {
    setDetections(prev => [det, ...prev].slice(0, 50));
    setAlerts(prev => [det, ...prev].slice(0, 10));
    setTotalScanned(n => n + 1);
  }, []);

  return (
    <div className="app">
      {/* ── Top bar ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <span className="brand-eye">◉</span>
          <span className="brand-name">ALERT<span className="brand-accent">EYE</span></span>
        </div>
        <div className="topbar-center">
          AI-POWERED WEAPON DETECTION SYSTEM
        </div>
        <div className="topbar-right">
          <span className={`server-dot ${serverOk ? "server-dot--ok" : "server-dot--err"}`} />
          <span className="server-label">{serverOk ? "BACKEND ONLINE" : "BACKEND OFFLINE"}</span>
        </div>
      </header>

      {/* ── Alert banner ── */}
      <AlertBanner alerts={alerts} />

      {/* ── Tab nav ── */}
      <nav className="tab-nav">
        <button className={`tab-btn ${tab === "live"    ? "tab-btn--active" : ""}`} onClick={() => setTab("live")}>LIVE FEED</button>
        <button className={`tab-btn ${tab === "image"   ? "tab-btn--active" : ""}`} onClick={() => setTab("image")}>IMAGE SCAN</button>
        <button className={`tab-btn ${tab === "reports" ? "tab-btn--active" : ""}`} onClick={() => setTab("reports")}>REPORTS</button>
      </nav>

      {/* ── Main layout ── */}
      <main className={`main-grid ${tab === "reports" ? "main-grid--full" : ""}`}>
        <div className="col-left">
          {tab === "live"    && <LiveFeed    onDetection={handleDetection} />}
          {tab === "image"   && <ImageUpload onDetection={handleDetection} />}
          {tab === "reports" && <ReportsTab />}
        </div>
        {tab !== "reports" && (
          <div className="col-right">
            <DetectionLog
              detections={detections}
              totalScanned={totalScanned}
              alerts={alerts.length}
            />
          </div>
        )}
      </main>

      <footer className="footer">
        AlertEye © 2025 &nbsp;·&nbsp; Real-Time AI Weapon Detection &nbsp;·&nbsp; Powered by YOLOv8
      </footer>
    </div>
  );
}