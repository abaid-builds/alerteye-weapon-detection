# AlertEye — Real-Time Weapon Detection System

A real-time AI-powered weapon detection system that monitors live video streams 
and instantly alerts operators upon detecting threats. Built with a multi-stage 
verification pipeline to minimize false positives in production environments.

![Live Detection](live-detection.png)

---

## The Problem

Standard CCTV systems record footage but cannot detect threats in real time. 
Security personnel monitoring multiple feeds simultaneously cannot realistically 
catch every incident. AlertEye solves this by automatically detecting weapons 
the moment they appear on camera and alerting operators instantly.

---

## Key Features

- **Real-time detection** at 15-16 FPS on live video streams
- **Multi-stage AI pipeline** — YOLOv8 detection + OpenAI CLIP semantic verification
- **Temporal consistency filter** — eliminates single-frame false positives
- **Dual dashboard** — React + FastAPI WebSocket and standalone Flask fallback
- **Instant alerts** — visual red border flash + 880Hz audio beep on detection
- **Analytics** — detections by class, hourly activity, confidence distribution
- **Reports** — full detection history with timestamps, confidence scores, CSV export
- **Auto snapshots** — saves JPEG of exact frame where weapon was detected
- **Configurable** — adjustable confidence threshold via settings panel

---

## Multi-Stage AI Pipeline

```text
Live Video Stream
      ↓
Stage 1: YOLOv8 — object detection, class/size/aspect ratio filtering
      ↓
Stage 2: OpenAI CLIP (ViT-B/32) — semantic verification
         "Is this actually a weapon or a phone/remote/tool?"
      ↓
Stage 3: Temporal Filter — confirms detection across 2 consecutive frames
      ↓
Alert Triggered
```

This three-stage approach significantly reduces false positives compared to single-model detection systems.

---

## Screenshots

### Live Feed — Pistol Detected
![Live Detection](live-detection.png)

### Detection Snapshot — Auto Saved
![Detection Snapshot](detection-snapshot.png)

### Analytics Dashboard
![Analytics](analytics.png)

### Reports & Detection History
![Reports](reports.png)

### Settings — Confidence Threshold
![Settings](settings.png)

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI / ML | YOLOv8 (Ultralytics), OpenAI CLIP (ViT-B/32), PyTorch |
| Computer Vision | OpenCV |
| Backend (Primary) | Python, FastAPI, WebSocket |
| Backend (Fallback) | Python, Flask, MJPEG streaming |
| Frontend | React 19, JavaScript, CSS3, Web Audio API |
| Data | Pandas, CSV, JPEG snapshots |
| Inference Backends | PyTorch, TensorRT, OpenVINO |

---

## Detection Classes

- Pistol
- Rifle
- Knife
- Explosive
- Shotgun

---

## Architecture

**Dual-backend design:**
- FastAPI serves the React SPA via WebSocket for real-time bidirectional 
  communication
- Flask provides a self-contained fallback dashboard requiring zero frontend 
  setup — run instantly with a single Python command

**Performance optimizations:**
- Detection runs every 5th frame, bounding boxes drawn on all frames using 
  latest result
- `cap.grab()` + `cap.retrieve()` instead of `cap.read()` to always serve 
  the latest frame
- ThreadPoolExecutor keeps detection off the async event loop
- CSV writes protected by `threading.Lock()` to prevent file corruption

**Inference flexibility:**
- Switch between PyTorch, TensorRT, and OpenVINO backends with a single 
  line change

---

## Target Use Cases

- Educational institutions
- Banks and financial institutions
- Government buildings
- Shopping malls and public spaces
- Any environment with existing CCTV infrastructure

---

## Team

Built as a Technopreneurship course project at Superior University, Lahore.

---

## Note

This repository contains project documentation and screenshots only. 
Source code is available upon request for academic or professional review.
