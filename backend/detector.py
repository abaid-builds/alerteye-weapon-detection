from ultralytics import YOLO
import cv2
import numpy as np
import os
import clip
import torch
from PIL import Image
from collections import deque
import threading

# ── Temporal Filter ───────────────────────────────────────────────────────────
class TemporalFilter:
    def __init__(self, required_frames=2):
        self.history = deque(maxlen=6)
        self.required_frames = required_frames

    def is_confirmed(self, detected: bool) -> bool:
        self.history.append(detected)
        return sum(list(self.history)[-self.required_frames:]) >= self.required_frames

# ── Async CLIP Verifier ───────────────────────────────────────────────────────
class CLIPVerifier:
    def __init__(self, persist_frames=30):
        self.device = "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.labels = [
            "a weapon like a gun knife or rifle",
            "an everyday object like a phone remote control or tool"
        ]
        self.text = clip.tokenize(self.labels).to(self.device)

        # ── Async state ───────────────────────────────────────
        self.persist_frames   = persist_frames
        self.verified_classes = {}   # class_name → frames_remaining
        self.rejected_classes = {}   # class_name → frames_remaining
        self.last_crops       = {}   # class_name → last crop sent to CLIP
        self.lock             = threading.Lock()
        self._clip_thread     = None

    def _run_clip(self, class_name, crop_bgr):
        """Runs in background thread — never blocks main loop."""
        try:
            pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            image   = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _ = self.model(image, self.text)
                probs     = logits.softmax(dim=-1).cpu().numpy()[0]

            is_weapon = probs[0] > 0.50
            print(f"[CLIP] {class_name} → Weapon: {probs[0]:.2%} | Object: {probs[1]:.2%} | Verdict: {'✅ WEAPON' if is_weapon else '❌ REJECTED'}")
            with self.lock:
                if is_weapon:
                    self.verified_classes[class_name] = self.persist_frames
                    self.rejected_classes.pop(class_name, None)
                else:
                    self.rejected_classes[class_name] = self.persist_frames
                    self.verified_classes.pop(class_name, None)
        except Exception as e:
            print(f"CLIP error: {e}")

    def _crop_moved_significantly(self, class_name, new_crop):
        """Check if object moved enough to warrant a new CLIP check."""
        old_crop = self.last_crops.get(class_name)
        if old_crop is None:
            return True
        if old_crop.shape != new_crop.shape:
            return True
        diff = cv2.absdiff(
            cv2.resize(old_crop, (64, 64)),
            cv2.resize(new_crop, (64, 64))
        )
        return diff.mean() > 15  # significant movement threshold

    def is_weapon(self, frame, bbox, class_name) -> bool:
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        with self.lock:
            # ── Decrement persistence counters ────────────────
            for d in [self.verified_classes, self.rejected_classes]:
                for k in list(d.keys()):
                    d[k] -= 1
                    if d[k] <= 0:
                        del d[k]

            # ── Already verified → allow for N more frames ────
            if class_name in self.verified_classes:
                return True

            # ── Already rejected → block for N more frames ────
            if class_name in self.rejected_classes:
                return False

        # ── First time or object moved → send to CLIP async ───
        if self._crop_moved_significantly(class_name, crop):
            self.last_crops[class_name] = crop.copy()
            # Only start new thread if previous one finished
            if self._clip_thread is None or not self._clip_thread.is_alive():
                self._clip_thread = threading.Thread(
                    target=self._run_clip,
                    args=(class_name, crop.copy()),
                    daemon=True
                )
                self._clip_thread.start()

        # ── While CLIP is thinking → trust YOLO for now ───────
        # High confidence YOLO detection = allow through while CLIP decides
        return True

# ── Main Detector ─────────────────────────────────────────────────────────────
class WeaponDetector:
    def __init__(self, model_path="best.pt"):
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, model_path)
        self.model                = YOLO(full_path)
        self.confidence_threshold = 0.25
        self.temporal_filter      = TemporalFilter(required_frames=2)
        self.clip_verifier        = CLIPVerifier(persist_frames=30)
        self.non_weapon_classes   = {"person", "hand"}

    def is_valid_detection(self, box, frame_shape, conf):
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2  = box
        box_w = x2 - x1
        box_h = y2 - y1

        if (box_w * box_h) < (frame_w * frame_h * 0.01):
            return False

        aspect_ratio = box_h / box_w if box_w > 0 else 0
        if aspect_ratio > 8:
            return False

        return True

    def detect_image(self, image_bytes):
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame, detections, _ = self._run_detection(frame, use_temporal=False)
        return frame, detections

    def detect_frame(self, frame):
        return self._run_detection(frame, use_temporal=True)

    def _run_detection(self, frame, use_temporal=True):
        results      = self.model(frame, conf=self.confidence_threshold)[0]
        detections   = []
        weapon_found = False

        for box in results.boxes:
            confidence = float(box.conf[0])
            class_id   = int(box.cls[0])
            class_name = self.model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Filter 1 — non-weapon classes
            if class_name in self.non_weapon_classes:
                continue

            # Filter 2 — size + aspect ratio
            if not self.is_valid_detection(
                (x1, y1, x2, y2), frame.shape, confidence
            ):
                continue

            # Filter 3 — Async CLIP (non-blocking)
            if not self.clip_verifier.is_weapon(
                frame, (x1, y1, x2, y2), class_name
            ):
                continue

            weapon_found = True
            detections.append({
                "class":      class_name,
                "confidence": round(confidence, 2),
                "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Filter 4 — temporal consistency
        confirmed = (
            self.temporal_filter.is_confirmed(weapon_found)
            if use_temporal else weapon_found
        )

        return frame, detections, confirmed