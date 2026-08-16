"""
detector.py — YOLO-based cricket ball detector.

Responsibilities:
  - Load the YOLO11 model once.
  - Run inference on a resized frame (for speed).
  - Return the best ball bounding-box centre or None.
  - Reject obviously-wrong detections (size, aspect ratio).
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best.pt")

CONF_THRESHOLD = 0.30   # minimum detection confidence
RESIZE_W       = 640    # inference width  (keeps aspect-ratio for standard feeds)
RESIZE_H       = 360    # inference height

# Reject detections whose bounding-box diagonal is outside this pixel range
# (relative to RESIZE_W × RESIZE_H — helps filter sponsor logos / crowd noise)
MIN_DIAG_PX = 4
MAX_DIAG_PX = 80


class BallDetector:
    """Wraps YOLO inference for cricket ball detection."""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"🚀 Detector using device: {self.device}")
        try:
            self.model = YOLO(MODEL_PATH)
            print("✅ YOLO model loaded")
        except Exception as exc:
            print(f"❌ Failed to load model: {exc}")
            self.model = None

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> tuple | None:
        """
        Run inference on *frame* (full-resolution BGR).
        Returns (cx, cy) in full-resolution pixel coords, or None.

        The frame is internally downscaled to RESIZE_W×RESIZE_H for speed,
        then the result is scaled back to the original resolution.
        """
        if self.model is None:
            return None

        h_full, w_full = frame.shape[:2]
        small  = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        result = self.model.predict(
            source=small, conf=CONF_THRESHOLD, verbose=False, device=self.device
        )[0]

        sx = w_full / RESIZE_W
        sy = h_full / RESIZE_H

        best_center = None
        best_conf   = -1.0

        if result.boxes is None:
            return None

        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            if int(cls) != 0:
                continue

            # Size sanity check on resized coords
            diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if not (MIN_DIAG_PX <= diag <= MAX_DIAG_PX):
                continue

            # Prefer highest-confidence detection
            if score > best_conf:
                best_conf   = score
                best_center = (
                    int((x1 + x2) / 2 * sx),
                    int((y1 + y2) / 2 * sy),
                )

        return best_center
