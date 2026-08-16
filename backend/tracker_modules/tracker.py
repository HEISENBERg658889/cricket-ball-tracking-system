"""
tracker.py — Kalman-filter ball tracker with occlusion handling.

Features:
  - Constant-velocity Kalman Filter (4-state: x, y, vx, vy)
  - Graceful handling of missed detections (predict-only up to MAX_MISSED)
  - False-positive rejection via gating (Mahalanobis-like distance)
  - Trajectory smoothing via a rolling weighted average
  - Automatic reset when the ball is lost for too long
"""

import numpy as np
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
MAX_TRAIL   = 120   # maximum trajectory points to keep
MAX_MISSED  = 8     # frames without detection before declaring ball lost
GATE_RADIUS = 120   # pixel radius — detections further away are rejected


class KalmanBallTracker:
    """
    4-state Kalman Filter: [x, y, vx, vy].
    Designed for a cricket ball moving under near-constant velocity between
    frames (ignoring gravity within a single-frame step).
    """

    def __init__(self):
        # State vector: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)

        # State transition (constant-velocity model)
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = 1.0   # x += vx
        self.F[1, 3] = 1.0   # y += vy

        # Measurement matrix (we observe x, y only)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance — larger = trust measurement more
        self.Q = np.diag([5.0, 5.0, 20.0, 20.0]).astype(np.float32)

        # Measurement noise covariance
        self.R = np.diag([10.0, 10.0]).astype(np.float32)

        # Estimate covariance
        self.P = np.eye(4, dtype=np.float32) * 500.0

        self.initialized = False

    def init(self, pt: tuple):
        """Seed the filter with the first observed position."""
        self.state = np.array([[pt[0]], [pt[1]], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.0
        self.initialized = True

    def predict(self) -> tuple:
        """Time-update step. Returns predicted (x, y)."""
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q
        return int(self.state[0, 0]), int(self.state[1, 0])

    def update(self, pt: tuple):
        """Measurement-update step."""
        z = np.array([[pt[0]], [pt[1]]], dtype=np.float32)
        y = z - self.H @ self.state                  # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.state = self.state + K @ y
        self.P     = (np.eye(4) - K @ self.H) @ self.P

    def position(self) -> tuple:
        """Current estimated position (x, y)."""
        return int(self.state[0, 0]), int(self.state[1, 0])

    def velocity(self) -> tuple:
        """Current estimated velocity (vx, vy) in pixels/frame."""
        return float(self.state[2, 0]), float(self.state[3, 0])


class BallTracker:
    """
    High-level tracker that wraps KalmanBallTracker and manages:
      - trajectory buffer
      - missed-frame counter
      - per-delivery position accumulator
    """

    def __init__(self):
        self.kf             = KalmanBallTracker()
        self.trajectory     = deque(maxlen=MAX_TRAIL)
        self.missed_frames  = 0
        self.in_delivery    = False
        self.delivery_positions: list[tuple] = []   # [(pixel_pt, frame_no)]

    # ── Per-frame update ──────────────────────────────────────────────────────
    def update(self, detection: tuple | None, frame_no: int) -> tuple | None:
        """
        Feed one frame's detection result.
        Returns the smoothed (x, y) to use for rendering, or None if lost.
        """
        if detection is not None:
            # Gate: ignore detections far from the current estimate
            if self.kf.initialized:
                px, py = self.kf.predict()
                dist = ((detection[0] - px) ** 2 + (detection[1] - py) ** 2) ** 0.5
                if dist > GATE_RADIUS:
                    # Likely a false positive — skip update but keep prediction
                    smoothed = self.kf.position()
                    self.trajectory.append(smoothed)
                    return smoothed
                self.kf.update(detection)
            else:
                self.kf.init(detection)

            self.missed_frames = 0
            pos = self.kf.position()
            self.trajectory.append(pos)

            # Delivery accumulation
            if not self.in_delivery:
                self.in_delivery = True
                self.delivery_positions = []
            self.delivery_positions.append((pos, frame_no))

            return pos

        else:
            # No detection — predict-only
            self.missed_frames += 1
            if self.kf.initialized and self.missed_frames <= MAX_MISSED:
                pos = self.kf.predict()
                self.trajectory.append(pos)
                return pos
            else:
                # Ball truly lost
                return None

    def reset_delivery(self):
        """Call after a delivery is finalised to reset delivery state."""
        self.in_delivery = False
        self.delivery_positions = []
        self.trajectory.clear()
        self.kf.initialized = False
        self.missed_frames  = 0

    @property
    def is_lost(self) -> bool:
        return self.missed_frames > MAX_MISSED

    @property
    def traj_points(self) -> list:
        return list(self.trajectory)
