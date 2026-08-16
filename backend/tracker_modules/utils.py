"""
utils.py — Shared utility functions for CrickTrack pipeline.

Covers:
  - Savitzky-Golay smoothing (no scipy dep)
  - Homography build / pixel-to-ground transform
  - FPS extraction
  - Angle computation helpers
"""

import math
import cv2
import numpy as np

# ── Real-world pitch constants (metres) ─────────────────────────────────────
PITCH_LENGTH_M = 20.12   # bowling crease to bowling crease
CREASE_WIDTH_M  = 2.64   # full crease line width (horizontal reference)


# ── Savitzky-Golay smoother (pure numpy, no scipy) ───────────────────────────
def savgol_smooth(values: list, window: int = 7, poly: int = 2) -> list:
    """
    Smooth a 1-D list with a Savitzky-Golay-style polynomial fit.
    Falls back to the raw values if there are too few points.
    """
    n = len(values)
    if n < window:
        return list(values)
    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = values[lo:hi]
        x   = list(range(len(seg)))
        try:
            coeffs = np.polyfit(x, seg, poly)
            center = i - lo
            val    = sum(coeffs[-(k + 1)] * (center ** k) for k in range(poly + 1))
        except Exception:
            val = values[i]
        smoothed.append(float(val))
    return smoothed


# ── Homography calibration ────────────────────────────────────────────────────
def build_homography(width: int, height: int) -> np.ndarray:
    """
    Build a pixel → ground-plane homography for the standard side-on
    broadcast feed used by this app (TATA IPL broadcast template).

    Four pitch-corner pixel fractions are mapped to real-world (metres):
      TL → (0, 0)              (bowling end, leg side)
      TR → (CREASE_WIDTH_M, 0) (bowling end, off side)
      BL → (0, PITCH_LENGTH_M) (batting end, leg side)
      BR → (CREASE_WIDTH_M, PITCH_LENGTH_M)

    These fractions were measured directly off a real reference frame from
    this broadcast feed (far crease ≈27% down the frame spanning x 42-56%;
    near crease ≈66% down spanning x 43-57%) — NOT a generic guess. The
    previous values assumed the crease spanned ~40-50% of the frame width;
    it actually only spans ~14%, which was throwing every speed calculation
    off by roughly 3x. Re-measure and update these if the camera framing
    changes (different zoom/crop/broadcast template).
    """
    frac = {
        "TL": (0.418, 0.268),
        "TR": (0.560, 0.268),
        "BL": (0.432, 0.655),
        "BR": (0.572, 0.655),
    }
    src = np.float32([
        [frac["TL"][0] * width,  frac["TL"][1] * height],
        [frac["TR"][0] * width,  frac["TR"][1] * height],
        [frac["BL"][0] * width,  frac["BL"][1] * height],
        [frac["BR"][0] * width,  frac["BR"][1] * height],
    ])
    dst = np.float32([
        [0,              0              ],
        [CREASE_WIDTH_M, 0              ],
        [0,              PITCH_LENGTH_M ],
        [CREASE_WIDTH_M, PITCH_LENGTH_M ],
    ])
    H, _ = cv2.findHomography(src, dst)
    return H


def pixel_to_ground(pt: tuple, H: np.ndarray) -> tuple:
    """Map a single pixel (x, y) to ground-plane coordinates (metres) via H."""
    p  = np.float32([[pt]]).reshape(-1, 1, 2)
    gp = cv2.perspectiveTransform(p, H)
    return float(gp[0][0][0]), float(gp[0][0][1])


# ── FPS extraction ────────────────────────────────────────────────────────────
def get_fps(cap: cv2.VideoCapture, default: float = 30.0) -> float:
    """Extract FPS from VideoCapture; fall back to *default* if invalid."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 1000:
        print(f"⚠️  Invalid FPS from metadata ({fps}), defaulting to {default}")
        return default
    print(f"📹 Video FPS (from metadata): {fps:.3f}")
    return fps


# ── Angle helpers ─────────────────────────────────────────────────────────────
def compute_angle_deg(p1: tuple, p2: tuple) -> float:
    """Angle (degrees) of the vector from p1 to p2 relative to horizontal."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def angle_between_vectors(v1: tuple, v2: tuple) -> float:
    """Angle (degrees) between two 2-D vectors."""
    dot   = v1[0] * v2[0] + v1[1] * v2[1]
    mag1  = math.hypot(*v1) + 1e-9
    mag2  = math.hypot(*v2) + 1e-9
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))
