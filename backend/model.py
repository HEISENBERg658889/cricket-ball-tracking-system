"""
model.py — CrickTrack main processing pipeline.

Orchestrates:
  - BallDetector  (YOLO11)
  - BallTracker   (Kalman Filter)
  - Trajectory analysis (speed, bounce, swing, angles, classification)
  - On-screen rendering: ball tracking trajectory trail only (no HUD/markers/
    mini pitch-map — those overlays were removed; the underlying analysis
    data is still computed and saved to analysis.json / returned to the API)

Modular architecture — each concern lives in tracker_modules/.
"""

import cv2
import json
import os
import sys
import statistics

# ── Make tracker_modules importable when called from the backend dir ───────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker_modules import (
    # utils
    build_homography, get_fps,
    # detector
    BallDetector,
    # tracker
    BallTracker,
    # trajectory
    compute_release_speed, detect_bounce,
    estimate_swing, compute_release_angle, compute_bounce_angle,
    classify_length, classify_line,
    MAX_EARLY_POSITIONS,
    # renderer — trajectory trail only (see process_video for why the rest
    # of the HUD/marker/minimap overlays were dropped from the render loop)
    draw_trajectory_trail,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# Minimum frames between two deliveries (debounce)
MIN_FRAMES_BETWEEN_DELIVERIES = 60
# Minimum detections for a trajectory to be analysed
MIN_DELIVERY_FRAMES = 8

# ── Module-level singletons (loaded once) ─────────────────────────────────────
_detector = None

def _get_detector() -> BallDetector:
    global _detector
    if _detector is None:
        _detector = BallDetector()
    return _detector


# ── Main pipeline ─────────────────────────────────────────────────────────────
def process_video(input_path: str, output_path: str) -> dict:
    """
    Full end-to-end processing pipeline.

    1. Reads the video frame-by-frame.
    2. Detects the ball with YOLO.
    3. Tracks it with a Kalman filter.
    4. Accumulates delivery positions.
    5. On ball loss: computes speed, bounce, swing, angles, classification.
    6. Renders the ball tracking trajectory trail onto each frame and writes
       the output video.
    7. Returns a JSON-serialisable analysis dict.
    """
    detector = _get_detector()
    if detector.model is None:
        return {"error": "Model not loaded. Ensure models/best.pt is present."}

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = get_fps(cap)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐 Resolution: {width}×{height}   Frames: {total_frames}   FPS: {fps:.2f}")

    # Build homography once (pixel → ground plane in metres)
    H = build_homography(width, height)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Pipeline objects
    tracker = BallTracker()

    # Resolve the progress-status hook once, outside the hot loop, instead of
    # re-importing main on every single frame.
    _progress_status = None
    if total_frames > 0:
        try:
            import main as _main
            _progress_status = _main.processing_status
        except Exception:
            _progress_status = None

    # ── State ────────────────────────────────────────────────────────────────
    frame_number         = 0
    trajectory_data      = []   # full frame-by-frame log
    deliveries           = []   # one dict per confirmed delivery
    delivery_id          = 1
    last_delivery_frame  = -MIN_FRAMES_BETWEEN_DELIVERIES

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # ── Progress report ───────────────────────────────────────────────────
        if _progress_status is not None:
            _progress_status["progress"] = int(frame_number / total_frames * 100)

        # ── Detect ───────────────────────────────────────────────────────────
        detection = detector.detect(frame)

        # ── Track ────────────────────────────────────────────────────────────
        pos = tracker.update(detection, frame_number)

        if pos:
            trajectory_data.append({"frame": frame_number, "x": pos[0], "y": pos[1]})

        # ── Ball truly lost → finalise delivery ───────────────────────────────
        # (Bounce/speed/swing/angles are computed ONCE here from the complete
        # delivery trajectory — this is the analysis pipeline, and it is
        # untouched by the render-loop simplification below.)
        if tracker.is_lost and tracker.in_delivery:
            dp = tracker.delivery_positions
            if len(dp) >= MIN_DELIVERY_FRAMES:
                _finalise_delivery(
                    dp, H, fps, height, width,
                    delivery_id, frame_number,
                    last_delivery_frame, deliveries,
                )
                if deliveries:
                    last_delivery_frame  = frame_number
                    delivery_id += 1

            tracker.reset_delivery()

        # ── Render: ball tracking trajectory ONLY ─────────────────────────────
        # All other on-screen overlays (speed/length/line HUD, bounce/release
        # markers, predicted arc, mini pitch-map) were removed on request —
        # the analysis data for them still lives in `deliveries` / analysis.json,
        # it just isn't drawn onto the video anymore.
        traj = tracker.traj_points
        if len(traj) >= 2:
            draw_trajectory_trail(frame, traj)

        out.write(frame)

    # ── Handle delivery still in progress at end of video ────────────────────
    dp = tracker.delivery_positions
    if tracker.in_delivery and len(dp) >= MIN_DELIVERY_FRAMES:
        _finalise_delivery(
            dp, H, fps, height, width,
            delivery_id, frame_number,
            last_delivery_frame, deliveries,
        )

    cap.release()
    out.release()

    if not trajectory_data:
        return {"error": "No ball detected in video. Try a clearer cricket video."}

    # ── Fill missing speeds (None / "—") with the median of the deliveries ───
    # that *were* computed reliably. Median (not mean) so a couple of fast or
    # slow outlier balls don't drag the fill-in value off-centre. Each filled
    # value is flagged with speed_estimated=True so the API/UI can show it
    # differently from a directly-measured speed if you want to.
    valid_speeds = [d["speed"] for d in deliveries if isinstance(d["speed"], (int, float))]
    if valid_speeds:
        fallback_speed = round(statistics.median(valid_speeds), 1)
        for d in deliveries:
            if d["speed"] is None:
                d["speed"] = fallback_speed
                d["speed_estimated"] = True

    analysis = {"trajectory": trajectory_data, "deliveries": deliveries}
    path = os.path.join(BASE_DIR, "analysis.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)

    n_estimated = sum(1 for d in deliveries if d.get("speed_estimated"))
    print(
        f"✅ Saved → {path}  |  {len(deliveries)} deliveries detected"
        + (f"  ({n_estimated} speeds filled in via median imputation)" if n_estimated else "")
    )
    return analysis


# ── Delivery finalisation helper ──────────────────────────────────────────────
def _finalise_delivery(
    delivery_positions: list,
    H,
    fps: float,
    height: int,
    width: int,
    delivery_id: int,
    frame_number: int,
    last_delivery_frame: int,
    deliveries: list,
):
    """
    Compute all analytics for one delivery and append to *deliveries*.
    Called when the ball is confirmed lost (end of delivery).
    """
    traj_list = [p for p, _ in delivery_positions]

    # ── Bounce detection ──────────────────────────────────────────────────────
    bounce_idx = detect_bounce(traj_list, H)

    if bounce_idx is not None:
        bx, by = traj_list[bounce_idx]
    else:
        # Fallback: lowest point in image (highest y value)
        fallback = max(traj_list, key=lambda p: p[1])
        bx, by   = fallback
        bounce_idx = traj_list.index(fallback)

    # ── Speed ─────────────────────────────────────────────────────────────────
    early = delivery_positions[:MAX_EARLY_POSITIONS]
    spd   = compute_release_speed(early, fps, H, debug=True)
    if spd is None and len(delivery_positions) >= 4:
        spd = compute_release_speed(delivery_positions[:6], fps, H, debug=False)

    # ── Full classification (5 length zones + 5 line zones) ──────────────────
    delivery_type = classify_length(by, height)
    line          = classify_line(bx, width)
    swing         = estimate_swing(traj_list, bounce_idx, H)
    release_angle = compute_release_angle(delivery_positions)
    bounce_angle  = compute_bounce_angle(traj_list, bounce_idx, H)

    # ── Cooldown guard ────────────────────────────────────────────────────────
    frames_since = frame_number - last_delivery_frame
    if frames_since < 60:
        return

    speed_val = spd   # None when not reliably computable — never fabricated
    record = {
        "ball":          delivery_id,
        "speed":         speed_val,
        "length":        delivery_type,
        "line":          line,
        "swing":         swing,
        "release_angle": release_angle,
        "bounce_angle":  bounce_angle,
        "bounce_x":      round(bx / width,  3),
        "bounce_y":      round(by / height, 3),
    }
    deliveries.append(record)
    speed_txt = f"{speed_val} km/h" if speed_val is not None else "speed N/A"
    print(
        f"🏏 Ball {delivery_id}: {delivery_type} | {line} | "
        f"{speed_txt} | {swing} | "
        f"Release {release_angle}° | Bounce {bounce_angle}°"
    )
