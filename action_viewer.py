#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "opencv-python", "imageio[ffmpeg]", "physicalai[so101]"]
# ///
"""
Interactive action viewer — scrub through a recorded video frame-by-frame
while sending the corresponding joint actions to the SO-101 robot arm.

Controls:
  ← / →   Previous / next frame   (or A / D)
  R        Toggle robot actuation on/off
  Q / Esc  Quit

Usage:
    uv run action_viewer.py <dataset_dir>
    uv run action_viewer.py <video.mp4> <actions.npy>
    uv run action_viewer.py <video.mp4> <actions.npy> \\
        --port /dev/ttyACM0 --calibration ~/.lerobot/calibration/so101.json
    uv run action_viewer.py <dataset_dir> --no-robot
"""

import argparse
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np

PANEL_W = 640
PANEL_H = 480
INFO_H  = 100
WIN_W   = PANEL_W * 2
WIN_H   = PANEL_H + INFO_H

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex",  "wrist_roll",    "gripper"]

# Hardcoded SO-101 calibration
_CALIBRATION_DATA = {
    "shoulder_pan":  dict(id=1, drive_mode=0, homing_offset=-577,  range_min=711,  range_max=3307),
    "shoulder_lift": dict(id=2, drive_mode=0, homing_offset=-472,  range_min=820,  range_max=3005),
    "elbow_flex":    dict(id=3, drive_mode=0, homing_offset=1593,  range_min=907,  range_max=3133),
    "wrist_flex":    dict(id=4, drive_mode=0, homing_offset=1813,  range_min=704,  range_max=3150),
    "wrist_roll":    dict(id=5, drive_mode=0, homing_offset=1009,  range_min=7,    range_max=4091),
    "gripper":       dict(id=6, drive_mode=0, homing_offset=-1332, range_min=1969, range_max=3352),
}

def _make_calibration():
    from physicalai.robot.so101.calibration import SO101Calibration, SO101JointCalibration
    return SO101Calibration(joints={
        name: SO101JointCalibration(**vals)
        for name, vals in _CALIBRATION_DATA.items()
    })

# BGR colours
BG     = (20,  20,  20)
ACCENT = (80,  200, 120)
WARN   = (60,  100, 220)
WHITE  = (240, 240, 240)
GRAY   = (130, 130, 130)
DARK   = (40,  40,  40)

# cv2.waitKeyEx arrow codes (Linux X11)
KEY_LEFT  = 65361
KEY_RIGHT = 65363
KEY_ESC   = 27


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_video_frames(path: Path) -> list[np.ndarray]:
    """Load all frames as HWC uint8 RGB arrays."""
    reader = imageio.get_reader(str(path), "ffmpeg")
    frames = [np.array(f) for f in reader]
    reader.close()
    return frames


def fit_frame(frame_rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize + letterbox RGB frame into a w×h BGR panel."""
    src_h, src_w = frame_rgb.shape[:2]
    scale = min(w / src_w, h / src_h)
    nw, nh = int(src_w * scale), int(src_h * scale)
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    panel = np.full((h, w, 3), DARK, dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    panel[y0:y0 + nh, x0:x0 + nw] = bgr
    return panel


def placeholder(text: str, w: int, h: int) -> np.ndarray:
    panel = np.full((h, w, 3), DARK, dtype=np.uint8)
    cv2.putText(panel, text, (w // 2 - len(text) * 4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1, cv2.LINE_AA)
    return panel


def draw_text(canvas: np.ndarray, text: str, x: int, y: int,
              color=WHITE, scale: float = 0.45, thickness: int = 1) -> int:
    """Draw text and return the y position below it."""
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
    return y + int(18 * scale / 0.45)


# ── Robot ─────────────────────────────────────────────────────────────────────

def try_connect_robot(port: str, calibration_path: str | None):
    try:
        from physicalai.robot.so101 import SO101
        if calibration_path:
            cal = calibration_path  # SO101 accepts a path string directly
        else:
            cal = _make_calibration()
            print("  Using hardcoded calibration")
        robot = SO101(port=port, calibration=cal)
        robot.connect()
        print(f"  SO-101 connected on {port}")
        return robot
    except Exception as e:
        print(f"  Could not connect: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video",   nargs="?")
    parser.add_argument("actions", nargs="?")
    parser.add_argument("--port",        default="/dev/ttyACM1")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--camera",      type=int, default=0)
    parser.add_argument("--no-robot",    action="store_true")
    args = parser.parse_args()

    if args.video is None:
        parser.print_help()
        sys.exit(1)

    video_path   = Path(args.video)
    actions_path = Path(args.actions) if args.actions else None

    if video_path.is_dir():
        candidates   = sorted(video_path.glob("videos/**/*.mp4"))
        if not candidates:
            print("No .mp4 found under videos/ in dataset dir")
            sys.exit(1)
        video_path   = candidates[0]
        actions_path = Path(args.video) / "actions.npy"

    if not video_path.exists():
        print(f"Video not found: {video_path}"); sys.exit(1)
    if actions_path is None or not actions_path.exists():
        print(f"Actions not found: {actions_path}"); sys.exit(1)

    print(f"Loading video:   {video_path}")
    video_frames = load_video_frames(video_path)

    print(f"Loading actions: {actions_path}")
    actions = np.load(str(actions_path))
    n_frames, action_dim = actions.shape

    n = min(len(video_frames), n_frames)
    if n < len(video_frames) or n < n_frames:
        print(f"  Trimming to {n} frames (video={len(video_frames)}, actions={n_frames})")
    video_frames, actions, n_frames = video_frames[:n], actions[:n], n

    print(f"  {n_frames} frames, {action_dim} dims")

    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("  Warning: camera unavailable")
        cap = None

    robot = None
    if not args.no_robot:
        print(f"Connecting robot on {args.port}...")
        robot = try_connect_robot(args.port, args.calibration)

    robot_active = robot is not None
    frame_idx    = 0

    def send(idx: int) -> None:
        if robot and robot_active:
            try:
                robot.send_action(actions[idx].astype(np.float32))
            except Exception as e:
                print(f"  Send error: {e}")

    def build_canvas(idx: int, cam_bgr: np.ndarray | None) -> np.ndarray:
        canvas = np.full((WIN_H, WIN_W, 3), BG, dtype=np.uint8)

        # ── Panels ──
        src_panel = fit_frame(video_frames[idx], PANEL_W, PANEL_H)
        canvas[:PANEL_H, :PANEL_W] = src_panel

        if cam_bgr is not None:
            cam_rgb   = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
            cam_panel = fit_frame(cam_rgb, PANEL_W, PANEL_H)
        else:
            cam_panel = placeholder("no camera", PANEL_W, PANEL_H)
        canvas[:PANEL_H, PANEL_W:] = cam_panel

        # ── Panel labels ──
        draw_text(canvas, "SOURCE VIDEO", 8,          14, GRAY, 0.4)
        draw_text(canvas, "ROBOT CAMERA", PANEL_W + 8, 14, GRAY, 0.4)

        # ── Divider ──
        canvas[PANEL_H, :] = (60, 60, 60)

        # ── Info bar ──
        info_y = PANEL_H + 4

        # Progress bar
        bar_x, bar_y, bar_h = 8, info_y + 4, 5
        bar_w_total = WIN_W - 16
        pct = idx / max(n_frames - 1, 1)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + bar_w_total, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(canvas, (bar_x, bar_y),
                      (bar_x + int(bar_w_total * pct), bar_y + bar_h), ACCENT, -1)

        # Frame counter
        y = draw_text(canvas, f"Frame {idx + 1}/{n_frames}   <- ->  navigate   R  toggle robot",
                      8, info_y + 22, WHITE, 0.45, 1)

        # Robot status
        if robot:
            status, col = ("ROBOT ON", ACCENT) if robot_active else ("ROBOT OFF", WARN)
        else:
            status, col = "NO ROBOT", GRAY
        (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        draw_text(canvas, status, WIN_W - tw - 10, info_y + 22, col, 0.5, 1)

        # Joint values
        joint_labels = JOINT_NAMES if action_dim >= 6 else [f"d{i}" for i in range(action_dim)]
        vals = "  ".join(
            f"{joint_labels[i]}: {actions[idx, i]:+7.2f}"
            for i in range(min(action_dim, len(joint_labels)))
        )
        draw_text(canvas, vals, 8, y + 4, GRAY, 0.38, 1)

        return canvas

    cv2.namedWindow("Action Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Action Viewer", WIN_W, WIN_H)

    send(frame_idx)

    while True:
        # Grab camera frame
        cam_bgr = None
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cam_bgr = frame

        canvas = build_canvas(frame_idx, cam_bgr)
        cv2.imshow("Action Viewer", canvas)

        key = cv2.waitKeyEx(33)  # ~30 fps; waitKeyEx for arrow keys
        if key in (KEY_ESC, ord('q'), ord('Q')):
            break
        elif key in (KEY_RIGHT, ord('d'), ord('D')):
            frame_idx = min(frame_idx + 1, n_frames - 1)
            send(frame_idx)
        elif key in (KEY_LEFT, ord('a'), ord('A')):
            frame_idx = max(frame_idx - 1, 0)
            send(frame_idx)
        elif key in (ord('r'), ord('R')):
            robot_active = not robot_active
            if robot_active:
                send(frame_idx)

    if cap is not None:
        cap.release()
    if robot is not None:
        try:
            robot.disconnect()
            print("Robot disconnected.")
        except Exception:
            pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
