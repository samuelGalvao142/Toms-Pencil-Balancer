"""
Calibration test tool — click to move the table.

Opens a cv2 workspace window. Left-click anywhere inside the workspace square
to send that position to the servo via the TPS calibration model. The green dot
tracks the last command sent.

Run:
    python -m src.system.actuator.test_calibration
    python -m src.system.actuator.test_calibration --port /dev/ttyUSB1
    python -m src.system.actuator.test_calibration --half-side 0.06
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import serial

from src.shared import ControlInput
from src.system.actuator.mech import MechanismTPS, MechanismTPSParams, MECHANISM_TPS_PRESETS


# ── Rendering ─────────────────────────────────────────────────────────────────

CANVAS_SIZE  = 500
MARGIN_PX    = 30
WINDOW_NAME  = "Calibration Test"


def _world_to_px(x: float, y: float, x_ref: float, y_ref: float, scale: float) -> tuple[int, int]:
    cx = CANVAS_SIZE // 2
    return int(cx + (x - x_ref) * scale), int(cx - (y - y_ref) * scale)


def _px_to_world(px: int, py: int, x_ref: float, y_ref: float, scale: float) -> tuple[float, float]:
    cx = CANVAS_SIZE // 2
    return x_ref + (px - cx) / scale, y_ref - (py - cx) / scale


def _render(
    half_side: float,
    x_ref: float,
    y_ref: float,
    scale: float,
    command: ControlInput | None,
) -> np.ndarray:
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 30, dtype=np.uint8)
    cx = CANVAS_SIZE // 2

    grid_color    = (55, 55, 55)
    border_color  = (110, 110, 110)
    cross_color   = (90, 90, 90)
    label_color   = (130, 130, 130)
    command_color = (0, 220, 80)

    # Grid lines at 1 cm intervals
    grid_step_m = 0.01
    n = int(np.ceil(half_side / grid_step_m)) + 1
    for k in range(-n, n + 1):
        # vertical
        px, _ = _world_to_px(x_ref + k * grid_step_m, y_ref, x_ref, y_ref, scale)
        if 0 <= px < CANVAS_SIZE:
            cv2.line(canvas, (px, 0), (px, CANVAS_SIZE - 1), grid_color, 1)
        # horizontal
        _, py = _world_to_px(x_ref, y_ref + k * grid_step_m, x_ref, y_ref, scale)
        if 0 <= py < CANVAS_SIZE:
            cv2.line(canvas, (0, py), (CANVAS_SIZE - 1, py), grid_color, 1)

    # Workspace square border
    s_px = int(half_side * scale)
    top_left  = (cx - s_px, cx - s_px)
    bot_right = (cx + s_px, cx + s_px)
    cv2.rectangle(canvas, top_left, bot_right, border_color, 1)

    # Center crosshair
    arm = 12
    cv2.line(canvas, (cx - arm, cx), (cx + arm, cx), cross_color, 1)
    cv2.line(canvas, (cx, cx - arm), (cx, cx + arm), cross_color, 1)

    # Axis labels (cm)
    for k in range(-int(half_side * 100), int(half_side * 100) + 1, 2):
        val_m = k * 0.01
        if abs(val_m) > half_side:
            continue
        px, _ = _world_to_px(x_ref + val_m, y_ref, x_ref, y_ref, scale)
        _, py = _world_to_px(x_ref, y_ref + val_m, x_ref, y_ref, scale)
        if k != 0:
            cv2.putText(canvas, f"{k}", (px - 8, cx + 14),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, label_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{k}", (cx + 4, py + 5),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, label_color, 1, cv2.LINE_AA)

    # Current command dot
    if command is not None:
        cx_cmd, cy_cmd = _world_to_px(command.px_cmd, command.py_cmd, x_ref, y_ref, scale)
        cv2.circle(canvas, (cx_cmd, cy_cmd), 6, command_color, -1)
        cv2.circle(canvas, (cx_cmd, cy_cmd), 8, command_color, 1)
        label = f"({command.px_cmd * 100:+.1f}, {command.py_cmd * 100:+.1f}) cm"
        cv2.putText(canvas, label, (cx_cmd + 10, cy_cmd - 6),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, command_color, 1, cv2.LINE_AA)

    return canvas


# ── Serial helpers ────────────────────────────────────────────────────────────

def _send_angles(ser: serial.Serial, theta1_deg: float, theta4_deg: float) -> None:
    ser.write(f"CMD,{theta1_deg:.2f},{theta4_deg:.2f}\r\n".encode("utf-8"))
    ser.flush()


def _apply_command(
    cmd: ControlInput,
    mechanism: MechanismTPS,
    ser: serial.Serial,
) -> None:
    try:
        _, (t1, t4) = mechanism.command_geometry(cmd)
        _send_angles(ser, t1, t4)
    except Exception as e:
        print(f"  [warn] command_geometry failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port",             default="COM7")
    parser.add_argument("--baud",             type=int,   default=115200)
    parser.add_argument("--half-side", type=float, default=0.07,
                        help="half the square side in meters (default: 0.07 → 14×14 cm square)")
    parser.add_argument("--x-ref",     type=float, default=0.0)
    parser.add_argument("--y-ref",     type=float, default=0.0)
    args = parser.parse_args()

    half_side    = args.half_side
    x_ref, y_ref = args.x_ref, args.y_ref
    scale = (CANVAS_SIZE - 2 * MARGIN_PX) / (2 * half_side)

    # Build MechanismTPS (loads TPS calibration file automatically)
    preset = MECHANISM_TPS_PRESETS["default"]
    mech_params = MechanismTPSParams(
        O=preset["O"],
        B=preset["B"],
        la=preset["la"],
        lb=preset["lb"],
    )
    mechanism = MechanismTPS(mech_params)

    if mechanism._interp_theta1 is None:
        print("[warn] No TPS calibration file found — falling back to FK/IK.")
    else:
        print(f"Loaded TPS calibration: {mechanism._calibration_xy.shape[0]} points.")

    # Open serial
    print(f"Opening {args.port} at {args.baud} baud…")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)
    ser.write(b"MODE,EXP\r\n")
    ser.flush()
    print("Serial ready. Click in the window to move the table.")

    current_command: ControlInput | None = None

    def on_mouse(event, px, py, _flags, _userdata):
        nonlocal current_command
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x, y = _px_to_world(px, py, x_ref, y_ref, scale)

        # Clamp to workspace square
        cx_new = float(np.clip(x - x_ref, -half_side, half_side)) + x_ref
        cy_new = float(np.clip(y - y_ref, -half_side, half_side)) + y_ref
        clamped = (cx_new != x or cy_new != y)
        x, y = cx_new, cy_new
        if clamped:
            print(f"Clamped → ({x * 100:+.2f}, {y * 100:+.2f}) cm")
        else:
            print(f"Moving  → ({x * 100:+.2f}, {y * 100:+.2f}) cm")

        cmd = ControlInput(px_cmd=x, py_cmd=y)
        _apply_command(cmd, mechanism, ser)
        current_command = cmd

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_SIZE, CANVAS_SIZE)

    # Must imshow before setMouseCallback — Qt backend creates the handle on first draw
    frame = _render(half_side, x_ref, y_ref, scale, current_command)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            frame = _render(half_side, x_ref, y_ref, scale, current_command)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
