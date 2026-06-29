"""
TPS actuator calibrator — interactive dense-grid servo calibration.

For each grid point the servo auto-moves to the nominal FK/IK position; the
operator WASD-nudges the end effector to the desired physical position by eye,
then presses Enter to record (desired_xy, servo_xy, theta1, theta4).

No cameras or DVS involved — purely servo + terminal.

Calibration domain is a square of side 2*half_side centered at (0,0).  Only
interpolation is supported — all live points must lie inside this square.

Usage:
    python -m src.system.actuator.calibrator
    python -m src.system.actuator.calibrator --grid-step 0.02
    python -m src.system.actuator.calibrator --grid-step 0.005 --half-side 0.06
"""

from __future__ import annotations

import argparse
import curses
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import serial

from src.shared import ControlInput
from src.system.actuator.mech import (
    Mechanism, MechanismParams, MECHANISM_PRESETS,
    MechanismTPS, MechanismTPSParams, MECHANISM_TPS_PRESETS,
)


# ── Grid ──────────────────────────────────────────────────────────────────────

def generate_grid(half_side_m: float, step_m: float) -> list[np.ndarray]:
    """
    Row-by-row snake scan over a square [-half_side, +half_side]^2.
    Grid starts at -half_side with the given step; any point that would fall
    beyond +half_side is clamped to the edge (duplicates removed).
    Odd rows reverse direction to minimize servo travel between rows.
    """
    raw = np.arange(-half_side_m, half_side_m + step_m, step_m)
    vals = np.unique(np.clip(raw, -half_side_m, half_side_m))
    pts: list[np.ndarray] = []
    for row_idx, y in enumerate(vals):
        row = [np.array([x, y]) for x in vals]
        if row_idx % 2 == 1:
            row = row[::-1]
        pts.extend(row)
    return pts


# ── Serial helpers ─────────────────────────────────────────────────────────────

def _serial_send(ser: serial.Serial, cmd: str) -> None:
    ser.write((cmd.strip() + "\r\n").encode("utf-8"))
    ser.flush()


def _send_angles(ser: serial.Serial, theta1_deg: float, theta4_deg: float) -> None:
    _serial_send(ser, f"CMD,{theta1_deg:.2f},{theta4_deg:.2f}")


def _get_angles(mechanism: Mechanism, servo_xy: np.ndarray) -> tuple[float, float]:
    _, angles = mechanism.command_geometry(ControlInput(float(servo_xy[0]), float(servo_xy[1])))
    return angles  # (theta1_deg, theta4_deg)


# ── Smooth move ────────────────────────────────────────────────────────────────

def _smooth_move(
    mechanism: Mechanism,
    ser: serial.Serial,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    duration_s: float = 0.4,
) -> None:
    start_xy = np.asarray(start_xy, dtype=float)
    end_xy   = np.asarray(end_xy,   dtype=float)
    if np.allclose(start_xy, end_xy):
        t1, t4 = _get_angles(mechanism, end_xy)
        _send_angles(ser, t1, t4)
        return
    n = max(int(np.ceil(duration_s / 0.02)), 1)
    for i in range(1, n + 1):
        alpha = i / n
        xy = (1.0 - alpha) * start_xy + alpha * end_xy
        t1, t4 = _get_angles(mechanism, xy)
        _send_angles(ser, t1, t4)
        if i < n:
            time.sleep(0.02)


# ── Warm-start seeds ──────────────────────────────────────────────────────────

def _load_existing(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if data.get("version") != "v1_tps_actuator":
            return {}
        return {_xy_key(np.array(r["desired_xy_m"])): r for r in data["points"]}
    except Exception:
        return {}


def _xy_key(xy: np.ndarray) -> str:
    return f"{xy[0]:.6f},{xy[1]:.6f}"


# ── Curses UI ─────────────────────────────────────────────────────────────────

def _draw(stdscr, row: int, text: str) -> None:
    h, w = stdscr.getmaxyx()
    if row < h:
        stdscr.addstr(row, 0, text[: max(0, w - 1)])


def _calibration_loop(
    stdscr,
    *,
    mechanism: Mechanism,
    ser: serial.Serial,
    grid_points: list[np.ndarray],
    nudge_step_m: float,
    existing: dict[str, dict],
    args,
) -> list[dict]:
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(50)

    accepted: list[dict] = []
    last_action = "Starting calibration."
    current_xy = np.zeros(2)

    total = len(grid_points)
    idx = 0

    while idx < total:
        desired_xy = grid_points[idx]
        key_str = _xy_key(desired_xy)

        saved = existing.get(key_str)
        if args.cmd == "interpolation":
            seed_xy  = desired_xy.copy()
            seed_src = "desired (interpolation mode)"
        else:
            seed_xy  = np.array(saved["servo_xy_m"], dtype=float) if saved else desired_xy.copy()
            seed_src = "saved" if saved else "FK/IK nominal"

        servo_xy = seed_xy.copy()
        _smooth_move(mechanism, ser, current_xy, servo_xy)
        current_xy = servo_xy.copy()
        last_action = f"Moved to {seed_src} seed."

        while True:
            key = stdscr.getch()
            dx = dy = 0.0

            if key != -1:
                if key in (ord("w"), ord("W"), curses.KEY_UP):
                    dy = +nudge_step_m;  last_action = "Nudged +Y"
                elif key in (ord("s"), ord("S"), curses.KEY_DOWN):
                    dy = -nudge_step_m;  last_action = "Nudged -Y"
                elif key in (ord("a"), ord("A"), curses.KEY_LEFT):
                    dx = -nudge_step_m;  last_action = "Nudged -X"
                elif key in (ord("d"), ord("D"), curses.KEY_RIGHT):
                    dx = +nudge_step_m;  last_action = "Nudged +X"
                elif key in (ord("r"), ord("R")):
                    servo_xy = seed_xy.copy()
                    _smooth_move(mechanism, ser, current_xy, servo_xy, duration_s=0.2)
                    current_xy = servo_xy.copy()
                    last_action = "Reset to seed."
                elif key in (ord("b"), ord("B")):
                    idx = max(0, idx - 1)
                    last_action = "Back to previous point."
                    break
                elif key == ord(" "):
                    t1, t4 = _get_angles(mechanism, servo_xy)
                    accepted.append({
                        "desired_xy_m": desired_xy.tolist(),
                        "servo_xy_m":   servo_xy.tolist(),
                        "theta1_deg":   t1,
                        "theta4_deg":   t4,
                        "skipped":      False,
                    })
                    idx += 1
                    last_action = "Accepted."
                    break
                elif key in (10, 13, curses.KEY_ENTER):
                    t1, t4 = _get_angles(mechanism, servo_xy)
                    accepted.append({
                        "desired_xy_m": desired_xy.tolist(),
                        "servo_xy_m":   servo_xy.tolist(),
                        "theta1_deg":   t1,
                        "theta4_deg":   t4,
                        "skipped":      True,
                    })
                    idx += 1
                    last_action = "Skipped (recorded as-is)."
                    break
                elif key in (ord("q"), ord("Q")):
                    return accepted

                if dx != 0.0 or dy != 0.0:
                    servo_xy = servo_xy + np.array([dx, dy])
                    t1, t4 = _get_angles(mechanism, servo_xy)
                    _send_angles(ser, t1, t4)
                    current_xy = servo_xy.copy()

            # ── Render ───────────────────────────────────────────────────────
            t1, t4 = _get_angles(mechanism, servo_xy)
            stdscr.erase()
            _draw(stdscr, 0, f"TPS Actuator Calibrator  [{idx + 1}/{total}]")
            _draw(stdscr, 1, "WASD/arrows: nudge  Space: accept  Enter: skip  R: reset  B: back  Q: quit+save")
            _draw(stdscr, 2, f"Grid step: {args.grid_step * 1000:.0f} mm   Nudge: {nudge_step_m * 1000:.1f} mm   Half-side: {args.half_side * 100:.1f} cm   Cmd: {args.cmd}")
            _draw(stdscr, 4, f"Target:  ({desired_xy[0]:+.4f}, {desired_xy[1]:+.4f}) m  =  ({desired_xy[0]*100:+.2f}, {desired_xy[1]*100:+.2f}) cm")
            _draw(stdscr, 5, f"Command: ({servo_xy[0]:+.4f}, {servo_xy[1]:+.4f}) m")
            _draw(stdscr, 6, f"Angles:  th1={t1:.2f} deg   th4={t4:.2f} deg")
            _draw(stdscr, 8, f"Seed: {seed_src}")
            _draw(stdscr, 9, f"Accepted so far: {len(accepted)}/{total}")
            _draw(stdscr, 10, f"Last action: {last_action}")
            stdscr.refresh()

    return accepted

def _auto_calibrate(
    stdscr,
    *,
    mechanism: Mechanism,
    ser: serial.Serial,
    grid_points: list[np.ndarray],
    nudge_step_m: float,
    existing: dict[str, dict],
    args,
) -> list[dict]:
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(50)

    accepted: list[dict] = []
    last_action = "Starting calibration."
    current_xy = np.zeros(2)

    total = len(grid_points)
    idx = 0

    while idx < total:
        desired_xy = grid_points[idx]
        key_str = _xy_key(desired_xy)

        saved = existing.get(key_str)
        if args.cmd == "interpolation":
            seed_xy  = desired_xy.copy()
            seed_src = "desired (interpolation mode)"
        else:
            seed_xy  = np.array(saved["servo_xy_m"], dtype=float) if saved else desired_xy.copy()
            seed_src = "saved" if saved else "FK/IK nominal"

        servo_xy = seed_xy.copy()
        _smooth_move(mechanism, ser, current_xy, servo_xy)
        current_xy = servo_xy.copy()
        last_action = f"Moved to {seed_src} seed."

        while True:
            key = stdscr.getch()
            dx = dy = 0.0

            if key != -1:
                if key in (ord("w"), ord("W"), curses.KEY_UP):
                    dy = +nudge_step_m*10;  last_action = "Nudged +Y"
                elif key in (ord("s"), ord("S"), curses.KEY_DOWN):
                    dy = -nudge_step_m*10;  last_action = "Nudged -Y"
                elif key in (ord("a"), ord("A"), curses.KEY_LEFT):
                    dx = -nudge_step_m*10;  last_action = "Nudged -X"
                elif key in (ord("d"), ord("D"), curses.KEY_RIGHT):
                    dx = +nudge_step_m*10;  last_action = "Nudged +X"
                elif key in (ord("r"), ord("R")):
                    servo_xy = seed_xy.copy()
                    _smooth_move(mechanism, ser, current_xy, servo_xy, duration_s=0.2)
                    current_xy = servo_xy.copy()
                    last_action = "Reset to seed."
                elif key in (ord("b"), ord("B")):
                    idx = max(0, idx - 1)
                    last_action = "Back to previous point."
                    break
                elif key == ord(" "):
                    t1, t4 = _get_angles(mechanism, servo_xy)
                    accepted.append({
                        "desired_xy_m": desired_xy.tolist(),
                        "servo_xy_m":   servo_xy.tolist(),
                        "theta1_deg":   t1,
                        "theta4_deg":   t4,
                        "skipped":      False,
                    })
                    idx += 1
                    last_action = "Accepted."
                    break
                elif key in (10, 13, curses.KEY_ENTER):
                    t1, t4 = _get_angles(mechanism, servo_xy)
                    accepted.append({
                        "desired_xy_m": desired_xy.tolist(),
                        "servo_xy_m":   servo_xy.tolist(),
                        "theta1_deg":   t1,
                        "theta4_deg":   t4,
                        "skipped":      True,
                    })
                    idx += 1
                    last_action = "Skipped (recorded as-is)."
                    break
                elif key in (ord("q"), ord("Q")):
                    return accepted

                if dx != 0.0 or dy != 0.0:
                    servo_xy = servo_xy + np.array([dx, dy])
                    t1, t4 = _get_angles(mechanism, servo_xy)
                    _send_angles(ser, t1, t4)
                    current_xy = servo_xy.copy()

            # ── Render ───────────────────────────────────────────────────────
            t1, t4 = _get_angles(mechanism, servo_xy)
            stdscr.erase()
            _draw(stdscr, 0, f"TPS Actuator Calibrator  [{idx + 1}/{total}]")
            _draw(stdscr, 1, "WASD/arrows: nudge  Space: accept  Enter: skip  R: reset  B: back  Q: quit+save")
            _draw(stdscr, 2, f"Grid step: {args.grid_step * 1000:.0f} mm   Nudge: {nudge_step_m * 1000:.1f} mm   Half-side: {args.half_side * 100:.1f} cm   Cmd: {args.cmd}")
            _draw(stdscr, 4, f"Target:  ({desired_xy[0]:+.4f}, {desired_xy[1]:+.4f}) m  =  ({desired_xy[0]*100:+.2f}, {desired_xy[1]*100:+.2f}) cm")
            _draw(stdscr, 5, f"Command: ({servo_xy[0]:+.4f}, {servo_xy[1]:+.4f}) m")
            _draw(stdscr, 6, f"Angles:  th1={t1:.2f} deg   th4={t4:.2f} deg")
            _draw(stdscr, 8, f"Seed: {seed_src}")
            _draw(stdscr, 9, f"Accepted so far: {len(accepted)}/{total}")
            _draw(stdscr, 10, f"Last action: {last_action}")
            stdscr.refresh()

    return accepted


# ── Save ──────────────────────────────────────────────────────────────────────

def _save(output_path: Path, accepted: list[dict], args, existing: dict[str, dict]) -> None:
    # Merge: existing points as base, newly accepted points override by key.
    merged = {**existing, **{_xy_key(np.array(p["desired_xy_m"])): p for p in accepted}}
    points = list(merged.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version":            "v1_tps_actuator",
        "updated_at_utc":     datetime.now(timezone.utc).isoformat(),
        "grid_step_m":          args.grid_step,
        "workspace_half_side_m": args.half_side,
        "n_points":           len(points),
        "points":             points,
        "metadata": {
            "port": args.port,
            "tool": "src.system.actuator.calibrator",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(points)} points ({len(accepted)} new/updated, {len(existing)} preserved) -> {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive TPS actuator calibration -- WASD dense-grid tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--grid-step",  type=float, default=0.01,
                        help="grid spacing in meters (default: 0.01 = 10 mm)")
    parser.add_argument("--half-side",  type=float, default=0.07,
                        help="half the square side in meters (default: 0.07 → 14×14 cm square)")
    parser.add_argument("--cmd",        choices=["kinematics", "interpolation"], default="kinematics",
                        help="command model: 'kinematics' uses FK/IK (default), 'interpolation' uses saved RBF calibration")
    parser.add_argument("--nudge-step",       type=float, default=0.001,
                        help="WASD nudge size in meters (default: 0.001 = 1 mm)")
    parser.add_argument("--port",             default="/dev/ttyUSB0",
                        help="serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--baud",             type=int, default=115200,
                        help="baud rate (default: 115200)")
    parser.add_argument("--output",           default="src/system/actuator/mech/five_bar_tps_calibration.json",
                        help="output JSON file path")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    # Build command mechanism
    if args.cmd == "interpolation":
        mech_params = MechanismTPSParams(**{
            k: v for k, v in MECHANISM_TPS_PRESETS["default"].items()
            if k != "calibration_file"
        }, calibration_file=args.output)
        mechanism = MechanismTPS(mech_params)
        mechanism.set_workspace_offset(0.0, 0.0)
        print(f"Command model: RBF interpolation from {output_path}")
    else:
        mech_params = MechanismParams(**{
            k: v for k, v in MECHANISM_PRESETS["default"].items()
            if k != "calibration_file"
        }, calibration_file=None)
        mechanism = Mechanism(mech_params)
        mechanism.set_calibration_enabled(False)
        mechanism.set_workspace_offset(0.0, 0.0)
        print("Command model: FK/IK kinematics")

    # Open serial
    print(f"Opening {args.port} at {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)
    _serial_send(ser, "MODE,EXP")
    print("Serial ready.")

    grid_points = generate_grid(args.half_side, args.grid_step)
    existing    = _load_existing(output_path)
    print(f"Grid: {len(grid_points)} points  |  Warm-start hits: {len(existing)}")

    try:
        accepted = curses.wrapper(
            lambda stdscr: _calibration_loop(
                stdscr,
                mechanism=mechanism,
                ser=ser,
                grid_points=grid_points,
                nudge_step_m=args.nudge_step,
                existing=existing,
                args=args,
            )
        )
    except KeyboardInterrupt:
        accepted = []
    finally:
        ser.close()

    if accepted:
        _save(output_path, accepted, args, existing)
    else:
        print("No points accepted; nothing saved.")


if __name__ == "__main__":
    main()
