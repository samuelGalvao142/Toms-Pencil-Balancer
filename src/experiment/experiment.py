from dataclasses import dataclass, field
import time
from pathlib import Path

import numpy as np

from src.experiment.logger import TerminalInfo
from src.experiment.log_paths import allocate_run_log_dir
from src.shared import TimingParams, default_timing


EXPERIMENT_PRESETS = {
    "sim": {
        "system":              "default:simple_sim",
        "logger":              "default:default",
        "stop_condition":      "max_steps:default",
        "realtime_visualizer": "null:default",
        "offline_visualizer":  "3d:default",
        "progress":            "default:default",
        "pacing":              "null:default",
        "scheduler":           "realtime:default",
        "n_trials":            1,
    },
    "new_sim": {
        "base": "sim",
        "system":"default:new_sim",
        "stop_condition":  "max_steps:default",
        "realtime_visualizer": "null:default",
        "offline_visualizer":  "3d:default",
    },
    "real_new_sim": {
        "base": "sim",
        "system":"real:real_new_sim",
        "stop_condition":  "infinite:default",
        "realtime_visualizer": "real_ws:default",
        "pacing":              "realtime:default",
        "offline_visualizer":  "null:default",
    },
    "montecarlo": {
        "base": "sim",
        "system":"default:new_sim",
        "stop_condition":  "any:early_stop",
        "realtime_visualizer": "null:default",
        "offline_visualizer":  "null:default",
        "n_trials":            10,
    },
    "test_sim_dvs": {
        "base": "sim",
        "system":"default:placing_only",
        "realtime_visualizer": "real_ws:default",
        "offline_visualizer":  "3d:default",
    },
    "placing": {
        "base": "sim",
        "system":"default:placing_only",
    },
    "dynamic_sim": {
        "base": "sim",
        "system":"default:dynamic_sim",
        "offline_visualizer":  "3d:default",
    },
    "realtime_sim": {
        "base": "sim",
        "realtime_visualizer": "sim:default",
        "offline_visualizer":  "null:default",
        "pacing":              "realtime:default",
    },
    "real_vision": {
        "base": "sim",
        "system":              "real:real_vision",
        "stop_condition":      "infinite:default",
        "realtime_visualizer": "real_ws:default",
        "offline_visualizer":  "null:default",
        "pacing":              "realtime:default",
    },
    "real": {
        "base":   "real_vision",
        "system": "real:real",
    },
    "real_supervised": {
        "base":   "real_vision",
        "system": "real:real_supervised",
    },
    "real_supervised_dynamic": {
        "base":   "real_vision",
        "system": "real:real_dynamic_supervised",
    },
}


@dataclass
class ExperimentParams:
    system:               object
    logger:               object
    stop_condition:       object
    realtime_visualizer:  object
    offline_visualizer:   object
    progress:             object
    pacing:               object
    scheduler:            object
    n_trials:             int
    timing:               TimingParams = field(default_factory=default_timing)


class Experiment:
    def __init__(self, params: ExperimentParams):
        p = params

        self.system              = p.system
        self.logger               = p.logger
        self.stop_condition       = p.stop_condition
        self.realtime_visualizer  = p.realtime_visualizer
        self.offline_visualizer   = p.offline_visualizer
        self.progress             = p.progress
        self.pacing               = p.pacing
        self.scheduler            = p.scheduler
        self.dt                   = getattr(self.scheduler, "dt", p.timing.dt)
        self.n_trials             = p.n_trials
        self._run_log_dir: Path | None = None

        if hasattr(self.pacing, "dt"):
            self.pacing.dt = self.dt

        if hasattr(self.realtime_visualizer, "_event_frames_fn"):
            if self.realtime_visualizer._event_frames_fn is None and hasattr(
                self.system.sensor, "get_event_accumulator_frames"
            ):
                self.realtime_visualizer._event_frames_fn = (
                    self.system.sensor.get_event_accumulator_frames
                )
        self._configure_run_log_dir()

    def _supervisor_title(self) -> str | None:
        state_name = getattr(self.system.supervisor, "state_name", None)
        if state_name == "ACQUISITION":
            return (
                "Ready | "
                f"WASD: tilt trim {self._tilt_trim_text()} | "
                f"{self._recording_hint_text()}"
            )
        if state_name in {"STABILIZATION", "STABILIZING", "stabilizing"}:
            return (
                "Stabilizing | "
                f"WASD: tilt trim {self._tilt_trim_text()} | "
                f"{self._recording_hint_text()}"
            )
        if state_name == "BALANCED":
            return (
                "Balanced | "
                f"WASD: trim {self._tilt_trim_text()} | "
                f"{self._recording_hint_text()}"
            )
        return None

    def _tilt_trim_text(self) -> str:
        angle_offset = getattr(self.system.supervisor, "measurement_angle_offset", (0.0, 0.0))
        ax_deg, ay_deg = np.rad2deg(np.asarray(angle_offset, dtype=float).reshape(2))
        return f"ax={ax_deg:+.1f} ay={ay_deg:+.1f} deg"

    def _recording_hint_text(self) -> str:
        sensor = getattr(self.system, "sensor", None)
        if sensor is None or not bool(getattr(sensor, "recording_enabled", False)):
            return "R: rec unavailable"
        if bool(getattr(sensor, "recording_active", False)):
            return "REC ON | R: stop rec"
        return "REC OFF | R: start rec"

    def run_trial(self):
        self.reset()
        i = 0
        while not self.stop_condition.should_stop(i, self.system.x, self.dt):
            control_tick = self.scheduler.should_actuate()
            self.system.step(self.dt, control_tick=control_tick)
            self.logger.record(self.system.step_data)
            self.scheduler.tick()
            if self.scheduler.should_render():
                meas = getattr(self.system.sensor, "last_line_observation", None)
                vr = self.realtime_visualizer.render(
                    measurement=meas,
                    command=self.system.u,
                    y_meas=self.system.last_y_meas,
                    paused=False,
                    title=self._supervisor_title(),
                )
                if vr.key is not None and hasattr(self.system.supervisor, "handle_key"):
                    if not self._handle_runtime_key(vr.key):
                        self.system.supervisor.handle_key(vr.key)
                if vr.quit:
                    break
            self.pacing.pace()
            i += 1

        if hasattr(self.logger, "flush_pending_chunks"):
            self.logger.flush_pending_chunks(
                force_full_trial=bool(getattr(self.system, "is_simulation", False))
            )
        result = self.logger.get_result()
        result.terminal = TerminalInfo(
            stabilized=self.stop_condition.is_stabilized(),
            settling_time=self.stop_condition.settling_time(),
        )
        self.offline_visualizer.finalize(result, dt=self.dt)
        return result

    def reset(self):
        self.system.reset()
        self.logger.reset(self.system.step_data)
        self.stop_condition.reset()
        self.scheduler.reset()
        if hasattr(self.pacing, "dt"):
            self.pacing.dt = self.dt
        if hasattr(self.pacing, "reset"):
            self.pacing.reset()
        elif hasattr(self.pacing, "next_time"):
            self.pacing.next_time = time.perf_counter()

    def run_experiment(self):
        results = []
        try:
            for _ in range(self.n_trials):
                result = self.run_trial()
                results.append(result)
            return results
        finally:
            sensor = getattr(self.system, "sensor", None)
            if sensor is not None and hasattr(sensor, "close"):
                sensor.close()

    def _configure_run_log_dir(self) -> None:
        run_log_dir = allocate_run_log_dir("logs")
        self._run_log_dir = run_log_dir

        if hasattr(self.logger, "set_save_dir"):
            self.logger.set_save_dir(run_log_dir)

        sensor = getattr(self.system, "sensor", None)
        writer = getattr(sensor, "_writer", None)
        if writer is not None and hasattr(writer, "set_output_dir"):
            writer.set_output_dir(run_log_dir)

        print(f"[experiment] logging to {run_log_dir}")
        if sensor is not None and bool(getattr(sensor, "recording_enabled", False)):
            print("[experiment] recording armed: focus the visualization window and press R to start/stop.")

    def _handle_runtime_key(self, key: int | None) -> bool:
        if key is None:
            return False

        key_low = key & 0xFF
        if key_low not in (ord("r"), ord("R")):
            return False

        sensor = getattr(self.system, "sensor", None)
        if sensor is None or not hasattr(sensor, "toggle_recording"):
            return False
        if not bool(getattr(sensor, "recording_enabled", False)):
            return False

        is_recording = sensor.toggle_recording()
        state_text = "started" if is_recording else "stopped"
        if is_recording:
            print(f"[experiment] recording {state_text}")
            paths = getattr(sensor, "recording_paths", {})
            if isinstance(paths, dict):
                for cam_id in sorted(paths):
                    cam_paths = paths.get(cam_id)
                    if not isinstance(cam_paths, tuple) or len(cam_paths) != 2:
                        continue
                    aedat4_path, csv_path = cam_paths
                    if aedat4_path is not None:
                        print(f"[experiment] cam{cam_id} aedat4 -> {aedat4_path}")
                    if csv_path is not None:
                        print(f"[experiment] cam{cam_id} hough csv -> {csv_path}")
            elif isinstance(paths, tuple) and len(paths) == 2:
                aedat4_path, csv_path = paths
                if aedat4_path is not None:
                    print(f"[experiment] aedat4 -> {aedat4_path}")
                if csv_path is not None:
                    print(f"[experiment] hough csv -> {csv_path}")
        else:
            print(f"[experiment] recording {state_text}")
        return True
