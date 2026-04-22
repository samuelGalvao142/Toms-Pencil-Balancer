from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy import deg2rad
from numpy.linalg import norm

from src.shared import ControlInput, WorkspaceParams, clamp_control_input_to_workspace


class Supervisor:
    """Base class for supervisors."""

    def update(
        self,
        x_hat_0,
        innovation_0,
        x_hat_1,
        innovation_1,
        dt,
    ) -> tuple[int, float]:
        """Return (controller_index, est_k) where est_k blends estimator 0 -> 1."""
        raise NotImplementedError

    @property
    def active_output(self) -> tuple[int, float]:
        """Return the currently desired (controller_index, est_k)."""
        raise NotImplementedError

    @property
    def is_offset_latched(self) -> bool:
        """Whether the measurement offset should remain frozen."""
        return False

    @property
    def is_ready_to_run(self) -> bool:
        """Whether the supervisor has left any pre-start state."""
        return True

    @property
    def is_prestart_state(self) -> bool:
        return not self.is_ready_to_run

    @property
    def command_override(self) -> ControlInput | None:
        """Optional manual command that overrides controller output for this step."""
        return None

    @property
    def measurement_angle_offset(self) -> tuple[float, float]:
        """Optional measurement-space trim applied to (ax, ay), radians."""
        return 0.0, 0.0

    @property
    def state_name(self) -> str:
        return self.__class__.__name__

    @property
    def last_transition(self) -> dict | None:
        """Metadata from last update, if available."""
        return None

    def attach_runtime(self, actuator=None, workspace=None):
        del actuator, workspace

    def handle_key(self, key: int | None) -> bool:
        del key
        return False

    def reset(self):
        pass

    def note_applied_command(self, command: ControlInput) -> None:
        del command

    def note_unoffset_measurement(self, measurement) -> None:
        del measurement

    def notify_fall_detected(self) -> None:
        pass


@dataclass(kw_only=True)
class RealStartupParams:
    centering_controller_index: int
    run_controller_index: int
    stable_threshold_deg: float
    stable_threshold_m: float
    stable_hold_s: float
    manual_step_m: float
    workspace: WorkspaceParams
    reacquire_ramp_s: float = 0.25
    tilt_trim_step_deg: float = 0.1
    tilt_trim_file: str = "src/system/supervisor/real_tilt_trim.json"


class RealServoSupervisorBase(Supervisor):
    _UP_KEYS = {82, 2490368, 65362, ord("w"), ord("W")}
    _DOWN_KEYS = {84, 2621440, 65364, ord("s"), ord("S")}
    _LEFT_KEYS = {81, 2424832, 65361, ord("a"), ord("A")}
    _RIGHT_KEYS = {83, 2555904, 65363, ord("d"), ord("D")}
    _ACCEPT_KEYS = {10, 13}
    _RESET_KEYS = {ord("c"), ord("C")}
    _REACQUIRE_KEYS = {ord(" ")}

    def __init__(self, params: RealStartupParams):
        self.params = params
        self.workspace = params.workspace
        self.actuator = None
        self.state = "ACQUISITION"
        self._manual_target = self._workspace_center_command()
        self._last_applied_command = self._workspace_center_command()
        self._reacquire_start_command = self._workspace_center_command()
        self._reacquire_elapsed_s = 0.0
        self._reacquire_active = False
        self._t_stable = 0.0
        self._last_transition: dict | None = None
        self._tilt_trim_rad = np.zeros(2, dtype=float)
        self._latest_unoffset_position: np.ndarray | None = None
        self._load_tilt_trim()

    @property
    def is_offset_latched(self) -> bool:
        return self._measurement_offset_latched()

    @property
    def is_ready_to_run(self) -> bool:
        return self._measurement_offset_latched()

    @property
    def command_override(self) -> ControlInput | None:
        if self.state == "SERVO_CENTERING":
            return self._manual_target
        if self.state == "ACQUISITION":
            return self._acquisition_command()
        return None

    @property
    def measurement_angle_offset(self) -> tuple[float, float]:
        return float(self._tilt_trim_rad[0]), float(self._tilt_trim_rad[1])

    @property
    def state_name(self) -> str:
        return self.state

    @property
    def last_transition(self) -> dict | None:
        return self._last_transition

    @property
    def manual_target(self) -> ControlInput:
        return self._manual_target

    @property
    def top_radius(self) -> float:
        """Real experiments do not know the top constraint, so keep it fully open."""
        if self.workspace.safe_radius is None:
            return 0.0
        return max(float(self.workspace.safe_radius), 0.0)

    def attach_runtime(self, actuator=None, workspace=None):
        if actuator is not None:
            self.actuator = actuator
        if workspace is not None:
            self.workspace = workspace
        self._manual_target = self._workspace_center_command()

    def handle_key(self, key: int | None) -> bool:
        if key is None:
            return False

        key_low = key & 0xFF
        if key in self._REACQUIRE_KEYS or key_low in self._REACQUIRE_KEYS:
            if self.state != "SERVO_CENTERING":
                self._reset_to_acquisition_state()
                return True
            return False

        if self.state != "SERVO_CENTERING":
            if key in self._UP_KEYS or key_low in self._UP_KEYS:
                self._nudge_tilt_trim(0.0, self.params.tilt_trim_step_deg)
                return True
            if key in self._DOWN_KEYS or key_low in self._DOWN_KEYS:
                self._nudge_tilt_trim(0.0, -self.params.tilt_trim_step_deg)
                return True
            if key in self._LEFT_KEYS or key_low in self._LEFT_KEYS:
                self._nudge_tilt_trim(-self.params.tilt_trim_step_deg, 0.0)
                return True
            if key in self._RIGHT_KEYS or key_low in self._RIGHT_KEYS:
                self._nudge_tilt_trim(self.params.tilt_trim_step_deg, 0.0)
                return True
            return False

        if key in self._UP_KEYS or key_low in self._UP_KEYS:
            self._nudge(0.0, self.params.manual_step_m)
            return True
        if key in self._DOWN_KEYS or key_low in self._DOWN_KEYS:
            self._nudge(0.0, -self.params.manual_step_m)
            return True
        if key in self._LEFT_KEYS or key_low in self._LEFT_KEYS:
            self._nudge(-self.params.manual_step_m, 0.0)
            return True
        if key in self._RIGHT_KEYS or key_low in self._RIGHT_KEYS:
            self._nudge(self.params.manual_step_m, 0.0)
            return True
        if key in self._RESET_KEYS or key_low in self._RESET_KEYS:
            self._reset_manual_target()
            return True
        if key in self._ACCEPT_KEYS or key_low in self._ACCEPT_KEYS:
            self._accept_manual_target()
            return True
        return False

    def reset(self):
        self.state = "ACQUISITION"
        self._t_stable = 0.0
        self._last_transition = None
        self._latest_unoffset_position = None
        self._last_applied_command = self._workspace_center_command()
        self._reacquire_start_command = self._workspace_center_command()
        self._reacquire_elapsed_s = 0.0
        self._reacquire_active = False
        self._reset_manual_target()

    def note_applied_command(self, command: ControlInput) -> None:
        self._last_applied_command = command

    def note_unoffset_measurement(self, measurement) -> None:
        self._latest_unoffset_position = np.array(
            [float(measurement.px), float(measurement.py)],
            dtype=float,
        )

    def _update_startup(self, x_est, dt: float) -> None:
        if self.state != "ACQUISITION":
            return
        if self._reacquire_active:
            self._reacquire_elapsed_s += dt
            if self._reacquire_elapsed_s >= self.params.reacquire_ramp_s:
                self._reacquire_active = False
        if self._is_upright(x_est):
            self._t_stable += dt
        else:
            self._t_stable = 0.0
        if self._t_stable >= self.params.stable_hold_s:
            self._t_stable = 0.0
            self._on_upright_ready()

    def _finish_update(self, prev_state: str) -> None:
        left_acquisition = (prev_state == "ACQUISITION" and self.state != "ACQUISITION")
        self._last_transition = {
            "prev_state": prev_state,
            "new_state": self.state,
            "left_acquisition": left_acquisition,
            "left_prestart": left_acquisition,
        }

    def _workspace_center_command(self) -> ControlInput:
        return ControlInput(
            px_cmd=float(self.workspace.x_ref),
            py_cmd=float(self.workspace.y_ref),
        )

    def _acquisition_command(self) -> ControlInput:
        center = self._workspace_center_command()
        if not self._reacquire_active:
            return center

        ramp_s = max(float(self.params.reacquire_ramp_s), 1e-9)
        alpha = min(max(self._reacquire_elapsed_s / ramp_s, 0.0), 1.0)
        start = self._reacquire_start_command
        return ControlInput(
            px_cmd=float((1.0 - alpha) * start.px_cmd + alpha * center.px_cmd),
            py_cmd=float((1.0 - alpha) * start.py_cmd + alpha * center.py_cmd),
        )

    def _reset_manual_target(self) -> None:
        self._manual_target = self._workspace_center_command()
        self._apply_workspace_offset(0.0, 0.0)

    def _nudge(self, dx: float, dy: float) -> None:
        candidate = ControlInput(
            px_cmd=self._manual_target.px_cmd + dx,
            py_cmd=self._manual_target.py_cmd + dy,
        )
        self._manual_target = clamp_control_input_to_workspace(candidate, self.workspace)

    def _accept_manual_target(self) -> None:
        dx = float(self._manual_target.px_cmd - self.workspace.x_ref)
        dy = float(self._manual_target.py_cmd - self.workspace.y_ref)
        self._apply_workspace_offset(dx, dy)
        self.state = "ACQUISITION"
        self._t_stable = 0.0
        self._reacquire_elapsed_s = 0.0
        self._reacquire_active = False

    def _apply_workspace_offset(self, dx: float, dy: float) -> None:
        actuator = self.actuator
        if actuator is not None and hasattr(actuator, "set_workspace_offset"):
            actuator.set_workspace_offset(dx, dy)

    def _nudge_tilt_trim(self, dax_deg: float, day_deg: float) -> None:
        delta_rad = np.deg2rad(np.array([dax_deg, day_deg], dtype=float))
        self._tilt_trim_rad = self._tilt_trim_rad + delta_rad
        self._save_tilt_trim()
        ax_deg, ay_deg = np.rad2deg(self._tilt_trim_rad)
        print(f"tilt trim -> ax={ax_deg:+.3f} deg, ay={ay_deg:+.3f} deg")

    def _tilt_trim_path(self) -> Path:
        path = Path(self.params.tilt_trim_file)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _load_tilt_trim(self) -> None:
        path = self._tilt_trim_path()
        if not path.exists():
            self._tilt_trim_rad = np.zeros(2, dtype=float)
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ax_deg = float(data.get("ax_trim_deg", 0.0))
            ay_deg = float(data.get("ay_trim_deg", 0.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"warning: failed to load tilt trim from {path}: {exc}")
            self._tilt_trim_rad = np.zeros(2, dtype=float)
            return

        self._tilt_trim_rad = np.deg2rad(np.array([ax_deg, ay_deg], dtype=float))

    def _save_tilt_trim(self) -> None:
        path = self._tilt_trim_path()
        payload = {
            "version": 1,
            "ax_trim_deg": float(np.rad2deg(self._tilt_trim_rad[0])),
            "ay_trim_deg": float(np.rad2deg(self._tilt_trim_rad[1])),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _reset_to_acquisition_state(self) -> None:
        self.state = "ACQUISITION"
        self._t_stable = 0.0
        self._reacquire_start_command = self._last_applied_command
        self._reacquire_elapsed_s = 0.0
        self._reacquire_active = self.params.reacquire_ramp_s > 0.0
        self._on_reset_to_acquisition()

    def notify_fall_detected(self) -> None:
        if self.state in {"STABILIZING", "BALANCED"}:
            self._reset_to_acquisition_state()

    def _is_upright(self, x_est) -> bool:
        position = self._latest_unoffset_position
        if position is None:
            position = np.array([float(x_est.px), float(x_est.py)], dtype=float)
        px = float(position[0] - self.workspace.x_ref)
        py = float(position[1] - self.workspace.y_ref)
        ax = float(x_est.ax)
        ay = float(x_est.ay)
        
        thresh_ang = deg2rad(self.params.stable_threshold_deg)
        thresh_m = self.params.stable_threshold_m
        return (
            (norm([ax, ay]) < thresh_ang) and (norm([px,py,]) < thresh_m)
        )

    def _measurement_offset_latched(self) -> bool:
        raise NotImplementedError

    def _on_upright_ready(self) -> None:
        raise NotImplementedError

    def _on_reset_to_acquisition(self) -> None:
        pass
