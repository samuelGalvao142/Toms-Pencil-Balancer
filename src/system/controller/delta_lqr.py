from dataclasses import dataclass, field

import control as ct
import numpy as np

from src.shared import (
    ControlInput,
    PlantParams,
    State,
    TimingParams,
    WorkspaceParams,
    default_plant,
    default_timing,
    default_workspace,
    make_reference_state,
)

from .base import BaseController


@dataclass
class DeltaLQRParams:
    pos_scale: float
    vel_scale: float
    tilt_scale: float
    tilt_rate_scale: float
    delta_u_scale: float

    q_pos: float         # cares about bringing table to center
    q_vel: float         # tries to eliminate drift
    q_tilt: float        # tries to set pencil upright
    q_tilt_rate: float   # penalty on angular acceleration

    # penalizing u_err from u_ref_lqr, which itself is moving.
    # This is fine and actually helpful
    # it stops u_prev from wandering away from the shifted reference.
    q_command: float

    r_delta_u: float     # penalty on control command

    pos_ref_ki: float
    max_pos_ref_shift: float

    # Tilt feedforward: accumulated I-term on position error, injected into u
    tilt_ff_ki: float
    max_tilt_ff: float

    max_delta_u: float | None = None
    max_command_radius: float | None = None
    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)
    workspace: WorkspaceParams = field(default_factory=default_workspace)


DELTA_LQR_PRESETS = {
    "default": {

        # "scale of concern", not "scale of catastrophe"
        # at what scale do they matter?
        # we care about pos when we are about 2cm away right?
        # or about tilt when at more than 2 degrees
        "pos_scale": 1e-2,
        "vel_scale": 3e-2,          # /s
        "tilt_scale": np.deg2rad(2.0),
        "tilt_rate_scale": np.deg2rad(10),  # /s
        "delta_u_scale": 1e-3,

        "q_pos": 1.0,
        "q_vel": 0.0,
        "q_tilt": 1.0,
        "q_tilt_rate": 0.0,

        "q_command": 0.06,

        "r_delta_u": 5.0,

        "max_delta_u": 2.0e-3,
        "max_command_radius": 7.0e-2,

        "pos_ref_ki": 0.1,
        "max_pos_ref_shift": 10e-3,

        # Tilt feedforward: slow I-term on position error -> injected into u
        # Units: [actuator_units / (m * s)] — tune this carefully,
        # start very small and increase until you see compensation
        "tilt_ff_ki": 0.0,
        "max_tilt_ff": 5.0e-2,
    },
    "gentle": {
        "base": "default",
        "r_delta_u": 850,
    },
    "strong": {
        "base": "default",
        "q_pos": 0.5,
        "q_vel": 0.0,
        "q_tilt": 10.0,
        "q_tilt_rate": 0.0,
        "max_delta_u": 5.0e-3,
        "r_delta_u": 1050,
    },
}


class DeltaLQRController(BaseController):
    """Discrete LQR that optimizes command increments and returns absolute position.

    Tilt compensation is handled as a feedforward term accumulated from position
    error (slow I-term), injected directly into u after the LQR step.  The LQR's
    own tilt reference stays fixed at x_ref_true so the error signal is never
    corrupted by the compensation intent.
    """

    def __init__(self, params: DeltaLQRParams):
        from src.system.plant.dynamics_model import BuildLinearModel

        A_c, B_c = BuildLinearModel(params.plant)
        dt = float(params.timing.actuator_dt)
        x_ref = make_reference_state(params.workspace)

        n, m = A_c.shape[0], B_c.shape[1]
        sys_c = ct.ss(A_c, B_c, np.eye(n), np.zeros((n, m)))
        sys_d = ct.c2d(sys_c, dt)
        A_d = np.asarray(sys_d.A, dtype=float)
        B_d = np.asarray(sys_d.B, dtype=float)

        A_aug = np.block([
            [A_d, B_d],
            [np.zeros((m, n)), np.eye(m)],
        ])
        B_aug = np.vstack([B_d, np.eye(m)])

        # Normalized Q
        Q_axis = np.diag([
            params.q_pos / params.pos_scale**2,
            params.q_vel / params.vel_scale**2,
            params.q_tilt / params.tilt_scale**2,
            params.q_tilt_rate / params.tilt_rate_scale**2,
        ])

        Q_x = np.block([
            [Q_axis, np.zeros_like(Q_axis)],
            [np.zeros_like(Q_axis), Q_axis],
        ])
        Q_command = float(params.q_command) * np.eye(m)
        Q = np.block([
            [Q_x, np.zeros((n, m))],
            [np.zeros((m, n)), Q_command],
        ])
        R = (params.r_delta_u / params.delta_u_scale**2) * np.eye(m)

        self.K, _, _ = ct.dlqr(A_aug, B_aug, Q, R)
        self.K = np.asarray(self.K, dtype=float)

        self._x_ref_true = x_ref.as_vector()   # never changes
        self.x_ref_lqr = x_ref.as_vector().copy()  # only position entries drift;
                                                    # tilt entries stay == _x_ref_true

        self._u_ref_true = -np.linalg.pinv(B_c) @ A_c  # [m x n], computed once
        self.u_ref_lqr = (self._u_ref_true @ self.x_ref_lqr).ravel()

        self._u_prev = self.u_ref_lqr.copy()
        self.max_delta_u = params.max_delta_u
        self.max_command_radius = params.max_command_radius
        self.workspace_ref = np.array(
            [float(params.workspace.x_ref), float(params.workspace.y_ref)],
            dtype=float,
        )

        # Position integrator: shifts position reference to cancel steady-state offset
        self.pos_ref_ki = params.pos_ref_ki
        self.max_pos_ref_shift = params.max_pos_ref_shift

        # Tilt feedforward: accumulated I-term on position error.
        # Injected as a direct additive offset to u, bypassing the LQR error signal.
        self.tilt_ff_ki = float(params.tilt_ff_ki)
        self.max_tilt_ff = float(params.max_tilt_ff)
        self._tilt_ff = np.zeros(m, dtype=float)  # [u_x_ff, u_y_ff]

        self.actuator_dt = float(params.timing.actuator_dt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _limit_delta_u(self, delta_u: np.ndarray) -> np.ndarray:
        if self.max_delta_u is None:
            return delta_u

        max_delta = float(self.max_delta_u)
        if max_delta <= 0.0:
            return np.zeros_like(delta_u)

        delta_norm = float(np.linalg.norm(delta_u))
        if delta_norm <= max_delta or delta_norm <= 0.0:
            return delta_u

        return delta_u * (max_delta / delta_norm)

    def _limit_command_radius(self, u: np.ndarray) -> np.ndarray:
        if self.max_command_radius is None:
            return u

        max_radius = float(self.max_command_radius)
        if max_radius <= 0.0:
            return self.workspace_ref.copy()

        radial = u - self.workspace_ref
        radius = float(np.linalg.norm(radial))
        if radius <= max_radius or radius <= 0.0:
            return u

        return self.workspace_ref + radial * (max_radius / radius)

    def _clamp(self, value: float, max_value: float, init_val: float) -> float:
        return float(np.clip(value, init_val - max_value, init_val + max_value))

    def _print_ref(self, state: State) -> None:
        state.print_est()
        print(
            f"ref: px: {self.x_ref_lqr[0]*1000:.2f}  py: {self.x_ref_lqr[4]*1000:.2f}  "
            f"ax: {np.rad2deg(self.x_ref_lqr[2]):.3f}  ay: {np.rad2deg(self.x_ref_lqr[6]):.3f}  "
            f"tilt_ff: {self._tilt_ff}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reference_state(self) -> State:
        return State.from_iterable(self.x_ref_lqr)

    def reference_command(self) -> ControlInput:
        return ControlInput(float(self.u_ref_lqr[0]), float(self.u_ref_lqr[1]))

    def _update_integrator(self, state: State) -> None:
        # ---- Position reference integrator (unchanged) ----
        pos_err_x = state.px - self._x_ref_true[0]
        pos_err_y = state.py - self._x_ref_true[4]

        self.x_ref_lqr[0] -= self.pos_ref_ki * pos_err_x * self.actuator_dt
        self.x_ref_lqr[4] -= self.pos_ref_ki * pos_err_y * self.actuator_dt

        self.x_ref_lqr[0] = self._clamp(self.x_ref_lqr[0], self.max_pos_ref_shift, self._x_ref_true[0])
        self.x_ref_lqr[4] = self._clamp(self.x_ref_lqr[4], self.max_pos_ref_shift, self._x_ref_true[4])

        # Tilt entries in x_ref_lqr are intentionally NOT touched here.
        # They stay equal to _x_ref_true so the LQR error signal is clean.

        # Refresh u_ref from the (position-shifted) reference
        self.u_ref_lqr = (self._u_ref_true @ self.x_ref_lqr).ravel()

        # ---- Tilt feedforward I-term ----
        # Accumulate position error into a direct actuator offset.
        # A persistent positive pos_err_x means the table is consistently off-center;
        # we build up a tilt compensation command to fight whatever is causing it.
        self._tilt_ff[0] += self.tilt_ff_ki * pos_err_x * self.actuator_dt
        self._tilt_ff[1] += self.tilt_ff_ki * pos_err_y * self.actuator_dt

        # Clamp each axis independently
        self._tilt_ff[0] = float(np.clip(self._tilt_ff[0], -self.max_tilt_ff, self.max_tilt_ff))
        self._tilt_ff[1] = float(np.clip(self._tilt_ff[1], -self.max_tilt_ff, self.max_tilt_ff))

    def compute(self, state: State) -> ControlInput:
        x_err = state.as_vector() - self.x_ref_lqr

        u_err = self._u_prev - self.u_ref_lqr
        xi_err = np.concatenate([x_err, u_err])

        delta_u = -(self.K @ xi_err).ravel()
        delta_u = self._limit_delta_u(delta_u)

        # LQR command + tilt feedforward injected here, outside the error signal
        u = self._u_prev + delta_u + self._tilt_ff

        u = self._limit_command_radius(u)
        
        #self._print_ref(state)

        return ControlInput(float(u[0]), float(u[1]))

    def set_applied_command(self, u: ControlInput, state: State) -> None:
        self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)
        self._update_integrator(state)

    def reset(self, x_hat: State | None = None):
        print("delta_lqr reset", x_hat)

        # Reset position integrator
        self.x_ref_lqr = self._x_ref_true.copy()
        self.u_ref_lqr = (self._u_ref_true @ self.x_ref_lqr).ravel()

        # Reset tilt feedforward accumulator
        self._tilt_ff = np.zeros_like(self._tilt_ff)

        self._u_prev = self.u_ref_lqr.copy()
        if x_hat is not None:
            u = self.compute(x_hat)
            self._u_prev = np.array([u.px_cmd, u.py_cmd], dtype=float)