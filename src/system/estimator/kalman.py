from dataclasses import dataclass, field

import numpy as np

from src.shared import (
    State,
    Measurement,
    ControlInput,
    PlantParams,
    TimingParams,
    default_plant,
    default_timing,
)

from .base import BaseEstimator
from .dynamics_disc import discretize_AB, measurement_H


@dataclass
class KalmanParams:
    # -----------------------------
    # State / measurement scales
    # -----------------------------
    # These define what "1 meaningful unit" is for each channel.
    # The normalized q/r values below are then divided by scale^2.
    pos_scale: float
    vel_scale: float
    ang_scale: float
    ang_vel_scale: float

    meas_pos_scale: float
    meas_ang_scale: float

    # -----------------------------
    # Normalized process-noise weights
    # -----------------------------
    # These are NOT raw covariances.
    # Actual covariance entries are:
    #   q_pos      / pos_scale^2
    #   q_vel      / vel_scale^2
    #   q_ang      / ang_scale^2
    #   q_ang_vel  / ang_vel_scale^2
    q_pos: float
    q_vel: float
    q_ang: float
    q_ang_vel: float

    # -----------------------------
    # Normalized measurement-noise weights
    # -----------------------------
    # Actual covariance entries are:
    #   r_meas_pos / meas_pos_scale^2
    #   r_meas_ang / meas_ang_scale^2
    r_meas_pos: float
    r_meas_ang: float

    # -----------------------------
    # Initial covariance scaling
    # -----------------------------
    p0_pos: float = 1.0
    p0_vel: float = 1.0
    p0_ang: float = 1.0
    p0_ang_vel: float = 1.0

    plant: PlantParams = field(default_factory=default_plant)
    timing: TimingParams = field(default_factory=default_timing)


KALMAN_PRESETS = {
    "default": {
        # Meaningful state scales
        "pos_scale": 2.0e-3,                 # 2 mm
        "vel_scale": 3.0e-2,                 # 3 cm/s
        "ang_scale": np.deg2rad(0.15),       # ~ real measured angle noise scale
        "ang_vel_scale": np.deg2rad(10.0),   # 10 deg/s

        # Measured-channel scales
        "meas_pos_scale": 2.0e-3,            # 2 mm
        "meas_ang_scale": np.deg2rad(0.15),  # 0.15 deg

        # Normalized process noise
        # Start modest on measured states, larger on hidden derivative states.
        "q_pos": 0.05,
        "q_vel": 2.0,
        "q_ang": 0.05,
        "q_ang_vel": 2.0,

        # Normalized measurement noise
        # If scale ~= actual sigma, then r ~= 1 means "about that noisy".
        "r_meas_pos": 1.0,
        "r_meas_ang": 1.0,

        # Initial covariance confidence
        "p0_pos": 4.0,
        "p0_vel": 25.0,
        "p0_ang": 4.0,
        "p0_ang_vel": 25.0,
    },

    "trust_measurement_more": {
        "base": "default",
        "q_vel": 10.0,
        "q_ang_vel": 10.0,
        "r_meas_pos": 0.5,
        "r_meas_ang": 0.5,
    },

    "trust_model_more": {
        "base": "default",
        "q_pos": 0.01,
        "q_vel": 0.2,
        "q_ang": 0.01,
        "q_ang_vel": 0.2,
        "r_meas_pos": 4.0,
        "r_meas_ang": 4.0,
    },
}


class KalmanEstimator(BaseEstimator):
    def __init__(self, params: KalmanParams):
        super().__init__()

        self.params = params

        # Fixed discrete model from nominal timing
        # If your real dt varies a lot, this is one of the first things to fix.
        self.A, self.B = discretize_AB(params.plant, params.timing)
        self.H = measurement_H()

        p = params

        # -----------------------------
        # Build normalized process covariance Q
        # -----------------------------
        q_pos = float(p.q_pos) / float(p.pos_scale) ** 2
        q_vel = float(p.q_vel) / float(p.vel_scale) ** 2
        q_ang = float(p.q_ang) / float(p.ang_scale) ** 2
        q_ang_vel = float(p.q_ang_vel) / float(p.ang_vel_scale) ** 2

        self.Q = np.diag([
            q_pos, q_vel, q_ang, q_ang_vel,
            q_pos, q_vel, q_ang, q_ang_vel,
        ])

        # -----------------------------
        # Build normalized measurement covariance R
        # -----------------------------
        r_pos = float(p.r_meas_pos) / float(p.meas_pos_scale) ** 2
        r_ang = float(p.r_meas_ang) / float(p.meas_ang_scale) ** 2

        self.R = np.diag([
            r_pos, r_ang,
            r_pos, r_ang,
        ])

        # -----------------------------
        # Initial covariance P0
        # Same normalized idea as Q/R
        # -----------------------------
        p0_pos = float(p.p0_pos) / float(p.pos_scale) ** 2
        p0_vel = float(p.p0_vel) / float(p.vel_scale) ** 2
        p0_ang = float(p.p0_ang) / float(p.ang_scale) ** 2
        p0_ang_vel = float(p.p0_ang_vel) / float(p.ang_vel_scale) ** 2

        self.P_init = np.diag([
            p0_pos, p0_vel, p0_ang, p0_ang_vel,
            p0_pos, p0_vel, p0_ang, p0_ang_vel,
        ])

        self.x_hat_init_0 = np.zeros((8, 1), dtype=float)
        self.P = self.P_init.copy()
        self.x_hat = self.x_hat_init_0.copy()

    def estimate(
        self,
        y_meas: Measurement,
        dt: float,
        u_cmd: ControlInput | None,
    ) -> tuple[State, np.ndarray]:
        # Note:
        # dt is currently unused because A,B are pre-discretized from params.timing.
        # If real dt varies meaningfully, rebuild A,B online or cache by dt.

        z = self.measurement_z(y_meas)
        u = self.control_u(u_cmd, y_meas)

        innovation, x_pred = self._step_prediction(z, self.x_hat, u)
        P_pred = self.A @ self.P @ self.A.T + self.Q

        S = self.H @ P_pred @ self.H.T + self.R

        # Better than inv(S)
        PHt = P_pred @ self.H.T
        K = np.linalg.solve(S.T, PHt.T).T

        self.x_hat = x_pred + K @ innovation.reshape(-1, 1)

        # Joseph form: numerically safer than (I-KH)P
        I = np.eye(self.P.shape[0])
        IKH = I - K @ self.H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

        # Force symmetry against numerical drift
        self.P = 0.5 * (self.P + self.P.T)

        x_hat = State.from_iterable(self.x_hat.flatten())
        return x_hat, innovation

    def reset(self, x_hat: State | None = None):
        self.P = self.P_init.copy()

        if x_hat is not None:
            self.x_hat = x_hat.as_vector().reshape(-1, 1)
        else:
            self.x_hat = self.x_hat_init_0.copy()