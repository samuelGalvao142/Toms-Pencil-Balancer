from dataclasses import dataclass, field, fields, is_dataclass
from types import UnionType
from typing import Sequence, get_args, get_origin
from pathlib import Path
import pickle
import numpy as np
from numpy import rad2deg, deg2rad


# ── Plant ─────────────────────────────────────────────────────────────────────

@dataclass
class PlantParams:
    """Pure dynamics: no workspace geometry."""
    g:          float
    com_length: float
    tau:        float
    zeta:       float
    num_states: int
    max_acc:    float | None = None

PLANT_PRESETS = {
    "default": {
        "g": 9.81, "com_length": 12.5e-2, "tau": 0.03, "zeta": 0.8,
        "max_acc": 9.81 * 5, "num_states": 8,
    }
}

def default_plant() -> PlantParams:
    return PlantParams(**PLANT_PRESETS["default"])


# ── Workspace ─────────────────────────────────────────────────────────────────

@dataclass
class WorkspaceParams:
    """Workspace geometry — separate concern from plant dynamics."""
    x_ref:       float
    y_ref:       float
    safe_radius: float | None = None

WORKSPACE_PRESETS = {
    "default": {"x_ref": 0.0, "y_ref": 0.0, "safe_radius": 7.0e-2}
}

def default_workspace() -> WorkspaceParams:
    return WorkspaceParams(**WORKSPACE_PRESETS["default"])


# ── Timing ────────────────────────────────────────────────────────────────────

@dataclass
class TimingParams:
    total_time: float
    dt:         float
    actuator_dt: float

TIMING_PRESETS = {
    "default": {"total_time": 10.0, "dt": 3e-3, "actuator_dt": 3e-3},
    "ideal": {"total_time": 5.0, "dt": 1e-3},
    "long":    {"total_time": 30.0, "dt": 4e-3},
}

def default_timing() -> TimingParams:
    return TimingParams(**TIMING_PRESETS["default"])

# ── Initial Conditions ────────────────────────────────────────────────────────

@dataclass
class InitConditionsSpread:
    pos_m:   float
    ang_deg: float
    vel_mps: float
    w_degps: float
    
INIT_CONDITIONS_SPREAD_PRESETS = {
    "easy": {
        "pos_m":   0, 
        "ang_deg": 2,
        "vel_mps": 0,
        "w_degps": 0,
    },
    "angle": {
        "pos_m":   0, 
        "ang_deg": 8,
        "vel_mps": 0,
        "w_degps": 0,
    },
    "real": {
        "pos_m": 10e-3, 
        "ang_deg": 5,
        "vel_mps": 0,
        "w_degps": 0,
    },
    "hard": {
        "pos_m": 30e-3, 
        "ang_deg": 8,
        "vel_mps": 5e-3,
        "w_degps": 4,
    },
}

def default_spread() -> InitConditionsSpread:
    return InitConditionsSpread(**INIT_CONDITIONS_SPREAD_PRESETS["easy"])
    
# ── Null ──────────────────────────────────────────────────────────────────────

@dataclass
class NullParams:
    pass

NULL_PRESETS = {"default": {}}

# ── Spec ──────────────────────────────────────────────────────────────────────

@dataclass
class Spec:
    cls:        type
    Params:     type
    Presets:    dict
    registries: dict | None = None
    sim_only:   bool | None = None


# ── Loop / actuation types ────────────────────────────────────────────────────

@dataclass
class State:
    px: float
    vx: float
    ax: float
    wx: float
    py: float
    vy: float
    ay: float
    wy: float
    
    @classmethod
    def from_iterable(cls, data: Sequence[float]) -> "State":
        if len(data) != 8:
            raise ValueError(f"Expected 8 values, got {len(data)}")

        return cls(*map(float, data))

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.px, self.vx, self.ax, self.wx,
            self.py, self.vy, self.ay, self.wy,
        ])
    
    def state_str(self):
        return (
            f"px={self.px*1000:+.2f} mm, vx={self.vx*1000:+.2f} mm/s, "
            f"ax={np.rad2deg(self.ax):+.2f}°, wx={np.rad2deg(self.wx):+.2f}°/s | "
            f"py={self.py*1000:+.2f} mm, vy={self.vy*1000:+.2f} mm/s, "
            f"ay={np.rad2deg(self.ay):+.2f}°, wy={np.rad2deg(self.wy):+.2f}°/s"
        )
    
    def print_state(self):
        print(f"x:   {self.state_str()}")

    def print_est(self):
        print(f"x_hat:   {self.state_str()}")
    
    def print_vel(self):
        print(
            f"x_vel:   "
            f"vx={self.vx*1000:+.2f} mm/s, "
            f"wx={np.rad2deg(self.wx):+.2f}°/s | "
            f"vy={self.vy*1000:+.2f} mm/s, "
            f"wy={np.rad2deg(self.wy):+.2f}°/s"
        )


@dataclass
class ControlInput:
    px_cmd: float
    py_cmd: float


@dataclass
class TableAccel:
    x_ddot: float
    y_ddot: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.x_ddot, self.y_ddot])


@dataclass
class Measurement:
    px: float
    py: float
    ax: float
    ay: float
    
    def as_vector(self) -> np.ndarray:
        return np.array([
            self.px, self.ax, self.py, self.ay,
        ])
    
    def print_y_meas(self):
        print(
            f"y:   "
            f"px={self.px*1000:+.2f} mm, "
            f"ax={np.rad2deg(self.ax):+.2f}° | "
            f"py={self.py*1000:+.2f} mm, "
            f"ay={np.rad2deg(self.ay):+.2f}°"
        )


# ── Logger ────────────────────────────────────────────────────────────────────

@dataclass
class StepData:
    x:     State
    u:     ControlInput
    acc:   TableAccel
    # Controller-facing estimated state for this sample (typically blended x_hat).
    # Falls back to ``x`` when an estimate is unavailable.
    x_hat: State | None = None
    # Measurement-space residual [px, ax, py, ay]; estimators return ndarray (1d or column).
    # None when unavailable (e.g. some hardware paths).
    innovation: np.ndarray | None = None
    # Five-bar joint points in mechanism global frame, mm: rows A, C, P; cols x, y.
    # NaN when the actuator has no mechanism or IK/workspace fails.
    mech_joints: np.ndarray | None = None
    # XY offset used to zero measured position channels [px, py], meters.
    offset_xy: np.ndarray | None = None
    # True when the offset has been frozen after acquisition exit.
    offset_latched: bool = False
    # Supervisor state associated with this sample.
    supervisor_state: str | None = None
    # Adaptive-Kalman LPF fallback weight in [0, 1]: 0 = Kalman-like, 1 = LPF-like.
    adaptive_lpf_weight: float | None = None
    # Current PlacingPlant top constraint radius, meters. NaN in logs when unavailable.
    top_radius: float | None = None
    # Controller reference offsets [px, ax, py, ay] relative to workspace/upright.
    # Meters for position channels, radians for angle channels.
    x_ref: np.ndarray | None = None
    # Controller tilt bias [ax, ay], radians.
    tilt_bias: np.ndarray | None = None


def plot_logger_chunk(path: str | Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    kalman_color = np.asarray(mcolors.to_rgb("tab:blue"), dtype=float)
    lpf_color = np.asarray(mcolors.to_rgb("tab:red"), dtype=float)

    def _mix_mode_color(weight: float) -> tuple[float, float, float]:
        weight = float(np.clip(weight, 0.0, 1.0))
        return tuple((1.0 - weight) * kalman_color + weight * lpf_color)

    def _plot_tilt_with_mode_color(
        ax_plot,
        x_data: np.ndarray,
        y_data: np.ndarray,
        mode_weight: np.ndarray | None,
        tilt_label: str,
        cmd_line,
        cmd_label: str,
        vel_line,
        vel_label: str,
    ) -> None:
        if mode_weight is None or x_data.size < 2:
            tilt_line = ax_plot.plot(x_data, y_data, color="tab:blue", label=tilt_label)
            lines = tilt_line + [cmd_line, vel_line]
            labels = [line.get_label() for line in lines]
            ax_plot.legend(lines, labels, loc="upper right")
            return

        points = np.column_stack((x_data, y_data)).reshape(-1, 1, 2)
        segments = np.concatenate((points[:-1], points[1:]), axis=1)
        segment_weights = 0.5 * (mode_weight[:-1] + mode_weight[1:])
        segment_colors = [_mix_mode_color(weight) for weight in segment_weights]
        tilt_segments = LineCollection(
            segments,
            colors=segment_colors,
            linewidths=2.0,
        )
        ax_plot.add_collection(tilt_segments)
        ax_plot.autoscale_view(scalex=False, scaley=True)

        legend_handles = [
            Line2D([0], [0], color=kalman_color, linewidth=2.0, label=f"{tilt_label} Kalman"),
            Line2D([0], [0], color=lpf_color, linewidth=2.0, label=f"{tilt_label} LPF"),
            cmd_line,
            vel_line,
        ]
        ax_plot.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper right")

    def _history_or_zeros(
        result,
        attr_name: str,
        n_rows: int,
        n_cols: int,
        chunk_path: Path,
    ) -> np.ndarray:
        history = getattr(result, attr_name, None)
        if history is None:
            return np.zeros((n_rows, n_cols), dtype=float)
        history = np.asarray(history, dtype=float)
        if history.shape != (n_rows, n_cols):
            raise ValueError(f"Invalid {attr_name} in {chunk_path}")
        return np.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)

    def _plot_reference_drift(
        ax_plot,
        x_data: np.ndarray,
        ref_angle_deg: np.ndarray,
        ref_pos_mm: np.ndarray,
        title: str,
        ref_angle_label: str,
        ref_pos_label: str,
    ) -> None:
        pos_ax = ax_plot.twinx()
        ref_angle_line = ax_plot.plot(
            x_data,
            ref_angle_deg,
            color="tab:green",
            label=ref_angle_label,
        )[0]
        ref_pos_line = pos_ax.plot(
            x_data,
            ref_pos_mm,
            color="tab:brown",
            label=ref_pos_label,
        )[0]

        ax_plot.set_title(title)
        ax_plot.set_ylabel("Angle (deg)", color="tab:green")
        pos_ax.set_ylabel("Position (mm)", color="tab:brown")
        ax_plot.set_ylim(-3.0, 3.0)
        pos_ax.set_ylim(-10.0, 10.0)
        ax_plot.grid(True, alpha=0.3)

        lines = [ref_angle_line, ref_pos_line]
        ax_plot.legend(lines, [line.get_label() for line in lines], loc="upper right")

    chunk_path = Path(path)
    with chunk_path.open("rb") as fh:
        payload = pickle.load(fh)

    result = payload.get("result", payload)
    estimate_history = getattr(result, "estimate_history", None)
    state_history = np.asarray(
        estimate_history if estimate_history is not None else result.state_history,
        dtype=float,
    )
    cmd_history = np.asarray(result.cmd_history, dtype=float)
    adaptive_lpf_weight_history = getattr(result, "adaptive_lpf_weight_history", None)
    if state_history.ndim != 2 or state_history.shape[1] < 8:
        raise ValueError(f"Invalid state_history in {chunk_path}")
    if cmd_history.ndim != 2 or cmd_history.shape[1] < 2:
        raise ValueError(f"Invalid cmd_history in {chunk_path}")

    sample_idx = np.arange(state_history.shape[0], dtype=float)
    x_ref_history = _history_or_zeros(
        result,
        "x_ref_history",
        state_history.shape[0],
        4,
        chunk_path,
    )
    ax_deg = np.rad2deg(state_history[:, 2])
    ay_deg = np.rad2deg(state_history[:, 6])
    vx_mm_s = 1000.0 * state_history[:, 1]
    vy_mm_s = 1000.0 * state_history[:, 5]
    cmd_x_mm = 1000.0 * cmd_history[:, 0]
    cmd_y_mm = 1000.0 * cmd_history[:, 1]
    x_ref_px_mm = 1000.0 * x_ref_history[:, 0]
    x_ref_ax_deg = np.rad2deg(x_ref_history[:, 1])
    x_ref_py_mm = 1000.0 * x_ref_history[:, 2]
    x_ref_ay_deg = np.rad2deg(x_ref_history[:, 3])
    if adaptive_lpf_weight_history is not None:
        adaptive_lpf_weight_history = np.asarray(adaptive_lpf_weight_history, dtype=float).reshape(-1)
        if adaptive_lpf_weight_history.shape[0] != state_history.shape[0]:
            adaptive_lpf_weight_history = None
        elif not np.isfinite(adaptive_lpf_weight_history).any():
            adaptive_lpf_weight_history = None
        else:
            adaptive_lpf_weight_history = np.clip(
                np.nan_to_num(adaptive_lpf_weight_history, nan=0.0),
                0.0,
                1.0,
            )

    for i in range(-min(800, len(sample_idx)), 0 ,1):
        print(f"step:{i} | alpha_x: {ax_deg[i]:.2f}° cmd_x: {cmd_x_mm[i]:.1f}mm | alpha_y: {ay_deg[i]:.2f}° cmd_y: {cmd_y_mm[i]:.1f}mm")
    
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 11))
    fig.suptitle(f"Logger Chunk: {chunk_path.name}")

    axis_specs = [
        (axes[0], ax_deg, cmd_x_mm, vx_mm_s, "X Axis", "Tilt ax (deg)", "Cmd x (mm)", "Pencil vx (mm/s)"),
        (axes[1], ay_deg, cmd_y_mm, vy_mm_s, "Y Axis", "Tilt ay (deg)", "Cmd y (mm)", "Pencil vy (mm/s)"),
    ]

    for ax_plot, tilt_deg, cmd_mm, vel_mm_s, title, tilt_label, cmd_label, vel_label in axis_specs:
        vel_ax = ax_plot.twinx()
        vel_ax.spines["right"].set_position(("outward", 52))
        vel_line = vel_ax.plot(
            sample_idx,
            vel_mm_s,
            color="tab:orange",
            alpha=0.35,
            label=vel_label,
            zorder=1,
        )[0]
        cmd_ax = ax_plot.twinx()
        cmd_line = cmd_ax.plot(sample_idx, cmd_mm, color="tab:green", label=cmd_label, zorder=3)[0]
        _plot_tilt_with_mode_color(
            ax_plot,
            sample_idx,
            tilt_deg,
            adaptive_lpf_weight_history,
            tilt_label,
            cmd_line,
            cmd_label,
            vel_line,
            vel_label,
        )

        ax_plot.set_title(title)
        ax_plot.set_ylabel("Angle (deg)", color="tab:blue")
        cmd_ax.set_ylabel("Command (mm)", color="tab:green")
        vel_ax.set_ylabel("Velocity (mm/s)", color="tab:orange")
        ax_plot.set_ylim(-15.0, 15.0)
        cmd_ax.set_ylim(-70.0, 70.0)
        vel_ax.set_ylim(-100.0, 100.0)
        ax_plot.grid(True, alpha=0.3)

    _plot_reference_drift(
        axes[2],
        sample_idx,
        x_ref_ax_deg,
        x_ref_px_mm,
        "X Reference",
        "x_ref ax offset (deg)",
        "x_ref px offset (mm)",
    )
    _plot_reference_drift(
        axes[3],
        sample_idx,
        x_ref_ay_deg,
        x_ref_py_mm,
        "Y Reference",
        "x_ref ay offset (deg)",
        "x_ref py offset (mm)",
    )

    axes[-1].set_xlabel("Sample")
    fig.tight_layout()
    plt.show()


# ── Camera / vision types ─────────────────────────────────────────────────────

@dataclass
class CameraParams:
    xr: float
    yr: float
    y_mask_top_line_1: float
    y_mask_top_line_2: float
    y_mask_line_1:   float
    y_mask_line_2:   float
    DAVIS346_WIDTH:  float
    DAVIS346_HEIGHT: float

CAMERA_PRESETS = {
    "default": {
        "xr": -160,
        "yr": -150,
        "y_mask_top_line_1": 30,
        "y_mask_top_line_2": 70,
        "y_mask_line_1":   162, #160
        "y_mask_line_2":   170, #150
        "DAVIS346_WIDTH":  346,
        "DAVIS346_HEIGHT": 260,
    },
}


def default_camera_params() -> CameraParams:
    return CameraParams(**CAMERA_PRESETS["default"])




@dataclass
class CameraObservation:
    slope: float
    intercept: float


@dataclass
class CameraPair:
    cam1: CameraObservation
    cam2: CameraObservation

    def unpack(self) -> tuple[float, float, float, float]:
        return (
            self.cam1.intercept, 
            self.cam1.slope, 
            self.cam2.intercept, 
            self.cam2.slope)
    


# ── Utilities ─────────────────────────────────────────────────────────────────

def make_reference_state(workspace: WorkspaceParams) -> State:
    """Build reference state from workspace params."""
    return State(
        px=workspace.x_ref, vx=0.0, ax=0.0, wx=0.0,
        py=workspace.y_ref, vy=0.0, ay=0.0, wy=0.0,
    )


def clamp_control_input_to_workspace(
    cmd: ControlInput, workspace: WorkspaceParams
) -> ControlInput:
    """Clamp a control input to the circular workspace boundary."""
    px_cmd, py_cmd = cmd.px_cmd, cmd.py_cmd
    safe_radius = workspace.safe_radius

    if safe_radius is None:
        return ControlInput(px_cmd, py_cmd)

    dx = px_cmd - workspace.x_ref
    dy = py_cmd - workspace.y_ref
    dist = float(np.sqrt(dx * dx + dy * dy))

    if dist > safe_radius and dist > 0:
        scale = safe_radius / dist
        px_cmd = workspace.x_ref + dx * scale
        py_cmd = workspace.y_ref + dy * scale

    return ControlInput(px_cmd, py_cmd)


# ── Registry builders ─────────────────────────────────────────────────────────

def resolve_preset(presets, name):
    p = presets[name]
    if "base" in p:
        base = resolve_preset(presets, p["base"])
        return {**base, **{k: v for k, v in p.items() if k != "base"}}
    return p


def _dataclass_type(tp):
    """Return the concrete dataclass type hidden inside ``tp``, if any."""
    if tp is None:
        return None
    if isinstance(tp, type) and is_dataclass(tp):
        return tp

    origin = get_origin(tp)
    if origin in (None,):
        return None
    if origin in (UnionType,):
        for arg in get_args(tp):
            dc = _dataclass_type(arg)
            if dc is not None:
                return dc
        return None

    if str(origin) == "typing.Union":
        for arg in get_args(tp):
            dc = _dataclass_type(arg)
            if dc is not None:
                return dc
    return None


def _build_dataclass_value(params_cls, raw):
    """Recursively instantiate nested dataclasses from plain dicts."""
    if not isinstance(raw, dict):
        return raw

    resolved = {}
    field_map = {f.name: f for f in fields(params_cls)}
    for key, value in raw.items():
        field_info = field_map.get(key)
        nested_cls = None if field_info is None else _dataclass_type(field_info.type)
        if nested_cls is not None and isinstance(value, dict):
            resolved[key] = _build_dataclass_value(nested_cls, value)
        else:
            resolved[key] = value
    return params_cls(**resolved)


def build_from_registry(registry, spec_string):
    type_, preset = spec_string.split(":")
    try:
        spec = registry[type_]
    except KeyError:
        raise ValueError(f"Unknown type: {type_!r} — available: {list(registry)}")

    raw = resolve_preset(spec.Presets, preset)

    resolved = {}
    param_fields = {f.name: f for f in fields(spec.Params)}
    for k, v in raw.items():
        sub_registry = (spec.registries or {}).get(k)
        if isinstance(v, str) and ":" in v:
            resolved[k] = build_from_registry(sub_registry, v)
        elif isinstance(v, list) and sub_registry:
            # Homogeneous collaborators (e.g. controllers): order is part of the
            # contract — supervisors pick active instances by index.
            resolved[k] = [build_from_registry(sub_registry, s) for s in v]
        elif isinstance(v, dict):
            field_info = param_fields.get(k)
            nested_cls = None if field_info is None else _dataclass_type(field_info.type)
            if nested_cls is not None:
                resolved[k] = _build_dataclass_value(nested_cls, v)
            else:
                resolved[k] = v
        else:
            resolved[k] = v

    return spec.cls(spec.Params(**resolved))
