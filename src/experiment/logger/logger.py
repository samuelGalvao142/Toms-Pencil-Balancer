from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pickle

import numpy as np

from src.shared import StepData, TableAccel


@dataclass
class LoggerParams:
    save_dir: str = "logs/logger_histories"
    autosave_on_reacquire: bool = True


@dataclass
class TerminalInfo:
    stabilized: bool
    settling_time: float | None


@dataclass
class SimulationResult:
    state_history: np.ndarray
    acc_history: np.ndarray
    estimate_history: np.ndarray | None = None
    adaptive_lpf_weight_history: np.ndarray | None = None
    mech_history: np.ndarray | None = None
    innovation_history: np.ndarray | None = None
    cmd_history: np.ndarray | None = None
    offset_history: np.ndarray | None = None
    offset_latched_history: np.ndarray | None = None
    supervisor_state_history: np.ndarray | None = None
    top_radius_history: np.ndarray | None = None
    x_ref_history: np.ndarray | None = None
    tilt_bias_history: np.ndarray | None = None
    terminal: TerminalInfo | None = None


def _acc_to_row(acc: TableAccel | np.ndarray) -> np.ndarray:
    if isinstance(acc, TableAccel):
        return acc.as_vector()
    a = np.asarray(acc, dtype=float).reshape(-1)
    if a.size != 2:
        raise ValueError(f"acc must have 2 elements, got shape {a.shape}")
    return a


def _mech_to_row(mech_joints: np.ndarray | None) -> np.ndarray:
    if mech_joints is None:
        return np.full((3, 2), np.nan, dtype=float)
    a = np.asarray(mech_joints, dtype=float).reshape(3, 2)
    return a


def _innovation_to_row(innovation: np.ndarray | None) -> np.ndarray:
    if innovation is None:
        return np.full(4, np.nan, dtype=float)
    a = np.asarray(innovation, dtype=float).reshape(-1)
    if a.size != 4:
        raise ValueError(f"innovation must have 4 elements, got shape {a.shape}")
    return a


def _offset_to_row(offset_xy: np.ndarray | None) -> np.ndarray:
    if offset_xy is None:
        return np.zeros(2, dtype=float)
    a = np.asarray(offset_xy, dtype=float).reshape(-1)
    if a.size != 2:
        raise ValueError(f"offset_xy must have 2 elements, got shape {a.shape}")
    return a


def _adaptive_lpf_weight_to_value(adaptive_lpf_weight: float | None) -> float:
    if adaptive_lpf_weight is None:
        return float("nan")
    value = float(adaptive_lpf_weight)
    if not np.isfinite(value):
        return float("nan")
    return float(np.clip(value, 0.0, 1.0))


def _top_radius_to_value(top_radius: float | None) -> float:
    if top_radius is None:
        return float("nan")
    value = float(top_radius)
    if not np.isfinite(value):
        return float("nan")
    return max(value, 0.0)


def _x_ref_to_row(x_ref: np.ndarray | None) -> np.ndarray:
    if x_ref is None:
        return np.zeros(4, dtype=float)
    a = np.asarray(x_ref, dtype=float).reshape(-1)
    if a.size != 4:
        raise ValueError(f"x_ref must have 4 elements, got shape {a.shape}")
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _tilt_bias_to_row(tilt_bias: np.ndarray | None) -> np.ndarray:
    if tilt_bias is None:
        return np.zeros(2, dtype=float)
    a = np.asarray(tilt_bias, dtype=float).reshape(-1)
    if a.size != 2:
        raise ValueError(f"tilt_bias must have 2 elements, got shape {a.shape}")
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


class Logger:
    def __init__(self, params: LoggerParams):
        self.params = params
        self._save_dir = Path(params.save_dir)
        self._states = None
        self._commands = None
        self._acc = None
        self._estimates = None
        self._innovation = None
        self._mech = None
        self._offset = None
        self._offset_latched = None
        self._supervisor_states = None
        self._adaptive_lpf_weight = None
        self._top_radius = None
        self._x_ref = None
        self._tilt_bias = None
        self._active_chunk_start_idx: int | None = None
        self._saved_chunks: set[tuple[int, int]] = set()

    def set_save_dir(self, save_dir: str | Path) -> None:
        self.params.save_dir = str(save_dir)
        self._save_dir = Path(save_dir)

    def reset(self, initial_step_data: StepData):
        self._states = [initial_step_data.x.as_vector()]
        x_hat0 = initial_step_data.x_hat if initial_step_data.x_hat is not None else initial_step_data.x
        self._estimates = [x_hat0.as_vector()]
        self._commands = [
            [initial_step_data.u.px_cmd, initial_step_data.u.py_cmd]
        ]
        self._mech = [_mech_to_row(initial_step_data.mech_joints)]
        self._acc = []
        self._innovation = []
        self._offset = [_offset_to_row(initial_step_data.offset_xy)]
        self._offset_latched = [bool(initial_step_data.offset_latched)]
        self._supervisor_states = [self._normalize_state_name(initial_step_data.supervisor_state)]
        self._adaptive_lpf_weight = [
            _adaptive_lpf_weight_to_value(initial_step_data.adaptive_lpf_weight)
        ]
        self._top_radius = [_top_radius_to_value(initial_step_data.top_radius)]
        self._x_ref = [_x_ref_to_row(initial_step_data.x_ref)]
        self._tilt_bias = [_tilt_bias_to_row(initial_step_data.tilt_bias)]
        self._active_chunk_start_idx = None
        self._saved_chunks = set()

    def record(self, step_data: StepData):
        current_state = self._normalize_state_name(step_data.supervisor_state)
        prev_offset_latched = self._offset_latched[-1] if self._offset_latched else False
        current_offset_latched = bool(step_data.offset_latched)
        self._states.append(step_data.x.as_vector())
        x_hat = step_data.x_hat if step_data.x_hat is not None else step_data.x
        self._estimates.append(x_hat.as_vector())
        self._commands.append([step_data.u.px_cmd, step_data.u.py_cmd])
        self._acc.append(_acc_to_row(step_data.acc))
        self._innovation.append(_innovation_to_row(step_data.innovation))
        self._mech.append(_mech_to_row(step_data.mech_joints))
        self._offset.append(_offset_to_row(step_data.offset_xy))
        self._offset_latched.append(current_offset_latched)
        self._supervisor_states.append(current_state)
        self._adaptive_lpf_weight.append(
            _adaptive_lpf_weight_to_value(step_data.adaptive_lpf_weight)
        )
        self._top_radius.append(_top_radius_to_value(step_data.top_radius))
        self._x_ref.append(_x_ref_to_row(step_data.x_ref))
        self._tilt_bias.append(_tilt_bias_to_row(step_data.tilt_bias))

        current_idx = len(self._supervisor_states) - 1
        if (not prev_offset_latched) and current_offset_latched:
            self._active_chunk_start_idx = current_idx
        elif (
            self.params.autosave_on_reacquire
            and self._active_chunk_start_idx is not None
            and prev_offset_latched
            and not current_offset_latched
        ):
            self._save_chunk(self._active_chunk_start_idx, current_idx, reason="reacquire")
            self._active_chunk_start_idx = None

    def flush_pending_chunks(self, *, force_full_trial: bool = False) -> None:
        if self._supervisor_states is None:
            return

        if force_full_trial:
            self._save_chunk(0, len(self._states), reason="full_trial")
            self._active_chunk_start_idx = None
            return

        stop_idx = len(self._supervisor_states)
        if self._active_chunk_start_idx is not None:
            self._save_chunk(self._active_chunk_start_idx, stop_idx, reason="trial_end")
            self._active_chunk_start_idx = None
            return

        if self._should_save_full_trial():
            self._save_chunk(0, len(self._states), reason="full_trial")

    def get_result(self) -> SimulationResult:
        return self._build_result(0, len(self._states))

    def _build_result(self, start_idx: int, stop_idx: int) -> SimulationResult:
        state_history = np.array(self._states[start_idx:stop_idx])
        estimate_history = np.array(self._estimates[start_idx:stop_idx])
        cmd_history = np.array(self._commands[start_idx:stop_idx])
        mech_history = np.array(self._mech[start_idx:stop_idx])
        offset_history = np.array(self._offset[start_idx:stop_idx])
        offset_latched_history = np.array(
            self._offset_latched[start_idx:stop_idx],
            dtype=bool,
        )
        supervisor_state_history = np.array(
            self._supervisor_states[start_idx:stop_idx],
            dtype=str,
        )
        adaptive_lpf_weight_history = np.array(
            self._adaptive_lpf_weight[start_idx:stop_idx],
            dtype=float,
        )
        if not np.isfinite(adaptive_lpf_weight_history).any():
            adaptive_lpf_weight_history = None
        top_radius_history = np.array(
            self._top_radius[start_idx:stop_idx],
            dtype=float,
        )
        if not np.isfinite(top_radius_history).any():
            top_radius_history = None
        x_ref_history = np.array(self._x_ref[start_idx:stop_idx], dtype=float)
        tilt_bias_history = np.array(self._tilt_bias[start_idx:stop_idx], dtype=float)

        acc_start = min(start_idx, len(self._acc))
        acc_stop = min(max(stop_idx - 1, acc_start), len(self._acc))
        if acc_stop > acc_start:
            acc_history = np.array(self._acc[acc_start:acc_stop])
        else:
            acc_history = np.zeros((0, 2))

        innov_start = min(start_idx, len(self._innovation))
        innov_stop = min(max(stop_idx - 1, innov_start), len(self._innovation))
        if innov_stop > innov_start:
            innovation_history = np.array(self._innovation[innov_start:innov_stop])
        else:
            innovation_history = np.zeros((0, 4))

        return SimulationResult(
            state_history=state_history,
            acc_history=acc_history,
            estimate_history=estimate_history,
            adaptive_lpf_weight_history=adaptive_lpf_weight_history,
            mech_history=mech_history,
            cmd_history=cmd_history,
            innovation_history=innovation_history,
            offset_history=offset_history,
            offset_latched_history=offset_latched_history,
            supervisor_state_history=supervisor_state_history,
            top_radius_history=top_radius_history,
            x_ref_history=x_ref_history,
            tilt_bias_history=tilt_bias_history,
        )

    def _normalize_state_name(self, state_name: str | None) -> str:
        return "UNKNOWN" if state_name is None else str(state_name)

    def _should_save_full_trial(self) -> bool:
        if not self._states or not self._supervisor_states:
            return False
        if any(bool(v) for v in self._offset_latched):
            return False
        state_names = set(self._supervisor_states)
        return state_names.issubset({"STATIC", "UNKNOWN"})

    def _build_chunk_logger(self, start_idx: int, stop_idx: int) -> "Logger":
        chunk_logger = Logger(self.params)
        chunk_logger._states = list(self._states[start_idx:stop_idx])
        chunk_logger._estimates = list(self._estimates[start_idx:stop_idx])
        chunk_logger._commands = list(self._commands[start_idx:stop_idx])
        chunk_logger._mech = list(self._mech[start_idx:stop_idx])
        chunk_logger._offset = list(self._offset[start_idx:stop_idx])
        chunk_logger._offset_latched = list(self._offset_latched[start_idx:stop_idx])
        chunk_logger._supervisor_states = list(self._supervisor_states[start_idx:stop_idx])
        chunk_logger._adaptive_lpf_weight = list(
            self._adaptive_lpf_weight[start_idx:stop_idx]
        )
        chunk_logger._top_radius = list(self._top_radius[start_idx:stop_idx])
        chunk_logger._x_ref = list(self._x_ref[start_idx:stop_idx])
        chunk_logger._tilt_bias = list(self._tilt_bias[start_idx:stop_idx])

        acc_start = min(start_idx, len(self._acc))
        acc_stop = min(max(stop_idx - 1, acc_start), len(self._acc))
        chunk_logger._acc = list(self._acc[acc_start:acc_stop])

        innov_start = min(start_idx, len(self._innovation))
        innov_stop = min(max(stop_idx - 1, innov_start), len(self._innovation))
        chunk_logger._innovation = list(self._innovation[innov_start:innov_stop])

        chunk_logger._active_chunk_start_idx = None
        chunk_logger._saved_chunks = set()
        return chunk_logger

    def _save_chunk(self, start_idx: int, stop_idx: int, *, reason: str) -> Path | None:
        if stop_idx <= start_idx:
            return None
        key = (start_idx, stop_idx)
        if key in self._saved_chunks:
            return None

        result = self._build_result(start_idx, stop_idx)
        chunk_logger = self._build_chunk_logger(start_idx, stop_idx)
        if result.state_history.size == 0:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._save_dir.mkdir(parents=True, exist_ok=True)
        path = self._save_dir / f"logger_chunk_{timestamp}.pkl"
        payload = {
            "saved_at": timestamp,
            "reason": reason,
            "start_index": start_idx,
            "stop_index": stop_idx,
            "logger": chunk_logger,
            "result": result,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

        self._saved_chunks.add(key)
        print(f"saved logger chunk to {path}")
        return path
