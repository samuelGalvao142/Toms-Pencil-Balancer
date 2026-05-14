from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional

from src.shared import CameraObservation, CameraParams, default_camera_params

from .base import DVSLineAlgorithm
from .model_definition import CONFIG, SNN_Net

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some environments
    cv2 = None


_LEGACY_SLOPE_MODEL_PATH = (
    r"C:\Users\pcadm\Documents\SNN-Lucca-Meg\SNN-Regression-Pencil-Balancer-True"
    r"\SNN-regression-main\models\new normalization\modelSlope"
    r"\model_SEW_BN\checkpoints_pendulum\best_model_weights.pth"
)
_LEGACY_INTERCEPT_MODEL_PATH = (
    r"C:\Users\pcadm\Documents\SNN-Lucca-Meg\SNN-Regression-Pencil-Balancer-True"
    r"\SNN-regression-main\models\new normalization\modelIntercept"
    r"\model_SEW_BN\checkpoints_pendulum\best_model_weights.pth"
)


@dataclass
class SNNLineParams:
    cam_params: CameraParams = field(default_factory=default_camera_params)
    frame_decay: float = 0.0
    max_events: int = 0
    normalize_input: bool = False
    gaussian_sigma: float = 0.0
    clamp_output: bool = False
    max_slope: float = 0.0
    max_intercept: float = 0.0
    reset_each_step: bool = False
    slope_model_path: str | None = None
    intercept_model_path: str | None = None
    slope_scale: float = 1.76e6
    slope_offset: float = -0.44e6
    intercept_scale: float = 44000.0
    intercept_offset: float = -27500.0
    enable_event_refinement: bool = True
    refinement_sigma_px: float = 6.0
    refinement_min_weight_sum: float = 20.0
    refinement_max_delta_slope: float = 0.50
    refinement_max_delta_intercept: float = 80.0
    refinement_target_side: str = "left"
    refinement_side_search_px: float = 40.0
    line_smoothing: float = 0.35


class SNN_regression(DVSLineAlgorithm):
    def __init__(self, params: SNNLineParams):
        cam = params.cam_params
        self.W = int(cam.DAVIS346_WIDTH)
        self.H = int(cam.DAVIS346_HEIGHT)

        self.frame_decay = float(params.frame_decay)
        self.max_events = int(params.max_events) if params.max_events else None
        self.normalize_input = bool(params.normalize_input)
        self.gaussian_sigma = float(params.gaussian_sigma)
        self.clamp_output = bool(params.clamp_output)
        self.max_slope = float(params.max_slope)
        self.max_intercept = float(params.max_intercept)
        self.reset_each_step = bool(params.reset_each_step)
        self.slope_scale = float(params.slope_scale)
        self.slope_offset = float(params.slope_offset)
        self.intercept_scale = float(params.intercept_scale)
        self.intercept_offset = float(params.intercept_offset)
        self.enable_event_refinement = bool(params.enable_event_refinement)
        self.refinement_sigma_px = float(params.refinement_sigma_px)
        self.refinement_min_weight_sum = float(params.refinement_min_weight_sum)
        self.refinement_max_delta_slope = float(params.refinement_max_delta_slope)
        self.refinement_max_delta_intercept = float(params.refinement_max_delta_intercept)
        self.refinement_target_side = str(params.refinement_target_side).lower()
        self.refinement_side_search_px = float(params.refinement_side_search_px)
        self.line_smoothing = float(np.clip(params.line_smoothing, 0.0, 0.99))
        self._last_slope: float | None = None
        self._last_intercept: float | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.slope_model = self._build_model()
        self.intercept_model = self._build_model()

        slope_model_path = self._resolve_model_path(
            params.slope_model_path,
            env_var="PBR_SNN_SLOPE_MODEL",
            legacy_path=_LEGACY_SLOPE_MODEL_PATH,
        )
        intercept_model_path = self._resolve_model_path(
            params.intercept_model_path,
            env_var="PBR_SNN_INTERCEPT_MODEL",
            legacy_path=_LEGACY_INTERCEPT_MODEL_PATH,
        )

        self._load_weights(self.slope_model, slope_model_path)
        self._load_weights(self.intercept_model, intercept_model_path)

        self.pos_frame = np.zeros((self.H, self.W), dtype=np.float32)
        self.neg_frame = np.zeros((self.H, self.W), dtype=np.float32)

    def _accumulate_events_bincount(self, xs: np.ndarray, ys: np.ndarray, polarity: np.ndarray) -> None:
        if xs.size == 0:
            return

        flat = ys.astype(np.int64, copy=False) * self.W + xs.astype(np.int64, copy=False)

        pos_flat = flat[polarity]
        if pos_flat.size:
            self.pos_frame.reshape(-1)[:] += np.bincount(
                pos_flat,
                minlength=self.pos_frame.size,
            ).astype(np.float32, copy=False)

        neg_flat = flat[~polarity]
        if neg_flat.size:
            self.neg_frame.reshape(-1)[:] += np.bincount(
                neg_flat,
                minlength=self.neg_frame.size,
            ).astype(np.float32, copy=False)

    def _build_model(self) -> SNN_Net:
        model = SNN_Net(
            tau=CONFIG["tau"],
            final_tau=CONFIG["final_tau"],
            hidden=CONFIG["hidden"],
            norm_type=CONFIG["norm_type"],
            learnable_norm=CONFIG["learnable_norm"],
            init_scale=CONFIG["init_scale"],
        )
        model.to(self.device)
        model.eval()
        functional.reset_net(model)
        return model

    @staticmethod
    def _resolve_model_path(
        configured_path: str | None,
        *,
        env_var: str,
        legacy_path: str,
    ) -> Path:
        candidate = configured_path or os.environ.get(env_var) or legacy_path
        path = Path(candidate).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"SNN model weights not found: {path}. Configure {env_var} or the SNN preset paths."
            )
        return path

    def _load_weights(self, model: SNN_Net, weights_path: Path) -> None:
        state_dict = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(state_dict)

    def _extract_polarity(self, events_np: np.ndarray) -> np.ndarray | None:
        dtype_names = getattr(getattr(events_np, "dtype", None), "names", None)
        if not dtype_names:
            return None
        if "p" in dtype_names:
            return np.asarray(events_np["p"], dtype=bool)
        if "polarity" in dtype_names:
            return np.asarray(events_np["polarity"], dtype=bool)
        return None

    def _maybe_blur(self, frame: np.ndarray) -> np.ndarray:
        if self.gaussian_sigma <= 0.0:
            return frame
        if cv2 is None:
            raise ModuleNotFoundError(
                "opencv-python is required when gaussian_sigma > 0 for the SNN algorithm."
            )
        ksize = max(3, int(round(self.gaussian_sigma * 6)) | 1)
        return cv2.GaussianBlur(frame, (ksize, ksize), self.gaussian_sigma)

    def _frame_tensor(self) -> torch.Tensor:
        frame = np.stack([self.pos_frame, self.neg_frame], axis=0).astype(np.float32, copy=False)
        if self.normalize_input:
            max_abs = float(np.max(np.abs(frame)))
            if max_abs > 0.0:
                frame = frame / max_abs
        return torch.from_numpy(frame).unsqueeze(0).to(self.device)

    def _refine_line_from_events(
        self,
        slope: float,
        intercept: float,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> tuple[float, float]:
        if xs.size < 2:
            return slope, intercept

        xs64 = xs.astype(np.float64, copy=False)
        ys64 = ys.astype(np.float64, copy=False)

        predicted_x = slope * ys64 + intercept
        residual = xs64 - predicted_x

        target_offset = 0.0
        if self.refinement_target_side in ("left", "right"):
            if self.refinement_target_side == "left":
                side_mask = (residual < 0.0) & (residual >= -self.refinement_side_search_px)
            else:
                side_mask = (residual > 0.0) & (residual <= self.refinement_side_search_px)

            if int(np.count_nonzero(side_mask)) >= 2:
                side_residual = residual[side_mask]
                bins = np.arange(
                    -self.refinement_side_search_px,
                    self.refinement_side_search_px + 1.0,
                    1.0,
                    dtype=np.float64,
                )
                hist, edges = np.histogram(side_residual, bins=bins)
                if hist.size and int(hist.max()) > 0:
                    peak_idx = int(np.argmax(hist))
                    target_offset = float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))

        sigma = max(self.refinement_sigma_px, 1e-6)
        weights = np.exp(-((residual - target_offset) ** 2) / (2.0 * sigma * sigma))
        if float(weights.sum()) < self.refinement_min_weight_sum:
            return slope, intercept

        A = np.column_stack([ys64, np.ones_like(ys64)])
        Aw = A * weights[:, None]
        bw = xs64 * weights

        try:
            refined_slope, refined_intercept = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return slope, intercept

        refined_slope = float(refined_slope)
        refined_intercept = float(refined_intercept)

        if abs(refined_slope - slope) > self.refinement_max_delta_slope:
            refined_slope = slope + np.sign(refined_slope - slope) * self.refinement_max_delta_slope
        if abs(refined_intercept - intercept) > self.refinement_max_delta_intercept:
            refined_intercept = intercept + np.sign(refined_intercept - intercept) * self.refinement_max_delta_intercept

        return refined_slope, refined_intercept

    def update(self, events_np):
        if events_np is None or len(events_np) == 0:
            return None, None

        if self.max_events is not None and len(events_np) > self.max_events:
            events_np = events_np[-self.max_events:]

        polarity = self._extract_polarity(events_np)
        if polarity is None:
            return None, None

        if self.reset_each_step:
            functional.reset_net(self.slope_model)
            functional.reset_net(self.intercept_model)

        self.pos_frame *= self.frame_decay
        self.neg_frame *= self.frame_decay

        xs = np.asarray(events_np["x"], dtype=np.intp)
        ys = np.asarray(events_np["y"], dtype=np.intp)
        self._accumulate_events_bincount(xs, ys, polarity)

        self.pos_frame = self._maybe_blur(self.pos_frame)
        self.neg_frame = self._maybe_blur(self.neg_frame)

        frame = self._frame_tensor()
        with torch.no_grad():
            intercept_out = self.intercept_model(frame)
            slope_out = self.slope_model(frame)

        slope = float(slope_out.item()) * self.slope_scale + self.slope_offset
        intercept = float(intercept_out.item()) * self.intercept_scale + self.intercept_offset

        if not np.isfinite(slope) or not np.isfinite(intercept):
            return None, None

        if self.enable_event_refinement:
            slope, intercept = self._refine_line_from_events(slope, intercept, xs, ys)

        if self._last_slope is not None and self._last_intercept is not None and self.line_smoothing > 0.0:
            slope = self.line_smoothing * self._last_slope + (1.0 - self.line_smoothing) * slope
            intercept = self.line_smoothing * self._last_intercept + (1.0 - self.line_smoothing) * intercept

        self._last_slope = slope
        self._last_intercept = intercept

        if self.clamp_output:
            if self.max_slope > 0.0:
                slope = float(np.clip(slope, -self.max_slope, self.max_slope))
            if self.max_intercept > 0.0:
                intercept = float(np.clip(intercept, -self.max_intercept, self.max_intercept))

        return CameraObservation(slope=slope, intercept=intercept)

    def reset(self):
        functional.reset_net(self.slope_model)
        functional.reset_net(self.intercept_model)
        self.pos_frame.fill(0.0)
        self.neg_frame.fill(0.0)
        self._last_slope = None
        self._last_intercept = None
        return CameraObservation(
            slope=0.0,
            intercept=self.W / 2,
        )


SNN_PRESETS = {
    "default": {
        "frame_decay": 0.0,
        "max_events": 5000,
        "normalize_input": True,
        "gaussian_sigma": 0.0,
        "clamp_output": True,
        "max_slope": 2.0,
        "max_intercept": 700.0,
        "reset_each_step": False,
        "slope_scale": 3.135188,
        "slope_offset": -1.578480,
        "intercept_scale": 741.832452,
        "intercept_offset": -179.728476,
        "enable_event_refinement": True,
        "refinement_sigma_px": 6.0,
        "refinement_min_weight_sum": 20.0,
        "refinement_max_delta_slope": 0.50,
        "refinement_max_delta_intercept": 80.0,
        "refinement_target_side": "left",
        "refinement_side_search_px": 40.0,
        "line_smoothing": 0.35,
    }
}
