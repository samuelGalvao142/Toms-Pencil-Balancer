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
    r"C:\Users\pcadm\Downloads\SNN\SNN-Regression-Pencil-Balancer-True\models\modelSlope"
    r"\model_SEW_BN\checkpoints_pendulum\best_model_weights.pth"
)
_LEGACY_INTERCEPT_MODEL_PATH = (
    r"C:\Users\pcadm\Downloads\SNN\SNN-Regression-Pencil-Balancer-True\models\modelIntercept"
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
        self.pos_frame[ys[polarity], xs[polarity]] += 1.0
        self.neg_frame[ys[~polarity], xs[~polarity]] += 1.0

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
        return CameraObservation(
            slope=0.0,
            intercept=self.W / 2,
        )


SNN_PRESETS = {
    "default": {
        "frame_decay": 0.90,
        "max_events": 5000,
        "normalize_input": True,
        "gaussian_sigma": 3.5,
        "clamp_output": True,
        "max_slope": 1e6,
        "max_intercept": 5e4,
        "reset_each_step": False,
    }
}
