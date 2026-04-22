from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
from src.shared import CameraObservation, Measurement, ControlInput, default_camera_params
from src.system.sensor.observation_model.camera_model import CameraModel
from src.system.sensor.algo.dvs_algorithms import line_x_at_pixel_y
from .multi_panel_layout import build_composite

from .base import RealtimeVisualizerBase, EventFramesFn, VizResult, _window_closed


_DEFAULT_CAMERA_PARAMS = default_camera_params()


@dataclass
class RealDvsVisualizerParams:
    width:          int
    height:         int
    top_mask_y_cam1: int
    top_mask_y_cam2: int
    mask_y_cam1:    int
    mask_y_cam2:    int
    event_frames_fn: EventFramesFn | None = None


REAL_DVS_VISUALIZER_PRESETS = {
    "default": {
        "width":       int(_DEFAULT_CAMERA_PARAMS.DAVIS346_WIDTH),
        "height":      int(_DEFAULT_CAMERA_PARAMS.DAVIS346_HEIGHT),
        "top_mask_y_cam1": int(_DEFAULT_CAMERA_PARAMS.y_mask_top_line_1),
        "top_mask_y_cam2": int(_DEFAULT_CAMERA_PARAMS.y_mask_top_line_2),
        "mask_y_cam1": int(_DEFAULT_CAMERA_PARAMS.y_mask_line_1),
        "mask_y_cam2": int(_DEFAULT_CAMERA_PARAMS.y_mask_line_2),
    }
}


class RealDvsVisualizer(RealtimeVisualizerBase):
    """Real DVS event-accumulator panels + line and mask overlays."""

    def __init__(self, params: RealDvsVisualizerParams):
        self._event_frames_fn = params.event_frames_fn
        self.width = params.width
        self.height = params.height
        self.cam = CameraModel(params.width, params.height)
        self.top_mask_y_cam1 = int(params.top_mask_y_cam1)
        self.top_mask_y_cam2 = int(params.top_mask_y_cam2)
        self.mask_y_cam1 = int(params.mask_y_cam1)
        self.mask_y_cam2 = int(params.mask_y_cam2)
        self._window_ready = False

    def _bgr_from_surfaces(self) -> tuple[np.ndarray, np.ndarray]:
        if self._event_frames_fn is None:
            z = np.zeros((self.height, self.width), dtype=np.uint8)
            bgr = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
            return bgr, bgr
        out = self._event_frames_fn()
        if out is None or len(out) != 2:
            z = np.zeros((self.height, self.width), dtype=np.uint8)
            bgr = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
            return bgr, bgr
        surface1, surface2 = out
        frame1 = np.clip(surface1 * 50, 0, 255).astype(np.uint8)
        frame2 = np.clip(surface2 * 50, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

    def _draw_line(
        self,
        frame: np.ndarray,
        b: float,
        s: float,
        top_mask_y: int | None = None,
        mask_y: int | None = None,
    ) -> None:
        obs_px = self.cam.camnorm_to_pixel(CameraObservation(slope=s, intercept=b))
        s_px, b_px = obs_px.slope, obs_px.intercept
        s_px = self._to_finite_scalar(s_px)
        b_px = self._to_finite_scalar(b_px)
        if s_px is None or b_px is None:
            return
        y0 = 0
        if top_mask_y is not None and 0 < top_mask_y < self.height:
            y0 = int(top_mask_y)
        if mask_y is not None and 0 < mask_y < self.height:
            y1 = min(mask_y - 1, self.height - 1)
        else:
            y1 = self.height - 1
        if y1 < y0:
            return
        x0 = int(round(s_px * y0 + b_px))
        x1 = int(round(s_px * y1 + b_px))
        x0 = max(-10_000, min(10_000, x0))
        x1 = max(-10_000, min(10_000, x1))
        try:
            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
        except cv2.error:
            return
        if mask_y is not None and 0 < mask_y < self.height:
            xi = int(round(line_x_at_pixel_y(obs_px, mask_y)))
            if 0 <= xi < self.width:
                cv2.circle(frame, (xi, mask_y), 5, (0, 255, 0), -1)

    def render(
        self,
        measurement,
        command: ControlInput | None = None,
        *,
        surfaces: tuple[np.ndarray, np.ndarray] | None = None,
        title: str | None = None,
        paused: bool = False,
        y_meas: Measurement | None = None,
    ) -> VizResult:
        del surfaces, command, paused
        self._ensure_window(has_workspace=False)
        frame1, frame2 = self._bgr_from_surfaces()
        if measurement is not None:
            b1, s1, b2, s2 = measurement.unpack()
            if 0 < self.top_mask_y_cam1 < self.height:
                cv2.line(frame1, (0, self.top_mask_y_cam1), (self.width - 1, self.top_mask_y_cam1), (255, 0, 255), 2)
            if 0 < self.top_mask_y_cam2 < self.height:
                cv2.line(frame2, (0, self.top_mask_y_cam2), (self.width - 1, self.top_mask_y_cam2), (255, 0, 255), 2)
            if 0 < self.mask_y_cam1 < self.height:
                cv2.line(frame1, (0, self.mask_y_cam1), (self.width - 1, self.mask_y_cam1), (0, 165, 255), 2)
            if 0 < self.mask_y_cam2 < self.height:
                cv2.line(frame2, (0, self.mask_y_cam2), (self.width - 1, self.mask_y_cam2), (0, 165, 255), 2)
            self._draw_line(frame1, b1, s1, top_mask_y=self.top_mask_y_cam1, mask_y=self.mask_y_cam1)
            self._draw_line(frame2, b2, s2, top_mask_y=self.top_mask_y_cam2, mask_y=self.mask_y_cam2)

        title_str = title if title is not None else "Experiment | R: start/stop rec | Q: quit"
        title_str = self._append_y_meas_banner(title_str, y_meas)
        composite = build_composite(title_str, frame1, frame2, None)
        cv2.imshow(self._window_name, composite)
        if _window_closed(self._window_name):
            return VizResult(quit=True)
        key = cv2.waitKeyEx(1)
        key_low = key & 0xFF if key != -1 else -1
        return VizResult(quit=key_low in (ord("q"), ord("Q"), 27), key=None if key == -1 else key)
