from __future__ import annotations
from dataclasses import dataclass, field

import cv2
import numpy as np
from src.shared import Measurement, ControlInput, WorkspaceParams, default_workspace, default_camera_params
from .multi_panel_layout import build_composite

from .base import WorkspacePanelRenderer, EventFramesFn, VizResult, _window_closed
from .real_dvs import RealDvsVisualizer, RealDvsVisualizerParams


_DEFAULT_CAMERA_PARAMS = default_camera_params()


@dataclass
class RealDvsWorkspaceVisualizerParams:
    width:           int
    height:          int
    top_mask_y_cam1: int
    top_mask_y_cam2: int
    mask_y_cam1:     int
    mask_y_cam2:     int
    workspace:       WorkspaceParams = field(default_factory=default_workspace)
    event_frames_fn: EventFramesFn | None = None


REAL_DVS_WORKSPACE_VISUALIZER_PRESETS = {
    "default": {
        "width":       int(_DEFAULT_CAMERA_PARAMS.DAVIS346_WIDTH),
        "height":      int(_DEFAULT_CAMERA_PARAMS.DAVIS346_HEIGHT),
        "top_mask_y_cam1": int(_DEFAULT_CAMERA_PARAMS.y_mask_top_line_1),
        "top_mask_y_cam2": int(_DEFAULT_CAMERA_PARAMS.y_mask_top_line_2),
        "mask_y_cam1": int(_DEFAULT_CAMERA_PARAMS.y_mask_line_1),
        "mask_y_cam2": int(_DEFAULT_CAMERA_PARAMS.y_mask_line_2),
    }
}


class RealDvsWorkspaceVisualizer(RealDvsVisualizer):
    """Real DVS + workspace: pause UI, space toggles pause (returned in VizResult)."""

    def __init__(self, params: RealDvsWorkspaceVisualizerParams):
        super().__init__(RealDvsVisualizerParams(
            width=params.width,
            height=params.height,
            top_mask_y_cam1=params.top_mask_y_cam1,
            top_mask_y_cam2=params.top_mask_y_cam2,
            mask_y_cam1=params.mask_y_cam1,
            mask_y_cam2=params.mask_y_cam2,
            event_frames_fn=params.event_frames_fn,
        ))
        self._ws = WorkspacePanelRenderer(params.workspace)

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
        del surfaces
        self._ensure_window(has_workspace=True)
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

        is_paused = paused is True
        workspace_canvas = self._ws.build(command, paused=is_paused, y_meas=None if is_paused else y_meas)
        if title is not None:
            title_str = title
        else:
            title_str = (
                "Paused - table at center | Space: resume | R: start/stop rec | Q: quit"
                if is_paused
                else "Experiment | Space: pause | R: start/stop rec | Q: quit"
            )
        title_str = self._append_y_meas_banner(title_str, y_meas)
        composite = build_composite(title_str, frame1, frame2, workspace_canvas)
        cv2.imshow(self._window_name, composite)
        if _window_closed(self._window_name):
            return VizResult(quit=True, toggle_pause=False)
        key = cv2.waitKeyEx(1)
        key_low = key & 0xFF if key != -1 else -1
        quit_requested = key_low in (ord("q"), ord("Q"), 27)
        toggle_pause = key_low == ord(" ")
        return VizResult(quit=quit_requested, toggle_pause=toggle_pause, key=None if key == -1 else key)
