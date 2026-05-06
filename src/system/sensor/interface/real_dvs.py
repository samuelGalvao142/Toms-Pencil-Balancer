from dataclasses import dataclass, field
from pathlib import Path
import threading
import time

import numpy as np
from src.shared import (
    CameraObservation,
    CameraPair,
    CameraParams,
    Measurement
)

from .base import VisionModelBase
from src.system.sensor.observation_model.camera_model import CameraModel
from src.system.sensor.reader.dvs_camera_reader import (
    DVSReader,
    discover_devices,
)
from src.system.sensor.algo.dvs_algorithms import mask_events_between_y_lines
from src.system.sensor.writer import DVSWriter, DVSWriterParams

from src.shared import default_camera_params


@dataclass
class RealDVSParams:
    algo:                     object
    obs_model:                object
    cam_params:               CameraParams = field(default_factory=default_camera_params)
    cam1_device:              str | None = None
    cam2_device:              str | None = None
    noise_filter_duration_ms: float | None = None
    writer:                   DVSWriterParams = field(default_factory=DVSWriterParams)


REAL_DVS_PRESETS = {
    "hough": {
        "algo":                     "hough:default",
        "obs_model":                "simple:default",
        "noise_filter_duration_ms": None,
        "cam1_device":              None,
        "cam2_device":              None,
        "writer":                   {"enabled": False},
    },
    "sam": {
        "base":                     "hough",
        "algo":                     "sam:default",
        "noise_filter_duration_ms": 5,
    },
    "analytic_hough": {
        "base":                     "hough",
        "obs_model":                "analytic:default",
    },
    "snn": {
        "base": "hough",
        "algo": "snn:default",
        "obs_model": "simple:default",
        "noise_filter_duration_ms": None,
        "cam1_device": None,
        "cam2_device": None,
        "writer": {"enabled": False},
    },
}


class RealEventCameraInterface(VisionModelBase):
    @staticmethod
    def _resolve_camera_devices(cam1_device: str | None, cam2_device: str | None) -> tuple[str, str]:
        """
        Match the legacy camera-loading behavior by resolving two distinct devices
        before opening either reader.
        """
        if cam1_device is not None and cam2_device is not None:
            if cam1_device == cam2_device:
                raise ValueError("cam1_device and cam2_device must refer to different cameras.")
            return cam1_device, cam2_device

        if cam1_device is None and cam2_device is None:
            devices = discover_devices()
            if len(devices) < 2:
                raise RuntimeError(
                    f"Need at least 2 DVS cameras, but discover_devices() found {len(devices)}: {devices}"
                )
            return devices[0], devices[1]

        raise ValueError(
            "Provide both cam1_device and cam2_device, or leave both unset to auto-discover two cameras."
        )

    def __init__(self, params: RealDVSParams):
        super().__init__(params.cam_params)
        import copy
        p = params
        
        #cam = params.cam_params
        
        self.cam_height_px = p.cam_params.DAVIS346_HEIGHT
        self.cam_width_px = p.cam_params.DAVIS346_WIDTH

        self.cam1_algo = copy.deepcopy(p.algo)
        self.cam2_algo = copy.deepcopy(p.algo)

        self.cam = CameraModel()

        self.obs_model = p.obs_model
  
        noise_filter_duration_ms = p.noise_filter_duration_ms
        
        cam1_device, cam2_device = self._resolve_camera_devices(p.cam1_device, p.cam2_device)
        
        self._dvs_mask_line_y_cam1 = int(p.cam_params.y_mask_line_1)
        self._dvs_mask_line_y_cam2 = int(p.cam_params.y_mask_line_2)
        self._dvs_top_mask_line_y_cam1 = int(p.cam_params.y_mask_top_line_1)
        self._dvs_top_mask_line_y_cam2 = int(p.cam_params.y_mask_top_line_2)

        self._reader1 = DVSReader(cam1_device, noise_filter_duration_ms=noise_filter_duration_ms)
        self._reader2 = DVSReader(cam2_device, noise_filter_duration_ms=noise_filter_duration_ms)
        self._writer = DVSWriter(
            p.writer,
            width=self.cam_width_px,
            height=self.cam_height_px,
        )

        self._latest1: CameraObservation | None = None
        self._latest2: CameraObservation | None = None
        self._last_valid_cams_camnorm: CameraPair | None = None
        self._last_valid_y_meas: Measurement | None = None
        
        self._surface1 = np.zeros((self.cam_height_px, self.cam_width_px), dtype=np.float32)
        self._surface2 = np.zeros((self.cam_height_px, self.cam_width_px), dtype=np.float32)
        
        self._decay_display = 0.5
        self._lock = threading.Lock()
        self._stop = threading.Event()
        
        self._thread1 = threading.Thread(target=self._reader_loop, args=(self._reader1, self.cam1_algo, 1))
        self._thread2 = threading.Thread(target=self._reader_loop, args=(self._reader2, self.cam2_algo, 2))
        
        self._thread1.daemon = True
        self._thread2.daemon = True
        
        self._thread1.start()
        self._thread2.start()

        self.last_line_observation = None

    def _reader_loop(self, reader, algo, _cam_id: int):
        """Background loop: drain all queued batches, update algo, store latest."""

        surface = self._surface1 if _cam_id == 1 else self._surface2
        mask_y = self._dvs_mask_line_y_cam1 if _cam_id == 1 else self._dvs_mask_line_y_cam2
        top_mask_y = self._dvs_top_mask_line_y_cam1 if _cam_id == 1 else self._dvs_top_mask_line_y_cam2
        while not self._stop.is_set() and reader.is_running():
            native_batches = []
            batches = []
            while True:
                native_batch = reader.get_event_batch_native()
                if native_batch is None:
                    break
                batch = native_batch.numpy()
                if len(batch) == 0:
                    break
                native_batches.append(native_batch)
                batches.append(batch)

            if batches:
                raw_events = np.concatenate(batches)
                events = mask_events_between_y_lines(
                    raw_events,
                    min_line_y=top_mask_y,
                    max_line_y=mask_y,
                    frame_height=self.cam_height_px,
                )
                surface *= self._decay_display
                if len(events) > 0:
                    np.add.at(surface, (events["y"], events["x"]), 1.0)
                result = algo.update(events)
                if self._writer.enabled and _cam_id == self._writer.camera_id:
                    self._writer.record_batch(
                        raw_event_batches=native_batches,
                        raw_events=raw_events,
                        processed_events=events,
                        observation=result,
                        algo=algo,
                    )
                if not isinstance(result, tuple):
                    with self._lock:
                        if _cam_id == 1:
                            self._latest1 = result
                        else:
                            self._latest2 = result
            else:
                time.sleep(0.0001)

    def get_surfaces(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return copy of current event surfaces for visualization."""
        with self._lock:
            return self._surface1.copy(), self._surface2.copy()

    def get_event_accumulator_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Alias for :meth:`get_surfaces` — decaying event-accumulator images for display."""
        return self.get_surfaces()

    @property
    def recording_enabled(self) -> bool:
        return bool(self._writer.enabled)

    @property
    def recording_active(self) -> bool:
        return bool(self._writer.is_recording)

    @property
    def recording_paths(self) -> tuple[Path | None, Path | None]:
        return self._writer.aedat4_path, self._writer.csv_path

    def start_recording(self) -> bool:
        return self._writer.start_recording()

    def stop_recording(self) -> bool:
        return self._writer.stop_recording()

    def toggle_recording(self) -> bool:
        return self._writer.toggle_recording()

    def get_y(self, state_true=None) -> Measurement:
        cams_camnorm = self.get_z()
        if cams_camnorm is not None:
            self._last_valid_cams_camnorm = cams_camnorm
        elif self._last_valid_cams_camnorm is not None:
            cams_camnorm = self._last_valid_cams_camnorm
        elif state_true is not None:
            # Startup fallback: keep the system stepping until both trackers lock on.
            cams_camnorm = super().project_state_to_z(state_true)
        elif self._last_valid_y_meas is not None:
            return self._last_valid_y_meas
        else:
            raise RuntimeError("No DVS observation available yet and no fallback measurement exists.")

        self.last_line_observation = cams_camnorm

        y_meas = self.cams_to_measurement(cams_camnorm=cams_camnorm)
        self._last_valid_y_meas = y_meas
        
        return y_meas
        
    def get_z(self) -> CameraPair | None:
        """
        Return latest CameraPair(camnorm) from Hough
        """
        with self._lock:
            obs1_px = self._latest1
            obs2_px = self._latest2

        if obs1_px is None or obs2_px is None:
            print("No observation vision.py line 226")
            return None

        obs1 = self.cam.pixel_to_camnorm(obs1_px)
        obs2 = self.cam.pixel_to_camnorm(obs2_px)

        return CameraPair(
            CameraObservation(slope=obs1.slope, intercept=obs1.intercept),
            CameraObservation(slope=obs2.slope, intercept=obs2.intercept),
        )

    def cams_to_measurement(self, cams_camnorm):
        cams_px = CameraPair(
            cam1=self.cam.camnorm_to_pixel(cams_camnorm.cam1),
            cam2=self.cam.camnorm_to_pixel(cams_camnorm.cam2),
        )

        y_meas = self.obs_model.estimate(cams_px)

        if y_meas is not None and self.is_valid_y_meas(y_meas):
            return y_meas

        return super().cams_to_measurement(cams_camnorm=cams_camnorm)

    def reset(self):
        """Reset both Hough algorithms."""
        self.last_line_observation = None
        self._last_valid_cams_camnorm = None
        self._last_valid_y_meas = None
        self.cam1_algo.reset()
        self.cam2_algo.reset()
        with self._lock:
            self._latest1 = None
            self._latest2 = None

    def close(self):
        """Stop reader threads and release cameras."""
        self._stop.set()
        self._thread1.join(timeout=1.0)
        self._thread2.join(timeout=1.0)
        self._writer.close()
        self._reader1.close()
        self._reader2.close()
