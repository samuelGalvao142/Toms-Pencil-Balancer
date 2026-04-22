"""
DVS camera reader for DAVIS346 event cameras.
Wraps dv-processing for opening cameras and reading event batches.
"""

import datetime
import numpy as np

# DAVIS346 resolution
DAVIS346_WIDTH = 346
DAVIS346_HEIGHT = 260


def discover_devices() -> list:
    """
    Discover connected DVS cameras.
    Returns a list of device identifiers suitable for passing to DVSReader.
    """
    import dv_processing as dv

    return dv.io.camera.discover()


class DVSReader:
    """
    Wraps a single dv-processing DAVIS346 camera.
    Opens by device (from discover_devices()), serial string, or first available if None.
    """

    def __init__(self, device_or_serial: str | None = None, noise_filter_duration_ms: float | None = None):
        """
        Open a DAVIS346 camera.
        Args:
            device_or_serial: Either a device from discover_devices() (e.g. devices[0]),
                             a serial string (e.g. "00000499"), or None for first available.
            noise_filter_duration_ms: None = no filter. > 0 = apply BackgroundActivityNoiseFilter
                with that duration (ms) before returning events.
        """
        import dv_processing as dv

        if device_or_serial is not None:
            self._capture = dv.io.camera.open(device_or_serial)
        else:
            self._capture = dv.io.camera.open()

        res = self._capture.getEventResolution()
        self._width = int(res[0])
        self._height = int(res[1])

        if self._width != DAVIS346_WIDTH or self._height != DAVIS346_HEIGHT:
            raise ValueError(
                f"Expected DAVIS346 resolution {DAVIS346_WIDTH}x{DAVIS346_HEIGHT}, "
                f"got {self._width}x{self._height}"
            )

        self._noise_filter = None
        if noise_filter_duration_ms is not None and noise_filter_duration_ms > 0:
            self._noise_filter = dv.noise.BackgroundActivityNoiseFilter(
                (self._width, self._height),
                backgroundActivityDuration=datetime.timedelta(microseconds=int(noise_filter_duration_ms * 1000)),
            )

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) for CameraModel / PaperHoughLineAlgorithm."""
        return (self._width, self._height)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def get_event_batch(self) -> np.ndarray | None:
        """
        Get next event batch from the camera.
        Returns numpy array with 'x', 'y' fields (and optionally 't', 'p').
        Returns None if no events available.
        If noise_filter_duration_ms was set, events are filtered before conversion to numpy.
        """
        events = self.get_event_batch_native()
        if events is None:
            return None
        return events.numpy()

    def get_event_batch_native(self):
        """
        Get next native dv-processing event batch from the camera.
        Returns None if no events are available.
        If noise_filter_duration_ms was set, events are filtered before being returned.
        """
        events = self._capture.getNextEventBatch()
        if events is None:
            return None
        if self._noise_filter is not None:
            self._noise_filter.accept(events)
            events = self._noise_filter.generateEvents()
        return events

    def is_running(self) -> bool:
        """Whether the camera is still running."""
        return self._capture.isRunning()

    def close(self) -> None:
        """Release camera resources."""
        # dv-processing uses RAII; capture is released when object is destroyed.
        # Explicit close may not exist; we rely on garbage collection or
        # deleting the reference.
        self._capture = None
