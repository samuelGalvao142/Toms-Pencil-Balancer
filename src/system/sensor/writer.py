from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
import threading
import time
from typing import Any

import numpy as np


@dataclass
class DVSWriterParams:
    enabled: bool = False
    camera_id: int = 1
    output_dir: str = "logs/dvs_recordings"
    file_stem: str | None = None
    camera_name: str = "DAVIS346"


class DVSWriter:
    DEFAULT_STREAM = "events"

    def __init__(self, params: DVSWriterParams, *, width: int, height: int):
        self.params = params
        self.camera_id = int(params.camera_id)
        self.width = int(width)
        self.height = int(height)

        self._lock = threading.RLock()
        self._csv_handle = None
        self._csv_writer = None
        self._dv = None
        self._aedat_writer = None
        self._last_timestamp_us = -1
        self._recording = False
        self._segment_index = 0
        self._queue: SimpleQueue | None = None
        self._worker: threading.Thread | None = None
        self._sentinel = object()
        self._output_dir = Path(params.output_dir)
        self.aedat4_path: Path | None = None
        self.csv_path: Path | None = None
        self._refresh_output_paths(increment_segment=False)

    @property
    def enabled(self) -> bool:
        return bool(self.params.enabled)

    @property
    def is_recording(self) -> bool:
        return bool(self._recording)

    def set_output_dir(self, output_dir: str | Path) -> None:
        with self._lock:
            was_open = self._csv_handle is not None or self._aedat_writer is not None
            if was_open:
                self.close()
            self._output_dir = Path(output_dir)
            self._refresh_output_paths(increment_segment=False)

    def start_recording(self) -> bool:
        if not self.enabled:
            return False

        with self._lock:
            if self._recording:
                return False
            self._refresh_output_paths(increment_segment=True)
            self._ensure_open()
            self._queue = SimpleQueue()
            self._worker = threading.Thread(target=self._writer_loop, daemon=True)
            self._worker.start()
            self._recording = True
            return True

    def stop_recording(self) -> bool:
        if not self.enabled:
            return False

        with self._lock:
            if not self._recording:
                return False
            self._recording = False
            self._stop_worker()
            self._close_open_files()
            return True

    def toggle_recording(self) -> bool:
        if self.is_recording:
            self.stop_recording()
            return False
        return bool(self.start_recording())

    def record_batch(
        self,
        *,
        raw_event_batches=None,
        raw_events: np.ndarray | None = None,
        processed_events: np.ndarray,
        observation,
        algo,
    ) -> None:
        if not self.enabled or not self.is_recording:
            return

        queue = self._queue
        if queue is None:
            return
        native_batches = self._normalize_native_batches(raw_event_batches)
        queued_raw_events = raw_events
        if native_batches:
            queued_raw_events = None
        queue.put(
            (
                native_batches,
                queued_raw_events,
                self._measurement_timestamp_us(processed_events, raw_events, native_batches),
                *self._observation_values(observation),
            )
        )

    def close(self) -> None:
        with self._lock:
            self._recording = False
            self._stop_worker()
            self._close_open_files()

    def _refresh_output_paths(self, *, increment_segment: bool) -> None:
        if increment_segment:
            self._segment_index += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if self.params.file_stem is None:
            stem = f"dvs_session_{timestamp}"
        elif self._segment_index <= 1:
            stem = self.params.file_stem
        else:
            stem = f"{self.params.file_stem}_rec{self._segment_index:02d}"
        self.aedat4_path = self._output_dir / f"{stem}_cam{self.camera_id}.aedat4"
        self.csv_path = self._output_dir / f"{stem}_cam{self.camera_id}_hough.csv"

    def _ensure_open(self) -> None:
        if self._csv_handle is not None and self._aedat_writer is not None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._open_csv()
        self._open_aedat_writer()

    def _open_csv(self) -> None:
        if self.csv_path is None:
            raise RuntimeError("CSV output path was not initialized.")
        self._csv_handle = self.csv_path.open("w", newline="", encoding="utf-8", buffering=1024 * 1024)
        self._csv_writer = csv.DictWriter(
            self._csv_handle,
            fieldnames=[
                "timestamp_us",
                "slope",
                "intercept",
            ],
        )
        self._csv_writer.writeheader()

    def _open_aedat_writer(self) -> None:
        try:
            import dv_processing as dv  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "dv_processing is required when DVSWriterParams.enabled=True."
            ) from exc

        self._dv = dv
        config = dv.io.MonoCameraWriter.Config(self.params.camera_name)
        resolution = (self.width, self.height)
        if not self._try_add_named_event_stream(config, resolution, self.DEFAULT_STREAM):
            raise RuntimeError(
                "The installed dv_processing Python bindings do not expose an event "
                "stream configuration compatible with AEDAT4 writing."
            )
        if self.aedat4_path is None:
            raise RuntimeError("AEDAT4 output path was not initialized.")
        self._aedat_writer = dv.io.MonoCameraWriter(str(self.aedat4_path), config)

    def _try_add_named_event_stream(self, config, resolution, stream_name: str) -> bool:
        for args in (
            (resolution, stream_name, self.params.camera_name),
            (resolution, stream_name),
        ):
            try:
                config.addEventStream(*args)
                return True
            except TypeError:
                continue
        return False

    def _write_event_store(self, store, stream_name: str) -> None:
        try:
            self._aedat_writer.writeEvents(store, stream_name)
            return
        except TypeError:
            pass

        try:
            self._aedat_writer.writeEvents(stream_name, store)
            return
        except TypeError as exc:
            raise RuntimeError(
                "The installed dv_processing Python bindings do not expose a named "
                "writeEvents overload for multiple event streams."
            ) from exc

    def _write_hough_row(self, *, timestamp_us: int, slope: float, intercept: float) -> None:
        self._csv_writer.writerow(
            {
                "timestamp_us": timestamp_us,
                "slope": slope,
                "intercept": intercept,
            }
        )

    def _events_to_store(self, events_np: np.ndarray):
        timestamps = self._event_timestamps(events_np)
        xs = np.asarray(events_np["x"], dtype=np.int64)
        ys = np.asarray(events_np["y"], dtype=np.int64)
        polarity = self._extract_field(events_np, ("p", "polarity"))
        all_store = self._dv.EventStore()
        if polarity is not None:
            polarity = np.asarray(polarity, dtype=bool)
        if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
            indices = np.argsort(timestamps, kind="stable")
        else:
            indices = range(len(timestamps))

        for idx in indices:
            event_polarity = bool(polarity[idx]) if polarity is not None else True
            event = (
                int(timestamps[idx]),
                int(xs[idx]),
                int(ys[idx]),
                event_polarity,
            )
            all_store.push_back(*event)
        return all_store

    def _event_timestamps(self, events_np: np.ndarray) -> np.ndarray:
        if events_np is None or len(events_np) == 0:
            return np.zeros(0, dtype=np.int64)

        timestamp_field = self._extract_field(events_np, ("t", "timestamp", "ts"))
        if timestamp_field is None:
            base = max(time.time_ns() // 1000, self._last_timestamp_us + 1)
            timestamps = base + np.arange(len(events_np), dtype=np.int64)
        else:
            timestamps = np.asarray(timestamp_field, dtype=np.int64).reshape(-1)
            if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
                timestamps = np.sort(timestamps, kind="stable")

        if timestamps.size > 0:
            self._last_timestamp_us = max(self._last_timestamp_us, int(timestamps[-1]))
        return timestamps

    def _extract_field(self, events_np: np.ndarray, candidates: tuple[str, ...]):
        names = getattr(getattr(events_np, "dtype", None), "names", None)
        if not names:
            return None
        for candidate in candidates:
            if candidate in names:
                return events_np[candidate]
        return None

    def _measurement_timestamp_us(
        self,
        processed_events: np.ndarray,
        raw_events: np.ndarray | None,
        raw_event_batches: list[Any] | None,
    ) -> int:
        if processed_events is not None and len(processed_events) > 0:
            return int(self._event_timestamps(processed_events)[-1])
        if raw_events is not None and len(raw_events) > 0:
            return int(self._event_timestamps(raw_events)[-1])
        if raw_event_batches:
            timestamp = self._native_batches_last_timestamp_us(raw_event_batches)
            if timestamp is not None:
                return timestamp
        return int(max(time.time_ns() // 1000, self._last_timestamp_us + 1))

    @staticmethod
    def _observation_values(observation) -> tuple[float, float]:
        if observation is None:
            return np.nan, np.nan
        if isinstance(observation, tuple):
            if len(observation) >= 2:
                slope, intercept = observation[0], observation[1]
            else:
                return np.nan, np.nan
        else:
            slope = getattr(observation, "slope", np.nan)
            intercept = getattr(observation, "intercept", np.nan)

        slope = np.nan if slope is None else float(slope)
        intercept = np.nan if intercept is None else float(intercept)
        return slope, intercept

    def _writer_loop(self) -> None:
        queue = self._queue
        if queue is None:
            return

        while True:
            payload = queue.get()
            if payload is self._sentinel:
                break

            native_batches, raw_events, timestamp_us, slope, intercept = payload
            if native_batches:
                self._write_native_batches(native_batches)
            elif raw_events is not None and len(raw_events) > 0:
                all_store = self._events_to_store(raw_events)
                self._write_event_store(all_store, self.DEFAULT_STREAM)
            self._write_hough_row(timestamp_us=timestamp_us, slope=slope, intercept=intercept)

    def _normalize_native_batches(self, raw_event_batches) -> list[Any] | None:
        if raw_event_batches is None:
            return None
        if isinstance(raw_event_batches, list):
            return [batch for batch in raw_event_batches if batch is not None]
        return [raw_event_batches] if raw_event_batches is not None else None

    def _write_native_batches(self, native_batches: list[Any]) -> None:
        for batch in native_batches:
            self._write_event_store(batch, self.DEFAULT_STREAM)

    def _native_batches_last_timestamp_us(self, native_batches: list[Any]) -> int | None:
        for batch in reversed(native_batches):
            timestamp = self._native_batch_last_timestamp_us(batch)
            if timestamp is not None:
                self._last_timestamp_us = max(self._last_timestamp_us, timestamp)
                return timestamp
        return None

    def _native_batch_last_timestamp_us(self, native_batch) -> int | None:
        highest_time = getattr(native_batch, "getHighestTime", None)
        if callable(highest_time):
            try:
                timestamp = highest_time()
            except TypeError:
                timestamp = None
            else:
                if timestamp is not None:
                    return int(timestamp)

        numpy_batch = native_batch.numpy()
        if len(numpy_batch) == 0:
            return None
        return int(self._event_timestamps(numpy_batch)[-1])

    def _stop_worker(self) -> None:
        queue = self._queue
        worker = self._worker
        if queue is not None:
            queue.put(self._sentinel)
        if worker is not None:
            worker.join()
        self._queue = None
        self._worker = None

    def _close_open_files(self) -> None:
        if self._csv_handle is not None:
            self._csv_handle.flush()
            self._csv_handle.close()
            self._csv_handle = None
            self._csv_writer = None

        writer = self._aedat_writer
        self._aedat_writer = None
        if writer is None:
            return

        close_fn = getattr(writer, "close", None)
        if callable(close_fn):
            close_fn()
