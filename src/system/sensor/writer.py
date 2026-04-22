from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
import threading
import time

import numpy as np


@dataclass
class DVSWriterParams:
    enabled: bool = False
    camera_id: int = 1
    output_dir: str = "logs/dvs_recordings"
    file_stem: str | None = None
    camera_name: str = "DAVIS346"


class DVSWriter:
    POSITIVE_STREAM = "events_positive"
    NEGATIVE_STREAM = "events_negative"

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
        raw_events: np.ndarray,
        processed_events: np.ndarray,
        observation,
        algo,
    ) -> None:
        if not self.enabled or not self.is_recording:
            return

        queue = self._queue
        if queue is None:
            return
        queue.put(
            (
                raw_events,
                self._measurement_timestamp_us(processed_events, raw_events),
                self._state_value(algo, "linear_m"),
                self._state_value(algo, "linear_b"),
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
                "lin_m",
                "lin_b",
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
        if not self._try_add_named_event_stream(config, resolution, self.POSITIVE_STREAM):
            raise RuntimeError(
                "The installed dv_processing Python bindings do not expose named "
                "event streams required to store positive/negative events separately."
            )
        if not self._try_add_named_event_stream(config, resolution, self.NEGATIVE_STREAM):
            raise RuntimeError(
                "The installed dv_processing Python bindings do not expose named "
                "event streams required to store positive/negative events separately."
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

    def _write_hough_row(self, *, timestamp_us: int, lin_m: float, lin_b: float) -> None:
        self._csv_writer.writerow(
            {
                "timestamp_us": timestamp_us,
                "lin_m": lin_m,
                "lin_b": lin_b,
            }
        )

    def _events_to_stores(self, events_np: np.ndarray):
        timestamps = self._event_timestamps(events_np)
        xs = np.asarray(events_np["x"], dtype=np.int64)
        ys = np.asarray(events_np["y"], dtype=np.int64)
        polarity = self._extract_field(events_np, ("p", "polarity"))
        if polarity is None:
            raise ValueError(
                "Event recording is enabled, but the event batch does not contain a polarity field."
            )
        polarity = np.asarray(polarity, dtype=bool)

        pos_store = self._dv.EventStore()
        neg_store = self._dv.EventStore()
        if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
            indices = np.argsort(timestamps, kind="stable")
        else:
            indices = range(len(timestamps))

        pos_count = 0
        neg_count = 0
        for idx in indices:
            if bool(polarity[idx]):
                target_store = pos_store
                pos_count += 1
            else:
                target_store = neg_store
                neg_count += 1
            target_store.push_back(
                int(timestamps[idx]),
                int(xs[idx]),
                int(ys[idx]),
                bool(polarity[idx]),
            )
        return pos_store, neg_store, pos_count, neg_count

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

    def _measurement_timestamp_us(self, processed_events: np.ndarray, raw_events: np.ndarray) -> int:
        if processed_events is not None and len(processed_events) > 0:
            return int(self._event_timestamps(processed_events)[-1])
        if raw_events is not None and len(raw_events) > 0:
            return int(self._event_timestamps(raw_events)[-1])
        return int(max(time.time_ns() // 1000, self._last_timestamp_us + 1))

    @staticmethod
    def _state_value(algo, attr_name: str) -> float:
        state = getattr(algo, "state", None)
        value = getattr(state, attr_name, np.nan)
        return float(value)

    def _writer_loop(self) -> None:
        queue = self._queue
        if queue is None:
            return

        while True:
            payload = queue.get()
            if payload is self._sentinel:
                break

            raw_events, timestamp_us, lin_m, lin_b = payload
            pos_store, neg_store, pos_count, neg_count = self._events_to_stores(raw_events)
            if pos_count > 0:
                self._write_event_store(pos_store, self.POSITIVE_STREAM)
            if neg_count > 0:
                self._write_event_store(neg_store, self.NEGATIVE_STREAM)
            self._write_hough_row(timestamp_us=timestamp_us, lin_m=lin_m, lin_b=lin_b)

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
