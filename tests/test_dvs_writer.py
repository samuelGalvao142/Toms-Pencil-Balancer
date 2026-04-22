from pathlib import Path
from datetime import datetime
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_dv_processing_stub():
    stub = types.ModuleType("dv_processing")

    class EventStore:
        def __init__(self):
            self.events = []

        def push_back(self, timestamp, x, y, polarity):
            self.events.append((timestamp, x, y, polarity))

    class Config:
        def __init__(self, camera_name):
            self.camera_name = camera_name
            self.event_streams = []

        def addEventStream(self, resolution, name="events", source=None):
            self.event_streams.append(
                {"resolution": tuple(resolution), "name": name, "source": source}
            )

    class MonoCameraWriter:
        last_instance = None

        def __init__(self, path, config):
            self.path = path
            self.config = config
            self.writes = []
            self.closed = False
            MonoCameraWriter.last_instance = self

        def writeEvents(self, events, stream_name):
            self.writes.append((stream_name, list(events.events)))

        def close(self):
            self.closed = True

    MonoCameraWriter.Config = Config
    stub.EventStore = EventStore
    stub.io = types.SimpleNamespace(MonoCameraWriter=MonoCameraWriter)
    sys.modules["dv_processing"] = stub
    return stub


def _events():
    return np.array(
        [
            (1000, 10, 20, True),
            (1001, 11, 21, False),
            (1002, 12, 22, True),
        ],
        dtype=[
            ("t", np.int64),
            ("x", np.int16),
            ("y", np.int16),
            ("p", np.bool_),
        ],
    )


def test_dvs_writer_records_two_event_streams_and_hough_csv(tmp_path: Path):
    _install_dv_processing_stub()

    from src.system.sensor.writer import DVSWriter, DVSWriterParams
    from src.shared import CameraObservation

    writer = DVSWriter(
        DVSWriterParams(
            enabled=True,
            camera_id=1,
            output_dir=str(tmp_path),
            file_stem="trial01",
            camera_name="DAVIS346-test",
        ),
        width=346,
        height=260,
    )
    assert writer.start_recording() is True

    algo = types.SimpleNamespace(
        current_centered_line=CameraObservation(slope=0.45, intercept=-3.0),
        state=types.SimpleNamespace(
            quadratic_m2=1.0,
            cross_mb=2.0,
            quadratic_b2=3.0,
            linear_m=4.0,
            linear_b=5.0,
        ),
    )

    raw_events = _events()
    processed_events = raw_events[:2]
    observation = CameraObservation(slope=0.5, intercept=120.0)

    writer.record_batch(
        raw_events=raw_events,
        processed_events=processed_events,
        observation=observation,
        algo=algo,
    )
    assert writer.stop_recording() is True
    writer.close()

    dv_stub = sys.modules["dv_processing"]
    writer_stub = dv_stub.io.MonoCameraWriter.last_instance
    assert writer_stub is not None
    assert writer_stub.closed is True
    assert [stream["name"] for stream in writer_stub.config.event_streams] == [
        "events_positive",
        "events_negative",
    ]
    assert [name for name, _events_written in writer_stub.writes] == [
        "events_positive",
        "events_negative",
    ]
    assert len(writer_stub.writes[0][1]) == 2
    assert len(writer_stub.writes[1][1]) == 1

    csv_path = tmp_path / "trial01_cam1_hough.csv"
    assert csv_path.exists()
    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert rows[0] == "timestamp_us,lin_m,lin_b"
    assert rows[1] == "1001,4.0,5.0"


def test_experiment_closes_sensor_after_run_experiment():
    from src.experiment.experiment import Experiment, ExperimentParams
    from src.shared import TimingParams

    class DummySystem:
        def __init__(self):
            self.sensor = types.SimpleNamespace(close=self._close_sensor)
            self.closed = False

        def _close_sensor(self):
            self.closed = True

    noop = types.SimpleNamespace()
    experiment = Experiment(
        ExperimentParams(
            system=DummySystem(),
            logger=noop,
            stop_condition=noop,
            realtime_visualizer=noop,
            offline_visualizer=noop,
            progress=noop,
            pacing=noop,
            scheduler=noop,
            n_trials=0,
            timing=TimingParams(total_time=0.0, dt=0.01, actuator_dt=0.01),
        )
    )

    results = experiment.run_experiment()

    assert results == []
    assert experiment.system.closed is True


def test_allocate_run_log_dir_creates_sequential_dated_folders(tmp_path: Path):
    from src.experiment.log_paths import allocate_run_log_dir

    first = allocate_run_log_dir(tmp_path, now=datetime(2026, 4, 22, 9, 0, 0))
    second = allocate_run_log_dir(tmp_path, now=datetime(2026, 4, 22, 9, 1, 0))
    third = allocate_run_log_dir(tmp_path, now=datetime(2026, 4, 23, 9, 2, 0))

    assert first.name == "log-1-04-22-2026"
    assert second.name == "log-2-04-22-2026"
    assert third.name == "log-3-04-23-2026"


def test_dvs_writer_can_be_redirected_to_run_log_dir(tmp_path: Path):
    _install_dv_processing_stub()

    from src.system.sensor.writer import DVSWriter, DVSWriterParams

    writer = DVSWriter(
        DVSWriterParams(
            enabled=True,
            output_dir=str(tmp_path / "initial"),
            file_stem="trial01",
        ),
        width=346,
        height=260,
    )

    run_dir = tmp_path / "log-7-04-22-2026"
    writer.set_output_dir(run_dir)

    assert writer.aedat4_path == run_dir / "trial01_cam1.aedat4"
    assert writer.csv_path == run_dir / "trial01_cam1_hough.csv"


def test_dvs_writer_toggle_recording_creates_new_segment_names(tmp_path: Path):
    _install_dv_processing_stub()

    from src.system.sensor.writer import DVSWriter, DVSWriterParams

    writer = DVSWriter(
        DVSWriterParams(
            enabled=True,
            output_dir=str(tmp_path),
            file_stem="trial01",
        ),
        width=346,
        height=260,
    )

    assert writer.is_recording is False
    assert writer.start_recording() is True
    assert writer.is_recording is True
    assert writer.aedat4_path == tmp_path / "trial01_cam1.aedat4"
    assert writer.stop_recording() is True
    assert writer.is_recording is False
    assert writer.start_recording() is True
    assert writer.aedat4_path == tmp_path / "trial01_rec02_cam1.aedat4"
