"""Recorder round-trips, crash tolerance, and the label-to-window join."""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.recorder import Recorder, Recording, latest, open_recording


def _fill(root, n: int, dim: int = 3, start: float = 1000.0, step: float = 10.0):
    with Recorder(root, dim, 300.0) as rec:
        for i in range(n):
            rec.record(
                np.full(dim, float(i), dtype=np.float32),
                sensors={"sensor.temp": 20.0 + i, "sensor.motion": float(i % 2)},
                t=start + i * step,
            )
    return root


def test_round_trip(tmp_path) -> None:
    root = _fill(tmp_path / "r1", 4)
    rec = open_recording(root)
    assert rec.features.shape == (4, 3)
    assert rec.features[2].tolist() == [2.0, 2.0, 2.0]
    assert len(rec.windows) == 4
    assert rec.windows[0]["sensors"]["sensor.temp"] == 20.0


def test_feature_dim_mismatch_is_rejected(tmp_path) -> None:
    root = _fill(tmp_path / "r2", 2, dim=3)
    with pytest.raises(ValueError, match="feature_dim"):
        Recorder(root, 5, 300.0)


def test_wrong_width_row_is_rejected(tmp_path) -> None:
    with Recorder(tmp_path / "r3", 3, 300.0) as rec, pytest.raises(ValueError, match="expected 3"):
        rec.record(np.zeros(4, dtype=np.float32))


def test_reopening_appends_and_counts_from_the_file(tmp_path) -> None:
    """n_windows comes from the file size, so a restart must not lose or double-count rows."""
    root = _fill(tmp_path / "r4", 3)
    with Recorder(root, 3, 300.0) as rec:
        assert rec.n_windows == 3
        rec.record(np.zeros(3, dtype=np.float32))
        assert rec.n_windows == 4
    assert open_recording(root).features.shape[0] == 4


def test_partial_trailing_row_is_dropped_not_fatal(tmp_path) -> None:
    """A hard kill mid-write leaves a short row; the recording must still open."""
    root = _fill(tmp_path / "r5", 2, dim=3)
    with (root / "features.f32").open("ab") as fh:
        fh.write(np.zeros(2, dtype=np.float32).tobytes())  # 2 of 3 floats
    assert open_recording(root).features.shape == (2, 3)


def test_missing_sensor_reads_as_nan_not_zero(tmp_path) -> None:
    """An absent sensor is not a sensor reporting zero."""
    with Recorder(tmp_path / "r6", 2, 300.0) as rec:
        rec.record(np.zeros(2, dtype=np.float32), sensors={"sensor.a": 1.0}, t=0.0)
        rec.record(np.zeros(2, dtype=np.float32), sensors={"sensor.a": 3.0}, t=1.0)
    matrix, present = open_recording(tmp_path / "r6").sensor_matrix(
        ["sensor.a", "sensor.never_seen"]
    )
    assert present == ["sensor.a"]
    assert matrix.tolist() == [[1.0], [3.0]]


def test_labelled_join_takes_the_most_recent_label(tmp_path) -> None:
    root = tmp_path / "r7"
    with Recorder(root, 2, 300.0) as rec:
        for i in range(6):
            rec.record(np.full(2, float(i), dtype=np.float32), t=float(i * 10))
        rec.label("quiet", t=0.0)
        rec.label("busy", t=30.0)
    x, y, used = open_recording(root).labelled(horizon_s=100.0)
    assert y.tolist() == ["quiet", "quiet", "quiet", "busy", "busy", "busy"]
    assert x.shape == (6, 2)
    assert len(used) == 6


def test_labelled_drops_windows_beyond_the_horizon(tmp_path) -> None:
    """Windows nobody annotated must not be guessed at."""
    root = tmp_path / "r8"
    with Recorder(root, 2, 300.0) as rec:
        for i in range(5):
            rec.record(np.full(2, float(i), dtype=np.float32), t=float(i * 100))
        rec.label("busy", t=0.0)
    x, y, _ = open_recording(root).labelled(horizon_s=150.0)
    # windows at t=0,100 are within 150s of the label; t=200,300,400 are not
    assert y.tolist() == ["busy", "busy"]
    assert x.shape == (2, 2)


def test_labelled_with_no_labels_is_empty(tmp_path) -> None:
    root = _fill(tmp_path / "r9", 3)
    x, y, _ = open_recording(root).labelled()
    assert x.shape[0] == 0 and y.shape[0] == 0


def test_empty_label_is_rejected(tmp_path) -> None:
    with Recorder(tmp_path / "r10", 2, 300.0) as rec, pytest.raises(ValueError):
        rec.label("   ")


def test_latest_picks_the_newest_recording(tmp_path) -> None:
    _fill(tmp_path / "old", 1)
    _fill(tmp_path / "new", 1)
    assert latest(tmp_path).name == "new"


def test_no_recordings_yields_none(tmp_path) -> None:
    assert latest(tmp_path / "nothing") is None


def test_recording_requires_meta(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Recording(tmp_path / "absent")


def test_meta_records_the_extra_context(tmp_path) -> None:
    with Recorder(tmp_path / "r11", 2, 300.0, meta={"roles": ["visual"], "rev": "abc"}):
        pass
    meta = open_recording(tmp_path / "r11").meta
    assert meta["roles"] == ["visual"] and meta["rev"] == "abc"
    assert meta["window_ms"] == 300.0
