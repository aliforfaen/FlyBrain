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


def test_a_partial_row_is_repaired_before_the_next_append(tmp_path) -> None:
    """The root cause: a fragment must never end up in the *middle* of the file.

    Regression. The reader already dropped a trailing partial row, but the writer reopened in
    append mode and wrote *after* it. From then on every row sat at a constant offset from
    ``windows.jsonl``, and the row count still looked correct — so a sensor reading for one
    moment would be silently paired with the spike vector from another.
    """
    root = _fill(tmp_path / "r5b", 2, dim=3)
    with (root / "features.f32").open("ab") as fh:
        fh.write(np.zeros(2, dtype=np.float32).tobytes())  # 2 of 3 floats: a torn write

    with Recorder(root, 3, 300.0) as rec:
        rec.record(np.full(3, 9.0, dtype=np.float32), sensors={"sensor.temp": 99.0}, t=5000.0)

    rec = open_recording(root)
    assert rec.features.shape == (3, 3)
    # Without the repair this row read back as [0.0, 0.0, 9.0] — the fragment, then one float
    # of the real row.
    assert rec.features[2].tolist() == [9.0, 9.0, 9.0]
    assert len(rec.windows) == 3
    assert rec.windows[2]["sensors"]["sensor.temp"] == 99.0


def test_an_extra_feature_row_is_not_paired_with_a_window(tmp_path) -> None:
    """A crash between the feature write and the window write leaves one orphan row."""
    root = _fill(tmp_path / "r5c", 3, dim=3)
    with (root / "features.f32").open("ab") as fh:
        fh.write(np.full(3, 77.0, dtype=np.float32).tobytes())  # row written, window line lost

    rec = open_recording(root)
    assert rec.paired_count() == 3
    assert rec.features.shape[0] == 3
    assert len(rec.windows) == 3


def test_a_lost_feature_row_does_not_index_past_the_features(tmp_path) -> None:
    """The dangerous direction: more windows than feature rows.

    Regression. ``labelled`` indexes features using window positions, so a short feature file
    raised ``IndexError`` — or, worse, paired windows with the wrong rows.
    """
    root = _fill(tmp_path / "r5d", 3, dim=3)
    with Recorder(root, 3, 300.0) as rec:
        rec.label("busy", t=1000.0)

    # Drop a whole feature row, leaving three window lines behind it.
    path = root / "features.f32"
    path.write_bytes(path.read_bytes()[: 2 * 3 * 4])

    rec = open_recording(root)
    assert rec.features.shape[0] == 2
    assert len(rec.windows) == 2

    features, targets, _ = rec.labelled(horizon_s=1e9)
    assert features.shape[0] == 2
    assert len(targets) == 2


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


# --------------------------------------------------- the provenance a recording carries


class TestRecordingProvenance:
    """Pacing is part of the training regime, so it belongs in the recording, not in a note.

    ``AGENTS.md`` #2 is "train and run in the same regime". With a change trigger enabled, a
    recording is a sample of *events*; with a plain heartbeat it is a sample of *time*. Those are
    different distributions, and a readout fitted on one and run against the other loses accuracy
    in exactly the silent way the rest-basin bug did. The check has to be mechanical.
    """

    def test_the_meta_includes_the_pacing_configuration(self) -> None:
        from flybrain.loop import LoopConfig
        from flybrain.server import recording_meta

        cfg = LoopConfig(interval_s=60.0, poll_s=5.0, burst_s=10.0, trigger_delta=0.3)
        pacing = recording_meta(cfg)["pacing"]
        assert pacing == {
            "heartbeat_s": 60.0,
            "poll_s": 5.0,
            "burst_s": 10.0,
            "trigger_delta": 0.3,
        }

    def test_the_meta_still_carries_what_it_always_did(self) -> None:
        """Adding a key must not quietly drop the entity ids a recording is useless without."""
        from flybrain.loop import LoopConfig
        from flybrain.server import recording_meta

        cfg = LoopConfig(temperature_entity="sensor.hall", light_entity="light.lamp")
        meta = recording_meta(cfg)
        assert meta["temperature_entity"] == "sensor.hall"
        assert meta["light_entity"] == "light.lamp"
        assert meta["mode"] == "mock"

    def test_the_pacing_config_reaches_the_file(self, tmp_path) -> None:
        """The round trip, not just the dictionary: this is what a later refit will read."""
        from flybrain.loop import LoopConfig
        from flybrain.server import recording_meta

        cfg = LoopConfig(trigger_delta=0.4, interval_s=30.0)
        root = tmp_path / "paced"
        with Recorder(root, 2, 300.0, meta=recording_meta(cfg)):
            pass
        meta = open_recording(root).meta
        assert meta["pacing"]["trigger_delta"] == 0.4
        assert meta["pacing"]["heartbeat_s"] == 30.0

    def test_a_fixed_heartbeat_is_still_recorded_explicitly(self) -> None:
        """`trigger_delta: 0` is a meaningful value, not a missing one.

        A readout fitted on a plain heartbeat is only valid at that heartbeat, so the number has
        to be in the file even when the trigger is off.
        """
        from flybrain.loop import LoopConfig
        from flybrain.server import recording_meta

        pacing = recording_meta(LoopConfig())["pacing"]
        assert pacing["trigger_delta"] == 0.0
        assert pacing["heartbeat_s"] == 15.0
