"""Multi-channel drive: the temperature path plus extra sensory pathways.

The critical property here is that every channel reaches the simulator in a **single**
``set_drive`` call. ``set_drive`` replaces the whole drive map rather than merging into it, so
driving temperature and then the extras separately would silently leave only the last one
applied — a bug that would look like "the extra sensors do nothing" rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.codec import ChannelSpec
from flybrain.experiment import ColourReadout, ExperimentConfig, drive_current_for_rate
from flybrain.loop import LiveLoop, LoopConfig
from flybrain.types import Signal, SignalKind

DIM = 4


class _CountingSim:
    """Records every ``set_drive`` call, so "one call for all channels" is testable."""

    def __init__(self) -> None:
        self.params = type("P", (), {"dt_ms": 0.1})()
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def set_drive(self, indices, current_mv) -> None:
        idx = np.asarray(indices).reshape(-1)
        cur = np.broadcast_to(np.asarray(current_mv, dtype=float), idx.shape).copy()
        self.calls.append((idx, cur))

    def clear_drive(self) -> None:
        self.calls.clear()

    @property
    def driven(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for idx, cur in self.calls:
            out.update(dict(zip(idx.tolist(), cur.tolist())))
        return out


def _loop(channels=(), smooth_ms=0.0):
    config = ExperimentConfig()
    readout = ColourReadout(n_features=DIM + 1, l2=0.0)
    readout.W = np.zeros(DIM + 1)
    sim = _CountingSim()
    loop = LiveLoop(
        sim,
        config,
        readout,
        input_indices=np.arange(2),
        readout_indices=np.arange(DIM),
        loop_config=LoopConfig(smooth_ms=smooth_ms),
    )
    if channels:
        loop.configure_channels(list(channels))
    return loop, sim, config


def _channel(entity_id="binary_sensor.hall_motion", indices=(10, 11, 12), **kw):
    return ChannelSpec(
        entity_id=entity_id,
        kind=kw.pop("kind", SignalKind.MOTION),
        neuron_indices=np.array(indices),
        vmin=kw.pop("vmin", 0.0),
        vmax=kw.pop("vmax", 1.0),
        **kw,
    )


def _signal(entity_id, value, state="on", kind=SignalKind.MOTION, attributes=None):
    return Signal(
        entity_id=entity_id,
        kind=kind,
        value=value,
        state=state,
        timestamp=0.0,
        attributes=dict(attributes or {}),
    )


# ------------------------------------------------------------------ configuration


def test_temperature_entity_is_excluded_from_extra_channels() -> None:
    """It keeps its trained path; routing it twice would double-drive the same neurons."""
    loop, _, config = _loop()
    loop.configure_channels(
        [
            _channel(config.temperature_entity, indices=(1, 2)),
            _channel("binary_sensor.hall_motion", indices=(10, 11)),
        ]
    )
    assert [c.entity_id for c in loop.channels] == ["binary_sensor.hall_motion"]


# ------------------------------------------------------------------ rate mapping


def test_channel_rate_maps_range_endpoints_to_the_trained_band() -> None:
    loop, _, config = _loop()
    ch = _channel()
    assert loop.channel_rate(ch, 0.0) == pytest.approx(config.min_rate_hz)
    assert loop.channel_rate(ch, 1.0) == pytest.approx(config.max_rate_hz)


def test_channel_rate_never_reaches_zero() -> None:
    """Like the thermosensory path, the floor is min_rate_hz so the readout always has signal."""
    loop, _, config = _loop()
    assert loop.channel_rate(_channel(), -5.0) == pytest.approx(config.min_rate_hz)


def test_channel_rate_clamps_above_the_top() -> None:
    loop, _, _ = _loop()
    assert loop.channel_rate(_channel(), 99.0) == pytest.approx(120.0)


def test_channel_rate_honours_invert() -> None:
    loop, _, _ = _loop()
    assert loop.channel_rate(_channel(invert=True), 0.0) == pytest.approx(120.0)
    assert loop.channel_rate(_channel(invert=True), 1.0) == pytest.approx(20.0)


def test_channel_rate_honours_gain() -> None:
    loop, _, _ = _loop()
    assert loop.channel_rate(_channel(gain=0.5), 1.0) == pytest.approx(60.0)


def test_channel_rate_uses_a_custom_range() -> None:
    loop, _, _ = _loop()
    ch = _channel(vmin=0.0, vmax=500.0, kind=SignalKind.ILLUMINANCE)
    assert loop.channel_rate(ch, 250.0) == pytest.approx(70.0)


# ------------------------------------------------------------------ drive


def test_all_channels_reach_the_simulator_in_one_call() -> None:
    """set_drive replaces the drive map, so a second call would erase the first."""
    loop, sim, _ = _loop(
        [_channel("binary_sensor.hall_motion", indices=(10, 11, 12)),
         _channel("sensor.hall_lux", indices=(20, 21), kind=SignalKind.ILLUMINANCE, vmax=500.0)]
    )
    loop.drive_channels(
        [
            _signal("sensor.living_room_temperature", 22.0, kind=SignalKind.TEMPERATURE),
            _signal("binary_sensor.hall_motion", 1.0),
            _signal("sensor.hall_lux", 250.0, kind=SignalKind.ILLUMINANCE),
        ]
    )
    assert len(sim.calls) == 1
    assert set(sim.driven) == {0, 1, 10, 11, 12, 20, 21}


def test_extra_channels_actually_apply_current() -> None:
    loop, sim, config = _loop([_channel("binary_sensor.hall_motion", indices=(10, 11))])
    loop.drive_channels(
        [
            _signal("sensor.living_room_temperature", 22.0, kind=SignalKind.TEMPERATURE),
            _signal("binary_sensor.hall_motion", 1.0),
        ]
    )
    expected = drive_current_for_rate(config.max_rate_hz, sim.params.dt_ms, config)
    assert sim.driven[10] == pytest.approx(expected)


def test_unavailable_sensor_is_skipped_not_driven_at_its_floor() -> None:
    """An unavailable entity is not a sensor reading zero."""
    loop, sim, _ = _loop([_channel("binary_sensor.hall_motion", indices=(10, 11))])
    loop.drive_channels(
        [
            _signal("sensor.living_room_temperature", 22.0, kind=SignalKind.TEMPERATURE),
            _signal("binary_sensor.hall_motion", 0.0, state="unavailable"),
        ]
    )
    assert 10 not in sim.driven


def test_temperature_still_drives_when_no_extra_channels_exist() -> None:
    loop, sim, _ = _loop()
    loop.drive_channels([_signal("sensor.living_room_temperature", 22.0, kind=SignalKind.TEMPERATURE)])
    assert set(sim.driven) == {0, 1}


def test_unavailable_temperature_is_not_driven_at_its_fallback() -> None:
    """The primary sensor gets the same dead-state check the extra channels always had.

    Regression. ``HAClient`` substitutes a ``0.0`` fallback for an unavailable state, so an
    offline thermometer arrived here as a plausible-looking 0 C and was driven like any other
    reading — which could produce a real light action from a sensor that was not reporting.
    The check below already existed for every channel *except* this one.
    """
    loop, sim, _ = _loop()
    loop.drive_channels(
        [
            _signal(
                "sensor.living_room_temperature",
                0.0,
                state="unavailable",
                kind=SignalKind.TEMPERATURE,
                attributes={"unavailable": True},
            )
        ]
    )
    assert sim.driven == {}
    assert loop.reading_stale is True


def test_unparseable_temperature_state_counts_as_missing() -> None:
    """A state that is not a number is flagged ``unavailable`` without joining the dead list.

    ``HAClient._parse_state`` sets the attribute when the value fails to parse, so a check on
    the state *string* alone would let this through as the ``0.0`` fallback.
    """
    loop, sim, _ = _loop()
    loop.drive_channels(
        [
            _signal(
                "sensor.living_room_temperature",
                0.0,
                state="not-a-number",
                kind=SignalKind.TEMPERATURE,
                attributes={"unavailable": True},
            )
        ]
    )
    assert sim.driven == {}
    assert loop.reading_stale is True


def test_a_recovered_reading_clears_staleness() -> None:
    """Staleness is a state the loop leaves again, not a latch."""
    loop, sim, _ = _loop()
    loop.drive_channels(
        [
            _signal(
                "sensor.living_room_temperature",
                0.0,
                state="unavailable",
                kind=SignalKind.TEMPERATURE,
                attributes={"unavailable": True},
            )
        ]
    )
    assert loop.reading_stale is True

    loop.drive_channels(
        [_signal("sensor.living_room_temperature", 22.0, kind=SignalKind.TEMPERATURE)]
    )
    assert loop.reading_stale is False
    assert loop.last_good_reading_at is not None
    assert set(sim.driven) == {0, 1}


def test_a_failed_fetch_clears_the_drive_rather_than_holding_it() -> None:
    """An empty signal list means the fetch failed, not that nothing changed.

    Regression. With no signals the drive was left exactly as it was, so the simulator kept
    running on the previous window's input while ``decide`` decoded a colour from it — a
    decision caused by a reading nobody supplied.
    """
    loop, sim, _ = _loop()
    loop.drive_channels(
        [_signal("sensor.living_room_temperature", 22.0, kind=SignalKind.TEMPERATURE)]
    )
    assert set(sim.driven) == {0, 1}

    loop.drive_channels([])
    assert sim.driven == {}
    assert loop.reading_stale is True


def test_missing_temperature_leaves_the_extras_driven() -> None:
    loop, sim, _ = _loop([_channel("binary_sensor.hall_motion", indices=(10, 11))])
    loop.drive_channels([_signal("binary_sensor.hall_motion", 1.0)])
    assert set(sim.driven) == {10, 11}


def test_reported_rates_cover_every_driven_channel() -> None:
    loop, _, _ = _loop([_channel("binary_sensor.hall_motion", indices=(10, 11))])
    rates = loop.drive_channels(
        [
            _signal("sensor.living_room_temperature", 22.0, kind=SignalKind.TEMPERATURE),
            _signal("binary_sensor.hall_motion", 1.0),
        ]
    )
    assert set(rates) == {"sensor.living_room_temperature", "binary_sensor.hall_motion"}


def test_snapshot_reports_the_channels() -> None:
    loop, _, _ = _loop([_channel("binary_sensor.hall_motion", indices=(10, 11))])
    loop.drive_channels([_signal("binary_sensor.hall_motion", 1.0)])
    channels = loop.snapshot()["channels"]
    assert channels[0]["entity_id"] == "binary_sensor.hall_motion"
    assert channels[0]["neurons"] == 2
    assert channels[0]["rate_hz"] == pytest.approx(120.0)


def test_clear_drive_forgets_channel_rates() -> None:
    loop, _, _ = _loop([_channel("binary_sensor.hall_motion", indices=(10, 11))])
    loop.drive_channels([_signal("binary_sensor.hall_motion", 1.0)])
    loop.clear_drive()
    assert loop.channel_rates == {}
