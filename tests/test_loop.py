"""Unit tests for the live control loop.

These avoid the 138,639-neuron connectome: they cover the parts where a mistake would be
silent — whether a real device gets called, how a decoded colour is clamped and labelled,
and what the dashboard is told.
"""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.experiment import COLOUR_BANDS, ColourReadout, ExperimentConfig

#: Endpoints of the trained colour range. Read from the source of truth rather than
#: hardcoded, so widening the range does not silently break these tests.
K_LO = COLOUR_BANDS[0][1]
K_HI = COLOUR_BANDS[-1][1]
from flybrain.loop import LiveLoop, LoopConfig, MissingReadout


class _StubSim:
    """Just enough simulator for the loop's non-simulation paths."""

    def __init__(self) -> None:
        self.params = type("P", (), {"dt_ms": 0.1})()
        self.drive = None

    def set_drive(self, indices, current_mv):
        self.drive = (np.asarray(indices), float(current_mv))

    def clear_drive(self):
        self.drive = None


def _loop(dry_run: bool, mode: str = "mock", w=None, **loop_kwargs) -> LiveLoop:
    cfg = ExperimentConfig()
    n = 4
    readout = ColourReadout(n_features=n + 1, l2=0.0)
    readout.W = np.zeros(n + 1) if w is None else np.asarray(w, dtype=float)
    kwargs = {"mode": mode, "dry_run": dry_run, "smooth_ms": 0.0}
    kwargs.update(loop_kwargs)
    return LiveLoop(
        _StubSim(), cfg, readout,
        input_indices=np.arange(2), readout_indices=np.arange(n),
        loop_config=LoopConfig(**kwargs),
    )


# ------------------------------------------------------------------ safety


def test_mock_mode_always_applies_its_call_whatever_dry_run_says():
    """The mock is not a real device, so skipping the call would only lose information."""
    assert _loop(dry_run=True, mode="mock").loop.will_send is True
    assert _loop(dry_run=False, mode="mock").loop.will_send is True


def test_real_home_assistant_is_never_called_unless_dry_run_is_off():
    assert _loop(dry_run=True, mode="rest").loop.will_send is False
    assert _loop(dry_run=False, mode="rest").loop.will_send is True


def test_missing_readout_raises_before_touching_the_simulator(tmp_path):
    with pytest.raises(MissingReadout):
        LiveLoop.from_artifact(None, tmp_path)


# ------------------------------------------------------------------ decoding


def test_decode_clamps_to_the_reported_band_range():
    n = 4
    # A weight vector that would predict an absurd Kelvin value if unclamped.
    w = np.zeros(n + 1)
    w[-1] = 1e9
    loop = _loop(dry_run=True, w=w)
    kelvin = loop.decode(np.zeros(n), window_ms=300.0)
    assert kelvin == COLOUR_BANDS[-1][1]


def test_decode_uses_the_window_duration_for_the_rate():
    n = 4
    w = np.zeros(n + 1)
    w[0] = 1.0            # predict directly from the first neuron's rate in Hz
    loop = _loop(dry_run=True, w=w)
    # White-box: widen the clamp purely so the raw rate is observable. The configured
    # limits are normally pinned inside the trained band, which would hide both values.
    loop.loop.kelvin_min, loop.loop.kelvin_max = 0.0, 20000.0
    counts = np.array([30, 0, 0, 0])          # 30 spikes is 100 Hz over 300 ms, 10 Hz over 3 s
    assert loop.decode(counts, window_ms=300.0) == pytest.approx(100.0)
    assert loop.decode(counts, window_ms=3000.0) == pytest.approx(10.0)


def test_band_names_follow_lighting_convention():
    """Low Kelvin must read as *warm*: the label sits next to a colour swatch."""
    loop = _loop(dry_run=True)
    assert loop.band_for(K_LO) == "warm"
    assert loop.band_for(COLOUR_BANDS[1][1]) == "neutral"
    assert loop.band_for(K_HI) == "cool"


# ------------------------------------------------------------------ snapshot


def test_snapshot_reports_the_error_against_the_ideal_mapping():
    n = 4
    w = np.zeros(n + 1)
    loop = _loop(dry_run=True, w=w)
    loop.temperature_c = 22.5
    loop.kelvin = 5200.0
    snap = loop.snapshot()
    assert snap["temperature_c"] == 22.5
    assert snap["kelvin"] == 5200
    # 22.5 C is the midpoint of 10-35 C, so the ideal is the midpoint of the band range.
    assert snap["ideal_kelvin"] == round((K_LO + K_HI) / 2)
    assert snap["error_k"] == 5200 - round((K_LO + K_HI) / 2)
    assert snap["band"] == "neutral"
    assert snap["history"] == []
    assert snap["dry_run"] is False       # mock mode always applies its calls


def test_drive_temperature_scales_with_the_configured_rate_range():
    loop = _loop(dry_run=True)
    cfg = loop.config
    loop.drive_temperature(loop.loop.source_min_c)
    cold = loop.sim.drive[1]
    loop.drive_temperature(loop.loop.source_max_c)
    warm = loop.sim.drive[1]
    assert cold < warm
    assert loop.snapshot()["sensor_rate_hz"] == pytest.approx(cfg.max_rate_hz)


# ------------------------------------------------------------- sensitivity


def test_sensitivity_stretches_a_narrow_room_onto_the_trained_range():
    """The whole point: a real room moves 2-3 degC, the readout knows 10-35."""
    loop = _loop(dry_run=True, source_min_c=18.0, source_max_c=24.0)
    cfg = loop.config
    assert loop.brain_temperature(18.0) == pytest.approx(cfg.temp_min_c)
    assert loop.brain_temperature(24.0) == pytest.approx(cfg.temp_max_c)
    # 21 degC is halfway through the room span, so it lands halfway through the trained
    # span - a 3 degC move becomes 12.5 degC of brain temperature.
    assert loop.brain_temperature(21.0) == pytest.approx(22.5, abs=0.01)
    # Readings outside the configured span are clamped, not extrapolated.
    assert loop.brain_temperature(5.0) == pytest.approx(cfg.temp_min_c)
    assert loop.brain_temperature(40.0) == pytest.approx(cfg.temp_max_c)


def test_sensitivity_defaults_to_no_stretching():
    loop = _loop(dry_run=True)
    assert loop.brain_temperature(10.0) == pytest.approx(10.0)
    assert loop.brain_temperature(35.0) == pytest.approx(35.0)


def test_invert_reverses_the_direction():
    normal = _loop(dry_run=True, source_min_c=18.0, source_max_c=24.0)
    flipped = _loop(dry_run=True, source_min_c=18.0, source_max_c=24.0, invert=True)
    cfg = normal.config
    assert normal.brain_temperature(18.0) == pytest.approx(cfg.temp_min_c)
    assert flipped.brain_temperature(18.0) == pytest.approx(cfg.temp_max_c)
    assert flipped.brain_temperature(24.0) == pytest.approx(cfg.temp_min_c)


def test_kelvin_per_degree_reports_the_stretch():
    loop = _loop(dry_run=True, source_min_c=18.0, source_max_c=24.0)
    # 6 degC of room now covers the whole trained colour range.
    assert loop.snapshot()["kelvin_per_degree"] == pytest.approx((K_HI - K_LO) / 6, rel=0.02)


# --------------------------------------------------------------- smoothing


def test_smoothing_lags_a_step_change_and_then_converges():
    loop = _loop(dry_run=True, smooth_ms=5000.0)
    loop.drive_temperature(10.0)
    first = loop.snapshot()["temperature_c"]
    loop.drive_temperature(35.0)
    second = loop.snapshot()["temperature_c"]
    # One 300 ms window against a 5 s time constant must move only a little.
    assert first == pytest.approx(10.0)
    assert 10.0 < second < 12.5
    for _ in range(200):
        loop.drive_temperature(35.0)
    assert loop.snapshot()["temperature_c"] == pytest.approx(35.0, abs=0.5)


def test_smoothing_can_be_turned_off():
    loop = _loop(dry_run=True, smooth_ms=0.0)
    loop.drive_temperature(10.0)
    loop.drive_temperature(35.0)
    assert loop.snapshot()["temperature_c"] == pytest.approx(35.0)


# ------------------------------------------------------- settings plumbing


def test_update_settings_clamps_limits_into_the_trained_band():
    loop = _loop(dry_run=True)
    out = loop.update_settings({"kelvin_min": 100.0, "kelvin_max": 99999.0})
    assert out["kelvin_min"] == K_LO
    assert out["kelvin_max"] == K_HI


def test_update_settings_rejects_a_zero_width_sensitivity_span():
    loop = _loop(dry_run=True)
    out = loop.update_settings({"source_min_c": 20.0, "source_max_c": 20.0})
    assert out["source_max_c"] > out["source_min_c"]


def test_update_settings_rejects_unknown_keys():
    loop = _loop(dry_run=True)
    with pytest.raises(KeyError):
        loop.update_settings({"history": 10})
    with pytest.raises(KeyError):
        loop.update_settings({"nonsense": 1})


def test_fit_range_to_observed_uses_the_readings_seen():
    loop = _loop(dry_run=True)
    for t in (19.4, 21.0, 22.6):
        loop.drive_temperature(t)
    out = loop.fit_range_to_observed(margin_c=0.5)
    assert out["source_min_c"] == pytest.approx(18.5)
    assert out["source_max_c"] == pytest.approx(23.5)


# ----------------------------------------------------------------- deadband


def test_deadband_suppresses_calls_but_still_reports_every_decision():
    import asyncio

    class _StubHA:
        def __init__(self):
            self.calls = []

        async def get_signals(self):
            return []

        async def call_service(self, entity_id, service, data=None):
            self.calls.append((entity_id, service, dict(data or {})))
            return True

    n = 4
    w = np.zeros(n + 1)
    w[0] = 1.0                      # 1 Hz of readout rate maps to 1 K
    loop = _loop(dry_run=True, mode="mock", w=w, deadband_k=100.0)
    loop.ha = _StubHA()
    # Counts / 0.3 s gives the readout rate, and w[0] = 1 means rate == Kelvin. Pick
    # values inside the trained band whichever range is configured.
    base = 1560                       # 5200 K
    nudge = 1570                      # 5233 K, +33 K: inside the 100 K deadband
    jump = base + 400                 # 6533 K -> clamps, clearly beyond the deadband

    async def run():
        loop.drive_temperature(20.0)
        await loop.decide(np.array([base, 0, 0, 0]), window_ms=300.0)    # sent
        await loop.decide(np.array([nudge, 0, 0, 0]), window_ms=300.0)   # suppressed
        await loop.decide(np.array([jump, 0, 0, 0]), window_ms=300.0)    # sent

    asyncio.run(run())
    assert loop.decisions == 3                    # the dashboard sees all three
    assert len(loop.ha.calls) == 2                # the light only hears about two
    assert loop.last_action["sent"] is True


def test_a_stale_reading_stops_the_loop_acting_and_says_so():
    """A sensor that stops reporting must stop the light, not freeze the last colour.

    Regression. A failed fetch left ``_active_source_c`` holding the previous reading, so
    ``decide`` kept decoding and could keep sending service calls driven by an input that no
    longer existed. The loop must refuse to act *and* report why: on a dashboard, a silent gap
    is indistinguishable from a crash.
    """
    import asyncio

    class _StubHA:
        def __init__(self):
            self.calls = []

        async def get_signals(self):
            return []

        async def call_service(self, entity_id, service, data=None):
            self.calls.append((entity_id, service, dict(data or {})))
            return True

    n = 4
    w = np.zeros(n + 1)
    w[0] = 1.0
    loop = _loop(dry_run=True, mode="mock", w=w)
    loop.ha = _StubHA()
    counts = np.array([1560, 0, 0, 0])            # 5200 K, inside the trained band

    async def run():
        loop.drive_temperature(20.0)              # a usable reading, and a real decision
        first = await loop.decide(counts, window_ms=300.0)
        assert first is not None
        assert len(loop.ha.calls) == 1
        assert loop.snapshot()["reading_age_s"] is not None

        loop.drive_channels([])                   # the fetch failed: nothing is readable
        stale = await loop.decide(counts, window_ms=300.0)
        assert stale is not None
        assert stale["stale"] is True
        assert loop.last_action["sent"] is False
        assert loop.last_action["reason"] == "sensor_stale"
        assert len(loop.ha.calls) == 1            # no second call, on any window

    asyncio.run(run())
    assert loop.reading_stale is True
    assert loop.decisions == 1                    # a refusal is not a decision
    assert len(loop.history) == 1                 # the chart gets no colourless point
    snap = loop.snapshot()
    assert snap["reading_stale"] is True
    assert snap["reading_age_s"] is not None


class TestFromEnv:
    """Config from the environment: the only way to point the loop at a real house."""

    def test_entity_defaults_are_the_mock_home(self) -> None:
        cfg = LoopConfig.from_env({})
        assert cfg.temperature_entity == "sensor.living_room_temperature"
        assert cfg.mode == "mock"
        assert cfg.dry_run is True

    def test_entities_come_from_the_environment(self) -> None:
        cfg = LoopConfig.from_env(
            {
                "FLYBRAIN_TEMPERATURE_ENTITY": "sensor.hallway_temperature",
                "FLYBRAIN_LIGHT_ENTITY": "light.hall_lamp",
            }
        )
        assert cfg.temperature_entity == "sensor.hallway_temperature"
        assert cfg.light_entity == "light.hall_lamp"

    def test_dry_run_stays_on_unless_explicitly_disabled(self) -> None:
        assert LoopConfig.from_env({"HA_DRY_RUN": "1"}).dry_run is True
        assert LoopConfig.from_env({"HA_MODE": "rest"}).dry_run is True
        assert LoopConfig.from_env({"HA_DRY_RUN": "0"}).dry_run is False
        assert LoopConfig.from_env({"HA_DRY_RUN": "false"}).dry_run is False

    def test_sensitivity_range_is_configurable(self) -> None:
        """A real room moves a couple of degrees; the range has to be settable without code."""
        cfg = LoopConfig.from_env({"FLYBRAIN_SOURCE_MIN_C": "18", "FLYBRAIN_SOURCE_MAX_C": "24"})
        assert (cfg.source_min_c, cfg.source_max_c) == (18.0, 24.0)
        assert cfg.source_span_c == 6.0

    def test_a_bad_number_falls_back_to_the_default(self) -> None:
        cfg = LoopConfig.from_env({"FLYBRAIN_SOURCE_MIN_C": "warm-ish"})
        assert cfg.source_min_c == 10.0

    def test_kelvin_bounds_are_configurable(self) -> None:
        """light.hall_lamp goes to 2000 K; the default floor is more conservative than the lamp."""
        cfg = LoopConfig.from_env({"FLYBRAIN_KELVIN_MIN": "2000", "FLYBRAIN_KELVIN_MAX": "6535"})
        assert (cfg.kelvin_min, cfg.kelvin_max) == (2000.0, 6535.0)

    def test_invert_flag(self) -> None:
        assert LoopConfig.from_env({"FLYBRAIN_INVERT": "yes"}).invert is True
        assert LoopConfig.from_env({"FLYBRAIN_INVERT": "0"}).invert is False


class TestPacing:
    """`interval_s` is the efficiency knob: it decides what it costs to leave this running."""

    def test_flat_out_by_default(self) -> None:
        assert LoopConfig().interval_s == 0.0

    def test_interval_comes_from_the_environment(self) -> None:
        assert LoopConfig.from_env({"FLYBRAIN_INTERVAL_S": "15"}).interval_s == 15.0

    def test_bad_interval_falls_back_to_flat_out(self) -> None:
        assert LoopConfig.from_env({"FLYBRAIN_INTERVAL_S": "soon"}).interval_s == 0.0

    def test_interval_survives_a_settings_round_trip(self) -> None:
        """The dashboard sends this as a patch, so `apply` must accept it."""
        cfg = LoopConfig()
        cfg.apply({"interval_s": 20})
        assert cfg.interval_s == 20.0
        assert cfg.to_dict()["interval_s"] == 20.0
