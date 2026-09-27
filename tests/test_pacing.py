"""Adaptive pacing: the heartbeat, the burst, and the change trigger.

Energy cannot be measured here — there is no GPU in the test environment, and no power meter on
any machine that runs CI. What *can* be pinned down is the thing that decides the energy: the
cadence. Every test below therefore drives the pacer with an explicit clock, so a simulated hour
is a few lines and the assertions are exact rather than "roughly".

The properties worth defending, in order of how badly they break the project if they regress:

1. ``heartbeat_s == 0`` reproduces the pre-pacing behaviour exactly. Pacing was added to a
   working demo, and the demo has to stay reachable.
2. The heartbeat is a *maximum* gap. A trigger can make decisions earlier; nothing can make
   them later. If that inverts, a house event is missed entirely.
3. The first poll is a baseline, not an event. Otherwise every restart bursts on nothing.
4. Dead sensors are not events. A thermometer going ``unavailable`` is a fault in the plumbing,
   and bursting the GPU for it spends power exactly when the input is unusable.
"""

from __future__ import annotations

import pytest

from flybrain.pacing import (
    BURST,
    FLAT_OUT,
    WAITING,
    ChangeTrigger,
    Pacer,
    estimated_watts,
)
from flybrain.types import Signal, SignalKind

TEMP = "sensor.room_temperature"
MOTION = "binary_sensor.hall_motion"


def sig(entity_id: str, value: float, state: str, kind=SignalKind.TEMPERATURE) -> Signal:
    return Signal(
        entity_id=entity_id,
        kind=kind,
        value=value,
        state=state,
        timestamp=0.0,
        attributes={},
    )


def dead(entity_id: str) -> Signal:
    """What ``HAClient._parse_state`` produces for a sensor that is not reporting."""
    return Signal(
        entity_id=entity_id,
        kind=SignalKind.TEMPERATURE,
        value=0.0,
        state="unavailable",
        timestamp=0.0,
        attributes={"unavailable": True},
    )


def pacer(**kwargs) -> Pacer:
    defaults = {"heartbeat_s": 60.0, "poll_s": 5.0, "burst_s": 10.0, "primary_entity": TEMP}
    defaults.update(kwargs)
    return Pacer(**defaults)


class TestFlatOut:
    """`heartbeat_s=0` must be indistinguishable from the loop before pacing existed."""

    def test_always_steps(self) -> None:
        p = pacer(heartbeat_s=0.0)
        for t in (0.0, 0.001, 1.0, 9999.0):
            assert p.should_step(t) is True

    def test_never_polls(self) -> None:
        """There is no waiting to fill, so polling would only add Home Assistant traffic."""
        p = pacer(heartbeat_s=0.0)
        assert p.should_poll(0.0) is False
        p.note_step(0.0)
        assert p.should_poll(100.0) is False

    def test_ignores_a_configured_trigger(self) -> None:
        """A trigger cannot shorten a gap that does not exist, so it is simply irrelevant."""
        p = pacer(heartbeat_s=0.0, trigger_delta=0.1)
        assert p.adaptive is False
        assert p.should_step(0.0) is True

    def test_reports_flat_out(self) -> None:
        p = pacer(heartbeat_s=0.0)
        assert p.snapshot(0.0)["mode"] == FLAT_OUT


class TestHeartbeat:
    """The heartbeat is the maximum gap, and it is the whole scheme when no trigger is set."""

    def test_the_first_decision_is_immediate(self) -> None:
        assert pacer().should_step(0.0) is True

    def test_then_it_waits_the_full_heartbeat(self) -> None:
        p = pacer(heartbeat_s=60.0)
        p.note_step(0.0)
        assert p.should_step(59.9) is False
        assert p.should_step(60.0) is True

    def test_it_keeps_stepping_after_each_decision(self) -> None:
        p = pacer(heartbeat_s=60.0)
        p.note_step(0.0)
        assert p.should_step(60.0) is True
        p.note_step(60.0)
        assert p.should_step(119.0) is False
        assert p.should_step(120.0) is True

    def test_a_heartbeat_of_zero_disables_the_wait(self) -> None:
        assert pacer(heartbeat_s=0.0).flat_out is True

    def test_mode_is_waiting_between_decisions(self) -> None:
        p = pacer(heartbeat_s=60.0)
        p.note_step(0.0)
        assert p.snapshot(30.0)["mode"] == WAITING

    def test_it_reports_how_long_until_the_next_one(self) -> None:
        p = pacer(heartbeat_s=60.0)
        p.note_step(0.0)
        assert p.snapshot(35.0)["next_in_s"] == 25.0


class TestChangeTrigger:
    """The v1 trigger: a comparison, not a model."""

    def test_the_first_poll_is_a_baseline_not_an_event(self) -> None:
        """Otherwise every restart bursts because there is nothing to compare against."""
        p = pacer(trigger_delta=0.5)
        assert p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")]) is False

    def test_a_large_move_fires(self) -> None:
        p = pacer(trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        assert p.note_poll(5.0, [sig(TEMP, 20.6, "20.6")]) is True

    def test_a_small_move_does_not(self) -> None:
        p = pacer(trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        assert p.note_poll(5.0, [sig(TEMP, 20.2, "20.2")]) is False

    def test_a_move_exactly_at_the_threshold_fires(self) -> None:
        """`>=`, not `>`: the threshold is a boundary the operator chose, not a fudge."""
        p = pacer(trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        assert p.note_poll(5.0, [sig(TEMP, 20.5, "20.5")]) is True

    def test_a_discrete_entity_changing_state_fires(self) -> None:
        p = pacer(trigger_delta=0.5)
        p.note_poll(0.0, [sig(MOTION, 0.0, "off", SignalKind.MOTION)])
        assert p.note_poll(5.0, [sig(MOTION, 1.0, "on", SignalKind.MOTION)]) is True

    def test_a_numeric_entity_does_not_fire_on_its_state_string(self) -> None:
        """The trap this rule exists for.

        Home Assistant reports a thermometer's state as its value, so ``"20.0" -> "20.1"`` is a
        state change on every *reported decimal*. Treating that as an event would make the
        trigger fire constantly, and pacing would quietly become "always burst" — the most
        expensive possible misreading of the setting.
        """
        trigger = ChangeTrigger(primary_entity=TEMP, delta=0.5)
        previous = [sig(TEMP, 20.0, "20.0")]
        current = [sig(TEMP, 20.1, "20.1")]  # state moved, value did not move enough
        assert trigger(previous, current) is False

    def test_a_dead_sensor_is_not_an_event(self) -> None:
        p = pacer(trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        assert p.note_poll(5.0, [dead(TEMP)]) is False

    def test_a_sensor_coming_back_is_not_an_event_either(self) -> None:
        """Coming back from ``unavailable`` reads as a huge move, and it is not one."""
        p = pacer(trigger_delta=0.5)
        p.note_poll(0.0, [dead(TEMP)])
        assert p.note_poll(5.0, [sig(TEMP, 20.0, "20.0")]) is False

    def test_an_entity_we_have_never_seen_does_not_fire(self) -> None:
        p = pacer(trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        assert p.note_poll(5.0, [sig(TEMP, 20.0, "20.0"), sig(MOTION, 1.0, "on")]) is False

    def test_a_zero_threshold_leaves_a_plain_heartbeat(self) -> None:
        p = pacer(trigger_delta=0.0)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        assert p.note_poll(5.0, [sig(TEMP, 30.0, "30.0")]) is False
        assert p.adaptive is False


class TestPolling:
    def test_it_polls_on_the_configured_cadence(self) -> None:
        p = pacer(poll_s=5.0, trigger_delta=0.5)
        assert p.should_poll(0.0) is True
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        assert p.should_poll(4.9) is False
        assert p.should_poll(5.0) is True

    def test_it_does_not_poll_during_a_burst(self) -> None:
        """The GPU is already the bottleneck; extra sensor reads would add nothing."""
        p = pacer(poll_s=5.0, burst_s=10.0, trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        p.note_poll(5.0, [sig(TEMP, 21.0, "21.0")])  # fires; burst until 15.0
        assert p.should_poll(6.0) is False
        assert p.should_poll(16.0) is True

    def test_no_poll_interval_means_no_polling(self) -> None:
        p = pacer(poll_s=0.0, trigger_delta=0.5)
        assert p.should_poll(0.0) is False

    def test_a_trigger_with_no_polling_says_so(self, caplog) -> None:
        """It still runs, but the trigger can never fire, and that is not obvious."""
        with caplog.at_level("WARNING"):
            pacer(poll_s=0.0, trigger_delta=0.5)
        assert "cannot fire" in caplog.text


class TestBurst:
    def test_a_fired_trigger_steps_immediately(self) -> None:
        """The event must not wait for the next heartbeat — that is the point of a trigger."""
        p = pacer(heartbeat_s=600.0, trigger_delta=0.5)
        p.note_step(0.0)
        assert p.should_step(1.0) is False
        p.note_poll(1.0, [sig(TEMP, 20.0, "20.0")])
        p.note_poll(6.0, [sig(TEMP, 21.0, "21.0")])
        assert p.should_step(6.0) is True

    def test_then_it_keeps_going_for_the_burst(self) -> None:
        p = pacer(heartbeat_s=600.0, burst_s=10.0, trigger_delta=0.5)
        p.note_step(0.0)
        p.note_poll(1.0, [sig(TEMP, 20.0, "20.0")])
        p.note_poll(6.0, [sig(TEMP, 21.0, "21.0")])
        p.note_step(6.0)
        assert p.should_step(7.0) is True
        p.note_step(7.0)
        assert p.should_step(14.0) is True
        p.note_step(14.0)
        assert p.should_step(16.1) is False  # burst expired at 16.0

    def test_the_burst_ends(self) -> None:
        p = pacer(heartbeat_s=600.0, burst_s=10.0, trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        p.note_poll(5.0, [sig(TEMP, 21.0, "21.0")])
        p.note_step(5.0)
        assert p.snapshot(6.0)["mode"] == BURST
        assert p.snapshot(16.1)["mode"] == WAITING

    def test_a_zero_burst_is_one_extra_decision(self) -> None:
        p = pacer(heartbeat_s=600.0, burst_s=0.0, trigger_delta=0.5)
        p.note_step(0.0)
        p.note_poll(1.0, [sig(TEMP, 20.0, "20.0")])
        p.note_poll(6.0, [sig(TEMP, 21.0, "21.0")])
        assert p.should_step(6.0) is True
        p.note_step(6.0)
        assert p.should_step(6.1) is False

    def test_the_heartbeat_still_bounds_the_gap_after_a_burst(self) -> None:
        """A burst must not become a licence to never wait again."""
        p = pacer(heartbeat_s=60.0, burst_s=10.0, trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        p.note_poll(5.0, [sig(TEMP, 21.0, "21.0")])
        p.note_step(5.0)
        p.note_step(15.5)  # last step of the burst
        assert p.should_step(74.0) is False
        assert p.should_step(75.5) is True


class TestPause:
    def test_resuming_waits_a_full_heartbeat(self) -> None:
        """A pause must not bank credit and fire the instant it resumes."""
        p = pacer(heartbeat_s=60.0)
        p.note_step(0.0)
        p.on_pause(1000.0)
        assert p.should_step(1000.0) is False
        assert p.should_step(1059.0) is False
        assert p.should_step(1060.0) is True

    def test_a_change_during_the_pause_is_still_caught(self) -> None:
        """The house changed while we were not looking. That is a real event."""
        p = pacer(heartbeat_s=3600.0, poll_s=5.0, trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        p.note_step(1.0)
        p.on_pause(100.0)
        assert p.should_poll(100.0) is True
        assert p.note_poll(100.0, [sig(TEMP, 25.0, "25.0")]) is True

    def test_a_pause_clears_a_pending_burst(self) -> None:
        p = pacer(heartbeat_s=600.0, burst_s=10.0, trigger_delta=0.5)
        p.note_poll(0.0, [sig(TEMP, 20.0, "20.0")])
        p.note_poll(5.0, [sig(TEMP, 21.0, "21.0")])
        p.on_pause(6.0)
        assert p.should_step(6.5) is False


class TestDelegation:
    """The seam the A3 novelty trigger will use, exercised with a stand-in."""

    def test_an_injected_trigger_enables_adaptive_pacing(self) -> None:
        class Never:
            def __call__(self, previous, current) -> bool:
                return False

        p = pacer(trigger_delta=0.0, trigger=Never())
        assert p.adaptive is True
        assert p.should_poll(0.0) is True

    def test_an_injected_trigger_fires(self) -> None:
        calls: list[int] = []

        class Surprise:
            def __call__(self, previous, current) -> bool:
                calls.append(len(previous))
                return True

        p = pacer(trigger_delta=0.0, trigger=Surprise())
        p.note_step(0.0)
        assert p.note_poll(1.0, [sig(TEMP, 20.0, "20.0")]) is True
        assert p.should_step(1.0) is True

    def test_the_trigger_name_is_reported(self) -> None:
        p = pacer(trigger_delta=0.5)
        assert p.snapshot(0.0)["trigger"] == "ChangeTrigger"


class TestReporting:
    def test_the_duty_cycle_is_none_until_something_has_run(self) -> None:
        """`0% duty` from no data is a claim, and the dashboard would draw it as a measurement."""
        assert pacer().observed_duty(0.0) is None

    def test_the_duty_cycle_is_measured_from_actual_steps(self) -> None:
        p = pacer(heartbeat_s=10.0)
        p.note_step(0.0)
        p.note_step(100.0)
        # two steps * 2.34 s of work, over 100 s of wall clock
        assert p.observed_duty(100.0) == pytest.approx(0.0468, abs=1e-4)

    def test_the_duty_cycle_cannot_exceed_one(self) -> None:
        p = pacer(heartbeat_s=1.0)
        p.note_step(0.0)
        p.note_step(0.001)
        assert p.observed_duty(0.001) == 1.0

    def test_it_reports_the_measured_power_model(self) -> None:
        """19 W idle plus duty x 146 W, the model measured in docs/live-view.md."""
        assert estimated_watts(0.0) == pytest.approx(19.0)
        assert estimated_watts(1.0) == pytest.approx(165.0)
        assert estimated_watts(0.16) == pytest.approx(42.4, abs=0.1)

    def test_the_snapshot_says_whether_adaptive_pacing_is_on(self) -> None:
        assert pacer(trigger_delta=0.5).snapshot(0.0)["adaptive"] is True
        assert pacer(trigger_delta=0.0).snapshot(0.0)["adaptive"] is False

    def test_the_snapshot_carries_every_setting_the_dashboard_patches(self) -> None:
        snap = pacer(trigger_delta=0.25).snapshot(0.0)
        for key in ("heartbeat_s", "poll_s", "burst_s", "trigger_delta"):
            assert key in snap
        assert snap["trigger_delta"] == 0.25


# --------------------------------------------------------------- loop integration


class _FakeHA:
    """Minimal Home Assistant stand-in: returns whatever signals it was handed."""

    def __init__(self, signals):
        self.signals = signals
        self.reads = 0

    async def get_signals(self):
        self.reads += 1
        return list(self.signals)


def _live_loop(ha, **loop_kwargs):
    import numpy as np

    from flybrain.experiment import ColourReadout, ExperimentConfig
    from flybrain.loop import LiveLoop, LoopConfig

    class _Sim:
        params = type("P", (), {"dt_ms": 0.1})()

        def set_drive(self, indices, current_mv): ...
        def clear_drive(self): ...

    readout = ColourReadout(n_features=5, l2=0.0)
    readout.W = np.zeros(5)
    # The trigger matches the *configured* primary entity rather than guessing, so the fixture
    # has to configure the same entity the signals use.
    loop_kwargs.setdefault("temperature_entity", TEMP)
    return LiveLoop(
        _Sim(),
        ExperimentConfig(),
        readout,
        input_indices=np.arange(2),
        readout_indices=np.arange(4),
        loop_config=LoopConfig(**loop_kwargs),
        ha=ha,
    )


class TestLoopIntegration:
    def test_the_pacer_is_built_from_the_loop_config(self) -> None:
        loop = _live_loop(_FakeHA([]), interval_s=42.0, burst_s=7.0, trigger_delta=0.3)
        assert loop.pacer.heartbeat_s == 42.0
        assert loop.pacer.burst_s == 7.0
        assert loop.pacer.trigger_delta == 0.3
        assert loop.pacer.primary_entity == loop.loop.temperature_entity

    def test_a_poll_does_not_overwrite_the_recorder_snapshot(self) -> None:
        """The misalignment this flag exists to prevent.

        The poll happens *between* decisions. If it became ``last_signals``, the next recorded
        window would carry the reading taken after the window was driven - so the recorded
        pairing would be wrong while the row count, the file sizes and every existing check
        still looked perfect.
        """
        import asyncio

        driving = sig(TEMP, 20.0, "20.0")
        ha = _FakeHA([driving])
        loop = _live_loop(ha, trigger_delta=0.5)

        async def run():
            await loop.read_signals()  # the reading that drives the window
            assert loop.last_signals[0].value == 20.0

            ha.signals = [sig(TEMP, 25.0, "25.0")]  # the pacer notices the room warmed up
            polled = await loop.read_signals(store=False)
            return polled

        polled = asyncio.run(run())
        assert polled[0].value == 25.0
        assert loop.last_signals[0].value == 20.0, "the poll clobbered the driving reading"
        assert loop.sensor_snapshot() == {TEMP: 20.0}

    def test_a_normal_read_does_update_the_snapshot(self) -> None:
        import asyncio

        ha = _FakeHA([sig(TEMP, 20.0, "20.0")])
        loop = _live_loop(ha)
        asyncio.run(loop.read_signals())
        assert loop.last_signals[0].value == 20.0

    def test_the_snapshot_reports_pacing_state(self) -> None:
        loop = _live_loop(_FakeHA([]), interval_s=0.0)
        assert loop.snapshot()["pacing"]["mode"] == FLAT_OUT

    def test_a_settings_patch_rebuilds_the_pacer(self) -> None:
        """Otherwise the dashboard control would appear to work and change nothing."""
        loop = _live_loop(_FakeHA([]), trigger_delta=0.0)
        assert loop.pacer.adaptive is False
        loop.update_settings({"trigger_delta": 0.5, "interval_s": 30.0})
        assert loop.pacer.adaptive is True
        assert loop.pacer.heartbeat_s == 30.0

    def test_the_pacer_survives_a_real_paced_iteration(self) -> None:
        """A change large enough to fire the trigger produces an immediate step."""
        import asyncio

        ha = _FakeHA([sig(TEMP, 20.0, "20.0")])
        loop = _live_loop(ha, interval_s=3600.0, poll_s=5.0, trigger_delta=0.5)
        now = 0.0
        assert loop.pacer.should_poll(now) is True
        loop.pacer.note_poll(now, asyncio.run(loop.read_signals(store=False)))
        assert loop.pacer.should_step(now) is True
        loop.pacer.note_step(now)

        ha.signals = [sig(TEMP, 23.0, "23.0")]
        assert loop.pacer.should_step(10.0) is False  # an hour from the heartbeat
        loop.pacer.note_poll(10.0, asyncio.run(loop.read_signals(store=False)))
        assert loop.pacer.should_step(10.0) is True


    def test_a_negative_interval_cannot_fail_open_into_flat_out(self) -> None:
        """A stray minus sign must not turn the cheapest setting into the most expensive one.

        The pacer reads a non-positive heartbeat as "never wait", so `interval_s = -5` is not
        "even faster" - it is flat out at ~165 W. The dashboard is the only thing that sends
        these values, which is exactly why the guard belongs on the receiving side.
        """
        loop = _live_loop(_FakeHA([]), interval_s=60.0)
        loop.update_settings({"interval_s": -5, "poll_s": -1, "burst_s": -2, "trigger_delta": -0.5})
        assert loop.loop.interval_s == 0.0
        assert loop.loop.poll_s == 0.0
        assert loop.loop.burst_s == 0.0
        assert loop.loop.trigger_delta == 0.0
        assert loop.loop.to_dict()["interval_s"] == 0.0, "the dashboard must see the clamped value"
