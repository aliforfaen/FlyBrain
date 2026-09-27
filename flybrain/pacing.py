"""Adaptive pacing: spend GPU time when the house is interesting, idle when it is not.

Stepping the connectome holds the GPU at ~165 W continuously, because the brain advances brain
time as fast as it can. A context layer needs a decision every few seconds at most, so the
cheapest optimisation in the project is to *not step the brain* most of the time. That is what
this module decides. It is deliberately free of torch and numpy: it is a pure function of a
monotonic clock and a list of :class:`~flybrain.types.Signal`, so it is testable in
microseconds and cannot accidentally prevent the process from idling.

The scheme is a heartbeat plus a trigger:

* **Heartbeat** -- one decision every ``heartbeat_s`` regardless of anything. This is not
  padding. The brain is never reset, so the heartbeat is what keeps the reservoir in the regime
  the readout was fitted on, and it produces the *quiet* windows that a ``house_activity``
  readout needs in order to have negatives to learn from. Without it, an event-driven recorder
  would collect only the house being busy, and the resulting readout would have no idea what
  "nothing is happening" looks like. That is how this scheme fails silently.
* **Burst** -- when a trigger fires, run at full rate for ``burst_s`` so the event is captured
  in detail rather than sampled once.

Both halves are load-bearing: the heartbeat is what makes the data learnable, the trigger is
what makes it cheap.

**The trigger is a comparison, not a model.** "Did something change?" is a local threshold
question, and a threshold beats a model on it every time -- the same rule that keeps a model out
of the regime probe. The intended long-term trigger is the reservoir's own prediction error
(roadmap A3), which is computed locally for nothing. Because the trigger is an injected callable
(:class:`Trigger`), that upgrade needs no change here: an A3 trigger is constructed in
``server.py`` where the reservoir is in scope, closes over whatever it needs, and replaces
:class:`ChangeTrigger` without this module learning about it.

Two traps, both documented at length in ``docs/engine.md``:

1. **Pacing is part of the training regime.** Event-weighted windows are a different
   distribution from fixed-interval ones, so the pacing configuration has to be recorded in the
   recording's ``meta.json`` and reproduced at inference time. It is not a runtime detail.
2. **Pacing is visible.** The 3D view advances in a burst and then holds still. That is
   configured behaviour, not a stall.

Energy is not measured here -- it cannot be. What this module *can* do is report the duty cycle
it actually achieved, so the dashboard shows a measured number rather than a predicted one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from flybrain.types import Signal, SignalKind, signal_is_dead

logger = logging.getLogger(__name__)

#: The three states the dashboard can draw. ``flat_out`` is not "waiting with a short
#: heartbeat" -- it means the loop never waits at all, and saying so is the honest label.
FLAT_OUT = "flat_out"
WAITING = "waiting"
BURST = "burst"

#: One decision consumes 300 ms of brain time, which costs ~2.34 s of GPU work at 0.13x
#: realtime, and the measured power model is ``mean = IDLE_W + duty * LOAD_W``.
#:
#: These three numbers are a *copy* of measurements that ``docs/live-view.md`` owns, and they
#: are here so that the duty cycle can be reported from one place for every pacing mode. If the
#: GPU or the engine changes, both the doc and these constants are wrong together rather than
#: the doc being right and the dashboard quietly lying.
STEP_COST_S = 2.34
IDLE_W = 19.0
LOAD_W = 146.0

#: Signal kinds whose *state string* is a meaningful event. Temperature, illuminance and the
#: rest are numeric: their state string changes on every reported decimal, so treating a string
#: change on those as an event would make the trigger fire constantly and pacing would silently
#: become "always burst". Motion and contact are genuinely discrete.
DISCRETE_KINDS = frozenset({SignalKind.MOTION, SignalKind.CONTACT})


def estimated_watts(duty: float) -> float:
    """Mean GPU power for a given duty cycle, from the measured model."""
    return IDLE_W + min(1.0, max(0.0, float(duty))) * LOAD_W


@runtime_checkable
class Trigger(Protocol):
    """Decides whether a sensor change is worth a burst.

    Takes the previous and current signal lists and returns whether this change is an *event*.

    Returning a boolean rather than a numeric score is deliberate. A score would have to live in
    one unit across entities that do not share one -- degrees Celsius, lux, and ``on``/``off`` --
    and the threshold would then belong to the pacer, which has no way to know what a large
    movement is for a given sensor. The threshold belongs to the trigger, which knows what it is
    measuring. For the learned A3 trigger that is doubly true: its threshold is calibrated
    against the predictor's own error distribution.
    """

    def __call__(self, previous: Sequence[Signal], current: Sequence[Signal]) -> bool: ...


@dataclass
class ChangeTrigger:
    """The v1 trigger: did the house move?

    Two rules, chosen so that neither needs to know a unit:

    * **A discrete entity changed state** (motion, contact). Exact, unit-free, and the most
      house-relevant thing that happens.
    * **The primary sensor moved by at least ``delta``**, in its own units. ``delta`` is the
      value the operator sets as ``FLYBRAIN_TRIGGER_DELTA``, so for a temperature sensor it is
      degrees Celsius.

    Entities that are not reporting are skipped, because a sensor going ``unavailable`` is not
    an event in the house -- it is an event in the plumbing, and bursting the GPU on it would
    burn power precisely when the input is unusable.

    **Known limitation, stated rather than hidden:** in v1 only the *primary* numeric entity is
    watched by value. Illuminance and humidity changes are not events unless they are discrete.
    The general answer is the A3 novelty score, which is multivariate by construction; this
    trigger is the cheap version that needs no training data and can be built today.
    """

    primary_entity: str = ""
    delta: float = 0.0

    def __call__(self, previous: Sequence[Signal], current: Sequence[Signal]) -> bool:
        if not previous:
            # Nothing to compare against. The first poll is not an event, it is the baseline.
            return False
        before = {s.entity_id: s for s in previous if not signal_is_dead(s)}
        for signal in current:
            if signal_is_dead(signal):
                continue
            old = before.get(signal.entity_id)
            if old is None:
                continue
            if (
                signal.kind in DISCRETE_KINDS
                and signal.state.strip().lower() != old.state.strip().lower()
            ):
                return True
            if (
                signal.entity_id == self.primary_entity
                and abs(float(signal.value) - float(old.value)) >= self.delta
            ):
                return True
        return False


class Pacer:
    """Decides, on every loop iteration, whether to poll, whether to step, and why.

    Every method takes ``now`` as an argument and none of them read a clock, so a test can drive
    hours of behaviour in a few lines and assert exact cadences. That is the whole reason this is
    a separate object rather than a handful of timestamps inside ``server.run_loop``.
    """

    def __init__(
        self,
        *,
        heartbeat_s: float = 15.0,
        poll_s: float = 5.0,
        burst_s: float = 10.0,
        trigger_delta: float = 0.0,
        trigger: Trigger | None = None,
        primary_entity: str = "",
    ) -> None:
        """``heartbeat_s == 0`` means flat out, which is the pre-pacing behaviour exactly.

        ``poll_s`` is the *sensor poll* cadence while waiting, and it is free: reading Home
        Assistant does no GPU work. It exists only so a change can be noticed before the next
        heartbeat would have happened anyway.

        Passing an explicit ``trigger`` enables adaptive pacing regardless of ``trigger_delta``;
        that is the seam the A3 novelty trigger will use. With no trigger passed and a positive
        ``trigger_delta``, a :class:`ChangeTrigger` is built from ``primary_entity``.
        """
        self.heartbeat_s = max(0.0, float(heartbeat_s))
        self.poll_s = max(0.0, float(poll_s))
        self.burst_s = max(0.0, float(burst_s))
        self.trigger_delta = max(0.0, float(trigger_delta))
        self.primary_entity = primary_entity

        if trigger is not None:
            self.trigger: Trigger | None = trigger
        elif self.trigger_delta > 0.0:
            self.trigger = ChangeTrigger(primary_entity=primary_entity, delta=self.trigger_delta)
        else:
            self.trigger = None

        if self.trigger is not None and self.poll_s <= 0.0:
            # Not fatal, and not worth refusing to run: the heartbeat still works, so the brain
            # still samples. But the trigger can never fire, and the reason is not obvious from
            # the configuration alone -- so say it once, at construction.
            logger.warning(
                "pacing has a trigger but poll_s is 0, so the house is never re-read between "
                "decisions and the trigger cannot fire; set FLYBRAIN_POLL_S above 0"
            )

        self._last_step: float | None = None
        self._last_poll: float | None = None
        self._burst_until: float | None = None
        self._triggered = False
        self._previous: list[Signal] = []
        self._steps = 0
        self._started_at: float | None = None

    @property
    def flat_out(self) -> bool:
        """True when the loop never waits. This is the pre-pacing behaviour, kept bit-exact."""
        return self.heartbeat_s <= 0.0

    @property
    def adaptive(self) -> bool:
        """True when something other than the clock can cause a decision."""
        return self.trigger is not None and not self.flat_out

    def _in_burst(self, now: float) -> bool:
        return self._burst_until is not None and now < self._burst_until

    def should_poll(self, now: float) -> bool:
        """Whether to re-read the sensors *without* touching the brain."""
        if self.trigger is None or self.poll_s <= 0.0 or self.flat_out:
            # Flat out never waits, so there is no gap in which a poll would be useful: the
            # brain is already being given the newest reading every iteration.
            return False
        if self._in_burst(now):
            # A burst is already spending the GPU; polling on top would only add Home Assistant
            # traffic while the brain is the bottleneck.
            return False
        if self._last_poll is None:
            return True
        return (now - self._last_poll) >= self.poll_s

    def should_step(self, now: float) -> bool:
        """Whether to advance the brain and make a decision."""
        if self.flat_out:
            return True
        if self._last_step is None:
            return True  # first decision after startup, or after a resume
        if self._triggered:
            return True
        if self._in_burst(now):
            return True
        return (now - self._last_step) >= self.heartbeat_s

    def note_poll(self, now: float, signals: Sequence[Signal]) -> bool:
        """Record a sensor poll; return whether it fired the trigger.

        Firing starts the burst immediately, so the next :meth:`should_step` is ``True`` and the
        event is captured from its first moment rather than up to a heartbeat later.
        """
        self._last_poll = now
        fired = False
        if self.trigger is not None:
            fired = bool(self.trigger(self._previous, signals))
        if fired:
            self._triggered = True
            self._burst_until = now + self.burst_s
        self._previous = list(signals)
        return fired

    def note_step(self, now: float) -> None:
        """Record that a decision was actually made."""
        self._last_step = now
        self._triggered = False
        self._steps += 1
        if self._started_at is None:
            self._started_at = now
        if self._burst_until is not None and now >= self._burst_until:
            self._burst_until = None

    def on_pause(self, now: float) -> None:
        """Reset the cadence so that resuming cannot fire a step instantly.

        The previous *signals* are deliberately kept. A pause is not a change in the house, but
        the house may well have changed during one -- so the first poll after a resume compares
        against the pre-pause reading and bursts if it moved. That is the desirable behaviour:
        the event happened, we simply were not looking.
        """
        self._last_step = now
        self._last_poll = None
        self._burst_until = None
        self._triggered = False

    def snapshot(self, now: float) -> dict:
        """Pacing state for the dashboard, including the duty it has actually achieved.

        Reporting the *observed* duty rather than the configured one matters once a trigger is
        in play: the whole point is that energy depends on how interesting the house has been,
        which no formula for a fixed interval can predict.
        """
        if self.flat_out:
            mode = FLAT_OUT
        elif self._in_burst(now):
            mode = BURST
        else:
            mode = WAITING

        next_in_s: float | None = None
        if mode == WAITING and self._last_step is not None:
            next_in_s = round(max(0.0, self.heartbeat_s - (now - self._last_step)), 1)

        duty = self.observed_duty(now)
        return {
            "mode": mode,
            "adaptive": self.adaptive,
            "heartbeat_s": self.heartbeat_s,
            "poll_s": self.poll_s,
            "burst_s": self.burst_s,
            "trigger_delta": self.trigger_delta,
            "trigger": type(self.trigger).__name__ if self.trigger is not None else None,
            "primary_entity": self.primary_entity,
            "triggered": self._triggered,
            "steps": self._steps,
            "next_in_s": next_in_s,
            "observed_duty": None if duty is None else round(duty, 4),
            "observed_watts": None if duty is None else round(estimated_watts(duty), 1),
        }

    def observed_duty(self, now: float) -> float | None:
        """Steps taken times the cost of a step, over the time in which they were taken.

        ``None`` until at least one step has happened, because "0% duty" from no data would be a
        claim rather than a measurement -- and the dashboard would draw it as real.
        """
        if self._started_at is None or self._steps <= 0:
            return None
        elapsed = now - self._started_at
        if elapsed <= 0.0:
            return None
        return min(1.0, (self._steps * STEP_COST_S) / elapsed)
