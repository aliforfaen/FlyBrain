"""The house pet: a closed vocabulary of states, derived from measurements only.

The pitch in ``docs/roadmap.md`` §9 is *observable personality with honest provenance*: let the
reservoir produce slow state, and let explicit software decide how that state is presented. Two
rules follow, and this module exists to make them structural rather than a matter of wording:

1. **A small, closed vocabulary**, each state derived from measured quantities — activity,
   change, and the house's own signals — and **never from a hand-written mood table**. There is
   no mapping from "temperature 19 °C" to "feels cosy" anywhere in here, because that would be
   an invention wearing a measurement's clothes.
2. **Every label shows its contributors.** :func:`derive` returns the numbers behind the word
   alongside the word, and the panel puts them on screen. A label whose inputs are visible is a
   measurement; a label whose inputs are hidden is a claim.

So the words are chosen to describe **the brain and the house**, not an inner life:
``resting``, ``curious``, ``startled``, ``settling``. They are answers to "what is it doing?",
not "how does it feel?". The distinction is deliberate and the panel says so out loud.

Two design consequences worth stating, because they are the parts most likely to be "improved"
into dishonesty later:

* **Activity is compared against the brain's own baseline**, an exponential moving average, and
  never against an absolute threshold. The number of neurons joining in depends on the drive, the
  connectome and the window length; a fixed cut-off would be a calibration that silently rots.
  A *ratio to its own recent self* stays meaningful across all of that.
* **A stale sensor forces ``resting``.** Not because the brain is resting, but because with no
  usable reading the house half of every other claim is unknown. Saying "curious" would be
  attributing curiosity to the loop's cleared drive. The sentence says exactly that instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: The whole vocabulary. Four states, closed on purpose: a longer list would be a taxonomy, and
#: the point is a glanceable word that always means the same kind of thing.
RESTING = "resting"
CURIOUS = "curious"
STARTLED = "startled"
SETTLING = "settling"
STATES = (RESTING, CURIOUS, STARTLED, SETTLING)

#: How long after a burst the pet is still willing to call itself startled. The burst lasts 10 s
#: by default; the extra few seconds are the part where the *reason* is still on screen.
STARTLE_S = 12.0

#: How far above its own baseline activity has to be before it counts as "more than usual". A
#: ratio rather than an absolute count, so it survives a change of window length or drive.
ELEVATED = 1.15

#: The comment shown beside the vocabulary, in the panel and in the API. The single most
#: important sentence in this module.
HONESTY = (
    "These words describe the brain and the house — activity, change, and what the sensors "
    "reported — not a feeling. Nothing here is a mood table."
)


@dataclass(frozen=True)
class Contributor:
    """One measured quantity behind a label, and where it came from.

    ``source`` names the part of the system it was read from, so a number on screen can always be
    traced: ``frame`` (the brain), ``pacer`` (the scheduler), ``sensor`` (the house), ``settings``.
    """

    label: str
    value: Any
    unit: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {"label": self.label, "value": self.value, "unit": self.unit, "source": self.source}


@dataclass(frozen=True)
class Observation:
    """Everything the pet is allowed to look at, as a value.

    Deliberately a frozen dataclass with no behaviour: :func:`derive` is then a pure function of
    its inputs, which is what makes every state and every boundary testable without a GPU, a
    house, or a clock.
    """

    active_neurons: int
    total_spikes: int
    #: Exponential moving average of ``active_neurons`` over recent windows. ``None`` until there
    #: is a second window to compare against, which is why the first state is not a guess.
    baseline_neurons: float
    #: Signed relative change in activity since the previous window.
    trend: float
    #: True while a trigger-driven burst is running.
    bursting: bool
    #: Seconds since the last trigger fired, or ``None`` if it never has.
    seconds_since_burst: float | None = None
    #: Per-entity movement since the previous window, in each entity's own units.
    sensor_changes: Mapping[str, float] = field(default_factory=dict)
    #: True when the configured sensor has no usable reading.
    stale: bool = False
    #: Age of the last good reading, in seconds, when stale.
    reading_age_s: float | None = None
    #: Names of the busiest cell classes this window, for the "brain did" layer.
    busiest: tuple[str, ...] = ()


@dataclass(frozen=True)
class PetState:
    """The pet's answer, with its evidence attached."""

    state: str | None
    sentence: str
    contributors: tuple[Contributor, ...]
    since_s: float
    honesty: str = HONESTY

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "sentence": self.sentence,
            "contributors": [c.to_dict() for c in self.contributors],
            "since_s": round(self.since_s, 1),
            "honesty": self.honesty,
            "vocabulary": list(STATES),
        }


def _ratio(active: int, baseline: float) -> float | None:
    if not baseline or baseline <= 0:
        return None
    return active / baseline


def _pct(ratio: float) -> str:
    return f"{ratio * 100:.0f}%"


def derive(obs: Observation, *, startle_s: float = STARTLE_S) -> tuple[str | None, str, tuple[Contributor, ...]]:
    """Turn one observation into a state, a sentence, and the numbers behind them.

    Ordered by precedence, and the order is the whole judgement: a house event outranks elevated
    activity, because an event is *why* the activity is elevated and the more specific word is
    the more useful one. Within "elevated", the sign of the trend separates still-climbing
    (``curious``) from coming-back-down (``settling``).
    """
    ratio = _ratio(obs.active_neurons, obs.baseline_neurons)
    base = [
        Contributor("active neurons", obs.active_neurons, "", "frame"),
        Contributor("spikes this window", obs.total_spikes, "", "frame"),
    ]
    if ratio is not None:
        base.append(Contributor("vs its own baseline", round(ratio, 2), "x", "frame"))

    # 1. No usable reading: say so, and refuse to describe the house at all. The brain keeps
    #    running on cleared drive, so any state word would be about the loop, not the room.
    if obs.stale:
        contributors = [
            Contributor("sensor", "no usable reading", "", "sensor"),
            Contributor(
                "age of last good reading",
                None if obs.reading_age_s is None else round(obs.reading_age_s, 1),
                "s",
                "sensor",
            ),
            *base,
        ]
        sentence = (
            "No usable sensor reading, so this says nothing about the house — the brain is "
            "running on a cleared drive and the loop refuses to act."
        )
        return RESTING, sentence, tuple(contributors)

    # 2. Something moved enough to fire the trigger. This is the loudest thing that happens.
    if obs.bursting or (
        obs.seconds_since_burst is not None and obs.seconds_since_burst <= startle_s
    ):
        # Drop movements that would *print* as zero. A delta of -0.004 formats as "-0.00", and
        # "-0.00" in a sentence a person reads is worse than useless: it looks like a display
        # rounding bug when it is really a real-but-imperceptible drift being reported as an
        # event. The trigger has its own threshold and is unaffected — this decides what is worth
        # *saying*, not what counts as a change.
        movers = sorted(
            ((e, v) for e, v in obs.sensor_changes.items() if round(v, 2)),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        who = (
            ", ".join(f"{entity} ({value:+.2f})" for entity, value in movers[:3])
            # Nothing numeric worth naming: the trigger fired on a discrete sensor changing
            # state, which has no magnitude to print.
            or "a sensor changed state"
        )
        contributors = [
            Contributor("change", who, "", "sensor"),
            Contributor(
                "since the change",
                None if obs.seconds_since_burst is None else round(obs.seconds_since_burst, 1),
                "s",
                "pacer",
            ),
            Contributor("bursting", bool(obs.bursting), "", "pacer"),
            *base,
        ]
        sentence = (
            f"The house moved — {who} — so the brain is running at full rate to capture it "
            "rather than sampling it once."
        )
        return STARTLED, sentence, tuple(contributors)

    # 3. Elevated and still climbing: something is building.
    if ratio is not None and ratio >= ELEVATED and obs.trend > 0:
        contributors = [
            *base,
            Contributor("trend since last window", f"{obs.trend:+.1%}", "", "frame"),
        ]
        if obs.busiest:
            contributors.append(Contributor("busiest region", obs.busiest[0], "", "frame"))
        sentence = (
            f"More of the brain is joining in than usual — {obs.active_neurons:,} neurons, "
            f"{_pct(ratio)} of its own baseline — and still climbing."
        )
        return CURIOUS, sentence, tuple(contributors)

    # 4. Elevated but coming down: the tail of something.
    if ratio is not None and ratio >= ELEVATED:
        contributors = [
            *base,
            Contributor("trend since last window", f"{obs.trend:+.1%}", "", "frame"),
            Contributor(
                "since the last change",
                None if obs.seconds_since_burst is None else round(obs.seconds_since_burst, 1),
                "s",
                "pacer",
            ),
        ]
        sentence = (
            f"Still above its own baseline ({_pct(ratio)}) but falling, so it is on the way "
            "back down rather than starting something."
        )
        return SETTLING, sentence, tuple(contributors)

    # 5. Nothing unusual. The interesting part of this branch is what it is *not*: it is not a
    #    claim that the house is empty, only that nothing moved and activity is unchanged.
    contributors = [
        *base,
        Contributor("trigger fired", False, "", "pacer"),
    ]
    if obs.seconds_since_burst is not None:
        contributors.append(
            Contributor("since the last change", round(obs.seconds_since_burst, 1), "s", "pacer")
        )
    sentence = (
        f"Nothing has moved and activity is steady at {_pct(ratio) if ratio else 'its own level'} "
        "of baseline."
    )
    return RESTING, sentence, tuple(contributors)


def sensor_deltas(
    previous: Mapping[str, float],
    current: Mapping[str, float],
    *,
    epsilon: float = 1e-9,
) -> dict[str, float]:
    """Per-entity movement between two reads, for the "the house moved" sentence.

    Takes plain ``{entity_id: value}`` maps rather than :class:`~flybrain.types.Signal` lists, so
    it is testable with a dict literal and has no opinion about where the values came from. The
    caller is responsible for having dropped dead readings: a sensor going ``unavailable`` is not
    the house moving, and putting that in a sentence a person reads would be worse than saying
    nothing.

    Entities present in only one of the two maps are skipped. A newly discovered entity is not a
    change, it is the first observation of it.
    """
    out: dict[str, float] = {}
    for entity_id, now_value in current.items():
        if entity_id not in previous:
            continue
        delta = float(now_value) - float(previous[entity_id])
        if abs(delta) > epsilon:
            out[entity_id] = round(delta, 3)
    return out


class PetWatcher:
    """Keeps the little bit of history a state word needs, and nothing else.

    The only genuinely stateful parts are the baseline (an EMA, so "unusual" means unusual *for
    this brain in this room*, not unusual against a number someone chose) and the clock that
    turns a burst into "3 seconds since the house moved".

    Kept small on purpose. Everything it holds is either a number the panel shows or something a
    test can read, and it must never become a place where behaviour is decided.
    """

    def __init__(
        self,
        *,
        baseline_alpha: float = 0.12,
        startle_s: float = STARTLE_S,
        log_limit: int = 200,
    ) -> None:
        self.baseline_alpha = float(baseline_alpha)
        self.startle_s = float(startle_s)
        self.log_limit = max(1, int(log_limit))
        self.baseline: float | None = None
        self.prev_activity: int | None = None
        self.last_burst_at: float | None = None
        self.state: str | None = None
        self.state_since: float | None = None
        #: ``(when, state, sentence)`` for each change, capped at ``log_limit``. In memory only:
        #: a restart is a new session, and persisting this would invent history.
        self.state_log: list[tuple[float, str, str]] = []
        #: Seconds spent in each state this run, for the journal. Nothing here is persisted: a
        #: restart is a new session, and pretending otherwise would invent history.
        self.time_in_state: dict[str, float] = {}
        self._last_seen: float | None = None

    # ---------------------------------------------------------------- inputs

    def note_burst(self, now: float) -> None:
        """Record that the trigger fired. Called by the loop, not inferred from the frame."""
        self.last_burst_at = now

    def observe(
        self,
        *,
        now: float,
        active_neurons: int,
        total_spikes: int,
        sensor_changes: Mapping[str, float] | None = None,
        stale: bool = False,
        reading_age_s: float | None = None,
        bursting: bool = False,
        busiest: tuple[str, ...] = (),
    ) -> PetState:
        """Fold one window in and return the new state."""
        trend = 0.0
        if self.prev_activity:
            trend = (active_neurons - self.prev_activity) / float(self.prev_activity)

        if self.baseline is None:
            # The first window *is* the baseline. It cannot be unusual yet, and saying so would
            # be an invention; from the next window on, the comparison is real.
            self.baseline = float(active_neurons)
        else:
            a = self.baseline_alpha
            self.baseline = a * float(active_neurons) + (1.0 - a) * self.baseline

        seconds_since_burst = (
            None if self.last_burst_at is None else max(0.0, now - self.last_burst_at)
        )

        observation = Observation(
            active_neurons=int(active_neurons),
            total_spikes=int(total_spikes),
            baseline_neurons=float(self.baseline),
            trend=trend,
            bursting=bursting,
            seconds_since_burst=seconds_since_burst,
            sensor_changes=dict(sensor_changes or {}),
            stale=stale,
            reading_age_s=reading_age_s,
            busiest=busiest,
        )
        state, sentence, contributors = derive(observation, startle_s=self.startle_s)

        if self._last_seen is not None and self.state is not None:
            self.time_in_state[self.state] = self.time_in_state.get(self.state, 0.0) + max(
                0.0, now - self._last_seen
            )
        self._last_seen = now

        if state != self.state:
            self.state = state
            self.state_since = now
            self.state_log.append((now, state, sentence))
            del self.state_log[: max(0, len(self.state_log) - self.log_limit)]
        self.prev_activity = int(active_neurons)

        since_s = 0.0 if self.state_since is None else max(0.0, now - self.state_since)
        return PetState(
            state=state,
            sentence=sentence,
            contributors=contributors,
            since_s=since_s,
        )

    # --------------------------------------------------------------- outputs

    def trail(self) -> list[dict]:
        """State changes, newest last, for the memory trail.

        Only *changes*, not every window: a trail of 1,440 identical entries would be a wall of
        noise, and the point of the trail is that each mark on it means something happened.
        """
        return [
            {"t": t, "kind": "state", "text": state, "detail": sentence}
            for t, state, sentence in self.state_log
        ]

    def journal(self, now: float) -> dict:
        """Time spent in each state, for the daily journal.

        Reports *all four* states, including the ones with no time, because the interesting and
        honest fact is usually the proportion — "resting 96% of the last hour" is a result, and a
        panel that only listed the states that happened would hide it.
        """
        totals = dict(self.time_in_state)
        if self._last_seen is not None and self.state is not None:
            totals[self.state] = totals.get(self.state, 0.0) + max(0.0, now - self._last_seen)
        return {
            "seconds_in_state": {name: round(totals.get(name, 0.0), 1) for name in STATES},
            "observed_s": round(sum(totals.values()), 1),
            "since_state_s": None if self.state_since is None else round(now - self.state_since, 1),
            "baseline_neurons": None if self.baseline is None else round(self.baseline, 1),
            "started_at": None if self.state_since is None else self.state_since,
        }


def summarise_journal(journal: Mapping[str, Any], *, windows: int, labels: int, watts: float | None) -> str:
    """One honest line for the day, from the counters the server already keeps.

    Deliberately plain and quantitative. The journal's job is to make behaviour legible over days,
    which is the timescale the reservoir's own memory cannot reach — so it states what happened
    and refuses to editorialise about it.
    """
    seconds = float(journal.get("observed_s") or 0.0)
    per_state = journal.get("seconds_in_state") or {}
    resting = float(per_state.get(RESTING, 0.0))
    share = 0.0 if seconds <= 0 else resting / seconds
    parts = [f"{windows:,} windows"]
    if labels:
        parts.append(f"{labels} label{'s' if labels != 1 else ''}")
    if seconds >= 60:
        # Minutes below the hour: "100% of 0.0 h" is technically true and useless to read.
        span = f"{seconds / 60:.0f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"
        parts.append(f"{share:.0%} of {span} resting")
    if watts is not None:
        parts.append(f"~{watts:.0f} W")
    return " · ".join(parts)
