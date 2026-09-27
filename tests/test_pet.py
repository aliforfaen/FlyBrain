"""The house pet: a closed vocabulary, derived from measurements, with its evidence attached.

Two properties are the whole point of this module, and both are easy to lose by "improving" it:

1. **Every state is a function of numbers that exist.** ``derive`` is pure, so each state and each
   boundary between states can be reached exactly, without a GPU, a house, or a clock.
2. **A label never arrives without its contributors.** A test asserts the evidence is present on
   every branch, including the boring one, because the boring branch is where a dashboard is most
   tempted to print a bare word.

The words describe *the brain and the house*. Nothing below asserts that the thing feels anything,
because nothing in the code can know that.
"""

from __future__ import annotations

import pytest

from flybrain.pet import (
    CURIOUS,
    ELEVATED,
    HONESTY,
    RESTING,
    SETTLING,
    STARTLED,
    STATES,
    Contributor,
    Observation,
    PetWatcher,
    derive,
    sensor_deltas,
    summarise_journal,
)


def obs(**kw) -> Observation:
    """An unremarkable window. Each test changes exactly the thing it is about."""
    base = {
        "active_neurons": 10_000,
        "total_spikes": 30_000,
        "baseline_neurons": 10_000.0,
        "trend": 0.0,
        "bursting": False,
        "seconds_since_burst": None,
        "sensor_changes": {},
        "stale": False,
        "reading_age_s": None,
        "busiest": (),
    }
    base.update(kw)
    return Observation(**base)


def state_of(observation: Observation) -> str | None:
    return derive(observation)[0]


class TestVocabulary:
    def test_it_is_closed_and_small(self) -> None:
        assert STATES == (RESTING, CURIOUS, STARTLED, SETTLING)

    def test_the_honesty_note_says_what_the_words_are_not(self) -> None:
        """The single most important sentence in the module, so it is asserted."""
        assert "not a feeling" in HONESTY
        assert "mood table" in HONESTY

    def test_every_state_carries_the_honesty_note(self) -> None:
        from flybrain.pet import PetState

        assert PetState(None, "", (), 0.0).honesty == HONESTY


class TestDerive:
    def test_nothing_unusual_is_resting(self) -> None:
        assert state_of(obs()) == RESTING

    def test_a_house_event_is_startled(self) -> None:
        assert state_of(obs(bursting=True, sensor_changes={"s": 0.4})) == STARTLED

    def test_a_recent_burst_is_still_startled(self) -> None:
        """The burst ends before the reason for it stops being worth showing."""
        assert state_of(obs(seconds_since_burst=3.0, sensor_changes={"s": 0.4})) == STARTLED

    def test_an_old_burst_is_not_startled(self) -> None:
        assert state_of(obs(seconds_since_burst=600.0)) == RESTING

    def test_elevated_and_climbing_is_curious(self) -> None:
        assert state_of(obs(active_neurons=13_000, baseline_neurons=10_000.0, trend=0.2)) == CURIOUS

    def test_elevated_and_falling_is_settling(self) -> None:
        assert state_of(obs(active_neurons=13_000, baseline_neurons=10_000.0, trend=-0.2)) == SETTLING

    def test_elevated_but_flat_is_not_curious(self) -> None:
        """`curious` means *starting something*, which needs a direction, not just a level."""
        assert state_of(obs(active_neurons=13_000, baseline_neurons=10_000.0, trend=0.0)) == (
            SETTLING
        )

    def test_activity_exactly_at_the_threshold_does_not_count(self) -> None:
        """The boundary is inclusive-by-name and asserted, so moving it is a visible decision."""
        at = obs(active_neurons=int(10_000 * ELEVATED), baseline_neurons=10_000.0, trend=0.2)
        assert state_of(at) == CURIOUS
        below = obs(active_neurons=int(10_000 * ELEVATED) - 1, baseline_neurons=10_000.0, trend=0.2)
        assert state_of(below) == RESTING

    def test_an_event_outranks_elevated_activity(self) -> None:
        """The event is *why* the activity is up, and the more specific word is the useful one."""
        both = obs(
            bursting=True,
            active_neurons=20_000,
            baseline_neurons=10_000.0,
            trend=0.9,
            sensor_changes={"s": 1.0},
        )
        assert state_of(both) == STARTLED

    def test_a_stale_sensor_forces_resting_and_says_so(self) -> None:
        """With no usable reading, the house half of every other claim is unknown.

        Calling this 'curious' would attribute curiosity to the loop's cleared drive. The state
        stays in the vocabulary; the sentence carries the refusal.
        """
        observation = obs(active_neurons=20_000, baseline_neurons=10_000.0, trend=0.9, stale=True,
                          reading_age_s=42.5)
        state, sentence, contributors = derive(observation)
        assert state == RESTING
        assert "nothing about the house" in sentence
        assert any(c.label == "age of last good reading" for c in contributors)

    def test_staleness_outranks_even_a_burst(self) -> None:
        assert state_of(obs(stale=True, bursting=True, sensor_changes={"s": 1.0})) == RESTING

    def test_no_baseline_means_no_ratio_claim(self) -> None:
        """A first window cannot be unusual. Claiming otherwise would be an invention."""
        state, sentence, contributors = derive(obs(baseline_neurons=0.0))
        assert state == RESTING
        assert "0%" not in sentence
        assert not any(c.label == "vs its own baseline" for c in contributors)


class TestContributors:
    """A label whose inputs are visible is a measurement; one whose inputs are hidden is a claim."""

    @pytest.mark.parametrize("observation", [
        obs(),
        obs(bursting=True, sensor_changes={"sensor.hall": 0.4}),
        obs(active_neurons=13_000, trend=0.2),
        obs(active_neurons=13_000, trend=-0.2),
        obs(stale=True, reading_age_s=10.0),
    ])
    def test_every_state_shows_its_numbers(self, observation: Observation) -> None:
        _, _, contributors = derive(observation)
        assert contributors, "a state with no contributors is a bare claim"
        for contributor in contributors:
            assert isinstance(contributor, Contributor)
            assert contributor.label
            assert contributor.source in {"frame", "sensor", "pacer", "settings"}

    def test_the_activity_numbers_are_always_present(self) -> None:
        _, _, contributors = derive(obs())
        labels = {c.label for c in contributors}
        assert "active neurons" in labels
        assert "spikes this window" in labels

    def test_the_burst_sentence_names_what_moved(self) -> None:
        _, sentence, contributors = derive(
            obs(bursting=True, sensor_changes={"sensor.hall_temperature": 0.4, "sensor.lux": 12.0})
        )
        assert "sensor.hall_temperature" in sentence
        assert any(c.label == "change" for c in contributors)

    def test_the_biggest_movement_is_named_first(self) -> None:
        _, sentence, _ = derive(
            obs(bursting=True, sensor_changes={"small": 0.1, "large": 9.0})
        )
        assert sentence.index("large") < sentence.index("small")


class TestWatcher:
    def test_the_first_window_establishes_the_baseline(self) -> None:
        watcher = PetWatcher()
        state = watcher.observe(now=0.0, active_neurons=10_000, total_spikes=30_000)
        assert state.state == RESTING
        assert watcher.baseline == pytest.approx(10_000.0)

    def test_the_baseline_follows_the_brain_rather_than_a_constant(self) -> None:
        """`unusual` has to mean unusual for *this* brain, or it silently rots."""
        watcher = PetWatcher()
        for i in range(40):
            watcher.observe(now=i * 60.0, active_neurons=10_000, total_spikes=30_000)
        before = watcher.baseline
        for i in range(40, 120):
            watcher.observe(now=i * 60.0, active_neurons=30_000, total_spikes=30_000)
        assert watcher.baseline > before * 2
        # And now 30k is normal, so a quiet brain is the unusual one.
        state = watcher.observe(now=120 * 60.0, active_neurons=30_000, total_spikes=30_000)
        assert state.state == RESTING

    def test_it_notices_a_climb_against_its_own_baseline(self) -> None:
        watcher = PetWatcher()
        for i in range(20):
            watcher.observe(now=i * 60.0, active_neurons=10_000, total_spikes=30_000)
        state = watcher.observe(now=20 * 60.0, active_neurons=16_000, total_spikes=30_000)
        assert state.state == CURIOUS

    def test_a_burst_is_noted_rather_than_inferred(self) -> None:
        """The loop knows when the trigger fired; guessing it from a frame would be a guess."""
        watcher = PetWatcher()
        watcher.observe(now=0.0, active_neurons=10_000, total_spikes=30_000)
        assert watcher.observe(now=60.0, active_neurons=10_000, total_spikes=30_000).state == RESTING
        watcher.note_burst(61.0)
        state = watcher.observe(now=62.0, active_neurons=10_000, total_spikes=30_000,
                                sensor_changes={"s": 0.5})
        assert state.state == STARTLED

    def test_the_state_keeps_its_since_time_until_it_changes(self) -> None:
        watcher = PetWatcher()
        watcher.observe(now=0.0, active_neurons=10_000, total_spikes=30_000)
        watcher.observe(now=100.0, active_neurons=10_000, total_spikes=30_000)
        state = watcher.observe(now=200.0, active_neurons=10_000, total_spikes=30_000)
        assert state.since_s >= 200.0

    def test_the_journal_accounts_for_the_time(self) -> None:
        watcher = PetWatcher()
        watcher.observe(now=0.0, active_neurons=10_000, total_spikes=30_000)
        watcher.observe(now=100.0, active_neurons=10_000, total_spikes=30_000)
        journal = watcher.journal(100.0)
        assert journal["seconds_in_state"][RESTING] == pytest.approx(100.0, abs=1.0)
        assert journal["observed_s"] == pytest.approx(100.0, abs=1.0)

    def test_the_journal_lists_every_state_including_the_empty_ones(self) -> None:
        """The proportion is the interesting fact; a list of only the states that happened hides it."""
        watcher = PetWatcher()
        watcher.observe(now=0.0, active_neurons=10_000, total_spikes=30_000)
        assert set(watcher.journal(0.0)["seconds_in_state"]) == set(STATES)

    def test_the_journal_reports_the_baseline_it_learned(self) -> None:
        watcher = PetWatcher()
        watcher.observe(now=0.0, active_neurons=12_345, total_spikes=1)
        assert watcher.journal(0.0)["baseline_neurons"] == pytest.approx(12_345.0)

    def test_an_empty_watcher_journals_no_claim(self) -> None:
        journal = PetWatcher().journal(0.0)
        assert journal["observed_s"] == 0.0
        assert journal["baseline_neurons"] is None


class TestSensorDeltas:
    def test_it_reports_movement_per_entity(self) -> None:
        assert sensor_deltas({"a": 1.0}, {"a": 1.5}) == {"a": 0.5}

    def test_no_movement_is_no_entry(self) -> None:
        assert sensor_deltas({"a": 1.0}, {"a": 1.0}) == {}

    def test_a_new_entity_is_not_a_change(self) -> None:
        """It is the first observation of it, not the house moving."""
        assert sensor_deltas({"a": 1.0}, {"a": 1.0, "b": 5.0}) == {}

    def test_a_disappearing_entity_is_not_a_change_either(self) -> None:
        assert sensor_deltas({"a": 1.0, "b": 5.0}, {"a": 1.0}) == {}

    def test_movement_is_signed(self) -> None:
        assert sensor_deltas({"a": 5.0}, {"a": 2.0})["a"] == pytest.approx(-3.0)


class TestSummary:
    def test_the_summary_states_the_proportion_rather_than_adjectives(self) -> None:
        journal = {"observed_s": 3600.0, "seconds_in_state": {RESTING: 3500.0, CURIOUS: 100.0}}
        line = summarise_journal(journal, windows=120, labels=3, watts=25.0)
        assert "120 windows" in line
        assert "3 labels" in line
        assert "97%" in line
        assert "25 W" in line

    def test_one_label_is_not_pluralised(self) -> None:
        line = summarise_journal({"observed_s": 0.0, "seconds_in_state": {}}, windows=1, labels=1,
                                 watts=None)
        assert "1 label" in line and "labels" not in line

    def test_with_no_data_it_says_only_what_it_knows(self) -> None:
        line = summarise_journal({"observed_s": 0.0, "seconds_in_state": {}}, windows=0, labels=0,
                                 watts=None)
        assert line == "0 windows"


# ------------------------------------------------------------------ server wiring


class _FakeBurstHost:
    """The smallest object that can act as ``self`` for ``BrainService._note_burst``."""

    def __init__(self) -> None:
        self.pet = PetWatcher()
        self._prev_sensor_values: dict[str, float] = {}
        self._bursts: list[dict] = []


def signal(entity_id: str, value: float, state: str = "21.0"):
    from flybrain.types import Signal, SignalKind

    return Signal(entity_id=entity_id, kind=SignalKind.TEMPERATURE, value=value,
                  state=state, timestamp=0.0, attributes={})


class TestBurstClock:
    """The pet is fed two clocks if you are not careful, and it shows as an absurd number.

    ``run_loop`` works in ``time.monotonic()`` (right for cadence — it cannot jump) while the
    windows the pet observes and the trail the browser draws both need ``time.time()``. Mixing
    them put the last burst ~1.79e9 seconds in the future, which rendered as
    "since the last change 1790499452s".
    """

    def test_a_burst_is_stamped_on_the_same_clock_as_the_windows(self) -> None:
        import time as _time

        from flybrain.server import BrainService

        host = _FakeBurstHost()
        before = _time.time()
        BrainService._note_burst(host, [signal("sensor.room", 21.4)])
        after = _time.time()

        stamped = host._bursts[-1]["t"]
        assert before <= stamped <= after, "the burst must be stamped in wall-clock time"
        assert host.pet.last_burst_at == stamped

    def test_the_age_of_a_burst_is_therefore_small(self) -> None:
        import time as _time

        from flybrain.server import BrainService

        host = _FakeBurstHost()
        BrainService._note_burst(host, [signal("sensor.room", 21.4)])
        state = host.pet.observe(
            now=_time.time(), active_neurons=10_000, total_spikes=1, bursting=False,
            sensor_changes={"sensor.room": 0.4},
        )
        # Well inside the startle window, rather than 1.79e9 seconds past it.
        assert state.state == STARTLED

    def test_the_burst_records_what_moved_using_the_previous_window(self) -> None:
        from flybrain.server import BrainService

        host = _FakeBurstHost()
        host._prev_sensor_values = {"sensor.room": 21.0}
        BrainService._note_burst(host, [signal("sensor.room", 21.5)])
        assert host._bursts[-1]["changes"] == {"sensor.room": 0.5}

    def test_a_dead_sensor_is_not_recorded_as_a_movement(self) -> None:
        from flybrain.server import BrainService
        from flybrain.types import Signal, SignalKind

        host = _FakeBurstHost()
        host._prev_sensor_values = {"sensor.room": 21.0}
        dead = Signal(entity_id="sensor.room", kind=SignalKind.TEMPERATURE, value=0.0,
                      state="unavailable", timestamp=0.0, attributes={"unavailable": True})
        BrainService._note_burst(host, [dead])
        assert host._bursts[-1]["changes"] == {}

    def test_the_span_is_in_minutes_when_it_is_under_an_hour(self) -> None:
        """"100% of 0.0 h" is technically true and useless to read."""
        line = summarise_journal(
            {"observed_s": 300.0, "seconds_in_state": {RESTING: 300.0}}, windows=5, labels=0,
            watts=None,
        )
        assert "5 min" in line
        assert "0.0 h" not in line
