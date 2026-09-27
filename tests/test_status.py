"""The status and timeline payloads: the contract the dashboard reads.

These exist because the panels are *pure consumers* of one JSON object. A rename or a missing key
does not raise anything server-side — it renders an em dash in the browser, which looks exactly
like "this feature is not built yet". So the keys the UI reads are pinned here, by name.

The stubs are deliberate: a real ``BrainService`` needs a loaded connectome and a GPU, and a test
that skipped without them would be a test that never runs.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from flybrain.loop import LoopConfig
from flybrain.pet import HONESTY, RESTING, STARTLED, STATES, PetWatcher
from flybrain.server import BrainService


class _StubLoop:
    """Just enough ``LiveLoop`` for the derived payloads."""

    def __init__(self, *, stale: bool = False, signals=(), channels=()) -> None:
        # A real LoopConfig, not a namespace: it is the object the payloads read their settings
        # from, and a stub with the same attribute names would pass while the real one broke.
        # `rest` + `dry_run` is the read-only real-house posture, which is the one that makes
        # `will_send` False and therefore exercises the interesting branch of `_outputs`.
        self.loop = LoopConfig(
            mode="rest",
            dry_run=True,
            light_entity="light.kitchen",
            temperature_entity="sensor.living_room_temperature",
        )
        self.channels = list(channels)
        self.history = [
            {"t": 1000.0, "temperature_c": 21.4, "kelvin": 4120, "band": "neutral"},
        ]
        self.last_action = {
            "at": 1000.0,
            "entity_id": "light.kitchen",
            "service": "turn_on",
            "data": {"color_temp_kelvin": 4120},
            "sent": False,
            "dry_run": True,
        }
        self.reading_stale = stale
        self.last_signals = list(signals)
        # A real pacer; the stub needs one because /api/status reports its state verbatim.
        from flybrain.pacing import Pacer

        self.pacer = Pacer(heartbeat_s=15.0, poll_s=5.0, burst_s=10.0, trigger_delta=0.3,
                           primary_entity="sensor.living_room_temperature")

    def snapshot(self) -> dict:
        return {
            "temperature_entity": "sensor.living_room_temperature",
            "light_entity": "light.kitchen",
            "temperature_c": 21.4,
            "reading_age_s": 2.5,
            "reading_stale": self.reading_stale,
            "kelvin": 4120,
            "ideal_kelvin": 4090,
            "error_k": 30,
            "band": "neutral",
            "driven_neurons": 256,
            "readout_neurons": 512,
            "decisions": 7,
            "channels": [
                {"entity_id": c.entity_id, "kind": "motion", "rate_hz": 12.0}
                for c in self.channels
            ],
            "last_action": self.last_action,
        }


def service(*, stale: bool = False, signals=(), channels=(), bursts=()) -> BrainService:
    """A ``BrainService`` with the connectome-shaped parts replaced by stubs."""
    host = BrainService.__new__(BrainService)  # bypass __init__: it reads the environment
    host.loop = _StubLoop(stale=stale, signals=signals, channels=channels)
    host.pet = PetWatcher()
    host.pet_state = None
    host._prev_sensor_values = {}
    host._last_regions = [{"name": "ALPN", "spikes": 812, "rate_hz": 41.2}]
    host._bursts = list(bursts)
    host._frame = {"seq": 42, "sim_ms": 300.0, "total_spikes": 38412, "active_neurons": 11204}
    host.recorder = None
    host.paused = False
    host.always_on = True
    host._jev = None
    return host


def channel(entity_id: str):
    return SimpleNamespace(entity_id=entity_id, kind="motion", neuron_indices=[1, 2, 3])


# --------------------------------------------------------------------- the three layers


class TestLayers:
    def test_it_reports_all_three_layers_by_name(self) -> None:
        layers = service()._layers()
        assert set(layers) == {"house", "brain", "mapped"}

    def test_the_house_layer_is_the_reading_that_drove_the_window(self) -> None:
        house = service()._layers()["house"]
        assert house["entity"] == "sensor.living_room_temperature"
        assert house["value"] == 21.4
        assert house["age_s"] == 2.5
        assert house["stale"] is False

    def test_a_stale_reading_is_flagged_in_the_house_layer(self) -> None:
        assert service(stale=True)._layers()["house"]["stale"] is True

    def test_the_brain_layer_reports_activity_and_the_busiest_regions(self) -> None:
        brain = service()._layers()["brain"]
        assert brain["active_neurons"] == 11204
        assert brain["spikes"] == 38412
        assert brain["regions"][0]["name"] == "ALPN"

    def test_the_mapped_layer_reports_the_colour_and_the_action(self) -> None:
        mapped = service()._layers()["mapped"]
        assert mapped["kelvin"] == 4120
        assert mapped["ideal_kelvin"] == 4090
        assert mapped["band"] == "neutral"
        assert mapped["action"]["dry_run"] is True

    def test_the_extra_senses_are_listed_in_the_house_layer(self) -> None:
        house = service(channels=[channel("binary_sensor.hall")])._layers()["house"]
        assert house["senses"][0]["entity_id"] == "binary_sensor.hall"

    def test_movement_is_reported_only_for_entities_seen_twice(self) -> None:
        from flybrain.types import Signal, SignalKind

        host = service(signals=[Signal("sensor.room", SignalKind.TEMPERATURE, 21.9, "21.9",
                                      0.0, "")])
        assert host._layers()["house"]["changed"] == {}, "nothing to compare against yet"
        host._prev_sensor_values = {"sensor.room": 21.4}
        assert host._layers()["house"]["changed"] == {"sensor.room": 0.5}


# ---------------------------------------------------------------- what it may touch


class TestOutputs:
    def test_the_light_is_named_with_what_is_written_to_it(self) -> None:
        outputs = service()._outputs()
        assert outputs[0]["entity_id"] == "light.kitchen"
        assert outputs[0]["what"] == "colour temperature only"

    def test_a_dry_run_says_so_rather_than_hiding_it(self) -> None:
        assert service()._outputs()[0]["enabled"] is False
        assert "dry run" in service()._outputs()[0]["reason"]

    def test_inputs_are_listed_as_read_only(self) -> None:
        outputs = service(channels=[channel("binary_sensor.hall")])._outputs()
        inbound = next(o for o in outputs if o["entity_id"] == "binary_sensor.hall")
        assert "read only" in inbound["what"]

    def test_no_loop_means_no_outputs_rather_than_an_exception(self) -> None:
        host = service()
        host.loop = None
        assert host._outputs() == []


# ----------------------------------------------------------------------- the trail


class TestTimeline:
    def test_it_merges_decisions_actions_and_state_changes(self) -> None:
        host = service()
        host.pet.observe(now=1000.0, active_neurons=10_000, total_spikes=1)
        host.pet.observe(now=2000.0, active_neurons=20_000, total_spikes=1)
        entries = host.timeline()["entries"]
        assert {e["kind"] for e in entries} >= {"state", "decision", "action"}

    def test_it_is_newest_first(self) -> None:
        host = service()
        host.pet.observe(now=1000.0, active_neurons=10_000, total_spikes=1)
        host.pet.observe(now=2000.0, active_neurons=20_000, total_spikes=1)
        times = [e["t"] for e in host.timeline()["entries"]]
        assert times == sorted(times, reverse=True)

    def test_a_burst_appears_with_what_moved(self) -> None:
        host = service(bursts=[{"t": 1500.0, "changes": {"sensor.room": 0.4}}])
        entries = host.timeline()["entries"]
        burst = next(e for e in entries if e["kind"] == "burst")
        assert "sensor.room +0.40" in burst["detail"]

    def test_every_entry_can_be_placed_on_a_timeline(self) -> None:
        """A missing timestamp means a mark the browser cannot draw, so it is filtered here."""
        host = service(bursts=[{"t": None, "changes": {}}])
        for entry in host.timeline()["entries"]:
            assert isinstance(entry["t"], (int, float))

    def test_it_is_capped(self) -> None:
        host = service()
        for i in range(400):
            host.pet.observe(now=float(i), active_neurons=10_000 + i * 100, total_spikes=1)
        assert len(host.timeline()["entries"]) <= 80

    def test_an_empty_service_still_returns_the_envelope(self) -> None:
        host = service()
        host.loop = None
        payload = host.timeline()
        assert payload["entries"] == [] and "now" in payload


# ----------------------------------------------------------------------- the status


class TestStatus:
    def _status(self, host) -> dict:
        # The Jev probe is stubbed: the real one makes an HTTP request, and a unit test that
        # needed the network would be a unit test that fails on a train. `probe` is accepted and
        # ignored so the stub keeps matching the real signature as it grows.
        async def fake_jev_status(*, refresh: bool = False, probe: bool = True) -> dict:
            return {"available": False, "reason": "no_key", "detail": "", "models": [],
                    "floor_ms": 47.0, "config": {"model": "jev-1.13.0"}, "calls": 0}

        host.jev_status = fake_jev_status
        return asyncio.run(host.status())

    def test_it_carries_every_panel_the_dashboard_draws(self) -> None:
        payload = self._status(service())
        assert set(payload) == {"pet", "journal", "layers", "pacing", "trust", "jev"}

    def test_before_any_window_the_pet_admits_it_has_nothing(self) -> None:
        """`None` rather than a default word: 'resting' would be a claim about a window that
        never happened."""
        pet = self._status(service())["pet"]
        assert pet["state"] is None
        assert "no completed window" in pet["sentence"]
        assert pet["vocabulary"] == list(STATES)
        assert pet["honesty"] == HONESTY

    def test_after_a_window_the_pet_carries_state_sentence_and_contributors(self) -> None:
        host = service()
        host.pet_state = host.pet.observe(now=time.time(), active_neurons=10_000, total_spikes=99)
        pet = self._status(host)["pet"]
        assert pet["state"] == RESTING
        assert pet["sentence"]
        assert pet["contributors"], "a label without its contributors is a claim"

    def test_the_trust_block_answers_what_it_may_touch(self) -> None:
        trust = self._status(service())["trust"]
        for key in ("mode", "dry_run", "will_send", "paused", "recording", "outputs", "trained"):
            assert key in trust
        assert trust["mode"] == "rest"
        assert trust["dry_run"] is True

    def test_the_trained_readout_is_reported_with_its_regime(self) -> None:
        """A readout is only valid under the regime it was fitted in, so the panel says which."""
        trained = self._status(service())["trust"]["trained"]
        assert trained is None or "regime" in trained

    def test_the_journal_reports_windows_labels_and_a_line(self) -> None:
        journal = self._status(service())["journal"]
        assert journal["windows"] == 0
        assert journal["labels"] == 0
        assert journal["recording"] is False
        assert isinstance(journal["line"], str) and journal["line"]

    def test_the_pacing_block_is_present_even_with_no_decisions(self) -> None:
        pacing = self._status(service())["pacing"]
        assert pacing is not None
        for key in ("mode", "heartbeat_s", "burst_s", "trigger_delta", "poll_s", "steps"):
            assert key in pacing

    def test_it_survives_a_service_with_no_loop_at_all(self) -> None:
        host = service()
        host.loop = None
        payload = self._status(host)
        assert payload["layers"]["house"]["value"] is None
        assert payload["trust"]["mode"] is None


class TestPetVocabularyIsClosed:
    def test_the_server_only_ever_reports_words_from_the_vocabulary(self) -> None:
        """The vocabulary is the contract. Anything outside it is a state the UI cannot colour."""
        host = service()
        for i, activity in enumerate((10_000, 40_000, 10_000, 10_500)):
            state = host.pet.observe(now=float(i), active_neurons=activity, total_spikes=1)
            assert state.state in STATES or state.state is None

    def test_a_stale_reading_reports_resting_with_the_refusal_in_the_sentence(self) -> None:
        host = service()
        state = host.pet.observe(now=0.0, active_neurons=50_000, total_spikes=1, stale=True,
                                 reading_age_s=99.0)
        assert state.state == RESTING
        assert "nothing about the house" in state.sentence

    def test_a_burst_reports_startled(self) -> None:
        host = service()
        host.pet.note_burst(time.time())
        state = host.pet.observe(now=time.time(), active_neurons=10_000, total_spikes=1,
                                 sensor_changes={"sensor.room": 0.6})
        assert state.state == STARTLED


@pytest.mark.parametrize("state", STATES)
def test_every_vocabulary_word_has_a_colour_in_the_stylesheet(state: str) -> None:
    """A word with no CSS rule renders as an unstyled glyph, which looks like a bug.

    Cheap check, but it is the one that catches 'added a fifth state, forgot the stylesheet'.
    """
    import pathlib

    css = pathlib.Path("web/app.css").read_text(encoding="utf-8")
    assert f'data-state="{state}"' in css, state
