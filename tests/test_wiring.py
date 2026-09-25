"""Wiring: entity -> fly pathway matching, discovery, and channel construction.

The matching tests are the important ones. A substring match on ``"temp"`` classifies
``sensor.backup_last_attempted_automatic_backup`` as a thermometer, which is the kind of error
that produces a plausible-looking wiring table and a confused brain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flybrain.ha import classify_kind
from flybrain.mapping import RoleResolver
from flybrain.types import Signal, SignalKind
from flybrain.wiring import (
    Pathway,
    discover,
    format_wiring,
    load_wiring,
    role_for,
)


def _signal(entity_id: str, state: str = "on", value: float = 1.0, dc: str | None = None):
    attrs = {"device_class": dc} if dc else {}
    return Signal(
        entity_id=entity_id,
        kind=classify_kind(entity_id),
        value=value,
        state=state,
        timestamp=0.0,
        attributes=attrs,
    )


# ------------------------------------------------------------------ matching


def test_temp_does_not_match_inside_attempted() -> None:
    """The false positive that discovery found on the real house."""
    assert role_for("sensor.backup_last_attempted_automatic_backup") is None


def test_gpu_temperature_is_not_a_room_thermometer() -> None:
    """``rack_gputemperature`` contains "temperature" as a substring, not a word."""
    assert role_for("sensor.rack_gputemperature") is None


def test_real_temperature_matches_thermosensory() -> None:
    role, _ = role_for("sensor.hallway_temperature")
    assert role == "thermosensory"


def test_temperature_matches_as_a_whole_word() -> None:
    assert role_for("sensor.hallway_temp")[0] == "thermosensory"


def test_motion_goes_to_the_visual_pathway() -> None:
    assert role_for("binary_sensor.hallway_motion")[0] == "visual"


def test_person_detection_goes_to_the_visual_pathway() -> None:
    assert role_for("binary_sensor.door_camera_person_detection")[0] == "visual"


def test_cell_motion_matches_as_a_phrase() -> None:
    assert role_for("binary_sensor.door_camera_cell_motion_detection")[0] == "visual"


def test_sound_is_matched_before_motion() -> None:
    """A camera can tag an acoustic event as device_class motion; hearing is not seeing."""
    role, note = role_for("binary_sensor.camera_bark_detection")
    assert role == "mechanosensory"
    assert "acoustic" in note


def test_glass_break_is_acoustic_not_visual() -> None:
    assert role_for("binary_sensor.door_camera_glass_break_detection")[0] == "mechanosensory"


def test_humidity_and_air_quality_roles() -> None:
    assert role_for("sensor.bathroom_humidity")[0] == "hygrosensory"
    assert role_for("sensor.air_quality_voc")[0] == "olfactory"


def test_unmatched_entity_has_no_role() -> None:
    assert role_for("sensor.google_wifi_uptime") is None


# ------------------------------------------------------------------ discovery


def _house():
    return [
        _signal("sensor.hallway_temperature", "23.6", 23.6),
        _signal("binary_sensor.hallway_motion", "off", 0.0),
        _signal("sensor.hallway_illuminance", "6", 6.0),
        _signal("binary_sensor.door_camera_person_detection", "unavailable", 0.0),
        _signal("light.hall_lamp", "off", 0.0),
        _signal("sensor.phone_audio_mode", "normal", 0.0, dc="enum"),
        _signal("sensor.google_wifi_uptime", "7.7", 7.7),
    ]


def test_discovery_separates_live_from_dormant() -> None:
    w = discover(_house())
    live = {p.entity_id for p in w.pathways}
    dormant = {p.entity_id for p, _ in w.dormant}
    assert live == {
        "sensor.hallway_temperature",
        "binary_sensor.hallway_motion",
        "sensor.hallway_illuminance",
    }
    assert dormant == {"binary_sensor.door_camera_person_detection"}


def test_dormant_keeps_the_reason() -> None:
    w = discover(_house())
    assert w.dormant[0][1] == "unavailable"


def test_actuators_are_not_sensed() -> None:
    """Feeding a lamp's own state back in as a sense would be circular."""
    w = discover(_house())
    assert all(not p.entity_id.startswith("light.") for p in w.pathways)
    assert all(not e.startswith("light.") for e, _ in w.ignored)


def test_enum_sensors_are_ignored_with_a_reason() -> None:
    w = discover(_house())
    reasons = dict(w.ignored)
    assert reasons["sensor.phone_audio_mode"] == "categorical state, not a measurement"


def test_unmatched_sensor_is_reported_not_dropped() -> None:
    w = discover(_house())
    assert "sensor.google_wifi_uptime" in dict(w.ignored)


def test_unreachable_roles_are_reported() -> None:
    """No olfactory sensor and no olfactory pathway must be distinguishable."""
    w = discover(_house())
    missing = w.unreachable_roles()
    assert "olfactory" in missing and "hygrosensory" in missing
    assert "visual" not in missing


def test_counts_and_roles() -> None:
    w = discover(_house())
    assert w.roles() == ["thermosensory", "visual"]
    assert w.counts() == {"thermosensory": 1, "visual": 2}


def test_format_wiring_mentions_missing_roles() -> None:
    text = format_wiring(discover(_house()))
    assert "no sensor at all for" in text
    assert "olfactory" in text


# ------------------------------------------------------------------ channels


def _resolver(n_visual: int = 200, n_thermo: int = 10) -> RoleResolver:
    rows = [
        {"cell_class": "visual", "super_class": "optic", "cell_type": f"R{i}"}
        for i in range(n_visual)
    ] + [
        {"cell_class": "thermosensory", "super_class": "central", "cell_type": f"T{i}"}
        for i in range(n_thermo)
    ]
    table = pd.DataFrame(rows)
    return RoleResolver(table, len(table))


def test_channels_are_disjoint_within_a_role() -> None:
    """Two sensors on one role must not share neurons, or their spikes are indistinguishable."""
    w = discover(_house())
    channels = w.to_channels(_resolver(), neurons_per_pathway=16)
    visual = [c for c in channels if c.kind in (SignalKind.MOTION, SignalKind.ILLUMINANCE)]
    assert len(visual) == 2
    a, b = (set(c.neuron_indices.tolist()) for c in visual)
    assert a and b and not (a & b)


def test_channel_width_is_capped() -> None:
    w = discover(_house())
    for ch in w.to_channels(_resolver(), neurons_per_pathway=16):
        assert ch.neuron_indices.size <= 16


def test_small_population_is_used_whole() -> None:
    """Thermosensory has only 29 neurons; all of them should be used, not padded."""
    w = discover(_house())
    channels = w.to_channels(_resolver(), neurons_per_pathway=64)
    thermo = next(c for c in channels if c.kind is SignalKind.TEMPERATURE)
    assert thermo.neuron_indices.size == 10          # the synthetic role size


def test_channel_building_is_deterministic() -> None:
    """A restart must re-create the identical wiring."""
    w = discover(_house())
    first = w.to_channels(_resolver(), neurons_per_pathway=16)
    second = w.to_channels(_resolver(), neurons_per_pathway=16)
    for a, b in zip(first, second):
        assert a.entity_id == b.entity_id
        assert np.array_equal(a.neuron_indices, b.neuron_indices)


def test_channels_match_the_pathway_range() -> None:
    w = discover(_house())
    by_entity = {c.entity_id: c for c in w.to_channels(_resolver())}
    thermo = by_entity["sensor.hallway_temperature"]
    assert (thermo.vmin, thermo.vmax) == (10.0, 35.0)
    motion = by_entity["binary_sensor.hallway_motion"]
    assert (motion.vmin, motion.vmax) == (0.0, 1.0)


def test_pathway_to_channel_round_trips() -> None:
    p = Pathway(
        entity_id="sensor.x_temperature",
        role="thermosensory",
        kind=SignalKind.TEMPERATURE,
        vmin=10.0,
        vmax=35.0,
    )
    ch = p.to_channel(np.arange(4))
    assert ch.entity_id == "sensor.x_temperature"
    assert ch.neuron_indices.tolist() == [0, 1, 2, 3]


def test_json_round_trip(tmp_path) -> None:
    import json

    w = discover(_house())
    path = tmp_path / "wiring.json"
    path.write_text(json.dumps(w.as_dict()), encoding="utf-8")
    back = load_wiring(path)
    assert [p.entity_id for p in back.pathways] == [p.entity_id for p in w.pathways]
    assert back.roles() == w.roles()
    assert len(back.dormant) == len(w.dormant)


def test_no_pathways_yields_no_channels() -> None:
    w = discover([_signal("sensor.google_wifi_uptime")])
    assert w.to_channels(_resolver()) == []


def test_unknown_role_is_skipped_not_fatal() -> None:
    """A wiring file naming a role that does not resolve must not take the loop down."""
    w = discover(_house())
    before = len(w.pathways)
    w.pathways.append(
        Pathway(
            entity_id="sensor.ghost",
            role="not_a_real_role",
            kind=SignalKind.OTHER,
            vmin=0.0,
            vmax=1.0,
        )
    )
    channels = w.to_channels(_resolver())
    assert len(channels) == before            # the ghost is dropped
    assert all(c.entity_id != "sensor.ghost" for c in channels)
