"""Tests for :mod:`flybrain.ha`.

Everything here is offline: the REST adapter is exercised through
``httpx.MockTransport`` so no socket is ever opened.  Async entry points are
driven with :func:`asyncio.run` because ``pytest-asyncio`` is not a project
dependency.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import httpx
import pytest

from flybrain.ha import (
    HomeAssistant,
    MockHomeAssistant,
    RestHomeAssistant,
    Scenario,
    classify_kind,
    make_home_assistant,
    parse_state_value,
)
from flybrain.types import Signal, SignalKind

T = TypeVar("T")


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


def signals_by_id(ha: MockHomeAssistant) -> dict[str, Signal]:
    """Fetch mock signals keyed by ``entity_id``."""
    return {signal.entity_id: signal for signal in run(ha.get_signals())}


# --------------------------------------------------------------------------- mock


def test_mock_call_service_mutates_state_and_records_calls() -> None:
    ha = MockHomeAssistant()
    assert ha.states["light.kitchen"] == "off"

    assert run(ha.call_service("light.kitchen", "turn_on")) is True
    assert ha.states["light.kitchen"] == "on"

    assert run(ha.call_service("light.kitchen", "toggle")) is True
    assert ha.states["light.kitchen"] == "off"

    assert run(ha.call_service("switch.fan", "turn_on", {"percentage": 40})) is True
    assert ha.states["switch.fan"] == "on"

    assert ha.calls == [
        ("light.kitchen", "turn_on", {}),
        ("light.kitchen", "toggle", {}),
        ("switch.fan", "turn_on", {"percentage": 40}),
    ]


def test_mock_unknown_entity_returns_false_and_is_not_recorded() -> None:
    ha = MockHomeAssistant()
    assert run(ha.call_service("light.does_not_exist", "turn_on")) is False
    assert ha.calls == []


def test_mock_get_signals_kinds_values_and_units() -> None:
    ha = MockHomeAssistant()
    signals = signals_by_id(ha)

    expected_kinds = {
        "sensor.living_room_temperature": SignalKind.TEMPERATURE,
        "sensor.living_room_humidity": SignalKind.HUMIDITY,
        "sensor.living_room_illuminance": SignalKind.ILLUMINANCE,
        "binary_sensor.hallway_motion": SignalKind.MOTION,
        "binary_sensor.window_contact": SignalKind.CONTACT,
        "sensor.kitchen_power": SignalKind.POWER,
        "light.kitchen": SignalKind.OTHER,
        "switch.fan": SignalKind.OTHER,
    }
    for entity_id, kind in expected_kinds.items():
        assert signals[entity_id].kind is kind, entity_id

    assert signals["sensor.living_room_temperature"].value == pytest.approx(21.0)
    assert signals["sensor.living_room_temperature"].unit == "\u00b0C"
    assert signals["sensor.living_room_humidity"].unit == "%"
    assert signals["binary_sensor.hallway_motion"].value == 0.0
    assert signals["light.kitchen"].state == "off"
    assert all(isinstance(signal, Signal) for signal in signals.values())


def test_mock_scenario_drifts_temperature_and_humidity_over_time() -> None:
    ha = MockHomeAssistant()
    before = signals_by_id(ha)

    assert ha.t == 0.0
    ha.advance(60.0)
    after = signals_by_id(ha)

    assert ha.t == 60.0
    temp_before = before["sensor.living_room_temperature"].value
    temp_after = after["sensor.living_room_temperature"].value
    hum_before = before["sensor.living_room_humidity"].value
    hum_after = after["sensor.living_room_humidity"].value
    assert temp_after > temp_before + 0.1
    assert hum_after > hum_before + 0.1

    # Light feedback drives illuminance and power.
    assert run(ha.call_service("light.kitchen", "turn_on")) is True
    ha.advance(0.0)
    lit = signals_by_id(ha)
    assert lit["sensor.living_room_illuminance"].value > 100.0
    assert lit["sensor.kitchen_power"].value > 5.0


def test_mock_scenario_is_deterministic_for_the_same_seed() -> None:
    first = MockHomeAssistant(Scenario(seed=7))
    second = MockHomeAssistant(Scenario(seed=7))
    other = MockHomeAssistant(Scenario(seed=8))
    for _ in range(5):
        first.advance(10.0)
        second.advance(10.0)
        other.advance(10.0)
    assert first.states == second.states
    assert first.states != other.states


def test_mock_advance_rejects_negative_time() -> None:
    ha = MockHomeAssistant()
    with pytest.raises(ValueError):
        ha.advance(-1.0)


def test_mock_motion_and_contact_events_expire() -> None:
    ha = MockHomeAssistant()

    ha.inject_motion(30.0)
    ha.inject_contact(10.0)
    active = signals_by_id(ha)
    assert active["binary_sensor.hallway_motion"].value == 1.0
    assert active["binary_sensor.hallway_motion"].state == "on"
    assert active["binary_sensor.window_contact"].value == 1.0

    ha.advance(15.0)
    moving = signals_by_id(ha)
    assert moving["binary_sensor.hallway_motion"].state == "on"
    assert moving["binary_sensor.window_contact"].state == "off"

    ha.advance(20.0)
    quiet = signals_by_id(ha)
    assert quiet["binary_sensor.hallway_motion"].state == "off"
    assert quiet["binary_sensor.hallway_motion"].value == 0.0


def test_mock_set_state_controls_entities() -> None:
    ha = MockHomeAssistant()
    ha.set_state("sensor.living_room_temperature", "26.5")
    ha.set_state("binary_sensor.hallway_motion", "unavailable")

    signals = signals_by_id(ha)
    assert signals["sensor.living_room_temperature"].value == pytest.approx(26.5)
    assert signals["binary_sensor.hallway_motion"].attributes["unavailable"] is True
    assert signals["binary_sensor.hallway_motion"].value == 0.0


def test_mock_get_states_alias_and_aclose() -> None:
    ha = MockHomeAssistant()
    assert run(ha.get_states()) == run(ha.get_signals())
    assert run(ha.aclose()) is None


def test_mock_satisfies_protocol() -> None:
    assert isinstance(MockHomeAssistant(), HomeAssistant)
    assert isinstance(RestHomeAssistant(), HomeAssistant)


def test_classify_kind_and_parse_state_value_helpers() -> None:
    assert classify_kind("sensor.office_temp") is SignalKind.TEMPERATURE
    assert classify_kind("binary_sensor.front_door") is SignalKind.MOTION
    assert classify_kind("binary_sensor.window_contact") is SignalKind.CONTACT
    assert classify_kind("sensor.kitchen_power") is SignalKind.POWER
    assert classify_kind("light.kitchen") is SignalKind.OTHER
    assert parse_state_value("21.5") == pytest.approx(21.5)
    assert parse_state_value("detected") == 1.0
    assert parse_state_value("clear") == 0.0
    assert parse_state_value("unavailable") is None


# --------------------------------------------------------------------------- rest

STATES_PAYLOAD: list[dict[str, Any]] = [
    {
        "entity_id": "sensor.living_room_temperature",
        "state": "21.5",
        "attributes": {"unit_of_measurement": "\u00b0C", "friendly_name": "Living Room"},
        "last_updated": "2024-01-01T00:00:00+00:00",
    },
    {"entity_id": "binary_sensor.hallway_motion", "state": "detected", "attributes": {}},
    {
        "entity_id": "sensor.office_temperature",
        "state": "19.0",
        "attributes": {"unit_of_measurement": "\u00b0C"},
        "last_updated": "2024-01-01T00:00:00Z",
    },
    {
        "entity_id": "sensor.kitchen_power",
        "state": "unavailable",
        "attributes": {"unit_of_measurement": "W"},
    },
]


def make_rest(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> tuple[RestHomeAssistant, list[httpx.Request]]:
    """Build a REST adapter backed by an in-memory mock transport."""
    requests: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(recording_handler))
    ha = RestHomeAssistant(
        base_url="http://ha.test:8123", token="secret-token", client=client, **kwargs
    )
    return ha, requests


def test_rest_get_signals_url_auth_and_normalization() -> None:
    ha, requests = make_rest(lambda request: httpx.Response(200, json=STATES_PAYLOAD))
    try:
        signals = {signal.entity_id: signal for signal in run(ha.get_signals())}
    finally:
        run(ha.aclose())

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert str(request.url) == "http://ha.test:8123/api/states"
    assert request.headers["Authorization"] == "Bearer secret-token"

    temperature = signals["sensor.living_room_temperature"]
    assert temperature.value == pytest.approx(21.5)
    assert temperature.kind is SignalKind.TEMPERATURE
    assert temperature.unit == "\u00b0C"
    assert temperature.timestamp == pytest.approx(1704067200.0)

    assert signals["binary_sensor.hallway_motion"].value == 1.0
    assert signals["binary_sensor.hallway_motion"].kind is SignalKind.MOTION

    # HA emits UTC "Z" timestamps; they must parse on Python 3.11.
    assert signals["sensor.office_temperature"].timestamp == pytest.approx(1704067200.0)
    assert signals["sensor.office_temperature"].value == pytest.approx(19.0)

    unavailable = signals["sensor.kitchen_power"]
    assert unavailable.value == 0.0
    assert unavailable.attributes["unavailable"] is True
    assert unavailable.state == "unavailable"


def test_rest_get_signals_uses_configured_fallback() -> None:
    payload = [{"entity_id": "sensor.unknown", "state": "unknown", "attributes": {}}]
    ha, _ = make_rest(lambda request: httpx.Response(200, json=payload), fallback=-1.0)
    try:
        signals = run(ha.get_signals())
    finally:
        run(ha.aclose())
    assert signals[0].value == -1.0
    assert signals[0].attributes["unavailable"] is True


def test_rest_get_signals_http_500_returns_empty_without_raising() -> None:
    ha, requests = make_rest(lambda request: httpx.Response(500, json={"message": "boom"}))
    try:
        signals = run(ha.get_signals())
    finally:
        run(ha.aclose())
    assert signals == []
    assert len(requests) == 1  # HTTP status errors are not retried


def test_rest_call_service_posts_domain_and_merges_data() -> None:
    ha, requests = make_rest(lambda request: httpx.Response(200, json=[]))
    try:
        ok = run(ha.call_service("light.kitchen", "turn_on", {"brightness_pct": 80}))
    finally:
        run(ha.aclose())

    assert ok is True
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://ha.test:8123/api/services/light/turn_on"
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert json.loads(request.content) == {"entity_id": "light.kitchen", "brightness_pct": 80}


def test_rest_call_service_without_data_and_invalid_entity() -> None:
    ha, requests = make_rest(lambda request: httpx.Response(200, json=[]))
    try:
        assert run(ha.call_service("switch.fan", "toggle")) is True
        assert json.loads(requests[0].content) == {"entity_id": "switch.fan"}
        assert run(ha.call_service("not-an-entity", "turn_on")) is False
    finally:
        run(ha.aclose())
    assert len(requests) == 1


def test_rest_call_service_500_returns_false_without_raising() -> None:
    ha, requests = make_rest(lambda request: httpx.Response(503))
    try:
        assert run(ha.call_service("light.office", "turn_off")) is False
    finally:
        run(ha.aclose())
    assert len(requests) == 1


def test_rest_connection_error_retries_then_returns_false() -> None:
    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    ha, requests = make_rest(failing_handler)
    try:
        assert run(ha.get_signals()) == []
        assert run(ha.call_service("light.kitchen", "turn_on")) is False
    finally:
        run(ha.aclose())

    # Two attempts per call (default retries=2), two calls.
    assert len(requests) == 4


def test_rest_health() -> None:
    ha, requests = make_rest(lambda request: httpx.Response(200, json={"message": "API running."}))
    try:
        assert run(ha.health()) is True
    finally:
        run(ha.aclose())
    assert str(requests[0].url) == "http://ha.test:8123/api/"

    failing, _ = make_rest(lambda request: httpx.Response(401))
    try:
        assert run(failing.health()) is False
    finally:
        run(failing.aclose())


def test_rest_get_states_alias() -> None:
    ha, requests = make_rest(lambda request: httpx.Response(200, json=STATES_PAYLOAD))

    def normalized(items: list[Signal]) -> list[tuple[str, SignalKind, float, str]]:
        return [(item.entity_id, item.kind, item.value, item.state) for item in items]

    try:
        assert normalized(run(ha.get_states())) == normalized(run(ha.get_signals()))
    finally:
        run(ha.aclose())
    assert len(requests) == 2


# ------------------------------------------------------------------------ factory


def test_factory_defaults_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HA_MODE", raising=False)
    assert isinstance(make_home_assistant(), MockHomeAssistant)


def test_factory_respects_ha_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HA_MODE", "mock")
    assert isinstance(make_home_assistant(), MockHomeAssistant)

    monkeypatch.setenv("HA_MODE", "rest")
    monkeypatch.setenv("HA_BASE_URL", "http://ha.local:8123/")
    monkeypatch.setenv("HA_TOKEN", "env-token")
    ha = make_home_assistant()
    assert isinstance(ha, RestHomeAssistant)
    assert ha.base_url == "http://ha.local:8123"
    assert ha.token == "env-token"


def test_factory_explicit_mode_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HA_MODE", "rest")
    assert isinstance(make_home_assistant("mock"), MockHomeAssistant)


def test_factory_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        make_home_assistant("carrier-pigeon")
