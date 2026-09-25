"""Home Assistant adapters for the flybrain controller.

This module defines the frozen :class:`HomeAssistant` protocol plus two
implementations:

* :class:`MockHomeAssistant` -- a fully in-process simulator driven by a
  :class:`Scenario` over *simulated* time (no sockets, no real sleeping).
* :class:`RestHomeAssistant` -- a thin ``httpx`` client for a real (or mocked)
  Home Assistant instance.

Configuration comes from the environment: ``HA_MODE`` (``mock`` | ``rest``),
``HA_BASE_URL`` and ``HA_TOKEN``.  :func:`make_home_assistant` is the factory.

Shared dataclasses come from :mod:`flybrain.types` and are never redefined here.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Protocol, runtime_checkable

import httpx

from flybrain.types import Signal, SignalKind

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8123"
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_RETRIES = 2

#: Raw HA states that carry no usable numeric value.
UNAVAILABLE_STATES = frozenset({"unavailable", "unknown", "none", ""})

#: Mapping of common non-numeric HA states onto a 0.0/1.0 float.
BINARY_STATES: dict[str, float] = {
    "on": 1.0,
    "true": 1.0,
    "yes": 1.0,
    "open": 1.0,
    "opening": 1.0,
    "home": 1.0,
    "detected": 1.0,
    "active": 1.0,
    "online": 1.0,
    "wet": 1.0,
    "locked": 1.0,
    "playing": 1.0,
    "off": 0.0,
    "false": 0.0,
    "no": 0.0,
    "closed": 0.0,
    "closing": 0.0,
    "not_home": 0.0,
    "clear": 0.0,
    "idle": 0.0,
    "standby": 0.0,
    "offline": 0.0,
    "dry": 0.0,
    "unlocked": 0.0,
    "paused": 0.0,
}

#: Ordered keyword -> :class:`SignalKind` rules, matched against the entity id.
#: Order matters: the first match wins, so sound-derived names come before the generic
#: motion rules. A Tapo camera exposes ``..._bark_detection`` and ``..._glass_break_detection``
#: as *motion* device-class binary sensors, and classifying a meow as "motion" would wire an
#: acoustic event to the visual pathway.
_KIND_KEYWORDS: tuple[tuple[tuple[str, ...], SignalKind], ...] = (
    (("bark", "meow", "glass", "sound", "audio", "noise", "siren"), SignalKind.AUDIO),
    (("temperature", "temp"), SignalKind.TEMPERATURE),
    (("humidity", "humid"), SignalKind.HUMIDITY),
    (("illuminance", "lux", "light_level", "brightness"), SignalKind.ILLUMINANCE),
    (("motion", "movement", "occupancy", "presence", "door"), SignalKind.MOTION),
    (("contact", "window", "opening", "garage"), SignalKind.CONTACT),
    (("power", "watt", "energy", "current", "voltage"), SignalKind.POWER),
)


def classify_kind(entity_id: str) -> SignalKind:
    """Map a Home Assistant ``entity_id`` to a :class:`SignalKind`.

    Matching is case-insensitive and keyword based; anything unrecognized maps
    to :attr:`SignalKind.OTHER`.
    """
    needle = entity_id.lower()
    for keywords, kind in _KIND_KEYWORDS:
        if any(keyword in needle for keyword in keywords):
            return kind
    return SignalKind.OTHER


def parse_state_value(state: str) -> float | None:
    """Return the numeric value of an HA state string, or ``None`` if unknown.

    Plain numbers are parsed directly (``"21.5"`` -> ``21.5``); common
    non-numeric states are mapped through :data:`BINARY_STATES`
    (``"on"``/``"detected"`` -> ``1.0``, ``"off"``/``"clear"`` -> ``0.0``).
    ``None`` means the state carries no usable value (e.g. ``"unavailable"``).
    """
    text = state.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return BINARY_STATES.get(text.lower())


def state_to_float(state: str, fallback: float = 0.0) -> float:
    """Like :func:`parse_state_value` but substitutes ``fallback`` for unknowns."""
    value = parse_state_value(state)
    return fallback if value is None else value


@runtime_checkable
class HomeAssistant(Protocol):
    """Frozen interface every Home Assistant adapter must satisfy."""

    async def get_signals(self) -> list[Signal]:
        """Return the current entity states, normalized to :class:`Signal`."""
        ...

    async def get_states(self) -> list[Signal]:
        """Alias of :meth:`get_signals` (name used in ``CONTRACT.md``)."""
        ...

    async def call_service(
        self, entity_id: str, service: str, data: dict[str, Any] | None = None
    ) -> bool:
        """Call a Home Assistant service; return ``True`` on success."""
        ...

    async def aclose(self) -> None:
        """Release any resources held by the adapter."""
        ...


@dataclass
class Scenario:
    """Deterministic recipe for how mock sensors evolve over simulated time.

    All rates are expressed per *simulated* minute.  ``noise`` is the standard
    deviation of the seeded Gaussian noise added on each :meth:`advance`; set it
    to ``0.0`` for a perfectly smooth ramp.
    """

    temperature_start: float = 21.0
    temperature_ramp_per_min: float = 0.5
    #: When > 0, the temperature **oscillates** around ``temperature_start`` with this
    #: half-amplitude instead of ramping. A linear ramp of 0.5 degC/min takes 50 minutes to
    #: cross the useful range, which is useless for watching a control loop work; a swing
    #: makes the whole temperature -> colour chain visibly cycle within a few minutes.
    temperature_swing_c: float = 0.0
    #: Period of that oscillation, in seconds of simulated time.
    temperature_period_s: float = 180.0
    humidity_start: float = 45.0
    humidity_drift_per_min: float = 0.25
    illuminance_on: float = 320.0
    illuminance_off: float = 4.0
    power_on: float = 12.0
    power_idle: float = 0.5
    noise: float = 0.05
    seed: int = 1234
    realtime: bool = False


class MockHomeAssistant:
    """Fully in-process Home Assistant simulator (no sockets, no real time).

    Entities are held as raw HA state strings in :attr:`states`.  Simulated time
    is advanced explicitly with :meth:`advance`, which applies the configured
    :class:`Scenario`: a temperature ramp, humidity drift, light-driven
    illuminance/power feedback, and scheduled motion/contact events.  Noise is
    drawn from a seeded :class:`random.Random`, so runs are reproducible.
    """

    #: Entity id -> unit of measurement.
    UNITS: ClassVar[dict[str, str]] = {
        "sensor.living_room_temperature": "\u00b0C",
        "sensor.living_room_humidity": "%",
        "sensor.living_room_illuminance": "lx",
        "sensor.kitchen_power": "W",
    }

    def __init__(
        self,
        scenario: Scenario | None = None,
        seed: int | None = None,
        realtime: bool | None = None,
    ) -> None:
        """Build the simulator.

        Args:
            scenario: Evolution recipe; a default :class:`Scenario` is used when
                omitted.
            seed: Overrides ``scenario.seed`` for the noise RNG.
            realtime: Overrides ``scenario.realtime``; when ``True``,
                :meth:`advance` actually sleeps for ``dt_seconds``.
        """
        self.scenario = scenario if scenario is not None else Scenario()
        self.realtime = self.scenario.realtime if realtime is None else bool(realtime)
        self._rng = random.Random(self.scenario.seed if seed is None else seed)
        self.t: float = 0.0
        """Simulated time in seconds since construction."""
        self._t0: float = time.time()
        self._events: dict[str, list[tuple[float, float]]] = {}
        self.attributes: dict[str, dict[str, Any]] = {
            entity_id: {"friendly_name": entity_id.split(".", 1)[-1].replace("_", " ").title()}
            for entity_id in self.UNITS
        }
        for entity_id in (
            "binary_sensor.hallway_motion",
            "binary_sensor.window_contact",
            "light.kitchen",
            "light.office",
            "switch.fan",
        ):
            self.attributes.setdefault(entity_id, {})
        self.states: dict[str, str] = {
            "sensor.living_room_temperature": f"{self.scenario.temperature_start:.2f}",
            "sensor.living_room_humidity": f"{self.scenario.humidity_start:.2f}",
            "sensor.living_room_illuminance": f"{self.scenario.illuminance_off:.1f}",
            "sensor.kitchen_power": f"{self.scenario.power_idle:.2f}",
            "binary_sensor.hallway_motion": "off",
            "binary_sensor.window_contact": "off",
            "light.kitchen": "off",
            "light.office": "off",
            "switch.fan": "off",
        }
        for entity_id in self.states:
            self.attributes.setdefault(entity_id, {})
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        """Every accepted service call as ``(entity_id, service, data)``."""

    @property
    def entities(self) -> dict[str, str]:
        """Alias of :attr:`states` (mutable raw state strings)."""
        return self.states

    def entity_ids(self) -> list[str]:
        """Return the tracked entity ids in insertion order."""
        return list(self.states)

    # ------------------------------------------------------------------ control
    def set_state(self, entity_id: str, state: str) -> None:
        """Force an entity state, creating the entity if it is unknown.

        Intended for test control; the scenario may overwrite sensor entities on
        the next :meth:`advance`.
        """
        self.states[entity_id] = str(state)
        self.attributes.setdefault(entity_id, {})

    def schedule_event(self, entity_id: str, start_s: float, end_s: float) -> None:
        """Schedule a binary sensor to read ``on`` during ``[start_s, end_s)``."""
        self._events.setdefault(entity_id, []).append((float(start_s), float(end_s)))

    def inject_event(self, entity_id: str, duration_s: float = 30.0) -> None:
        """Turn a binary sensor on now, for ``duration_s`` simulated seconds."""
        self._events.setdefault(entity_id, []).append((self.t, self.t + float(duration_s)))
        self.states[entity_id] = "on"

    def inject_motion(self, duration_s: float = 30.0) -> None:
        """Inject a hallway motion event starting at the current simulated time."""
        self.inject_event("binary_sensor.hallway_motion", duration_s)

    def inject_contact(self, duration_s: float = 30.0) -> None:
        """Inject a window-contact event starting at the current simulated time."""
        self.inject_event("binary_sensor.window_contact", duration_s)

    def advance(self, dt_seconds: float) -> None:
        """Advance simulated time by ``dt_seconds`` and evolve every sensor.

        Deterministic for a given seed.  Never touches the wall clock unless
        ``realtime`` was explicitly enabled.
        """
        dt = float(dt_seconds)
        if dt < 0:
            raise ValueError("dt_seconds must be non-negative")
        if self.realtime and dt > 0:
            time.sleep(dt)
        self.t += dt
        scenario = self.scenario
        noise = scenario.noise

        if scenario.temperature_swing_c > 0.0:
            period = max(float(scenario.temperature_period_s), 1e-6)
            temperature = scenario.temperature_start + scenario.temperature_swing_c * math.sin(
                2.0 * math.pi * self.t / period
            )
        else:
            temperature = (
                scenario.temperature_start + scenario.temperature_ramp_per_min * self.t / 60.0
            )
        humidity = scenario.humidity_start + scenario.humidity_drift_per_min * self.t / 60.0
        if noise > 0:
            temperature += self._rng.gauss(0.0, noise)
            humidity += self._rng.gauss(0.0, noise)
        humidity = min(100.0, max(0.0, humidity))

        self.states["sensor.living_room_temperature"] = f"{temperature:.2f}"
        self.states["sensor.living_room_humidity"] = f"{humidity:.2f}"

        for entity_id, windows in self._events.items():
            active = any(start <= self.t < end for start, end in windows)
            self.states[entity_id] = "on" if active else "off"

        kitchen_on = self.states.get("light.kitchen") == "on"
        illuminance = scenario.illuminance_on if kitchen_on else scenario.illuminance_off
        power = scenario.power_on if kitchen_on else scenario.power_idle
        if noise > 0:
            illuminance += self._rng.gauss(0.0, noise * 5.0)
            power += self._rng.gauss(0.0, noise * 0.1)
        self.states["sensor.living_room_illuminance"] = f"{max(0.0, illuminance):.1f}"
        self.states["sensor.kitchen_power"] = f"{max(0.0, power):.2f}"

    # --------------------------------------------------------------- protocol
    async def get_signals(self) -> list[Signal]:
        """Return one :class:`Signal` per tracked entity."""
        timestamp = self._t0 + self.t
        signals: list[Signal] = []
        for entity_id, state in self.states.items():
            attributes = dict(self.attributes.get(entity_id, {}))
            value = state_to_float(state, 0.0)
            if state.strip().lower() in UNAVAILABLE_STATES:
                attributes["unavailable"] = True
            signals.append(
                Signal(
                    entity_id=entity_id,
                    kind=classify_kind(entity_id),
                    value=value,
                    state=state,
                    timestamp=timestamp,
                    unit=self.UNITS.get(entity_id, ""),
                    attributes=attributes,
                )
            )
        return signals

    async def get_states(self) -> list[Signal]:
        """Alias of :meth:`get_signals` (``CONTRACT.md`` naming)."""
        return await self.get_signals()

    async def call_service(
        self, entity_id: str, service: str, data: dict[str, Any] | None = None
    ) -> bool:
        """Apply a service call to an actuator and record it in :attr:`calls`.

        Returns ``False`` for unknown entities; no exception is raised.
        """
        if entity_id not in self.states:
            logger.warning("MockHomeAssistant: unknown entity %r", entity_id)
            return False
        payload = dict(data) if data else {}
        domain = entity_id.split(".", 1)[0]
        if domain in {"light", "switch", "input_boolean"}:
            current = self.states[entity_id]
            if service == "turn_on":
                self.states[entity_id] = "on"
            elif service == "turn_off":
                self.states[entity_id] = "off"
            elif service == "toggle":
                self.states[entity_id] = "off" if current == "on" else "on"
            if "brightness_pct" in payload:
                self.attributes.setdefault(entity_id, {})["brightness_pct"] = payload[
                    "brightness_pct"
                ]
        self.calls.append((entity_id, service, payload))
        return True

    async def aclose(self) -> None:
        """No-op: the mock owns no resources."""
        return


class RestHomeAssistant:
    """Home Assistant REST adapter built on :class:`httpx.AsyncClient`.

    Failures (HTTP status errors, connection errors, malformed JSON) are logged
    and degrade to ``False`` / an empty signal list instead of raising.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        *,
        fallback: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Build the adapter.

        Args:
            base_url: HA root URL; defaults to ``$HA_BASE_URL`` or
                ``http://127.0.0.1:8123``.
            token: Long-lived access token; defaults to ``$HA_TOKEN``.
            fallback: Value used for ``unavailable``/``unknown`` states.
            timeout: Per-request timeout in seconds.
            retries: Total attempts for connection-level failures.
            client: Pre-built ``httpx.AsyncClient`` (e.g. carrying a
                ``MockTransport`` in tests).  When omitted, one is created
                lazily and closed by :meth:`aclose`.
        """
        resolved_url = base_url or os.environ.get("HA_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = resolved_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("HA_TOKEN", "")
        self.fallback = float(fallback)
        self.timeout = float(timeout)
        self.retries = max(1, int(retries))
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the underlying HTTP client, creating it on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        """Auth + content headers for every HA request."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response | None:
        """Perform a request, retrying connection errors and never raising.

        HTTP status errors fail immediately (no retry); transport errors are
        retried up to :attr:`retries` attempts.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = await self._get_client().request(
                    method, url, headers=self._headers(), timeout=self.timeout, **kwargs
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Home Assistant %s %s returned HTTP %s", method, url, exc.response.status_code
                )
                return None
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning(
                    "Home Assistant %s %s connection error (attempt %d/%d): %s",
                    method,
                    url,
                    attempt,
                    self.retries,
                    exc,
                )
        if last_error is not None:
            logger.error(
                "Home Assistant %s %s failed after %d attempts: %s",
                method,
                url,
                self.retries,
                last_error,
            )
        return None

    def _parse_state(self, item: dict[str, Any]) -> Signal:
        """Normalize one ``/api/states`` entry into a :class:`Signal`."""
        entity_id = str(item.get("entity_id", ""))
        raw_state = item.get("state")
        state = "" if raw_state is None else str(raw_state)
        attributes = dict(item.get("attributes") or {})
        unit = str(attributes.get("unit_of_measurement", ""))

        value = parse_state_value(state)
        if state.strip().lower() in UNAVAILABLE_STATES or value is None:
            attributes["unavailable"] = True
            value = self.fallback

        return Signal(
            entity_id=entity_id,
            kind=classify_kind(entity_id),
            value=value,
            state=state,
            timestamp=_parse_timestamp(item),
            unit=unit,
            attributes=attributes,
        )

    async def get_signals(self) -> list[Signal]:
        """Fetch ``/api/states`` and normalize every entity; ``[]`` on failure."""
        response = await self._request("GET", f"{self.base_url}/api/states")
        if response is None:
            return []
        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("Home Assistant /api/states returned invalid JSON: %s", exc)
            return []
        if not isinstance(payload, list):
            logger.error("Home Assistant /api/states returned %s, expected a list", type(payload))
            return []
        return [self._parse_state(item) for item in payload if isinstance(item, dict)]

    async def get_states(self) -> list[Signal]:
        """Alias of :meth:`get_signals` (``CONTRACT.md`` naming)."""
        return await self.get_signals()

    async def call_service(
        self, entity_id: str, service: str, data: dict[str, Any] | None = None
    ) -> bool:
        """POST ``/api/services/<domain>/<service>``; ``False`` on any failure."""
        domain, separator, _ = entity_id.partition(".")
        if not separator or not domain:
            logger.error("Home Assistant call_service: invalid entity_id %r", entity_id)
            return False
        url = f"{self.base_url}/api/services/{domain}/{service}"
        body: dict[str, Any] = {"entity_id": entity_id}
        if data:
            body.update(data)
        response = await self._request("POST", url, json=body)
        return response is not None

    async def health(self) -> bool:
        """Return ``True`` when ``GET /api/`` succeeds."""
        response = await self._request("GET", f"{self.base_url}/api/")
        return response is not None

    async def aclose(self) -> None:
        """Close the HTTP client if this adapter created it."""
        if self._client is not None:
            await self._client.aclose()
            if self._owns_client:
                self._client = None


def _parse_timestamp(item: dict[str, Any]) -> float:
    """Parse ``last_updated``/``last_changed`` into unix seconds."""
    raw = item.get("last_updated") or item.get("last_changed")
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            logger.debug("Home Assistant: unparseable timestamp %r", raw)
    return time.time()


def make_home_assistant(mode: str | None = None) -> HomeAssistant:
    """Build the adapter selected by ``mode`` or ``$HA_MODE`` (default ``mock``).

    Raises:
        ValueError: if the resolved mode is neither ``mock`` nor ``rest``.
    """
    resolved = (mode or os.environ.get("HA_MODE") or "mock").strip().lower()
    if resolved == "mock":
        return MockHomeAssistant()
    if resolved == "rest":
        return RestHomeAssistant()
    raise ValueError(f"unknown HA_MODE {resolved!r}; expected 'mock' or 'rest'")
