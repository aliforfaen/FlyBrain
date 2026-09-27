"""The live control loop: a Home Assistant temperature sensor drives a light's colour.

    HA sensor (°C) ──► rate-coded drive ──► frozen FlyWire v783 ──► spike rates on the
    readout population ──► learned ridge readout ──► colour temperature (K) ──► HA action

Why this is a separate module from ``experiment.py``: that one *trains* the readout by
simulating windows offline. This one *runs* it, against a live sensor, on a brain that is
already turning.

The loop deliberately owns no simulator. It is handed the same
:class:`~flybrain.sim.ConnectomeSim` that the dashboard is already stepping, so the picture
on screen and the decision being made come from the same running brain, and there is only
ever one copy of a 15-million-synapse network in memory. The cost of that choice is that the
readout must have been *trained* for a continuously-running brain rather than for windows
that start at rest — see ``TemperatureColourLoop.sweep``.

Safety: the loop will never touch a real device unless ``dry_run`` is explicitly turned off.
In ``mock`` mode the Home Assistant adapter is itself a simulation, so the call is always
made and the simulated light genuinely changes state.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flybrain.codec import ChannelSpec
from flybrain.env import load_dotenv
from flybrain.experiment import (
    ARTIFACT_DIR,
    COLOUR_BANDS,
    ColourReadout,
    ExperimentConfig,
    drive_current_for_rate,
    rate_for_temperature,
)
from flybrain.mapping import RoleResolver
from flybrain.pacing import Pacer
from flybrain.types import Action, Signal, coerce_patch, signal_is_dead

logger = logging.getLogger(__name__)

READOUT_FILE = "colour_readout.npz"
META_FILE = "colour_meta.json"


class MissingReadout(RuntimeError):
    """Raised when the trained colour readout has not been built yet."""


@dataclass
class LoopConfig:
    """Connection settings for the live loop.

    These are the knobs that adapt one trained brain to a *particular* home. The readout was
    fitted on a 10-35 degC sweep, but a real room moves by two or three degrees, so the
    sensor range has to be stretched onto the trained range or the light would barely
    change at all. Everything here is settable at runtime from the dashboard.
    """

    temperature_entity: str = "sensor.living_room_temperature"
    light_entity: str = "light.kitchen"
    #: ``mock`` or ``rest``; resolved from ``$HA_MODE`` by the server when omitted.
    mode: str = "mock"
    #: When True, a real Home Assistant is never called - the loop only records what it
    #: *would* do. Mock mode ignores this, because the mock is not a real device.
    dry_run: bool = True
    #: How many decisions to keep for the UI's temperature/colour history.
    history: int = 240

    # --- sensitivity -------------------------------------------------------
    #: The span of *real* sensor readings that should use the whole trained range.
    #: Defaults to the trained range itself, i.e. "no stretching". Set it to something like
    #: 18-24 degC and a two-degree room swing drives the entire colour range.
    source_min_c: float = 10.0
    source_max_c: float = 35.0
    #: Flip the direction, for a sensor that reads high when you want the light cool: an
    #: outdoor probe, a south-facing room, an air-conditioning return.
    invert: bool = False
    #: Milliseconds of exponential smoothing applied to the sensor reading. Real sensors
    #: jitter; without this the light would twitch on every stray decimal. 0 disables it.
    smooth_ms: float = 5000.0

    # --- what actually reaches the light -----------------------------------
    #: Keep the emitted colour inside these bounds. Both must sit inside the trained range,
    #: because the readout cannot honestly produce anything outside it - so they default to
    #: that range rather than to a hardcoded pair that would silently clip it.
    kelvin_min: float = COLOUR_BANDS[0][1]
    kelvin_max: float = COLOUR_BANDS[-1][1]
    #: Send nothing until the colour has moved at least this many Kelvin. The dashboard
    #: still shows every decision - this only gates the service call, so a real light is not
    #: spammed with changes nobody can see. 0 sends on every decision.
    deadband_k: float = 0.0

    #: Maximum wall-clock seconds between decisions - the **heartbeat**. 0 means "as fast as the
    #: GPU allows", which is what the demo wants; the default is paced because nothing in a room
    #: changes meaningfully in two seconds, and flat out holds an RTX 3070 at ~165 W continuously
    #: (~4 kWh/day) whether or not anything is watching. Measured: a decision every 15 s averages
    #: ~40 W. The pacing model and its limits are in ``docs/engine.md``.
    #:
    #: This is the *maximum* gap in every pacing mode rather than one of two competing knobs.
    #: Below, a change trigger can make decisions happen sooner; nothing can make them later.
    interval_s: float = 15.0

    #: Wall-clock seconds between sensor reads while the loop is waiting for the heartbeat.
    #: Reading Home Assistant costs no GPU time, so this is close to free - but it is not free:
    #: at 5 s it is ~17k requests a day, which a local instance will not notice. 0 disables
    #: polling, which also disables the change trigger, since nothing would be re-read to notice
    #: a change.
    poll_s: float = 5.0

    #: How long to keep running at full rate once the house does something interesting. The event
    #: is captured in detail rather than sampled once. 0 means "one extra decision, no burst".
    burst_s: float = 10.0

    #: How far the primary sensor must move, in its own units (degrees Celsius for the default
    #: temperature wiring) before that counts as an event worth bursting for. 0 disables the
    #: change trigger entirely, leaving a fixed heartbeat.
    #:
    #: The trigger is a comparison, not a model - deliberately. It is also the cheapest part of
    #: the scheme and the part most likely to be replaced: the intended long-term trigger is the
    #: reservoir's own prediction error (roadmap A3), which needs no threshold in degrees.
    trigger_delta: float = 0.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LoopConfig:
        """Build a config from environment variables.

        Entity ids have to be configurable for a real installation. The field defaults name the
        *mock* home, so pointing at a real house would otherwise mean editing source, or posting
        to the settings endpoint again after every restart. Precedence is environment -> defaults,
        and the dashboard can still override anything at runtime.

        ``.env`` is loaded only when the caller did not supply an environment of its own, so
        importing the loop picks up a real house without anyone having to `source` first, while
        tests that pass an explicit mapping stay hermetic.
        """
        if env is None:
            load_dotenv()
        e = os.environ if env is None else env

        def flag(name: str, default: bool) -> bool:
            raw = e.get(name)
            if raw is None:
                return default
            return raw.strip().lower() not in {"0", "false", "no", "off", ""}

        def number(name: str, default: float) -> float:
            raw = e.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                return float(raw)
            except ValueError:
                # A typo in a unit file must not take the loop down; the default is safe.
                logger.warning("ignoring non-numeric %s=%r", name, raw)
                return default

        return cls(
            temperature_entity=e.get("FLYBRAIN_TEMPERATURE_ENTITY", cls.temperature_entity),
            light_entity=e.get("FLYBRAIN_LIGHT_ENTITY", cls.light_entity),
            mode=(e.get("HA_MODE") or "mock").strip().lower(),
            dry_run=flag("HA_DRY_RUN", True),
            source_min_c=number("FLYBRAIN_SOURCE_MIN_C", cls.source_min_c),
            source_max_c=number("FLYBRAIN_SOURCE_MAX_C", cls.source_max_c),
            invert=flag("FLYBRAIN_INVERT", cls.invert),
            smooth_ms=number("FLYBRAIN_SMOOTH_MS", cls.smooth_ms),
            kelvin_min=number("FLYBRAIN_KELVIN_MIN", cls.kelvin_min),
            kelvin_max=number("FLYBRAIN_KELVIN_MAX", cls.kelvin_max),
            deadband_k=number("FLYBRAIN_DEADBAND_K", cls.deadband_k),
            interval_s=number("FLYBRAIN_INTERVAL_S", cls.interval_s),
            poll_s=number("FLYBRAIN_POLL_S", cls.poll_s),
            burst_s=number("FLYBRAIN_BURST_S", cls.burst_s),
            trigger_delta=number("FLYBRAIN_TRIGGER_DELTA", cls.trigger_delta),
        )

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"

    @property
    def will_send(self) -> bool:
        """Whether a service call is actually dispatched."""
        return self.is_mock or not self.dry_run

    @property
    def source_span_c(self) -> float:
        return max(float(self.source_max_c) - float(self.source_min_c), 1e-9)

    def to_dict(self) -> dict:
        return {
            "temperature_entity": self.temperature_entity,
            "light_entity": self.light_entity,
            "mode": self.mode,
            "dry_run": bool(self.dry_run),
            "source_min_c": self.source_min_c,
            "source_max_c": self.source_max_c,
            "invert": bool(self.invert),
            "smooth_ms": self.smooth_ms,
            "kelvin_min": self.kelvin_min,
            "kelvin_max": self.kelvin_max,
            "deadband_k": self.deadband_k,
            "interval_s": self.interval_s,
            "poll_s": self.poll_s,
            "burst_s": self.burst_s,
            "trigger_delta": self.trigger_delta,
        }

    def apply(self, patch: dict) -> None:
        """Apply a settings patch, coercing each value to its field's type.

        The patch is coerced *as a whole* before anything is assigned, so one bad value leaves
        the settings exactly as they were rather than half-updated. This matters here more than
        anywhere: ``dry_run`` lives in this dataclass, and a patch that half-applied could
        change it while the caller is told the update failed.
        """
        allowed = {f for f in self.__dataclass_fields__ if f != "history"}
        for key, value in coerce_patch(self, patch, allowed=allowed).items():
            setattr(self, key, value)


class LiveLoop:
    """Reads a temperature, decodes a colour from the running brain, emits an action."""

    def __init__(
        self,
        sim,
        config: ExperimentConfig,
        readout: ColourReadout,
        input_indices: np.ndarray,
        readout_indices: np.ndarray,
        loop_config: LoopConfig | None = None,
        ha=None,
    ) -> None:
        self.sim = sim
        self.config = config
        self.readout = readout
        self.input_indices = np.asarray(input_indices, dtype=np.int64)
        self.readout_indices = np.asarray(readout_indices, dtype=np.int64)
        self.loop = loop_config or LoopConfig()
        self.ha = ha

        self.band_centres = np.array([k for _, k in COLOUR_BANDS], dtype=np.float64)
        self.band_names = [name for name, _ in COLOUR_BANDS]

        self.decisions = 0
        self.temperature_c: float | None = None
        self.brain_temperature_c: float | None = None
        self.kelvin: float | None = None
        self.last_action: dict | None = None
        self.last_error: str | None = None
        self.history: list[dict] = []
        self._last_rate_hz: float | None = None
        self._smoothed_c: float | None = None
        self._active_source_c: float | None = None
        self._active_brain_c: float | None = None
        #: Smallest and largest raw readings seen, so the dashboard can offer to fit the
        #: sensitivity range to the room's actual behaviour rather than guesswork.
        self.observed_min_c: float | None = None
        self.observed_max_c: float | None = None
        self._last_sent_kelvin: float | None = None
        #: True when the configured temperature entity had no usable reading in the most
        #: recent window — offline, unknown, absent, or a state that is not a number. While
        #: this is set :meth:`decide` refuses to act, because the colour readout has no input
        #: and any output would be invented.
        self.reading_stale: bool = False
        #: Wall-clock time of the last usable temperature reading, so the dashboard can show
        #: the *age* of what the loop is acting on rather than an unexplained frozen number.
        self.last_good_reading_at: float | None = None
        #: Entity states captured by the most recent sensor read. Kept so the recorder can store
        #: what the house was doing without a second round-trip to Home Assistant.
        self.last_signals: list = []
        #: Additional sensory channels beyond the trained temperature path (see
        #: :meth:`configure_channels`), and the rates last applied to each.
        self.channels: list[ChannelSpec] = []
        self.channel_rates: dict[str, float] = {}
        #: Decides when the brain is worth stepping. Built from the config and rebuilt whenever
        #: the settings change, so a dashboard edit takes effect without a restart.
        self.pacer = self._build_pacer()
        # Keep the configured limits inside what the readout can honestly produce.
        self.loop.kelvin_min = max(float(self.loop.kelvin_min), float(self.band_centres[0]))
        self.loop.kelvin_max = min(float(self.loop.kelvin_max), float(self.band_centres[-1]))

    def _build_pacer(self) -> Pacer:
        """Create a pacer from the current config.

        The pacer owns the cadence and the trigger, and nothing else. Keeping it constructed here
        rather than in ``server.py`` means the settings that define the training regime travel
        with the loop that runs it - which is what makes recording them in ``meta.json`` a
        one-line consequence rather than a bookkeeping chore.
        """
        return Pacer(
            heartbeat_s=self.loop.interval_s,
            poll_s=self.loop.poll_s,
            burst_s=self.loop.burst_s,
            trigger_delta=self.loop.trigger_delta,
            primary_entity=self.loop.temperature_entity,
        )

    # ------------------------------------------------------------ construction

    @classmethod
    def from_artifact(
        cls,
        sim,
        directory: str | Path | None = None,
        loop_config: LoopConfig | None = None,
        ha=None,
    ) -> LiveLoop:
        """Rebuild a loop from the trained readout written by ``flybrain.experiment``.

        Every neuron pool is reconstructed from the recorded seed, not re-drawn, because
        the learned weights are tied to *those specific* neurons.
        """
        directory = Path(directory or ARTIFACT_DIR)
        readout_path = directory / READOUT_FILE
        meta_path = directory / META_FILE
        if not readout_path.exists() or not meta_path.exists():
            raise MissingReadout(
                f"no trained readout in {directory}. Build one first:\n"
                f"    .venv/bin/python -m flybrain.experiment"
            )
        meta = json.loads(meta_path.read_text())
        loop_cfg = loop_config or LoopConfig()
        cfg = ExperimentConfig(
            temperature_entity=loop_cfg.temperature_entity,
            light_entity=loop_cfg.light_entity,
            temp_min_c=float(meta["temp_range_c"][0]),
            temp_max_c=float(meta["temp_range_c"][1]),
            input_roles=list(meta["input_roles"]),
            input_pool_size=int(meta["input_pool"]),
            readout_roles=list(meta["readout_roles"]),
            readout_pool_size=int(meta["readout_pool"]),
            window_ms=float(meta["window_ms"]),
            current_per_spike_mv=float(meta["current_per_spike_mv"]),
            min_rate_hz=float(meta["sensor_rate_hz"][0]),
            max_rate_hz=float(meta["sensor_rate_hz"][1]),
            ridge_l2=float(meta.get("ridge_l2", 10.0)),
            seed=int(meta.get("seed", 0)),
        )

        roles = RoleResolver.from_sim(sim)
        input_indices = roles.pool(cfg.input_roles, cfg.input_pool_size, seed=cfg.seed)
        readout_indices = roles.pool(cfg.readout_roles, cfg.readout_pool_size, seed=cfg.seed + 1)
        readout = ColourReadout(n_features=int(readout_indices.size) + 1, l2=cfg.ridge_l2)
        readout.load(readout_path)

        if readout.n_features != readout_indices.size + 1:
            raise MissingReadout(
                f"readout expects {readout.n_features} features but the recorded pools "
                f"give {readout_indices.size + 1}; the artifact and its metadata disagree"
            )

        logger.info(
            "live loop ready: drives %d neurons from %s, reads %d neurons in %.0f ms windows",
            input_indices.size,
            cfg.input_roles,
            readout_indices.size,
            cfg.window_ms,
        )
        return cls(sim, cfg, readout, input_indices, readout_indices, loop_cfg, ha)

    # ------------------------------------------------------------------ drive

    def brain_temperature(self, celsius: float) -> float:
        """Map a real sensor reading onto the range the readout was trained on.

        This is the sensitivity control, and it is the difference between a working light
        and a useless one in a real home: the readout knows about 10-35 degC, a living room
        moves between maybe 19 and 23. Without stretching the sensor span onto the trained
        span, a two-degree swing would barely move the colour.
        """
        lo = float(self.loop.source_min_c)
        u = (float(celsius) - lo) / self.loop.source_span_c
        u = float(np.clip(u, 0.0, 1.0))
        if self.loop.invert:
            u = 1.0 - u
        return float(
            self.config.temp_min_c + u * (self.config.temp_max_c - self.config.temp_min_c)
        )

    def smooth(self, celsius: float) -> float:
        """Exponential smoothing on the raw reading, using one window as the timestep."""
        tau_ms = float(self.loop.smooth_ms)
        if tau_ms <= 0.0:
            return float(celsius)
        if self._smoothed_c is None:
            self._smoothed_c = float(celsius)
            return self._smoothed_c
        dt_ms = float(self.config.window_ms)
        a = math.exp(-dt_ms / tau_ms)
        self._smoothed_c = a * self._smoothed_c + (1.0 - a) * float(celsius)
        return self._smoothed_c

    def temperature_drive(self, celsius: float) -> tuple[np.ndarray, float, float]:
        """Compute the trained temperature channel, without touching the simulator.

        Returns ``(indices, current_mv, rate_hz)``. Split out from :meth:`drive_temperature` so
        that other channels can be applied in the *same* ``set_drive`` call: ``set_drive``
        replaces the whole drive map, so two calls would silently cancel the first.
        """
        raw = float(celsius)
        if self.observed_min_c is None or raw < self.observed_min_c:
            self.observed_min_c = raw
        if self.observed_max_c is None or raw > self.observed_max_c:
            self.observed_max_c = raw

        smoothed = self.smooth(raw)
        mapped = self.brain_temperature(smoothed)
        rate = rate_for_temperature(mapped, self.config)
        current = drive_current_for_rate(rate, self.sim.params.dt_ms, self.config)
        self._last_rate_hz = rate
        self._active_source_c = smoothed
        self._active_brain_c = mapped
        # Set here rather than in `drive_channels` because this is the only place a *value* is
        # actually applied, and it is reached by both entry points. The dead-state check lives
        # in `drive_channels`, which is the only caller that can see the sensor's state.
        self.last_good_reading_at = time.time()
        return self.input_indices, current, rate

    def drive_temperature(self, celsius: float) -> None:
        """Encode a temperature and apply it as a persistent drive to the sensory pool.

        The raw reading is smoothed, then mapped onto the trained range, then encoded. The
        values that actually drove this window are remembered so the dashboard reports what
        the brain was given rather than a later, different reading.
        """
        indices, current, _ = self.temperature_drive(celsius)
        self.sim.set_drive(indices, current)

    # ------------------------------------------------------- extra sensory channels

    def configure_channels(self, channels: Sequence[ChannelSpec]) -> None:
        """Attach additional sensory channels, each driving its own neuron pool.

        The configured temperature entity is excluded on purpose: it keeps its *trained* path
        (sensitivity stretch, smoothing, the exact rate curve the readout was fitted against).
        Routing it through the generic channel path as well would drive the same neurons twice.
        """
        temperature_entity = self.loop.temperature_entity
        self.channels = [c for c in channels if c.entity_id != temperature_entity]

    def channel_rate(self, channel: ChannelSpec, value: float) -> float:
        """Firing rate for one non-temperature channel.

        Uses the *same* rate band as the temperature path (``min_rate_hz``..``max_rate_hz``)
        rather than an arbitrary scale, so every sensory channel drives the brain with
        comparable weight. A channel reading its minimum still fires at ``min_rate_hz``: like
        the thermosensory path, the floor is not zero, so the readout always carries signal.
        """
        span = max(float(channel.vmax) - float(channel.vmin), 1e-9)
        u = (float(value) - float(channel.vmin)) / span
        if channel.invert:
            u = 1.0 - u
        u = float(np.clip(u, 0.0, 1.0))
        lo, hi = float(self.config.min_rate_hz), float(self.config.max_rate_hz)
        return (lo + u * (hi - lo)) * float(channel.gain)

    def drive_channels(self, signals: Sequence[Signal]) -> dict[str, float]:
        """Drive the temperature channel plus every configured channel, in one call.

        Sensors that are not reporting are skipped rather than driven at their floor: an
        ``unavailable`` entity is not a sensor reading zero, and silently encoding it as one
        would teach the brain something false.

        When **nothing** is readable — the usual cause being a failed ``/api/states`` fetch,
        which returns an empty list — the drive is *cleared* rather than left as it was.
        Leaving it would let the simulator keep running on the previous window's input while
        :meth:`decide` decoded a colour from it: a decision caused by a reading nobody
        supplied.

        Sets :attr:`reading_stale`, which :meth:`decide` uses to suppress the action.

        Returns the per-channel rates actually applied, for the dashboard.
        """
        by_entity = {s.entity_id: s for s in signals}
        indices: list[np.ndarray] = []
        currents: list[np.ndarray] = []
        rates: dict[str, float] = {}

        temperature = by_entity.get(self.loop.temperature_entity)
        # The primary sensor now gets the same dead-state check the extra channels always had.
        # An ``unavailable`` thermometer arrives as its fallback *value*, which is a perfectly
        # plausible-looking temperature — only the state distinguishes the two.
        if not signal_is_dead(temperature):
            idx, current, rate = self.temperature_drive(float(temperature.value))
            indices.append(idx)
            currents.append(np.broadcast_to(np.float32(current), idx.shape))
            rates[self.loop.temperature_entity] = rate
            self.reading_stale = False
        else:
            self.reading_stale = True

        for channel in self.channels:
            signal = by_entity.get(channel.entity_id)
            if signal_is_dead(signal):
                continue
            rate = self.channel_rate(channel, signal.value)
            current = drive_current_for_rate(rate, self.sim.params.dt_ms, self.config)
            idx = channel.neuron_indices
            if idx.size == 0:
                continue
            indices.append(idx)
            currents.append(np.broadcast_to(np.float32(current), idx.shape))
            rates[channel.entity_id] = rate

        if indices:
            self.sim.set_drive(
                np.concatenate(indices),
                np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in currents]),
            )
        else:
            # Nothing readable at all. Clear rather than hold, so the brain falls quiet.
            self.clear_drive()
        self.channel_rates = rates
        return rates

    def clear_drive(self) -> None:
        self.sim.clear_drive()
        self._last_rate_hz = None
        self.channel_rates = {}

    # ----------------------------------------------------------------- decode

    def features(self, window_counts: np.ndarray, window_ms: float) -> np.ndarray:
        """Per-neuron firing rates on the readout population: the readout's input vector.

        This is the single definition of "what the readout sees", shared by :meth:`decode` and by
        anything recording a window for later training. Duplicating the expression would let the
        recorded data drift out of step with the live decode, which is the kind of bug that only
        shows up as a readout that mysteriously fails to learn.
        """
        counts = np.asarray(window_counts)
        return counts[self.readout_indices].astype(np.float64) / max(window_ms / 1000.0, 1e-9)

    def decode(self, window_counts: np.ndarray, window_ms: float) -> float:
        """Turn one window of per-neuron spike counts into a colour temperature."""
        rates = self.features(window_counts, window_ms)
        lo, hi = float(self.loop.kelvin_min), float(self.loop.kelvin_max)
        if hi <= lo:                       # a bad patch must not invert the output
            lo, hi = float(self.band_centres[0]), float(self.band_centres[-1])
        return float(np.clip(self.readout.predict(rates)[0], lo, hi))

    def sensor_snapshot(self) -> dict[str, float]:
        """Every entity value from the last sensor read, for recording."""
        return {sig.entity_id: float(sig.value) for sig in self.last_signals}

    def band_for(self, kelvin: float) -> str:
        """Nearest band label, for display only - the output itself is continuous."""
        return self.band_names[int(np.argmin(np.abs(self.band_centres - kelvin)))]

    # ------------------------------------------------------------- one decision

    async def read_signals(self, *, store: bool = True) -> list[Signal]:
        """Fetch every entity state.

        ``store`` controls whether this read becomes :attr:`last_signals`, which is the snapshot
        the recorder writes beside a window. The pacing poll passes ``store=False``: it happens
        *between* decisions, so letting it overwrite the snapshot would stamp a later reading
        onto a window that an earlier reading actually drove. That is the same class of silent
        misalignment as the recorder's crashed-row bug, and it would be just as invisible - the
        row count would still be right and only the pairing would be wrong.
        """
        if self.ha is None:
            return []
        signals: list[Signal] = await self.ha.get_signals()
        if store:
            self.last_signals = signals
        return signals

    async def read_temperature(self) -> float | None:
        """Read the configured sensor entity, or ``None`` if it is missing."""
        signals = await self.read_signals()
        for sig in signals:
            if sig.entity_id == self.loop.temperature_entity:
                return float(sig.value)
        logger.warning("sensor %r not found in Home Assistant", self.loop.temperature_entity)
        return None

    async def decide(self, window_counts: np.ndarray, window_ms: float) -> dict | None:
        """Complete one control cycle: decode, act, record.

        The window being decoded must already have been driven by :meth:`drive_temperature`;
        the reading reported here is the one that actually drove it, not a fresh read. That
        ordering is what makes the decision causal rather than prophetic.

        Returns ``None`` only when there is genuinely nothing to report. A *stale* reading does
        return an entry, deliberately: the dashboard has to be able to show that the loop has
        stopped acting and why, and a silent gap is indistinguishable from a crash.
        """
        if self._active_source_c is None and not self.reading_stale:
            return None

        now = time.time()
        if self.reading_stale:
            # No usable input this window, so there is nothing to decode. Acting here would
            # mean inventing a colour from a reading nobody supplied — which is exactly what
            # used to happen, because an ``unavailable`` thermometer was encoded as 0 °C and
            # then decoded like any other reading.
            self.last_action = {
                "entity_id": self.loop.light_entity,
                "service": "turn_on",
                "data": {},
                "sent": False,
                "suppressed": True,
                "reason": "sensor_stale",
                "dry_run": not self.loop.will_send,
                "at": now,
            }
            # Deliberately *not* appended to ``history``: that is the X-Y plot of real
            # decisions, and a row with no colour on it is a hole in the chart rather than an
            # honest data point. The snapshot carries ``reading_stale`` and the age instead.
            return {"t": now, "temperature_c": None, "kelvin": None, "band": None, "stale": True}

        temperature = self._active_source_c
        kelvin = self.decode(window_counts, window_ms)

        # The deadband gates only the service call. The dashboard still gets every decoded
        # colour, so it shows the true behaviour rather than the throttle.
        moved = (
            self._last_sent_kelvin is None
            or abs(kelvin - self._last_sent_kelvin) >= float(self.loop.deadband_k)
        )
        action = Action(
            entity_id=self.loop.light_entity,
            service="turn_on",
            confidence=1.0,
            data={"color_temp_kelvin": round(kelvin)},
        )
        sent = False
        if self.loop.will_send and self.ha is not None and moved:
            sent = await self.ha.call_service(
                action.entity_id, action.service, dict(action.data)
            )
            if sent:
                self._last_sent_kelvin = kelvin
        now = time.time()
        self.decisions += 1
        self.temperature_c = temperature
        self.brain_temperature_c = self._active_brain_c
        self.kelvin = kelvin
        self.last_action = {
            "entity_id": action.entity_id,
            "service": action.service,
            "data": dict(action.data),
            "sent": bool(sent),
            "suppressed": bool(self.loop.will_send and not moved),
            "dry_run": not self.loop.will_send,
            "at": now,
        }
        entry = {
            "t": now,
            "temperature_c": round(temperature, 3),
            "kelvin": round(kelvin),
            "band": self.band_for(kelvin),
        }
        self.history.append(entry)
        if len(self.history) > self.loop.history:
            del self.history[: len(self.history) - self.loop.history]
        return entry

    # ------------------------------------------------------------------- view

    def snapshot(self) -> dict:
        """Everything the dashboard needs to draw the loop's current state."""
        # Report the reading the loop is currently acting on, which exists as soon as a
        # window has been driven - not only after the first decision.
        reading = (
            self._active_source_c if self._active_source_c is not None else self.temperature_c
        )
        # The ideal ramp must be evaluated at the temperature the *brain* was given, not the raw
        # room reading. When the sensitivity range is stretched (a 2-degree room driving the
        # whole trained range), those two differ by design - and comparing the decoded colour
        # against the raw reading reports a huge error for behaviour that is exactly correct.
        brain_c = self._active_brain_c
        if brain_c is None and reading is not None:
            brain_c = self.brain_temperature(reading)
        ideal = None
        if brain_c is not None:
            lo, hi = self.band_centres[0], self.band_centres[-1]
            u = (brain_c - self.config.temp_min_c) / max(
                self.config.temp_max_c - self.config.temp_min_c, 1e-9
            )
            ideal = float(lo + float(np.clip(u, 0.0, 1.0)) * (hi - lo))
        err = None if (ideal is None or self.kelvin is None) else self.kelvin - ideal
        return {
            "mode": self.loop.mode,
            "dry_run": bool(not self.loop.will_send),
            "temperature_entity": self.loop.temperature_entity,
            "light_entity": self.loop.light_entity,
            "temperature_c": None if reading is None else round(reading, 2),
            "brain_temperature_c": (
                None if self.brain_temperature_c is None else round(self.brain_temperature_c, 2)
            ),
            "kelvin": None if self.kelvin is None else round(self.kelvin),
            "ideal_kelvin": None if ideal is None else round(ideal),
            "error_k": None if err is None else round(err),
            "band": None if self.kelvin is None else self.band_for(self.kelvin),
            "sensor_rate_hz": None if self._last_rate_hz is None else round(self._last_rate_hz, 1),
            "driven_neurons": int(self.input_indices.size),
            "readout_neurons": int(self.readout_indices.size),
            # Extra sensory pathways driving the brain alongside temperature. The dashboard
            # shows these so it is visible that the fly is being driven by more than one sense.
            "channels": [
                {
                    "entity_id": c.entity_id,
                    "kind": c.kind.value,
                    "neurons": int(c.neuron_indices.size),
                    "rate_hz": (
                        None if c.entity_id not in self.channel_rates
                        else round(self.channel_rates[c.entity_id], 1)
                    ),
                }
                for c in self.channels
            ],
            "window_ms": self.config.window_ms,
            "decisions": self.decisions,
            "last_action": self.last_action,
            # A missing or offline sensor used to be invisible here: the loop reported its last
            # reading as though it were current and kept acting on it. The dashboard shows
            # these two so "the number is frozen" and "the number is wrong" look different.
            "reading_stale": bool(self.reading_stale),
            "reading_age_s": (
                None
                if self.last_good_reading_at is None
                else round(time.time() - self.last_good_reading_at, 1)
            ),
            "temp_range_c": [self.config.temp_min_c, self.config.temp_max_c],
            "band_centres_k": {n: k for n, k in COLOUR_BANDS},
            # Pacing state: whether the brain is waiting, bursting on something interesting, or
            # running flat out - and the duty cycle it has *actually* achieved. The dashboard
            # needs the first to explain a still picture, and the second because with a trigger
            # in play the energy cost is no longer predictable from the interval alone.
            "pacing": self.pacer.snapshot(time.monotonic()),
            "history": self.history,
            "error": self.last_error,
            # Connection settings + what the room has actually been doing, so the
            # dashboard can show them and offer to fit the range automatically.
            "settings": self.loop.to_dict(),
            "observed_range_c": (
                None
                if self.observed_min_c is None
                else [round(self.observed_min_c, 2), round(self.observed_max_c, 2)]
            ),
            # Kelvin of light per degree of room, i.e. how much a 1 degC change moves it.
            "kelvin_per_degree": round(
                (self.config.temp_max_c - self.config.temp_min_c)
                / self.loop.source_span_c
                * (self.band_centres[-1] - self.band_centres[0])
                / max(self.config.temp_max_c - self.config.temp_min_c, 1e-9),
                1,
            ),
        }

    # ------------------------------------------------------------------ settings

    def update_settings(self, patch: dict) -> dict:
        """Apply a connection-settings patch and return the new settings.

        Values are clamped to what the readout can honestly represent: limits outside the
        trained band range would produce colours the brain was never fitted to.
        """
        self.loop.apply(patch)
        lo = float(self.band_centres[0])
        hi = float(self.band_centres[-1])
        self.loop.kelvin_min = float(np.clip(self.loop.kelvin_min, lo, hi))
        self.loop.kelvin_max = float(np.clip(self.loop.kelvin_max, lo, hi))
        if self.loop.kelvin_max <= self.loop.kelvin_min:
            self.loop.kelvin_min, self.loop.kelvin_max = lo, hi
        if self.loop.source_max_c <= self.loop.source_min_c:
            # A zero-width span would divide by ~zero; give it a degree of room.
            self.loop.source_max_c = self.loop.source_min_c + 1.0
        if self.loop.smooth_ms < 0:
            self.loop.smooth_ms = 0.0
        if self.loop.deadband_k < 0:
            self.loop.deadband_k = 0.0
        # Pacing settings were just replaced wholesale, so the pacer holds a stale copy of them.
        # Rebuilding rather than mutating keeps one definition of each setting: the config.
        self.pacer = self._build_pacer()
        self._smoothed_c = None       # a new span invalidates the smooth history
        return self.loop.to_dict()

    def fit_range_to_observed(self, margin_c: float = 0.5) -> dict:
        """Set the sensitivity span to what this room has actually been doing.

        Guessing the span is the annoying part of setting this up; the loop has already
        been watching the sensor, so it can propose the range itself.
        """
        if self.observed_min_c is None or self.observed_max_c is None:
            return self.loop.to_dict()
        lo = math.floor((self.observed_min_c - margin_c) * 2) / 2
        hi = math.ceil((self.observed_max_c + margin_c) * 2) / 2
        if hi - lo < 1.0:
            hi = lo + 1.0
        return self.update_settings({"source_min_c": lo, "source_max_c": hi})


def build_loop(sim, loop_config: LoopConfig | None = None, directory: str | Path | None = None):
    """Load the Home Assistant adapter and wire up a :class:`LiveLoop`.

    Returns ``None`` if there is no trained readout yet, so the dashboard can still serve
    the brain view and explain what is missing instead of failing to start.
    """
    from flybrain.ha import make_home_assistant

    cfg = loop_config or LoopConfig()
    try:
        ha = make_home_assistant(cfg.mode)
    except Exception as exc:  # noqa: BLE001 - a missing HA must not take the dashboard down
        logger.warning("could not build a Home Assistant adapter (%s); loop disabled", exc)
        return None
    try:
        return LiveLoop.from_artifact(sim, directory, cfg, ha)
    except MissingReadout as exc:
        logger.warning("%s", exc)
        return None
