"""Live brain view: FastAPI server for the FlyWire connectome dashboard.

Architecture notes
------------------
The single most important performance decision is that **Python never sits in the
per-frame rendering path**. The neuron point cloud is uploaded to the GPU once; per frame
we push one compact ``uint8`` intensity buffer per neuron (~139 KB) over a **binary**
WebSocket, and a GLSL shader maps intensity to colour on the GPU. Sending JSON numbers
would be roughly 1 MB per frame plus a 139k-element parse, and would cap the dashboard far
below its frame budget.

Endpoints
---------
``GET  /``                    dashboard (static files)
``GET  /api/config``          connectome metadata + settings schema
``GET  /api/positions``       float32 LE, n*3, neuron soma positions (3D units)
``GET  /api/groups``          cell families + sensory pathways, with plain-English names
``GET  /api/groups/ids``      uint8, n*2, family id then sense id per neuron
``GET  /api/regions``         per-cell-class and per-family usage snapshot
``GET  /api/trace``           where a family or sense sends its signals (summary geometry)
``POST /api/jev``             classify one recorded decision (placement A)
``POST /api/jev/enabled``     turn the judgment layer on or off for this process
``GET  /api/settings``        current settings
``POST /api/settings``        update settings
``POST /api/drive``           set/clear input drive on a neuron population
``WS   /ws/activity``         binary heat frames + JSON events

Wire protocol (must match ``web/app.js``)
-----------------------------------------
Server -> client, binary frame::

    uint32 magic = 0x4642524E ("FBRN")
    uint32 seq
    uint32 n_neurons
    float32 sim_ms
    uint32 total_spikes
    uint32 active_neurons
    uint8  intensity[n_neurons]        # 0..255, one byte per neuron

Server -> client, text frames: JSON ``{"type": ...}`` for ``hello``, ``metrics``,
``regions``, ``settings``, ``info``.

Client -> server: JSON ``{"type": ...}`` for ``settings``, ``drive``, ``pause``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from flybrain.activity import ActivitySettings, MemoryBrain
from flybrain.env import load_dotenv
from flybrain.families import (
    FAMILY_BY_KEY,
    SENSE_BY_KEY,
    SENSE_GROUPS,
    family_ids,
    group_id_buffer,
    groups_payload,
    sense_ids,
)
from flybrain.mapping import RoleResolver, default_output_roles, default_sensor_roles
from flybrain.pet import HONESTY, STATES, PetWatcher, sensor_deltas, summarise_journal
from flybrain.types import signal_is_dead
from flybrain.wiring import role_for

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
POSITIONS = ROOT / "data/codex/positions_normalized.npy"
MAGIC = 0x4642524E  # "FBRN"

HEADER = struct.Struct("<III f II")


def recording_meta(loop_cfg) -> dict:
    """The provenance a recording needs in order to be reproducible later.

    A module-level function rather than an inline dict inside ``_start_recorder`` for one
    reason: ``_start_recorder`` needs a loaded brain and a live Home Assistant to reach, so
    anything written there is untestable. This is the part that can be wrong silently.

    **Why pacing is in here at all.** ``AGENTS.md`` #2 is "train and run in the same regime".
    Pacing looks like a runtime detail and is not: with a trigger enabled, a recording is a
    sample of *events*, while a fixed heartbeat records a sample of *time*. Those are different
    distributions, so a readout fitted on one and run against the other loses accuracy in
    exactly the silent way the rest-basin bug did. Recording the pacing config is what makes
    that checkable instead of remembered.

    ``session-20260924-135739`` predates this and is implicitly flat out; it is left as it is
    rather than rewritten, because inventing a pacing config for a file that did not have one
    would be a worse lie than the omission.
    """
    return {
        "temperature_entity": loop_cfg.temperature_entity,
        "light_entity": loop_cfg.light_entity,
        "mode": loop_cfg.mode,
        "pacing": {
            "heartbeat_s": loop_cfg.interval_s,
            "poll_s": loop_cfg.poll_s,
            "burst_s": loop_cfg.burst_s,
            "trigger_delta": loop_cfg.trigger_delta,
        },
    }


class BrainService:
    """Owns the simulator and the stepping loop, decoupled from the web layer."""

    def __init__(self) -> None:
        self.sim = None
        self.brain: MemoryBrain | None = None
        self.settings = ActivitySettings()
        self.paused = False
        self.positions: np.ndarray | None = None
        self.roles: RoleResolver | None = None
        self._frame: dict | None = None
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._drive_role: str | None = None
        # Live control loop state.
        #: Step the brain even with no dashboard watching. Off by default: the simulator
        #: is ~7.7x slower than wall-clock, so running it permanently is a real cost, and
        #: for a demo the GPU is better spent while someone is looking. Turn it on with
        #: ``FLYBRAIN_ALWAYS_ON=1`` when the loop is meant to control something for real.
        self.always_on = os.environ.get("FLYBRAIN_ALWAYS_ON", "0") not in {"0", "false", "no"}
        self.loop = None
        self.loop_cfg = None
        #: Optional recording of every completed window, so new readouts can be trained later.
        #: Off unless ``FLYBRAIN_RECORD=1``: it writes to disk continuously and holds a record of
        #: the inside of a real house, which should be an opt-in.
        self.recorder = None
        self._accum: np.ndarray | None = None
        self._accum_ms = 0.0
        self._last_mock_tick: float | None = None
        #: Extra sensory channels are discovered once, from the first real read of the house.
        self._channels_ready = False
        #: The Jev client is built lazily: with no API key there is nothing to build, and
        #: constructing it at import time would make an optional feature part of startup.
        self._jev = None
        #: Runtime on/off for the judgment layer, overriding ``JEV_ENABLED`` for this process
        #: only. ``None`` means "whatever the environment said". Kept separate from the config so
        #: flipping it needs no restart, the same way the dry-run switch works.
        self.jev_enabled_override: bool | None = None
        #: Verdicts already paid for, keyed by brain sequence number. A decision cannot change
        #: once recorded, so neither can its classification — which makes a re-click free.
        self._jev_verdicts: dict[int, dict] = {}
        #: The house pet. Holds only the activity baseline and the state clock - everything it
        #: says is derived from a window that actually happened, which is why it lives here
        #: rather than in the browser: a state computed client-side from the last frame would be
        #: describing a *display* slice, not the window the colour was decoded from.
        self.pet = PetWatcher()
        self.pet_state = None
        #: The previous window's sensor values, for "the house moved by X".
        self._prev_sensor_values: dict[str, float] = {}
        #: The last region snapshot, so the pet's sentence can name the busiest cell class. Read
        #: from the cached copy rather than recomputed, to keep the region poll the only caller.
        self._last_regions: list[dict] | None = None
        #: Bursts, for the memory trail. Capped: the trail is a glance, not a log.
        self._bursts: list[dict] = []
        self._started_at = time.time()
        #: Cell-family metadata and the 2-bytes-per-neuron id buffer. Built once on first
        #: request: the buffer is ~277 KB and the family join touches all 138,639 rows, so
        #: rebuilding it per poll would be the most expensive thing on the dashboard.
        self._groups: dict | None = None
        self._group_ids: bytes | None = None
        #: Connection-trace state. The per-synapse arrays are built once and kept because every
        #: trace request needs the same 15M-element expansion; the per-group answers are cached
        #: because a decision to look at one pathway does not change what that pathway connects to.
        self._trace_post = None
        self._trace_pre = None
        self._class_centroids_cache: np.ndarray | None = None
        self._trace_cache: dict[str, dict] = {}

    # ----------------------------------------------------------------- setup

    def load(self) -> None:
        from flybrain.sim import ConnectomeSim

        logger.info("loading connectome ...")
        self.sim = ConnectomeSim().load()
        self.brain = MemoryBrain(self.sim, self.settings)
        self.roles = RoleResolver.from_sim(self.sim)
        if POSITIONS.exists():
            self.positions = np.load(POSITIONS).astype(np.float32)
            if self.positions.shape[0] != self.sim.n_neurons:
                logger.warning(
                    "position count %d != neuron count %d; view will be degenerate",
                    self.positions.shape[0],
                    self.sim.n_neurons,
                )
        else:
            logger.warning("no positions file at %s; using origin cloud", POSITIONS)
            self.positions = np.zeros((self.sim.n_neurons, 3), dtype=np.float32)
        logger.info("ready: %d neurons, %d synapses", self.sim.n_neurons, self.sim._W.values().numel())
        self._setup_loop()

    def _setup_loop(self) -> None:
        """Build the temperature -> colour loop, if a trained readout exists.

        A missing readout is not fatal: the brain view is still worth serving, and the
        dashboard explains what to run. Failing to start would be a worse outcome than
        running with the loop marked unavailable.
        """
        from flybrain.ha import Scenario
        from flybrain.loop import LoopConfig, build_loop

        cfg = LoopConfig.from_env()
        self.loop_cfg = cfg
        try:
            self.loop = build_loop(self.sim, cfg)
        except Exception:
            logger.exception("live loop failed to start; the brain view will still serve")
            self.loop = None
            return
        if self.loop is None:
            return
        # Give the simulated room a visible swing so the whole chain can be watched.
        if cfg.is_mock and getattr(self.loop, "ha", None) is not None:
            scenario = Scenario(
                temperature_start=22.5, temperature_swing_c=11.0, temperature_period_s=180.0
            )
            self.loop.ha.scenario = scenario
            self.loop.ha.states["sensor.living_room_temperature"] = (
                f"{scenario.temperature_start:.2f}"
            )
        self._accum = np.zeros(self.sim.n_neurons, dtype=np.int64)
        self._start_recorder()

    def _start_recorder(self) -> None:
        """Open a recording session for the control loop, if one was asked for.

        Opt-in via ``FLYBRAIN_RECORD=1``, and never fatal: a read-only disk or a bad name must
        not stop the brain from running.
        """
        if os.environ.get("FLYBRAIN_RECORD", "0") in {"0", "false", "no"}:
            return
        from flybrain.recorder import DEFAULT_ROOT, Recorder

        name = os.environ.get("FLYBRAIN_RECORD_NAME") or time.strftime("session-%Y%m%d-%H%M%S")
        # Careful: on LiveLoop, ``.config`` is the ExperimentConfig (readout/window settings) and
        # ``.loop`` is the LoopConfig (entities, sensitivity, dry-run). The names invite a mix-up
        # and the AttributeError is only raised when recording is switched on.
        loop_cfg = self.loop.loop
        try:
            self.recorder = Recorder(
                DEFAULT_ROOT / name,
                feature_dim=int(self.loop.readout_indices.size),
                window_ms=float(self.loop.config.window_ms),
                meta=recording_meta(loop_cfg),
            )
        except Exception:
            logger.exception("could not open a recording; continuing without one")
            self.recorder = None
            return
        logger.info("recording windows to %s", self.recorder.root)

    # ------------------------------------------------------------ broadcast

    async def register(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def unregister(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast_binary(self, payload: bytes) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_bytes(payload)
            except Exception:  # noqa: BLE001 - a dead socket must not kill the broadcast
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def broadcast_json(self, message: dict) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_text(json.dumps(message))
            except Exception:  # noqa: BLE001 - a dead socket must not kill the broadcast
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    # ----------------------------------------------------------------- loop

    async def _advance_mock(self) -> None:
        """Move the simulated room clock on by the real time that has elapsed.

        The mock's sensors evolve in *simulated* minutes, so without this the temperature
        would never change and the dashboard would have nothing to show.

        The clamp is the heartbeat rather than a fixed five seconds, and that matters because
        this is called from :meth:`begin_window` - which, once pacing is on, only runs when a
        decision is made. With a fixed 5 s clamp and a 15 s heartbeat, the simulated house would
        advance at a third of real time and the demo would appear to have slowed down. It is a
        demo-only concern, but the mock is the default mode, so it is the first thing anyone
        sees.
        """
        ha = getattr(self.loop, "ha", None)
        if ha is None or self.loop_cfg is None or not self.loop_cfg.is_mock:
            return
        now = time.monotonic()
        if self._last_mock_tick is not None:
            limit = max(5.0, float(self.loop_cfg.interval_s))
            ha.advance(max(0.0, min(now - self._last_mock_tick, limit)))
        self._last_mock_tick = now

    async def begin_window(self) -> None:
        """Read the sensor and apply it as the drive for the window about to run.

        The drive is set *before* the window it will be judged on, so a decision is caused
        by the temperature it was given rather than by the one after it.
        """
        if self.loop is None:
            return
        await self._advance_mock()
        signals = await self.loop.read_signals()
        if signals and not self._channels_ready:
            self._configure_channels(signals)
        # One call for every channel: set_drive replaces the whole drive map, so driving
        # temperature and then the extras separately would silently cancel the first.
        self.loop.drive_channels(signals)

    def _configure_channels(self, signals) -> None:
        """Attach the extra sensory pathways discovery finds, if they were asked for.

        Off unless ``FLYBRAIN_CHANNELS`` is set. It is opt-in because of a real caveat: the
        colour readout was fitted with **only** the temperature channel driving the brain, and
        the readout is a function of the whole reservoir state. Driving the brain from motion
        and illuminance as well changes that state, so the decoded colour is no longer the
        thing that was trained until the readout is re-fitted. This is documented, not hidden.
        """
        self._channels_ready = True
        if os.environ.get("FLYBRAIN_CHANNELS", "off").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
            "",
        }:
            return
        from flybrain.wiring import discover

        try:
            wiring = discover(signals)
            channels = wiring.to_channels(self.roles, neurons_per_pathway=64)
            self.loop.configure_channels(channels)
        except Exception:
            logger.exception("could not configure extra sensory channels; continuing without")
            return
        if self.loop.channels:
            logger.warning(
                "driving %d extra sensory channel(s): %s. The colour readout was fitted on "
                "temperature alone, so its output is not trustworthy until it is re-fitted.",
                len(self.loop.channels),
                ", ".join(c.entity_id for c in self.loop.channels),
            )

    async def finish_window(self, counts, window_ms: float) -> None:
        """Decode the completed window, emit the action, and open the next window."""
        if self.loop is None:
            return
        try:
            entry = await self.loop.decide(
                counts, window_ms, context=self._decision_context(counts)
            )
        except Exception:
            logger.exception("control decision failed")
            # Tell the pacer anyway. Otherwise a persistent decode failure leaves the heartbeat
            # overdue, `should_step` stays true, and the loop retries at the frame rate - which
            # turns one broken decision into a log storm at 20 lines a second. Backing off to
            # the heartbeat is both quieter and cheaper.
            self.loop.pacer.note_step(time.monotonic())
            return
        # Record before begin_window(): the sensor snapshot must be the one that *drove* the
        # window just decoded, not the next window's reading.
        self._record_window(counts, window_ms)
        # Feed the pet from the window that was just decided on, so its state describes the same
        # moment the colour does. Reading it from the frame loop instead would compare a window
        # against a different window, which is how a "usual" baseline quietly stops meaning
        # anything.
        self._observe_pet(counts)
        # Tell the pacer a decision actually happened. This is what ends a burst's immediate
        # steps and restarts the heartbeat, so it must happen only after a real decode - not
        # when a window is merely accumulated, and not when the decision raised.
        self.loop.pacer.note_step(time.monotonic())
        if entry is not None:
            await self.broadcast_json({"type": "loop", "loop": self.loop.snapshot()})
        await self.begin_window()

    def _note_burst(self, signals) -> None:
        """Record that the trigger fired, for the pet's clock and the memory trail.

        The pet is *told* about a burst rather than inferring one from activity, because the pacer
        already knows exactly when it happened and why. Inferring it would be a second, worse
        answer to a question that was already answered.

        **Wall clock, not monotonic.** ``run_loop`` hands its pacer a ``time.monotonic()`` value,
        and using that here would put the burst on a different epoch from the windows the pet
        observes with ``time.time()``. It renders as "since the last change 1790499452s" — the
        kind of number that looks like a formatting bug and is really a clock bug. The trail needs
        wall clock anyway, because the browser places its marks on a timeline.
        """
        wall = time.time()
        self.pet.note_burst(wall)
        values = {
            s.entity_id: float(s.value) for s in (signals or []) if not signal_is_dead(s)
        }
        changes = sensor_deltas(self._prev_sensor_values, values)
        self._bursts.append({"t": wall, "changes": changes})
        del self._bursts[: max(0, len(self._bursts) - 200)]

    def _sensor_values(self) -> dict[str, float]:
        """``{entity_id: value}`` from the last read, skipping anything not reporting.

        The dead readings are dropped *here* rather than in the pet, because "the house moved" is a
        sentence a person reads: a thermometer going ``unavailable`` and coming back would
        otherwise appear as a large temperature change, which is the plumbing moving, not the room.
        """
        out: dict[str, float] = {}
        for signal in getattr(self.loop, "last_signals", []) or []:
            if signal_is_dead(signal):
                continue
            out[signal.entity_id] = float(signal.value)
        return out

    def _observe_pet(self, counts) -> None:
        """One pet observation per completed control window.

        Window-level counts, not the last published frame: a frame is a display slice and the pet
        should describe the same 300 ms the colour was decoded from.
        """
        if self.loop is None or counts is None:
            return
        values = self._sensor_values()
        changes = sensor_deltas(self._prev_sensor_values, values)
        self._prev_sensor_values = values
        counts = np.asarray(counts)
        observation = self.pet.observe(
            now=time.time(),
            active_neurons=int((counts > 0).sum()),
            total_spikes=int(counts.sum()),
            sensor_changes=changes,
            stale=bool(getattr(self.loop, "reading_stale", False)),
            reading_age_s=self.loop.snapshot().get("reading_age_s"),
            bursting=bool(self.loop.pacer.snapshot(time.monotonic())["mode"] == "burst"),
            busiest=tuple(row["name"] for row in (self._last_regions or ())[:3]),
        )
        self.pet_state = observation

    def _record_window(self, counts, window_ms: float) -> None:
        """Append one completed window to the recording, if recording is on.

        A failure here disables recording rather than repeating: a full disk would otherwise log
        an exception every couple of seconds for as long as the dashboard stays open.
        """
        if self.recorder is None or self.loop is None:
            return
        try:
            self.recorder.record(
                self.loop.features(counts, window_ms),
                sensors=self.loop.sensor_snapshot(),
            )
        except Exception:
            logger.exception("failed to record a window; recording is now off")
            self.recorder = None

    async def run_loop(self) -> None:
        """Step the brain, publish frames, and make one colour decision per window."""
        assert self.brain is not None
        window_ms = float(self.loop.config.window_ms) if self.loop is not None else 0.0
        await self.begin_window()

        while True:
            fps = max(1.0, float(self.settings.fps))
            await asyncio.sleep(1.0 / fps)
            if self.paused or not (self._clients or self.always_on):
                continue
            # Wall-clock pacing. Running flat out holds the GPU at ~165 W continuously, because
            # the brain steps as fast as it can to advance brain time. A context layer needs a
            # decision every few seconds at most, so pace the *decisions* and let the GPU idle in
            # between: average power falls roughly with the duty cycle.
            #
            # The pacer refines this from "one every N seconds" into "one every N seconds, or
            # now if the house did something" - see flybrain/pacing.py. Polling is a sensor read
            # only, so it costs no GPU time; it is what lets a change be noticed before the next
            # heartbeat was due.
            pacer = self.loop.pacer if self.loop is not None else None
            if pacer is not None:
                now = time.monotonic()
                if pacer.should_poll(now):
                    try:
                        # store=False: this read happens *between* decisions, and the snapshot
                        # the recorder writes must stay the reading that drove the window.
                        signals = await self.loop.read_signals(store=False)
                    except Exception:
                        logger.exception("pacing poll failed; waiting for the heartbeat")
                    else:
                        if pacer.note_poll(now, signals):
                            logger.info("house changed; bursting for %.0fs", pacer.burst_s)
                            self._note_burst(signals)
                if not pacer.should_step(now):
                    continue
            try:
                done = False
                counts = None
                async with self._lock:
                    frame = await asyncio.to_thread(self.brain.advance)
                    self._frame = {
                        "seq": frame.seq,
                        "sim_ms": frame.sim_ms,
                        "total_spikes": frame.total_spikes,
                        "active_neurons": frame.active_neurons,
                        "metrics": frame.metrics,
                    }
                    payload = self.encode_frame(frame)
                    if self._accum is not None and window_ms > 0:
                        self._accum += frame.counts
                        self._accum_ms += float(frame.metrics.get("window_ms", 0.0))
                        if self._accum_ms >= window_ms:
                            counts, ms = self._accum, self._accum_ms
                            self._accum = np.zeros_like(self._accum)
                            self._accum_ms = 0.0
                            done = True
                if self._clients:
                    await self.broadcast_binary(payload)
                if done and counts is not None:
                    await self.finish_window(counts, ms)
            except Exception:  # keep the dashboard alive on a transient failure
                logger.exception("activity loop iteration failed")

    @staticmethod
    def encode_frame(frame) -> bytes:
        """Pack a :class:`flybrain.activity.Frame` into the binary wire format."""
        intensity = np.ascontiguousarray(frame.intensity, dtype=np.uint8)
        header = HEADER.pack(
            MAGIC,
            int(frame.seq),
            int(intensity.size),
            float(frame.sim_ms),
            int(frame.total_spikes),
            int(frame.active_neurons),
        )
        return header + intensity.tobytes()

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def jev_status(self, *, refresh: bool = False, probe: bool = True) -> dict:
        """Report the judgment layer's availability, building the client on first use.

        ``probe`` defaults to True for the explicit endpoint and is passed False by the status
        path, so that a dashboard being *watched* cannot spend money.
        """
        from flybrain.jev import session_spend

        client = self._jev_client()
        status = await client.available(refresh=refresh, probe=probe)
        payload = status.to_dict()
        # The redacted config, so an operator can see *which* endpoint and model are configured
        # without the key ever leaving the process.
        payload["config"] = client.config.redacted()
        payload["calls"] = len(client.calls)
        # The accounting from the most recent real call. Shown because a credit balance reaching
        # zero is how a feature stops working without anyone noticing, and because the cost is the
        # number that decides whether generous labelling is affordable.
        if client.calls:
            last = client.calls[-1]
            payload["last_call"] = dict(last)
            payload["spent_usd"] = session_spend(client)
        return payload

    def _jev_client(self):
        """The Jev client, built on first use, with the runtime switch applied.

        One place rather than two, because a second construction site is exactly how the status
        badge and the inspector would end up disagreeing about whether Jev is on.
        """
        from flybrain.jev import JevClient, JevConfig

        if self._jev is None:
            self._jev = JevClient(JevConfig.from_env())
        if self.jev_enabled_override is not None:
            self._jev.config.enabled_flag = self.jev_enabled_override
        return self._jev

    async def jev_verdict(self, seq: int) -> dict:
        """Classify one recorded decision into a failure mode, caching per sequence number.

        Cached because the same click should never cost a second question: a decision is
        immutable once recorded, so its verdict is too.
        """
        from flybrain.jev import classify_decision

        cached = self._jev_verdicts.get(seq)
        if cached is not None:
            # Still served when the layer has since been switched off: it was paid for, the
            # decision has not changed, and re-asking would be spending to learn nothing.
            return {**cached, "cached": True}

        row = None
        if self.loop is not None:
            row = next((r for r in self.loop.history if r.get("seq") == seq), None)
        if row is None:
            raise KeyError(seq)

        client = self._jev_client()
        if not client.config.enabled:
            # Answer with the same vocabulary the badge uses, so the card and the badge cannot
            # tell two different stories about why there is no verdict.
            status = await client.available(probe=False)
            return {
                "available": False,
                "reason": status.reason,
                "detail": status.detail,
                "seq": seq,
            }

        verdict = await classify_decision(client, row)
        verdict.update({"available": True, "seq": seq, "cached": False})
        self._jev_verdicts[seq] = verdict
        return verdict

    async def aclose(self) -> None:
        """Release the Jev client's connection pool, if it was ever built."""
        if self._jev is not None:
            await self._jev.aclose()
            self._jev = None

    # ------------------------------------------------------------------ status

    def _trained(self) -> dict | None:
        """What the colour readout was fitted with, read from its own metadata.

        The dashboard shows this because "which brain is this" and "how was it trained" are the
        questions that decide whether a number on screen means anything — and because a readout
        fitted under one regime is not valid under another (AGENTS.md #2).
        """
        from flybrain.experiment import ARTIFACT_DIR
        from flybrain.loop import META_FILE, READOUT_FILE

        path = Path(ARTIFACT_DIR) / META_FILE
        readout = Path(ARTIFACT_DIR) / READOUT_FILE
        if not path.exists():
            return None
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return {
            "regime": meta.get("regime"),
            "window_ms": meta.get("window_ms"),
            "input_roles": meta.get("input_roles"),
            "readout_roles": meta.get("readout_roles"),
            "temp_range_c": meta.get("temp_range_c"),
            "band_centres_k": meta.get("band_centres_k"),
            "trained_at": datetime.fromtimestamp(
                readout.stat().st_mtime, tz=UTC
            ).isoformat(timespec="seconds")
            if readout.exists()
            else None,
        }

    def _journal(self, now: float, watts: float | None) -> dict:
        """Counters for "today so far" — what happened, in numbers, without adjectives."""
        journal = self.pet.journal(now)
        windows = int(self.recorder.n_windows) if self.recorder is not None else 0
        labels: list[dict] = []
        session = None
        if self.recorder is not None:
            session = self.recorder.root.name
            labels = self._labels()
        journal.update(
            {
                "windows": windows,
                "labels": len(labels),
                "session": session,
                "recording": self.recorder is not None,
                "outputs": self._outputs(),
            }
        )
        journal["line"] = summarise_journal(
            journal, windows=windows, labels=len(labels), watts=watts
        )
        return journal

    def _labels(self) -> list[dict]:
        """Labels written this session, newest last.

        Read from the file rather than kept in memory: the recorder is the owner of that file, and
        a second in-memory copy would be a second answer to "what has been labelled".
        """
        if self.recorder is None:
            return []
        path = self.recorder.root / "labels.jsonl"
        if not path.exists():
            return []
        out: list[dict] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        except (OSError, ValueError):
            return out
        return out[-200:]

    def _outputs(self) -> list[dict]:
        """Which Home Assistant entities this loop is allowed to write to, and how.

        Stated explicitly because it is the question that matters before leaving anything running:
        not "is it working" but "what can it touch". The answer is one light's colour temperature,
        and the dry-run flag decides whether even that is real.
        """
        if self.loop is None:
            return []
        cfg = self.loop.loop
        will_send = bool(cfg.will_send)
        outputs = [
            {
                "entity_id": cfg.light_entity,
                "what": "colour temperature only",
                "enabled": will_send,
                "reason": "dry run — logged, never sent" if not will_send else "live",
            }
        ]
        for channel in getattr(self.loop, "channels", []) or []:
            outputs.append(
                {
                    "entity_id": channel.entity_id,
                    "what": "read only (drives the brain)",
                    "enabled": True,
                    "reason": "inbound",
                }
            )
        return outputs

    def _layers(self) -> dict:
        """The three-layer explanation: house said, brain did, we mapped it to.

        Deliberately assembled server-side from the same values each layer *acts* on, rather than
        from the raw inputs the browser also receives. If the panel and the behaviour came from
        two places, the panel could explain something the loop did not do.
        """
        snapshot = self.loop.snapshot() if self.loop is not None else {}
        frame = self._frame or {}
        regions = (self._last_regions or [])[:3]
        action = snapshot.get("last_action") or {}

        house = {
            "entity": snapshot.get("temperature_entity"),
            "value": snapshot.get("temperature_c"),
            "unit": "°C",
            "age_s": snapshot.get("reading_age_s"),
            "stale": bool(snapshot.get("reading_stale")),
            "changed": sensor_deltas(
                self._prev_sensor_values, self._sensor_values()
            ),
            "senses": [
                {
                    "entity_id": ch.get("entity_id"),
                    "kind": ch.get("kind"),
                    "rate_hz": ch.get("rate_hz"),
                }
                for ch in snapshot.get("channels", [])
            ],
        }
        brain = {
            "active_neurons": frame.get("active_neurons"),
            "spikes": frame.get("total_spikes"),
            "sim_ms": frame.get("sim_ms"),
            "seq": frame.get("seq"),
            "driven_neurons": snapshot.get("driven_neurons"),
            "readout_neurons": snapshot.get("readout_neurons"),
            "regions": [
                {"name": r.get("name"), "spikes": r.get("spikes"), "rate_hz": r.get("rate_hz")}
                for r in regions
            ],
        }
        mapped = {
            "kelvin": snapshot.get("kelvin"),
            "ideal_kelvin": snapshot.get("ideal_kelvin"),
            "error_k": snapshot.get("error_k"),
            "band": snapshot.get("band"),
            "entity": snapshot.get("light_entity"),
            "action": action or None,
            "decisions": snapshot.get("decisions"),
        }
        return {"house": house, "brain": brain, "mapped": mapped}

    async def status(self) -> dict:
        """Everything the dashboard needs that is not the frame stream or the region list."""
        now = time.time()
        pacer = self.loop.pacer.snapshot(time.monotonic()) if self.loop is not None else None
        watts = pacer.get("observed_watts") if pacer else None
        pet = self.pet_state.to_dict() if self.pet_state is not None else {
            "state": None,
            "sentence": "waking up — no completed window yet, so there is nothing to describe.",
            "contributors": [],
            "since_s": 0.0,
            "vocabulary": list(STATES),
            "honesty": HONESTY,
        }
        config = self.loop.loop.to_dict() if self.loop is not None else {}
        will_send = bool(self.loop.loop.will_send) if self.loop is not None else False
        return {
            "pet": pet,
            "journal": self._journal(now, watts),
            "layers": self._layers(),
            "pacing": pacer,
            "trust": {
                "mode": config.get("mode"),
                "dry_run": not will_send,
                "will_send": will_send,
                "paused": bool(self.paused),
                "always_on": bool(self.always_on),
                "recording": self.recorder is not None,
                "sensor_entity": config.get("temperature_entity"),
                "trained": self._trained(),
                "outputs": self._outputs(),
            },
            "jev": await self.jev_status(probe=False),
        }

    # ------------------------------------------------------------------ groups

    def _decision_context(self, counts) -> dict:
        """The parts of a decision that only the server can see, for the decision inspector.

        ``seq`` is what gives a chart point and a verdict a shared identity, so clicking a point
        names a decision the server already recorded rather than re-deriving one. ``top_regions``
        is read here because the brain's per-window counts are overwritten by the next advance:
        a breakdown not captured at this moment cannot be recovered from a later read, and the
        inspector would have to send a thinner state than the design calls for.
        """
        brain = self.brain
        if brain is None or counts is None:
            return {}
        context: dict = {"seq": brain.seq}
        names = brain.class_names()
        index = brain.class_index()
        if index is not None and names:
            totals = brain.counts_by_class(np.asarray(counts))
            order = np.argsort(totals)[::-1][:5]
            context["top_regions"] = [
                {"name": names[int(i)], "spikes": int(totals[int(i)])}
                for i in order
                if totals[int(i)] > 0
            ]
        if self.loop is not None:
            # `self.loop` is the LiveLoop; `self.loop.loop` is its LoopConfig. The names are
            # genuinely inverted at that layer — `LiveLoop.config` is the *readout* config — so
            # the connection settings are reached through `.loop`.
            cfg = self.loop.loop
            # Recorded with the row rather than read at verdict time: the settings are
            # live-tunable, so asking later would describe the loop as it is now, not as it was
            # when the decision was made — and `throttled` is a verdict *about* the deadband.
            context["settings"] = {
                "interval_s": float(cfg.interval_s),
                "deadband_k": float(cfg.deadband_k),
                "temp_range_c": [float(cfg.source_min_c), float(cfg.source_max_c)],
            }
        return context

    def _wired_senses(self) -> dict[str, bool]:
        """Which sensory pathways a live Home Assistant entity is currently driving.

        Asked of :func:`flybrain.wiring.role_for` rather than mapped from the signal *kind*,
        because that is exactly the decision the wiring code makes: ``role_for`` keys off the
        entity *name*, since a bark detector and a motion detector both arrive as ``MOTION``.
        Deriving it any other way here would let the legend disagree with what is actually wired.
        """
        wired: dict[str, bool] = {}
        if self.loop is None:
            return wired
        # The trained temperature path is not in `channels` — it keeps its own encoding — so it
        # has to be added explicitly or the pathway the light is read from would look unwired.
        # Reached through `.loop`: `LiveLoop.config` is the readout config, not the connection one.
        if getattr(self.loop.loop, "temperature_entity", ""):
            wired["thermosensory"] = True
        for channel in getattr(self.loop, "channels", []) or []:
            match = role_for(channel.entity_id, channel.kind)
            if match:
                wired[str(match[0])] = True
        return wired

    def _build_groups(self) -> None:
        """Build the family/sense ids and their metadata, once.

        Every failure here degrades to "no families" rather than an exception: the point cloud
        still renders without them, and a missing annotation table is a legitimate state (the
        connectome is fetched separately), not an error.
        """
        if self._groups is not None:
            return
        sim, brain = self.sim, self.brain
        if sim is None or brain is None or self.positions is None:
            return
        table = getattr(sim, "annotation_table", None)
        n = int(sim.n_neurons)
        if table is None:
            logger.warning("no annotation table; cell families unavailable")
            self._groups = {"n": n, "families": [], "senses": [], "orientation": {}}
            self._group_ids = b""
            return

        family = (
            family_ids(table["super_class"].to_numpy())
            if "super_class" in table.columns
            else np.zeros(n, dtype=np.uint8)
        )
        role_indices: dict[str, np.ndarray] = {}
        if self.roles is not None:
            for sense in SENSE_GROUPS:
                try:
                    role_indices[sense.key] = self.roles.resolve(sense.key).indices
                except KeyError:
                    # A role this build does not know about is a missing overlay, not a fault.
                    logger.warning("no role %r; sense %r will have no cells", sense.key, sense.key)
        sense = sense_ids(role_indices, n)
        cell_type = table["cell_type"].to_numpy() if "cell_type" in table.columns else None

        self._groups = groups_payload(
            n_neurons=n,
            family=family,
            sense=sense,
            positions=self.positions,
            cell_type=cell_type,
            wired=self._wired_senses(),
        )
        self._group_ids = group_id_buffer(family, sense)
        logger.info(
            "cell families: %d neurons, %d families, %d senses wired",
            n,
            len(self._groups["families"]),
            sum(1 for s in self._groups["senses"] if s["wired"]),
        )

    def groups(self) -> dict:
        """Family and sense metadata, plus where the orientation markers go."""
        self._build_groups()
        return self._groups or {}

    def group_ids(self) -> bytes:
        """Two bytes per neuron — family id then sense id — aligned with ``/api/positions``."""
        self._build_groups()
        return self._group_ids or b""

    # ------------------------------------------------------------------- trace

    def _trace_arrays(self):
        """Per-synapse ``(post, pre)`` index arrays, expanded once and kept.

        ``_W`` is stored as CSR, which is indexed by *post*, so the row each synapse belongs to
        has to be expanded back out of ``crow_indices``. That is a single 15M-element pass, and
        every trace request needs the same answer, so it is done once rather than per group.
        """
        if self._trace_post is None:
            W = self.sim._W
            torch = self.sim._torch
            crow = W.crow_indices()
            self._trace_post = torch.repeat_interleave(
                torch.arange(crow.numel() - 1, dtype=torch.int32, device=crow.device),
                crow.diff().to(torch.long),
            )
            # In CSR the column index *is* the presynaptic neuron.
            self._trace_pre = W.col_indices()
        return self._trace_post, self._trace_pre

    def _class_centroids(self) -> np.ndarray:
        """Mean soma position per ``cell_class``, for drawing group-to-group lines."""
        if self._class_centroids_cache is None:
            index = self.brain.class_index()
            n = max(self.brain.n_classes, 1)
            out = np.zeros((n, 3), dtype=np.float32)
            if index is not None:
                sizes = np.bincount(index, minlength=n)
                for axis in range(3):
                    out[:, axis] = np.bincount(
                        index, weights=self.positions[:, axis], minlength=n
                    ) / np.maximum(sizes, 1)
            self._class_centroids_cache = out
        return self._class_centroids_cache

    def _group_indices(self, group: str) -> tuple[np.ndarray, str, str]:
        """Resolve a family or sense key to neuron indices, a kind, and a plain-English label."""
        family = FAMILY_BY_KEY.get(group)
        if family is not None:
            ids = self.brain.family_index()
            if ids is None:
                raise KeyError(group)
            return np.flatnonzero(ids == family.id), "family", family.label
        sense = SENSE_BY_KEY.get(group)
        if sense is not None and self.roles is not None:
            return self.roles.resolve(sense.key).indices, "sense", sense.label
        raise KeyError(group)

    def trace(self, group: str, top_k: int = 5) -> dict:
        """Where a family or sense sends its signals, as group-to-group lines.

        **This is a summary, not a synapse list.** 15,091,983 connections cannot be drawn, and a
        picture of all of them is a hairball that answers nothing. What is drawn is one line per
        target *cell class*, straight from the group's centre of mass to that class's centre of
        mass, weighted by how many synapses actually run that way. The counts are exact; the
        geometry is deliberately schematic, and the payload says so.
        """
        if group in self._trace_cache:
            return {**self._trace_cache[group], "cached": True}
        if self.sim is None or self.brain is None or self.positions is None:
            raise RuntimeError("brain still loading")
        indices, kind, label = self._group_indices(group)
        if indices.size == 0:
            raise KeyError(group)

        import torch

        post, pre = self._trace_arrays()
        index = self.brain.class_index()
        if index is None:
            raise RuntimeError("no annotation table, so targets cannot be named")

        member = torch.zeros(self.sim.n_neurons, dtype=torch.bool, device=pre.device)
        member[torch.as_tensor(np.asarray(indices, dtype=np.int64), device=pre.device)] = True
        targets = post[member[pre]]
        classes = torch.as_tensor(index.astype(np.int64), device=pre.device)
        counts = torch.bincount(classes[targets], minlength=self.brain.n_classes)
        counts = counts.cpu().numpy()

        names = self.brain.class_names()
        order = np.argsort(counts)[::-1]
        order = [int(i) for i in order if counts[int(i)] > 0][:top_k]
        centroids = self._class_centroids()
        origin = self.positions[np.asarray(indices)].mean(axis=0)

        sources = self.family_index_of(indices)
        payload = {
            "group": group,
            "kind": kind,
            "label": label,
            "neurons": int(indices.size),
            "total_synapses": int(counts.sum()),
            "note": (
                "Lines join group centres of mass, not individual synapses. The synapse counts "
                "are exact; the geometry is a summary."
            ),
            "targets": [
                {
                    "name": names[i],
                    "synapses": int(counts[i]),
                    "neurons": int((index == i).sum()),
                    "from": [round(float(v), 4) for v in origin],
                    "to": [round(float(v), 4) for v in centroids[i]],
                }
                for i in order
            ],
            "families": sources,
            "cached": False,
        }
        self._trace_cache[group] = payload
        return payload

    def family_index_of(self, indices: np.ndarray) -> list[dict]:
        """Which broad families a set of neurons falls into, largest first.

        Sent with a trace so the panel can say *what* is being followed in the same words the
        legend uses, rather than leaving the reader with a raw ``cell_class``.
        """
        self._build_groups()
        if not self._groups:
            return []
        ids = self.brain.family_index()
        if ids is None:
            return []
        present = ids[np.asarray(indices, dtype=np.int64)]
        counts = np.bincount(present, minlength=len(self._groups["families"]) + 1)
        out = [
            {"key": f["key"], "label": f["label"], "neurons": int(counts[f["id"]])}
            for f in self._groups["families"]
            if counts[f["id"]] > 0
        ]
        out.sort(key=lambda d: d["neurons"], reverse=True)
        return out

    def timeline(self) -> dict:
        """The memory trail: state changes, bursts, decisions, labels and actions, newest first.

        Five sources merged into one list because they are one story — something happened in the
        house, the brain did something, and the loop mapped it to a colour. Each entry carries its
        own ``kind`` so the client can colour it without knowing what produced it.
        """
        entries: list[dict] = []

        for state in self.pet.trail():
            entries.append(
                {
                    "t": state["t"],
                    "kind": "state",
                    "text": state["text"],
                    "detail": state["detail"],
                }
            )

        for burst in self._bursts[-40:]:
            movers = sorted(
                (burst.get("changes") or {}).items(), key=lambda kv: abs(kv[1]), reverse=True
            )
            detail = ", ".join(f"{k} {v:+.2f}" for k, v in movers[:3]) or "a change"
            entries.append(
                {"t": burst["t"], "kind": "burst", "text": "burst", "detail": detail}
            )

        if self.loop is not None:
            for row in self.loop.history[-60:]:
                entries.append(
                    {
                        "t": row.get("t"),
                        "kind": "decision",
                        "text": f"{row.get('kelvin')} K",
                        "detail": f"{row.get('temperature_c')} °C · {row.get('band')}",
                    }
                )
            action = self.loop.last_action
            if action:
                entries.append(
                    {
                        "t": action.get("at"),
                        "kind": "action",
                        "text": (
                            "sent" if action.get("sent")
                            else "suppressed" if action.get("suppressed")
                            else action.get("reason") or "not sent"
                        ),
                        "detail": f"{action.get('entity_id')} · {action.get('data') or {}}",
                    }
                )

        for label in self._labels():
            entries.append(
                {
                    "t": label.get("t"),
                    "kind": "label",
                    "text": str(label.get("label")),
                    "detail": f"source: {label.get('source', 'manual')}",
                }
            )

        entries = [e for e in entries if e.get("t") is not None]
        entries.sort(key=lambda e: e["t"], reverse=True)
        return {"entries": entries[:80], "now": time.time()}

    def release_gpu_cache(self) -> None:
        """Hand cached GPU blocks back to the driver while the brain is not stepping.

        This does **not** unload the connectome — the weights stay resident, which is why VRAM
        stays around 1.8 GB even when idle (measured). It only returns the allocator's unused
        cached blocks, which is worth doing and costs nothing. Freeing the model itself would
        mean reloading the connectome from disk on resume, which is far slower than leaving it.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("could not release cached GPU memory", exc_info=True)

    def set_paused(self, value: bool) -> bool:
        """Pause or resume, releasing the GPU cache when stopping."""
        self.paused = bool(value)
        if self.paused:
            self.release_gpu_cache()
            if self.loop is not None:
                # Do not let a pause bank burst credit, and do not fire a step the instant it
                # resumes. The store of previous sensor readings is deliberately kept: if the
                # house changed while we were not looking, that is a real event and the first
                # poll after resuming should catch it.
                self.loop.pacer.on_pause(time.monotonic())
        return self.paused


service = BrainService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(service.load)
    service._task = asyncio.create_task(service.run_loop())
    try:
        yield
    finally:
        service.stop()
        await service.aclose()


app = FastAPI(title="flybrain live view", lifespan=lifespan)


# ------------------------------------------------------------------- routes


@app.get("/api/config")
async def get_config() -> dict:
    sim = service.sim
    if sim is None:
        raise HTTPException(503, "brain still loading")
    return {
        "n_neurons": int(sim.n_neurons),
        "n_synapses": int(sim._W.values().numel()),
        "device": sim.device,
        "dt_ms": sim.params.dt_ms,
        "settings": service.settings.to_dict(),
        "roles": {k: int(v) for k, v in service.roles.summary().items()},
        "sensor_roles": default_sensor_roles(),
        "output_roles": default_output_roles(),
        "wire": {"magic": MAGIC, "header_bytes": HEADER.size, "dtype": "uint8"},
    }


@app.get("/api/positions")
async def get_positions() -> Response:
    if service.positions is None:
        raise HTTPException(503, "positions not loaded")
    return Response(
        content=service.positions.astype("<f4").tobytes(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/groups")
async def get_groups() -> dict:
    """Broad cell families and sensory pathways, with their plain-English names.

    One response rather than three: the legend, the id buffer and the orientation markers are
    only meaningful together, and fetching them separately would let the client label families
    with ids that do not match the buffer it is colouring.
    """
    if service.sim is None:
        raise HTTPException(503, "brain still loading")
    payload = await asyncio.to_thread(service.groups)
    if not payload.get("families"):
        raise HTTPException(503, "cell families unavailable (no annotation table)")
    return payload


@app.get("/api/groups/ids")
async def get_group_ids() -> Response:
    """The 2-bytes-per-neuron id buffer: family id, then sense id, in connectome order."""
    if service.sim is None:
        raise HTTPException(503, "brain still loading")
    payload = await asyncio.to_thread(service.group_ids)
    if not payload:
        raise HTTPException(503, "cell families unavailable (no annotation table)")
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/trace")
async def get_trace(group: str = "", top_k: int = 5) -> dict:
    """Where one family or sense sends its signals — the guided-exploration connection trace.

    Degrades rather than fails: a missing connectome, no annotation table or an unavailable
    accelerator all answer 503 with a reason, and the guides that use this are required to work
    without it. A trace is an overlay on an explanation, never the explanation itself.
    """
    if not group:
        raise HTTPException(400, "group is required")
    try:
        return await asyncio.to_thread(service.trace, group, max(1, min(int(top_k), 12)))
    except KeyError:
        raise HTTPException(404, f"unknown group {group!r}") from None
    except Exception as exc:
        logger.warning("trace for %r unavailable: %s", group, exc)
        raise HTTPException(503, f"trace unavailable: {exc}") from exc


@app.get("/api/regions")
async def get_regions() -> dict:
    if service.brain is None:
        raise HTTPException(503, "brain still loading")
    regions = await asyncio.to_thread(service.brain.region_usage)
    # The family grouping rides along on the poll that already runs, so the legend's live
    # activity can never be a different moment from the top-12 list beside it.
    families = await asyncio.to_thread(service.brain.family_usage)
    # Kept so the pet's sentence can name the busiest cell class without a second, slower
    # recomputation of the same thing on the status path.
    service._last_regions = regions
    return {"regions": regions, "families": families}


@app.get("/api/settings")
async def get_settings() -> dict:
    return service.settings.to_dict()


@app.post("/api/settings")
async def post_settings(patch: dict) -> dict:
    """Update the view settings, all-or-nothing.

    The patch is validated in full *before* any of it is applied, so a rejected update cannot
    leave the dashboard half-reconfigured. The previous version cast and assigned inside one
    loop, which meant the earlier keys of a bad patch took effect and were then reported as
    rejected.
    """
    try:
        service.settings.apply_patch(patch)
    except KeyError as exc:
        raise HTTPException(400, f"unknown settings: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await service.broadcast_json({"type": "settings", "settings": service.settings.to_dict()})
    return service.settings.to_dict()


@app.post("/api/drive")
async def post_drive(body: dict) -> dict:
    """Set or clear a persistent input drive on a named role."""
    if service.brain is None or service.roles is None:
        raise HTTPException(503, "brain still loading")
    role = body.get("role")
    current = float(body.get("current_mv", 0.0))
    if not role or current == 0.0:
        service.brain.clear_input_drive()
        service._drive_role = None
        return {"role": None, "current_mv": 0.0, "neurons": 0}
    resolved = service.roles.resolve(role)
    service.brain.set_input_drive(resolved.indices, current)
    service._drive_role = role
    return {"role": role, "current_mv": current, "neurons": len(resolved)}


@app.get("/api/frame")
async def get_frame() -> dict:
    return service._frame or {}


@app.get("/api/loop")
async def get_loop() -> dict:
    """Current state of the temperature -> colour control loop."""
    if service.loop is None:
        return {
            "available": False,
            "paused": bool(service.paused),
            "reason": (
                "no trained colour readout yet - run "
                "`.venv/bin/python -m flybrain.experiment` to build one"
            ),
        }
    return {"available": True, "paused": bool(service.paused), **service.loop.snapshot()}


@app.get("/api/jev/status")
async def get_jev_status(refresh: bool = False) -> dict:
    """Whether the Jev judgment layer can be used, and if not, *why*.

    The dashboard shows this beside the trust badges, because "Jev is off" and "Jev is broken"
    look identical in a log and identical in a UI that only renders a boolean. The endpoint
    keeps six reasons apart: ``disabled`` (switched off), ``no_key`` (nothing configured),
    ``unprobed`` (nobody has asked yet), ``unauthorized`` (a key was sent and refused),
    ``unreachable`` (no network), and ``ok``.

    ``refresh=true`` re-probes instead of using the cached answer. **The probe is not free**: this
    host has no ``/v1/models`` to call, so it spends one trivial *question* — which is why the
    result is cached and why ``/api/status`` passes ``probe=False`` rather than paying for a badge
    on a timer.
    """
    return await service.jev_status(refresh=refresh)


@app.post("/api/jev/enabled")
async def post_jev_enabled(body: dict) -> dict:
    """Turn the judgment layer on or off for this process.

    Runtime-only, like the dry-run switch: ``JEV_ENABLED`` in ``.env`` is the durable setting and
    this exists so the spend can be stopped without editing a file and restarting. Switching *on*
    re-probes, because the point of turning it on is to find out whether it works; switching off
    needs no round trip at all.
    """
    value = body.get("value")
    if not isinstance(value, bool):
        raise HTTPException(400, "value must be true or false")
    service.jev_enabled_override = value
    return await service.jev_status(refresh=value, probe=value)


@app.post("/api/jev")
async def post_jev(body: dict) -> dict:
    """Classify one recorded decision into a failure mode — placement A.

    A ``disabled``/``no_key`` layer answers **200** with ``available: false`` rather than an
    error: "Jev is off" and "Jev is broken" have to look different, and an HTTP failure would
    render both as the same red message. A genuine failure to get an answer is a 502, because
    that one *is* an error.
    """
    from flybrain.jev import JevError

    seq = body.get("seq")
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise HTTPException(400, "seq must be an integer")
    try:
        return await service.jev_verdict(seq)
    except KeyError:
        raise HTTPException(404, f"no decision recorded with seq {seq}") from None
    except JevError as exc:
        # Nothing cached, deliberately: a transient failure must not be remembered as an answer.
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/status")
async def get_status() -> dict:
    """The pet, the journal, the three layers, the pacing state and the trust badges.

    One endpoint rather than five, because the dashboard polls it at about 1 Hz and five round
    trips would be five chances to render a panel from a different moment than the one next to it.
    """
    return await service.status()


@app.get("/api/timeline")
async def get_timeline() -> dict:
    """The memory trail: state changes, bursts, decisions, labels and actions, newest first.

    Polled far more slowly than ``/api/status`` — the trail only changes when something happens,
    which is the point of it.
    """
    return service.timeline()


@app.post("/api/pause")
async def post_pause(payload: dict | None = None) -> dict:
    """Pause or resume the brain.

    Exposed over REST as well as the websocket so it is scriptable: a phone shortcut, a cron
    job, or an automation can stop the GPU without opening the dashboard. Pausing releases the
    allocator's cached GPU blocks; the connectome itself stays resident.
    """
    body = payload or {}
    return {"paused": service.set_paused(bool(body.get("value", True)))}


@app.post("/api/loop/settings")
async def post_loop_settings(patch: dict) -> dict:
    """Update the loop's connection settings: sensitivity, limits, entities, dry run."""
    if service.loop is None:
        raise HTTPException(503, "the control loop is not running")
    try:
        service.loop.update_settings(patch)
    except KeyError as exc:
        raise HTTPException(400, f"unknown or unsettable setting: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"bad value in {patch!r}") from exc
    snapshot = service.loop.snapshot()
    await service.broadcast_json({"type": "loop", "loop": snapshot})
    return snapshot


@app.post("/api/loop/fit-range")
async def post_fit_range() -> dict:
    """Fit the sensitivity span to the readings this room has actually produced."""
    if service.loop is None:
        raise HTTPException(503, "the control loop is not running")
    service.loop.fit_range_to_observed()
    snapshot = service.loop.snapshot()
    await service.broadcast_json({"type": "loop", "loop": snapshot})
    return snapshot


async def _apply_dashboard_settings(ws: WebSocket, msg: dict) -> None:
    """Apply a settings patch from the dashboard, reporting a bad one instead of raising.

    Extracted from the receive loop so the property that actually matters — a malformed message
    is *reported* and the socket survives it — is testable without a live server. Letting the
    error escape used to close the connection, and this socket is the dashboard's only view of
    the brain: one bad number was a self-inflicted outage in which the picture simply froze.

    Validation happens inside :meth:`ActivitySettings.apply_patch`, which commits all of a patch
    or none of it, so a rejected message cannot leave the view half-reconfigured either.
    """
    try:
        service.settings.apply_patch(msg.get("settings") or {})
    except (KeyError, ValueError) as exc:
        logger.warning("rejected dashboard settings patch: %s", exc)
        await ws.send_text(json.dumps({"type": "error", "error": str(exc)}))
        return
    await service.broadcast_json({"type": "settings", "settings": service.settings.to_dict()})


@app.websocket("/ws/activity")
async def ws_activity(ws: WebSocket) -> None:
    await ws.accept()
    await service.register(ws)
    try:
        if service.sim is not None:
            hello = {
                "type": "hello",
                "n_neurons": int(service.sim.n_neurons),
                "dt_ms": service.sim.params.dt_ms,
                "settings": service.settings.to_dict(),
                "roles": {k: int(v) for k, v in service.roles.summary().items()},
            }
            if service.loop is not None:
                hello["loop"] = service.loop.snapshot()
            await ws.send_text(json.dumps(hello))
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = msg.get("type")
            if kind == "pause":
                service.set_paused(bool(msg.get("value", True)))
            elif kind == "settings":
                await _apply_dashboard_settings(ws, msg)
            elif kind == "drive":
                await post_drive(msg)
            elif kind == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("websocket failed")
    finally:
        service.unregister(ws)


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main() -> None:
    import uvicorn

    # Before anything reads configuration, so `.env` drives the whole process without the
    # operator having to `source` it first. Real environment variables still take precedence.
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
