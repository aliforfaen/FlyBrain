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
``GET  /api/regions``         per-cell-class usage snapshot
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
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from flybrain.activity import ActivitySettings, MemoryBrain
from flybrain.env import load_dotenv
from flybrain.mapping import RoleResolver, default_output_roles, default_sensor_roles

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
            entry = await self.loop.decide(counts, window_ms)
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
        # Tell the pacer a decision actually happened. This is what ends a burst's immediate
        # steps and restarts the heartbeat, so it must happen only after a real decode - not
        # when a window is merely accumulated, and not when the decision raised.
        self.loop.pacer.note_step(time.monotonic())
        if entry is not None:
            await self.broadcast_json({"type": "loop", "loop": self.loop.snapshot()})
        await self.begin_window()

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

    async def jev_status(self, *, refresh: bool = False) -> dict:
        """Report the judgment layer's availability, building the client on first use."""
        from flybrain.jev import JevClient, JevConfig

        if self._jev is None:
            self._jev = JevClient(JevConfig.from_env())
        status = await self._jev.available(refresh=refresh)
        payload = status.to_dict()
        # The redacted config, so an operator can see *which* endpoint and model are configured
        # without the key ever leaving the process.
        payload["config"] = self._jev.config.redacted()
        payload["calls"] = len(self._jev.calls)
        return payload

    async def aclose(self) -> None:
        """Release the Jev client's connection pool, if it was ever built."""
        if self._jev is not None:
            await self._jev.aclose()
            self._jev = None

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


@app.get("/api/regions")
async def get_regions() -> dict:
    if service.brain is None:
        raise HTTPException(503, "brain still loading")
    return {"regions": await asyncio.to_thread(service.brain.region_usage)}


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

    There is no dashboard panel for this yet — the feature has no placement built, and a panel
    would imply otherwise. The endpoint exists so the credential surface is observable from
    ``curl`` and so "off", "misconfigured" and "unreachable" are distinguishable without reading
    a log: with no key it says ``no_key``; with a key the server rejects it says ``unauthorized``
    (which is what a revoked or mistyped key looks like); with no network it says ``unreachable``.

    ``refresh=true`` re-probes instead of using the cached answer. The probe is a real request, so
    it is cached — but it costs no input tokens, so a dashboard may call it freely.
    """
    return await service.jev_status(refresh=refresh)


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
