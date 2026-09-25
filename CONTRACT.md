# flybrain-ha — interface contract

Frozen interfaces for the connectome-driven Home Assistant controller.
Every module imports its types from `flybrain.types`. **Do not redefine them locally.**

> **Status note.** This file was written before implementation as a coordination contract between
> parallel workstreams. The modules are now built; where an implementation deliberately diverged,
> the note is inline below. For the current architecture read `docs/architecture.md`; for the
> live-view wire protocol read `docs/live-view.md`.

Repo root: the directory containing this file
Python: `.venv/bin/python` (CPython 3.11, managed by `uv`). Never use system `python3.14`.

## The core idea

A real *Drosophila* connectome (FlyWire v783, ~138,639 neurons, ~15M synapses) is run as a
LIF spiking network. It is **frozen** — we never train the brain. Home Assistant sensors are
encoded into spikes injected into chosen neuron populations; spikes from chosen output
populations are decoded into Home Assistant service calls. Only the small decode (and
optionally encode) weights are learned.

```
HA sensors ──► SignalBus ──► SpikeEncoder ──► injected current ──┐
                                                                │
                                              FlyWire LIF network (frozen)
                                                                │
HA services ◄── ActionBus ◄── SpikeDecoder ◄── output spike counts┘
                                     ▲
                                     └── ReadoutLearner (ridge / delta rule)
```

## Types (flybrain/types.py)

```python
class SignalKind(str, Enum):
    TEMPERATURE = "temperature"; HUMIDITY = "humidity"; ILLUMINANCE = "illuminance"
    MOTION = "motion"; CONTACT = "contact"; POWER = "power"; OTHER = "other"

@dataclass
class Signal:
    entity_id: str          # "sensor.living_room_temperature"
    kind: SignalKind
    value: float            # numeric, already unit-converted
    unit: str               # "°C", "%", "lx", ...
    state: str              # raw HA state string
    timestamp: float        # unix seconds
    attributes: dict = field(default_factory=dict)

@dataclass
class Action:
    entity_id: str          # "light.kitchen"
    service: str            # "turn_on" | "turn_off" | "toggle" | "set_temperature"
    confidence: float = 1.0
    data: dict = field(default_factory=dict)   # e.g. {"brightness_pct": 80}

@dataclass
class SpikeTrain:          # one sensor channel, over one integration window
    neuron_indices: np.ndarray   # int32, connectome indices that were driven
    spike_times_ms: np.ndarray   # float32, ascending, ms
    rate_hz: np.ndarray          # float32, per-neuron instantaneous rate

@dataclass
class BrainCommand:        # decoder output for one tick
    actions: list[Action]
    spike_counts: dict[int, int]    # connectome index -> spike count in window
    logits: dict[str, float]        # "<entity_id>.<service>" -> score
    window_ms: float
```

## Module boundaries

### `flybrain/ha.py` — MUST be written by one owner
- `class HomeAssistant(Protocol)`: `async get_states() -> list[Signal]`,
  `async call_service(entity_id, service, data=None) -> bool`, `async aclose()`.
- `class MockHomeAssistant`: in-process sim (no network) implementing the protocol,
  driven by a scenario (temperature ramp, motion events, light power feedback).
- `class RestHomeAssistant`: talks to a real or mock HA over HTTP
  (`GET /api/states`, `POST /api/services/<domain>/<service>`) using a long-lived token.
- Config comes from env: `HA_BASE_URL`, `HA_TOKEN`, `HA_MODE` (`mock`|`rest`).

### `flybrain/sim.py` — connectome simulator (controlled by the simulator owner)
- `class ConnectomeSim`: loads the v783 connectome, builds a sparse weight COO,
  steps LIF dynamics matching Shiu et al. (Nature 2024).
- Public surface:
  - `ConnectomeSim.load(connectivity_path, completeness_path, dt_ms=0.1, device="cuda")`
  - `sim.inject(indices, current_mv)` — add external current into `voltage_stim` for the
    *next* step only, or `sim.set_drive(indices, current_mv)` for a persistent drive.
  - `sim.step(n_steps)` / `sim.run(duration_ms)`
  - `sim.spike_counts(indices, reset=True) -> np.ndarray` — cumulative spike counts since
    last reset, for the decoder.
  - `sim.index_of_flywire_id(id)`, `sim.annotation_table` (DataFrame indexed by connectome index).

### `flybrain/codec.py` — `SpikeEncoder` / `SpikeDecoder` (owned by codec owner)
- `SpikeEncoder(channel_map, dt_ms)`: `encode(signals: list[Signal], duration_ms) -> list[SpikeTrain]`
  Rate coding: Gaussian tuning curves over each entity's configured range, Poisson sampling.
  Deterministic given a seed.
- `SpikeDecoder(output_map, window_ms)`: `decode(spike_counts: dict[int,int]) -> BrainCommand`
  Linear readout: `score = W @ features + b`, features = normalized output-population rates
  plus a bias. Threshold -> Action list. Weights persistable to `.npz`.

### `flybrain/learn.py` — `ReadoutLearner` (owned by codec owner)
- Collect `(features, target)` pairs from episodes, then `fit(method="ridge"|"delta") -> np.ndarray`
- `save(path)` / `load(path)`.

### `flybrain/mapping.py` — connectome cell selection (owned by simulator owner)
Resolves logical roles (`thermosensory_in`, `motor_out`, ...) to connectome indices using the
codex v783 annotation table, with a curated fallback list and a cached JSON artifact.

### `flybrain/scenario.py` / `flybrain/experiment.py` — `BehavioralExperiment` (owned by integrator)
Wires encoder -> sim -> decoder -> HA, runs the "when this sensor goes, turn on this light"
training loop, reports accuracy.

## Data
- `vendor/fly-brain/data/2025_Connectivity_783.parquet` — columns:
  `Presynaptic_ID, Postsynaptic_ID, Presynaptic_Index, Postsynaptic_Index, Connectivity, Excitatory, Excitatory x Connectivity`
- `vendor/fly-brain/data/2025_Completeness_783.csv` — `Unnamed: 0` (FlyWire root id) → row index
  == connectome index. 138,639 rows.
- Weight convention: `w_mV = 0.275 * Excitatory x Connectivity` (negative ⇒ inhibitory).
- Reference params: dt=0.1ms, v0=v_reset=v_rest=-52mV, v_th=-45mV, tau_mem=20ms, tau_syn=5ms,
  t_delay=1.8ms, t_refrac=2.2ms, w_scale=0.275mV, poisson_scale=250.

## Rules
1. Import shared types from `flybrain.types`; never duplicate them.
2. No network calls in unit tests. `MockHomeAssistant` is in-process.
3. Everything must run headless: `uv run python -m ...` or `.venv/bin/python`.
4. Type hints everywhere; `ruff` clean at line-length 100.
5. Do not modify files outside your assigned module list.

## Wire protocol for the live view (added after implementation)

The dashboard protocol is specified in **`docs/live-view.md`** and implemented in
`flybrain/server.py` + `web/app.js`. Two details drifted from the original draft contract in
this file and are now authoritative in the code:

- The binary frame header is **24 bytes** (`<IIIfII`), not 22. Total frame size is
  `24 + n_neurons = 138,663` bytes for the full connectome.
- The client should read `magic`, `header_bytes` and the neuron count from `GET /api/config`
  rather than hard-coding them; `web/app.js` does this and keeps fallback constants only for
  the window before config arrives.

`GET /api/config` is the source of truth for `wire.header_bytes` and `wire.magic`.
