# The live brain view

A local dashboard showing the connectome running: a 3D point cloud of all 138,639 neurons
coloured by live activity, plus panels for activity, usage and settings.

```
.venv/bin/python -m flybrain.server
# open http://127.0.0.1:8765/
```

## The one decision that makes it work

**The browser renders, Python does not paint.** Pushing per-neuron values through a
Python-serialised channel (Bokeh/Panel model sync, Streamlit reruns, Dash props, socket.io
outboxes) is the bottleneck in every comparable project — not the GPU, and not the browser.

So:

- Neuron positions are uploaded to the GPU **once** as a single `THREE.Points` geometry.
- Per frame we send exactly **one `uint8` per neuron** — 139 KB — over a **binary** WebSocket.
- A **GLSL fragment shader** maps intensity to colour on the GPU.

Sending the same data as JSON would be roughly 1 MB per frame plus a 139,000-element
`JSON.parse`, and would cap the dashboard far below its frame budget. Sending per-point RGB
would be 1.7 MB per frame instead of 139 KB.

This mirrors what the research found: the only projects that manage a live fly-brain view at
scale keep a 1-channel quantised intensity buffer and colour it in a shader.

## Data flow

```
ConnectomeSim.step()  ──►  MemoryBrain.advance()  ──►  Frame(intensity uint8[138639])
                                   │
                                   ├── quantize: clip(counts*gain/saturation)^gamma * 255
                                   │
                                   └── server.encode_frame()  ──►  binary WS frame
                                                                        │
                                                        web/app.js ──► THREE.Points
                                                        (ShaderMaterial colormap)
```

`MemoryBrain` lives in `flybrain/activity.py` and wraps the simulator so the dashboard never
touches simulation internals. It exposes exactly what a view needs — a dense intensity buffer,
a sparse list of the most active neurons, and aggregate usage — which is also the seam the fast
engine will slot into later without the frontend changing.

## Wire protocol

`WS /ws/activity`. On connect the server sends one **text** JSON frame:

```json
{"type":"hello","n_neurons":138639,"dt_ms":0.1,"settings":{...},"roles":{...}}
```

Then **binary** frames at `settings.fps` (default 20 Hz):

| Offset | Type | Field |
|---|---|---|
| 0 | uint32 LE | magic `0x4642524E` ("FBRN") |
| 4 | uint32 LE | sequence number |
| 8 | uint32 LE | n_neurons |
| 12 | float32 LE | simulated brain time, ms |
| 16 | uint32 LE | spikes this window |
| 20 | uint32 LE | active neurons this window |
| 24 | uint8 × n | intensity per neuron, 0–255 |

Total for the full connectome: **24 + 138,639 = 138,663 bytes** per frame.

Clients send text JSON: `{"type":"pause","value":true}`, `{"type":"settings","settings":{...}}`,
`{"type":"drive","role":"thermosensory","current_mv":8.0}`, `{"type":"ping"}`.

## HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/config` | connectome metadata, role sizes, settings, wire description |
| `GET /api/positions` | raw `Float32Array`, `n*3` — soma positions in normalised units |
| `GET /api/regions` | per-cell-class usage: neurons, spikes, rate |
| `GET`/`POST /api/settings` | read / patch the live settings |
| `POST /api/drive` | set or clear a persistent drive on a named role |
| `GET /api/frame` | last frame summary |

## Settings that are live-tunable

| Setting | Meaning |
|---|---|
| `gain` | multiplier on spike counts before display quantisation |
| `saturation` | spike count that maps to full brightness |
| `gamma` | <1 lifts dim activity into view |
| `window_ms` | brain time advanced per published frame |
| `fps` | frames published per second |
| `sparse_limit` | how many neurons go into the sparse "recent spikes" list |
| `background_drive_mv`, `background_fraction` | optional random background drive |

The drive control in the UI is the interesting one: point it at `thermosensory` and the fly's
real 29 hot/cold neurons light up, and you can watch activity propagate through the connectome
from there.

## Why the view is not a real-time brain

`ConnectomeSim` runs the full 138,639-neuron network at about **0.13× realtime** on the RTX 3070
(≈7 s of wall time per second of brain time). It cannot be driven at 20 Hz in real time.

The dashboard handles this honestly rather than pretending: it advances a fixed window of brain
time per published frame and displays the result at a comfortable rate, reporting both
`sim_ms` (brain time) and wall-clock FPS, so the ratio is always visible. Brain time and wall
time are deliberately decoupled.

The fix for actual real-time throughput is the **active-set integrator** — integrate only the
neurons that are active rather than sweeping all 138k every 0.1 ms step. See
[research/simulation-backends.md](research/simulation-backends.md).

## Frontend files

```
web/index.html          page shell, import map, styles
web/app.js              WebGL renderer, WebSocket client, panels
web/vendor/three/       three.js 0.186.0 (MIT), vendored locally — no CDN, no bundler
```

three.js is vendored deliberately: this is a local, offline-capable dashboard, and a CDN
dependency would break that.

## Verified end-to-end (2026-09-21)

Checked against the running server, not by inspection:

- `GET /api/config` reports `wire.header_bytes = 24` and `wire.magic = 1178751566`
  (`0x4642524E`). A binary frame is `24 + 138,639 = 138,663` bytes, confirmed on the wire.
- The client reads both values from `/api/config` rather than hard-coding them, so the
  header size and magic cannot silently drift out of sync with the server again.
- `GET /api/positions` returns exactly `n_neurons * 12 = 1,663,668` bytes.
- Pause freezes `seq`; killing the server puts the UI into `reconnecting` with backoff and
  it recovers on restart without a page reload.
- The 3D view was inspected as a screenshot at 1280x800 and 1920x1080: the point cloud is a
  recognisable fly brain, the panels dock around it without hiding it, and the usage table
  populates.

## Two backend bugs found while building the view

Both were silent and are now fixed, with regression tests in `tests/test_sim.py`.

**A drive could never be cleared.** `MemoryBrain.clear_input_drive()` cleared only its own
copy of the indices, while `ConnectomeSim._persistent_drive` kept re-applying the current on
every step. `POST /api/drive {"role": null}` reported success while the population kept
firing forever, and the only recovery was restarting the server. `clear_input_drive()` now
also calls `sim.clear_drive()`.

**The usage panel was racy.** `/api/regions` read `sim._spike_counts_gpu`, but
`MemoryBrain.advance()` had already replaced that tensor with a fresh zero tensor via
`spike_counts(reset=True)`. The endpoint therefore returned an empty list almost every time
and only occasionally caught data. `MemoryBrain` now caches the window's counts, and the
panel reads the cache.

## Note on defaults: a single spike saturates the colormap

With the default `gain = 8.0` and `saturation = 4.0`, one spike quantises to intensity 255 —
so any active neuron renders at full brightness and the activity view looks uniformly white.
Lower the gain or raise the saturation to see a gradient. The UI log-scales those two
controls for exactly this reason. The network is also **silent without a drive**: nothing
fires until you activate a sensory population or set a background drive.

## The control loop

Two pieces, deliberately separate: `flybrain/experiment.py` **trains** the readout offline, and
`flybrain/loop.py` **runs** it against a live sensor. The loop owns no simulator — it is handed
the same `ConnectomeSim` the dashboard is already stepping, so the picture on screen and the
decision being made come from the same running brain, and there is only one copy of a
15-million-synapse network in memory.

```
HA sensor (°C) ─► rate-coded drive ─► frozen FlyWire v783 ─► spike rates on the
readout population ─► learned ridge readout ─► colour temperature (K) ─► HA action
```

Train, then run:

```bash
.venv/bin/python -m flybrain.experiment    # ~50 s, writes data/experiments/
.venv/bin/python -m flybrain.server        # dashboard + live loop
```

### Training must match the regime the loop runs in

This is the least obvious thing in the whole project, and getting it wrong cost real accuracy.

The readout was originally trained on windows that each began from a **resting** brain. A live
loop never has one: it reads the most recent window of an already-running network. Those are
different states — at these time constants a window that starts from rest is a *transient*,
while a live window is a *steady* driven state — and a linear probe distinguishes them with
**100% accuracy** (`tools/loop_regime_probe.py`).

| trained on | evaluated on | mean abs error |
|---|---|---|
| rest windows | rest windows | 34 K |
| live windows | live windows | 34 K |
| rest windows | **live windows** | 50 K |
| live windows | rest windows | 250 K |

So `TemperatureColourLoop.sweep()` now drives the brain **continuously** through the
temperature range, one window per temperature, exactly as the loop does. The held-out number it
reports is therefore the number the loop will actually achieve, rather than a friendlier one.
It also happens to be cheaper: one continuous sweep costs ~50 s, against ~96 s for the
settle-per-temperature version it replaced.

Offline, `evaluate()` reports held-out temperatures that lie strictly between the training
points: monotone 8/8, correlation 0.9989, **mean error 41 K** (worst 106 K) against a 3800 K
ideal span — the trained range runs from a 2700 K candle-warm to a 6500 K daylight.

But the number that matters is the one the *live loop* achieves, so it is measured directly
(`tools/loop_accuracy.py`): the real `LiveLoop`, the real mock sensor, the brain running
continuously, one full three-minute cycle of the simulated room.

```
decisions 40   mean |err| 31 K   max 104 K   correlation 0.9971
mock service calls sent: 40
```

About 1% of full scale over a range of 11.5–33.5 °C, tracked both on the way up and the way
down — a difference nobody can see on a lamp. The live figure is slightly *better* than the
offline one because a real run visits each temperature with more history behind it.

The error is quoted in Kelvin, which flatters a *wide* range and punishes a narrow one, so
compare it against the span: 41 K on 3800 K and 27 K on 2500 K are the same ~1.1%. Widening
the range costs nothing in relative accuracy and buys a change you can actually see, which is
why the trained range is 2700–6500 K rather than the 4000–6500 K it started as.

> **Do not run the experiment while the dashboard is streaming.** They share the GPU: with a
> browser attached, training took **214 s** instead of 50 s for byte-identical results. The
> figures above are uncontended.

**Four mistakes worth recording**, because each one produced a plausible-looking but wrong
result rather than an obvious crash:

1. `current_per_spike_mv` was far too small. At 0.9 mV the drive never crossed the 7 mV gap to
   threshold, so nothing spiked anywhere and the network was silent.
2. The readout collapsed the population into one scalar per colour band, using a *distinct*
   slice of the pool for each. That fixed an earlier "all bands see the same neurons" bug, but
   it is still the wrong shape: three random slices of one population report almost exactly the
   same thing (r = 0.996–1.000), so the decoder had one effective dimension. The readout now
   takes the firing rate of **every** neuron in the pool and regresses the Kelvin value
   directly. On a held-out split that moved the achievable error from ~155 K to ~20 K.
3. The sensory rate ran from **0 Hz** at the cold end, so the coldest temperature injected no
   current at all and the readout population was completely silent — the solver had literally
   nothing to read, and could only fall back on its bias. Real thermoreceptors have spontaneous
   activity, so the encoder now runs 20–120 Hz.
4. `evaluate()` reused the training temperatures, so its reported error was in-sample fit
   quality presented as accuracy. It now evaluates strictly between training points. The
   *previous* 531 K figure was measured this way and was in fact an optimistic number.

Design detail worth knowing: the readout listens to the antennal-lobe populations
(`antennial_projection`, `antennial_local`) rather than descending/motor neurons, because a
sensory drive cannot reach the latter — a strong drive produced ~23 extra spikes across all
1,299 descending neurons (see `research/simulation-backends.md`). That is a compromise forced by
measurement, not the biologically ideal choice.

## What the dashboard shows

The dashboard was rewritten around the one thing it is for: **a reading goes in, a colour comes
out**. Controls that only changed how the picture was drawn were removed rather than tidied.

Removed: point size, exposure and floor-lift sliders; the colour-mode picker; the manual
sensory-drive panel (the sensor drives the brain now); and the raw simulation-settings panel,
which exposed dataclass fields as sliders. Kept: drag/scroll camera, **Pause** (it shows its own
state) and **Reset view** (a recovery affordance, not a dial).

The HUD now reads:

| Panel | What it is for |
|---|---|
| **Room → light colour** | The headline. Sensor reading in °C, the colour in Kelvin, a swatch painted in the actual light colour, the ideal for comparison, and the exact Home Assistant call |
| **Colour chosen** | An **X-Y plot**: room temperature on x, chosen colour on y, one point per decision, with the perfect mapping as a dashed line. Overlapping two time series would have hidden the very thing worth seeing — the mapping itself |
| **Light connection** | The settings that adapt the loop to a real home — see below |
| **Brain activity** | Plain language first ("the brain is busy"), then active neurons, spikes per window, simulated time, frames |
| **Which regions are busy** | Top 12 cell classes by spikes, for looking under the hood |
| **The brain itself** | The 3D point cloud, plus pause and reset |

### Sensitivity: the setting that makes it work in a real house

The readout was trained across a 10–35 °C sweep. **A living room does not move 23 degrees.** Left
alone, the loop would see a two-degree room swing, produce a two-degree-sized colour change, and
look broken.

The *Light connection* panel therefore maps *your* sensor's real span onto the trained span:

| Control | What it does |
|---|---|
| **Sensitivity** | The room range that should use the whole colour range. Set it to 18–24 °C and a six-degree room drives 2700–6500 K. Shows the resulting **K per °C** live |
| **Colour range sent** | Clamps the output, with a preview strip and a marker at the current colour |
| **Smoothing** | Exponential averaging on the sensor, in seconds. Real sensors jitter; this stops the light twitching |
| **Ignore changes under** | A deadband in Kelvin. Gates only the *service call* — the chart still shows every decision, so the throttle never hides the behaviour |
| **Reverse direction** | For a sensor that reads high when you want the light cool (an outdoor probe, an AC return) |
| **Dry run** | Never touch a real device. Mock mode disables this, since the mock is not real |
| **Fit sensitivity to this room** | Uses the min/max the loop has actually observed, so you do not have to guess the span |

Both preview strips are painted in the real light colour via a Kelvin→sRGB conversion, so the
panel shows the consequence of a setting before you commit to it.

One UI naming bug is worth recording because it was invisible in code review and obvious on
screen: the colour bands were named for the *room* (`cool` for the cold end) and then displayed
beside a colour swatch, so a light near 4000 K was labelled "cool white" while glowing orange. Lighting
convention is the opposite — low Kelvin looks warm. The bands are now named for the *light*, and
the dashboard derives its plain description from the Kelvin value itself so the words and the
swatch cannot disagree.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `HA_MODE` | `mock` | `mock` uses the in-process home simulator; `rest` talks to a real instance |
| `HA_DRY_RUN` | `1` | When true, a **real** Home Assistant is never called — the loop records what it *would* send. The mock is not a real device, so mock mode always applies its calls and the simulated light genuinely changes |
| `HA_BASE_URL`, `HA_TOKEN` | — | Required for `rest`. The token is a long-lived access token |
| `FLYBRAIN_TEMPERATURE_ENTITY` | `sensor.living_room_temperature` | **Must be set for a real house** — the default names the *mock* home |
| `FLYBRAIN_LIGHT_ENTITY` | `light.kitchen` | Target light; must actually support the colour command (see below) |
| `FLYBRAIN_SOURCE_MIN_C` / `_MAX_C` | `10` / `35` | Sensor span stretched onto the full trained colour range |
| `FLYBRAIN_INVERT` | `0` | Flip the direction for a sensor that reads high when you want it cool |
| `FLYBRAIN_SMOOTH_MS` | `5000` | Exponential smoothing on the reading |
| `FLYBRAIN_KELVIN_MIN` / `_MAX` | `2700` / `6500` | Clamp on the emitted colour; must sit inside the trained range |
| `FLYBRAIN_DEADBAND_K` | `0` | Send nothing until the colour has moved this far |
| `FLYBRAIN_ALWAYS_ON` | `0` | Step the brain (and therefore run the loop) even with no dashboard watching |
| `FLYBRAIN_RECORD` | `0` | Record every completed window to `data/recordings/` |
| `FLYBRAIN_RECORD_NAME` | timestamped | Name for that recording |
| `FLYBRAIN_CHANNELS` | `off` | `auto` discovers and drives every sensory pathway the house has, not just temperature. See the caveat below |
| `FLYBRAIN_INTERVAL_S` | `0` | Wall-clock seconds between decisions. `0` runs flat out (~165 W); `15` averages ~40 W |

All of it can also be changed at runtime from the dashboard, which overrides the environment.

### Extra sensory channels

`FLYBRAIN_CHANNELS=auto` turns on every pathway [discovery](wiring.md) finds — motion, illuminance,
humidity, whatever the house has — instead of driving the brain from the thermometer alone. The
dashboard's *brain activity* panel shows each channel, its neuron count and the rate it is
currently applying.

**The caveat is real:** the colour readout was fitted with *temperature only* driving the brain,
and a readout is a function of the whole reservoir state. Extra drive changes that state. Measured
against the live house with both channels sitting at their floor, it made no difference (−77…−141 K
off vs −65…−112 K on) — but motion while detected and a lit room were **not** exercised. Treat the
readout as unvalidated under real motion until it is re-fitted on recorded data. Details and
numbers: [`wiring.md`](wiring.md#what-it-does-to-the-colour-readout-measured).

The default is *off* for `FLYBRAIN_ALWAYS_ON` because the simulator runs at 0.13× realtime, so
keeping it turning permanently is a real cost. The consequence is worth stating plainly: **with
the default settings the control loop only advances while the dashboard is open.** Set
`FLYBRAIN_ALWAYS_ON=1` when it is meant to control something for real.

### Pointing it at a real instance

Copy [`.env.example`](../.env.example) to `.env` (gitignored) and edit it:

```bash
set -a; source .env; set +a
.venv/bin/python -m flybrain.server
```

Two traps, both of which fail *quietly* rather than loudly:

1. **The entity defaults are the mock home's.** `sensor.living_room_temperature` does not exist
   in a real instance, so the loop reads nothing and simply sits idle. Set
   `FLYBRAIN_TEMPERATURE_ENTITY`.
2. **Not every light accepts `color_temp_kelvin`.** Real installations mix `onoff` and
   `color_temp` lamps; sending a colour temperature to an `onoff`-only light is rejected. Check
   `supported_color_modes` before choosing a target — see
   [`ha-inventory.md`](ha-inventory.md#3-most-lights-cannot-accept-the-command-the-loop-sends).

## Recording windows

`FLYBRAIN_RECORD=1` writes every completed window to `data/recordings/<name>/`:

| File | Contents |
|---|---|
| `features.f32` | raw `float32`, one row of readout-population firing rates per window |
| `windows.jsonl` | one `{"t":…, "sensors":{…}}` per window — **every** entity the house reported |
| `labels.jsonl` | timestamped labels |
| `meta.json` | `feature_dim`, `window_ms`, the entities and mode in use |

Features are raw `float32` appended to one file rather than a database or a compressed archive,
specifically so that a power cut costs the last window and nothing else. Every write is flushed.

Labelling a moment needs no Python:

```bash
.venv/bin/python -m flybrain.recorder --label busy      # label the latest recording, now
.venv/bin/python -m flybrain.recorder --list            # what recordings exist
.venv/bin/python -m flybrain.recorder --summary         # windows, labels, times
```

Labels are recorded as **events with timestamps**, not as per-window fields, because a label
describes a moment and it is the surrounding windows that a readout learns from. `Recording.labelled()`
later joins them: each window takes the most recent preceding label, and windows further away than
a horizon are dropped rather than guessed at. That join is how "teaching by demonstration" works.

Recording is opt-in and its failure is never fatal — a full disk or a bad name disables recording
and logs once, rather than stopping the brain.

## What it costs to leave running

Measured on the RTX 3070 in this machine, sampling `nvidia-smi` power draw once a second.

| Mode | GPU util | Mean power | Share attributable to the fly |
|---|---|---|---|
| Flat out (`interval_s=0`) | 37–81% | **~165 W** | ~146 W |
| A decision every 5 s | bursts | ~86 W | ~67 W |
| A decision every 15 s | bursts | **~40 W** | ~21 W |
| Paused | idle | **~19 W** | ~0 |

CPU is a non-issue throughout: **0.1%** of one core, live or paused.

### Why flat out is the default, and why you should change it

The loop advances brain time as fast as the GPU allows, so it holds the card at **165 W
continuously**. That is right for watching the demo and wrong for a house: nothing in a room
changes meaningfully in two seconds.

The arithmetic is simple, and worth stating because the window size is what makes it
non-obvious. One decision consumes a **300 ms window of brain time**, which at ~0.78 ms per
0.1 ms step costs about **2.3 s of GPU work**. So:

```
duty cycle      = 2.34 s / interval_s
mean power      ≈ 19 W + duty × 146 W
```

Predicted 41.8 W at a 15 s interval; measured **39.8 W**. The model holds, which is why the
dashboard can show the cost next to the control.

`interval_s` is adjustable live from the dashboard (*Light connection* → *How often it
decides*), which displays the estimated wattage as you change it.

### Pause

Pause is the one control that really stops things, and it does so completely:
**165 W → 19 W**, with CPU at zero. It is reachable three ways, all going through the same
code path:

- the **Pause** button in the dashboard, or the **space bar**
- the websocket (`{"type": "pause", "value": true}`)
- `POST /api/pause` — so a phone shortcut, cron job or HA automation can stop the GPU without
  opening a browser

Pausing also calls `torch.cuda.empty_cache()` to hand the allocator's unused blocks back to the
driver. It does **not** unload the connectome: VRAM stays at ~1.8 GB because the weights stay
resident. Freeing them would mean re-reading the connectome from disk on resume, which costs far
more than the memory is worth.

### Notes for anyone touching this

- With pacing on, frames are *supposed* to be absent between decisions. The dashboard's
  staleness indicator accounts for this and reads **waiting** rather than crying "stalled" at
  configured behaviour — a regression that appeared the moment pacing was added.
- The loop already does no GPU work at all when paused or when no client is connected and
  `FLYBRAIN_ALWAYS_ON` is unset.
- Pacing also dilates brain time relative to wall clock: 300 ms of brain time per 15 s is
  **0.02× realtime** rather than 0.13×. Fine for context, and the same knob discussed in
  [`roadmap.md`](roadmap.md#6-two-honest-constraints).

## Ideas for the viewer that are not built

Recorded so they do not have to be re-derived. Roughly in order of value per unit of work.

- **Drive the light from something with more range than a room.** Colour *temperature* only
  spans white — amber to daylight. A hue light (or a set of lights) driven by the same readout
  would make the brain's behaviour far more legible, and is a one-line change to the action
  payload.
- **A second sensor — but not humidity.** The encoder already speaks humidity and drives real
  hygrosensory neurons, which makes it look like a one-line win. It is not: the live instance
  reports **no humidity sensor at all** ([`ha-inventory.md`](ha-inventory.md)). What it does have,
  on the same device as the thermometer, is `binary_sensor.hallway_motion` and
  `sensor.hallway_illuminance`. Illuminance maps to the fly's `visual` population, its largest;
  motion gives the readout a genuinely temporal question to answer.
- **Show the readout's weights.** The loop is a linear map over 512 neurons; drawing those 512
  coefficients as a bar strip beside the brain would show *which* neurons the colour actually
  depends on. It is the most direct answer to "is the brain doing something sensible?".
- **A "why" trace for one decision.** Record the per-neuron rates behind the current colour and
  let the user click a decision in the history to replay it. Turns the dashboard from a
  readout into a debugger.
- **Real-time engine.** Everything above is cheap; this is not. 0.13× realtime means a decision
  every ~2.3 s, which is fine for a thermostat and hopeless for anything that reacts. The
  active-set integrator is the documented route (see `engine.md`).
- **A time-of-day prior.** A fly brain has circadian clock neurons, and as of the `clock` role fix
  they resolve correctly (48 cells — see [`roadmap.md`](roadmap.md#resolved-the-circadian-clock-role-was-unreachable)).
  Driving them from `sun.sun` elevation would be a cheap contextual input. It is not free, though:
  it changes the input distribution, so the readout has to be re-fitted before it can be trusted.
