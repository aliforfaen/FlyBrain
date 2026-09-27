# AGENTS.md

Orientation for anyone (or anything) picking this up cold. **This file is deliberately
short — the detail lives in [`docs/`](docs/).** Read this, then jump.

## What this is

A whole fruit-fly brain — the FlyWire v783 connectome, 138,639 neurons and 15,091,983
synapses — running as a spiking neural network on a GPU, wired to a Home Assistant
temperature sensor so that a room reading becomes a light colour.

The brain is **frozen and never learns**. Only a small linear readout on top of it is
trained. Nothing in a fly brain knows what a kitchen light is; that mapping is ours, and
the code says so.

```
HA sensor (°C) → rate-coded drive → frozen connectome → spike rates on a readout
population → learned ridge readout → colour temperature (K) → HA light.turn_on
```

## Run it

```bash
uv sync                                        # Python 3.11 → .venv (torch + CUDA wheels)
.venv/bin/python tools/fetch_data.py           # connectome + annotations, ~140 MB, one time
.venv/bin/python -m pytest tests/ -q           # 344 tests, no connectome needed

.venv/bin/python -m flybrain.experiment        # train the readout, ~50 s
.venv/bin/python -m flybrain.server            # dashboard + live loop
.venv/bin/python -m flybrain.wiring            # propose a pathway map for the house
# → http://127.0.0.1:8765/
```

**The large tables are not committed.** `tools/fetch_data.py` downloads the FlyWire v783
connectivity, neuron list, cell-type annotations and soma coordinates from their public upstreams
and verifies each against its expected byte count. `--check` reports status without downloading.
`.venv-validation` is separate and built by hand — see Conventions.

The loop runs against a **simulated** home by default, so it works with nothing
configured. Settings live in the dashboard's *Light connection* panel.

To point it at a real house, copy [`.env.example`](.env.example) to `.env` (gitignored) and fill in
your URL, token and entity ids. [`flybrain/env.py`](flybrain/env.py) loads it automatically at
startup; real environment variables still win. `HA_DRY_RUN=1` keeps a real Home Assistant
**read-only** — the loop logs what it would send and changes nothing. Record windows for later
training with `FLYBRAIN_RECORD=1`, and label a moment with
`python -m flybrain.recorder --label busy`.

## The five things worth knowing

1. **`recurrent_scale` is `1.0` and `w_scale_mv` is the published `0.275 mV`.** There is no
   fudge factor. An earlier revision carried a "provisional 0.01" that was cancelling a
   missing `tau_mem` in the membrane update, which made every synapse 20× too strong. Do
   not reintroduce a gain knob without reading
   [`docs/architecture.md`](docs/architecture.md#resolved-the-recurrent-gain-problem-was-an-integrator-bug).

2. **Train and run in the same regime.** The readout is fitted on a *continuous* sweep of
   an already-running brain. A window that starts from rest is a different state (a linear
   probe separates them perfectly), and mixing the two silently costs accuracy. See
   [`docs/live-view.md`](docs/live-view.md#training-must-match-the-regime-the-loop-runs-in).

3. **A ~100× fudge factor is a bug report, not a calibration.** That lesson cost days.
   Verify arithmetic against a closed form on a two-neuron network before touching data:
   `validation/micro_gain_check.py`.

4. **The simulator is validated against Brian2, not assumed correct.** Three networks,
   `Jaccard` 1.000 / 1.000 / 0.964, rate correlation ≥ 0.998. If you change `sim.py`, re-run
   the harness — a green result is only meaningful if the thing measured is doing something.
   A previous "PASS" was degenerate: it agreed while producing zero recurrent spikes.

5. **The engine, not the physics, is the remaining compromise.** 0.13× realtime. The fix is
   an active-set integrator; batching does not help (measured: the GPU is already saturated
   at batch 1). See [`docs/engine.md`](docs/engine.md).

## Layout

| Path | What |
|---|---|
| `flybrain/sim.py` | Connectome as a sparse LIF network. The physics. |
| `flybrain/experiment.py` | Trains the colour readout offline |
| `flybrain/loop.py` | Runs it live: sensor → brain → colour → HA action |
| `flybrain/pacing.py` | When the brain is worth stepping: the heartbeat, the burst, the change trigger |
| `flybrain/wiring.py` | Which HA entity drives which fly sensory pathway |
| `flybrain/recorder.py` | Records every window so new readouts can be trained later |
| `flybrain/server.py` | FastAPI dashboard + the loop's clock |
| `flybrain/ha.py` | Home Assistant adapters (mock and REST) |
| `flybrain/env.py` | Loads `.env`, so no house-specific value is ever hardcoded or hand-sourced |
| `flybrain/jev.py` | The optional Jev judgment layer: typed questions, confidence routing, credential surface. Off unless `TYPESAFE_API_KEY` is set |
| `tools/fetch_data.py` | Downloads and byte-verifies the connectome and annotation tables |
| `web/` | three.js dashboard, no bundler, three.js vendored |
| `validation/` | Brian2 comparison and a closed-form integrator test |
| `tools/` | Probes: readout capacity, loop regime, loop accuracy |
| `docs/` | The real documentation; screenshots in `docs/images/` |
| `vendor/fly-brain/` | Upstream connectome **data only**, fetched not committed (gitignored) |

## Documentation map

- [`docs/architecture.md`](docs/architecture.md) — how it fits together, the equations, the
  integrator bug
- [`docs/live-view.md`](docs/live-view.md) — the dashboard, the control loop, every
  configuration flag
- [`docs/data.md`](docs/data.md) — schemas and provenance
- [`docs/engine.md`](docs/engine.md) — performance and the real-time plan
- [`docs/roadmap.md`](docs/roadmap.md) — what to build next, and what the brain is honestly
  good for
- [`docs/ha-inventory.md`](docs/ha-inventory.md) — the **real** house: which entities exist, and
  which documented ideas they rule out
- [`docs/wiring.md`](docs/wiring.md) — which sensor drives which fly pathway, and why the
  ordering of those rules matters
- [`docs/vision.md`](docs/vision.md) — the camera→visual-column design, **not built**
- [`docs/jev.md`](docs/jev.md) — the Jev judgment layer: three UI placements, their closed
  vocabularies, the four places it must **not** be used, and the credential surface. The client,
  config and confidence routing are **built** (`flybrain/jev.py`); **no placement has a UI yet**,
  and the most valuable content is still the negative recommendations
- [`docs/licensing.md`](docs/licensing.md) — **read before shipping anything**
- [`docs/research/`](docs/research/README.md) — asset inventory and the record of what went
  wrong along the way

## Conventions

- Python 3.11 via `uv`; `ruff check .` and `pytest` must both pass.
- `.venv-validation` exists only because Brian2 does not support NumPy 2.x. Anything needing
  brian2 runs there. It is built by hand, not by `uv sync`:
  `uv venv .venv-validation --python 3.11 && uv pip install --python
  .venv-validation/bin/python "numpy<2" brian2` (numpy 1.26.4 / brian2 2.8.0.4 here).
- Comments explain *why*, especially where the obvious-looking code is wrong. Several of
  the worst bugs here looked completely reasonable.
- Do not run the training loop while the dashboard is streaming — they share the GPU and
  training goes from 50 s to 215 s for identical output.

## Licensing

The connectome data is **CC BY-NC 4.0 — non-commercial**. Personal and homelab use is fine;
do not sell this. Two popular fly-brain projects have no licence at all and must not be
copied. Details in [`docs/licensing.md`](docs/licensing.md).
