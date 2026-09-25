# Architecture

## The idea

Run a real *Drosophila* connectome as a frozen spiking network, wire Home Assistant into it
as the sensory and motor surface, and learn a small readout so the brain can do things like
"when this sensor goes, turn on this light".

The fly brain is **not** trained. It is used as a fixed, richly structured dynamical system.
Only two small things are learned or configured:

1. **Encoding** — how a Home Assistant sensor value becomes drive onto a chosen population of
   real sensory neurons.
2. **Decoding** — how spike counts from a chosen population of real output neurons become
   Home Assistant service calls.

```
HA sensors ──► SignalBus ──► SpikeEncoder ──► injected current ──┐
                                                                │
                                          FlyWire v783 LIF network (frozen)
                                                                │
HA services ◄── ActionBus ◄── SpikeDecoder ◄── output spike counts┘
                                     ▲
                                     └── ReadoutLearner (ridge / delta rule)
```

## Why this shape, and not "the brain decides everything"

The most important architectural fact discovered during research: **the connectome gives you
structure and dynamics, not semantics.** Nothing in a fly brain knows what a "kitchen light"
is. The mapping from a home concept to a neuron population is a choice we make, and it is
where the interesting honesty lives:

- A **temperature** sensor drives the fly's genuine thermosensory neurons
  (`TRN_VP1m / TRN_VP2 / TRN_VP3a / TRN_VP3b`) — 29 cells that really do encode hot and cold.
- **Humidity** drives the real hygrosensory neurons (`HRN_VP1d/l/VP4/VP5`).
- **Motion/contact** drives mechanosensory neurons.
- **Output** is *read out* from a population the readout listens to. Measurement, not
  preference, chose it: the fly's descending neurons (1,299, brain→ventral nerve cord) and
  motor neurons are almost unreachable from a sensory drive (~23 extra spikes across all 1,299
  descending neurons), so the working loop reads **antennal-lobe** populations (`ALPN`, `ALLN`)
  instead. Descending and motor neurons remain available as roles
  (`resolve("descending")`, `resolve("motor")`) for experiments that drive the network
  differently.

## Modules

| Module | Responsibility |
|---|---|
| `flybrain/types.py` | Shared dataclasses: `Signal`, `Action`, `SpikeTrain`, `BrainCommand`, `Episode`. |
| `flybrain/sim.py` | `ConnectomeSim` — the connectome as a sparse LIF network, batchable, GPU or CPU. |
| `flybrain/activity.py` | `MemoryBrain` — incremental driver + `ActivitySettings` for the live view. |
| `flybrain/mapping.py` | `RoleResolver` — resolves logical roles (thermosensory, antennal, descending…) to connectome indices. |
| `flybrain/codec.py` | `SpikeEncoder`, `SpikeDecoder` — spike-rate coding in and out. |
| `flybrain/learn.py` | `ReadoutLearner` — ridge regression / delta rule for the decoder. |
| `flybrain/experiment.py` | `TemperatureColourLoop` — the first complete sensor→light loop. |
| `flybrain/ha.py` | Home Assistant client: `MockHomeAssistant`, `RestHomeAssistant`. |
| `flybrain/server.py` | FastAPI live-view server (binary WebSocket + static dashboard). |
| `web/` | three.js dashboard. |

## The control loop, concretely

`flybrain/experiment.py` trains the readout and `flybrain/loop.py` runs it live against a
sensor. The pipeline is the same in both:

```
temperature (°C)
   │  rate coding: warmer ⇒ faster, matching the fly's own thermosensory neurons
   ▼
256 neurons drawn from the real thermosensory + hygrosensory populations
   │  steady-current equivalent of that rate
   ▼
frozen FlyWire v783 network, 300 ms of brain time
   │
   ▼
firing rates of a responsive readout population (antennal-lobe ALPN/ALLN)
   │  ridge regression over 300 ms windows, straight onto the target Kelvin value
   ▼
colour temperature in Kelvin  →  Home Assistant light.turn_on {color_temp_kelvin}
```

Measured on **held-out** temperatures: monotone 8/8, correlation 0.9989, mean error **41 K**
against a 3800 K ideal span (~1% of full scale). See [live-view.md](live-view.md) for the full
table, and for the four design mistakes that had to be fixed to get there — including a readout
whose features were collinear at r = 0.996 and a label scheme that was itself 151 K away from
the target.

**The readout listens to antennal-lobe neurons, not descending neurons.** That is a compromise
forced by measurement: a strong sensory drive produced only ~23 extra spikes across all 1,299
descending neurons, while the antennal-lobe populations respond reliably. It is the least
biologically satisfying part of the design and is called out as such. (This measurement
survived the integrator fix described below — it is about where a sensory signal lands, not
about gain.)

## The simulator

`ConnectomeSim` reproduces the model of Shiu et al., *Nature* 2024, as implemented by the
reference project (`philshiu/Drosophila_brain_model`, MIT) and the benchmark harness at
`eonsystemspbc/fly-brain`.

```
dv/dt = (v0 - v + g) / tau_mem
dg/dt = -g / tau_syn
spike when v > v_th  →  v = v_reset, g = 0, refractory for t_refrac
```

| Parameter | Value |
|---|---|
| `dt` | 0.1 ms |
| `tau_mem` / `tau_syn` | 20 ms / 5 ms |
| `v_rest` / `v_reset` / `v_thresh` | −52 / −52 / −45 mV |
| axonal delay | 1.8 ms |
| refractory | 2.2 ms |
| weight | `0.275 mV × Excitatory × Connectivity` |
| sensory drive | an immediate `v += 0.275 mV × 250 = 68.75 mV` per Poisson event |

Three implementation details are easy to get wrong. All three *were* wrong at some point, and
each was caught by the validation harness in `validation/`:

- **Integration must be exact, not forward Euler.** Brian2's `linear` method solves the coupled
  linear (v, g) system analytically per step. Writing `a = exp(−dt/tau_mem)` and
  `b = exp(−dt/tau_syn)`:

  ```
  v(t+dt) = v_rest + (v(t) + voltage_stim − v_rest)·a + g(t)·alpha
  g(t+dt) = g(t)·b + (spikes arriving through the delay line)
  alpha   = tau_syn/(tau_mem − tau_syn) · (a − b)
  ```

  At `dt = 0.1 ms` this gives `alpha = 0.004938`. Two tempting alternatives are both wrong for
  this network: forward Euler (`dt/tau_mem = 0.005`, close but over-delivers), and
  `tau_mem·(1 − a) = 0.0998`, which is **20.2× too large**. That last one is not a rounding
  error — it silently scales every synapse in the connectome, and it is the bug described in
  the next section but one.

- **External drive is an immediate voltage step**, not a synaptic event through the axonal
  delay line. The reference model adds it to `v` directly (`voltage_stim`).

- **The sparse weight matrix index order is `[post, pre]`.** `torch.sparse.mm(W, spikes)`
  contracts over dim 1, so dim 1 must be the presynaptic axis. Reversed, the product is all
  zeros and every synapse in the network is silently disabled.

## Performance reality

Measured on the RTX 3070 with the full 138,639-neuron / 15,091,983-synapse network:

| Quantity | Measured |
|---|---|
| load + build sparse matrix | 3.1 s |
| VRAM for weights | ~184 MB (~200 MB peak while stepping) |
| throughput | **0.13× realtime** — 0.78 s wall per 100 ms of brain time (≈7.7 s per 1 s) |

This is why the design does not attempt naive real-time control. The reference benchmark
results show the same story and identify the fix (see `docs/research/simulation-backends.md`):
an **active-set integrator** that only integrates neurons which are actually active, rather
than sweeping all 138k neurons every timestep.

The live view sidesteps this by publishing at a display rate while advancing a fixed window
of brain time per frame; the control loop trains the readout offline over recorded episodes
rather than requiring a real-time loop.

## Resolved: the "recurrent gain" problem was an integrator bug

Earlier revisions of this project warned that the published `0.275 mV` weight did not transfer
to the v783 edge list, and set `recurrent_scale` to a "provisional" `0.01`. That diagnosis was
wrong. The correction is worth recording, because of how convincing the wrong answer looked.

The membrane update carried an extra factor of `tau_mem` on the conductance term —
`tau_mem·(1 − a)` instead of `alpha` — which inflated **every synapse by 20.2×**. A
network with 20× weights is explosively unstable, and that produced a persuasive story: gain
0.004 dead, 0.01 balanced, 0.05 runaway, no wide stable band. Two external recalibrations from
another project (0.179 mV and ~0.125 mV) seemed to corroborate it. They were about a different
network with different behavioural targets, but they made the story fit, so the search went into
the connectome instead of the arithmetic.

What actually exposed it was not a parameter sweep — it was a **closed-form micro-test**
(`validation/micro_gain_check.py`). One synaptic event of weight `w` must deflect the membrane
by `0.15749·w` mV. Ours delivered about 20× that, which points straight at the coefficient
rather than at the connectome. The general lesson: **when a physical parameter seems to need a
~100× fudge factor, suspect the units or the arithmetic, not the physics.**

With the integrator corrected to the exact form:

- `recurrent_scale` is `1.0` and `w_scale_mv` is the published `0.275 mV` — no fudge factor.
- The model reproduces Brian2 on identical networks with identical input: **Jaccard 1.000** on a
  4,000-neuron augmented slice **and** on a real 20,000-neuron / 318,232-synapse slice, both at
  rate correlation **1.000**.
- Driving sensory neurons no longer floods the brain; the active-neuron count now matches the
  reference's exactly (322/322 and 359/359).

The full post-mortem, including a validation run that *passed while proving nothing*, is in
[research/simulation-backends.md](research/simulation-backends.md).
