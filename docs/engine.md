# Engine decision

The project runs **two** simulation backends behind one interface, deliberately.

| Role | Engine | Why |
|---|---|---|
| Scientific reference | `ConnectomeSim` (`flybrain/sim.py`) — FlyWire v783, exact exponential integrator | Our own, fully inspected, mapped to real thermosensory and antennal populations, and validated against brian2 on identical networks and identical input. `recurrent_scale` is `1.0` and `w_scale_mv` is the published `0.275 mV` — there is no fudge factor. |
| Real-time engine (built, opt-in) | `ActiveSetSim` (`flybrain/sim.py`) — the active-set integrator from `fly-brain-minecraft` (MIT), ported to our tensor ops | **Bitwise-identical to the reference on CPU and CUDA, and validated against brian2 (Jaccard 1.000) — but measured *slower* than the dense engine on this GPU under real drive.** Opt-in via `FLYBRAIN_ENGINE=active`; numbers and the reason below. The GPU path to real time is compiled per-step code, not tensor fusion. |

Neither replaces the other. The reference answers "is this faithful?"; the fast engine was
meant to answer "can it keep up?" — and the honest measured answer is "not this way".

> **Resolved: the recurrent gain was never a gain problem.** Earlier revisions of this project
> set `recurrent_scale` to a "provisional" `0.01` and warned that the published weight did not
> transfer. That diagnosis was wrong. The membrane update multiplied the conductance term by an
> extra `tau_mem`, making **every synapse in the connectome 20.2× too strong**; the `0.01`
> fudge was cancelling it. With the update corrected to the exact solution, the published weight
> is both correct and stable. See [research/simulation-backends.md](research/simulation-backends.md)
> for the full post-mortem.

## Why the current engine is not enough

Measured here, on the RTX 3070, full 138,639 neurons / 15,091,983 synapses at
`dt = 0.1 ms`:

| Measurement | Value |
|---|---|
| load + sparse matrix build | ~3 s |
| VRAM for weights | ~184 MB (~200 MB peak while stepping) |
| 1,000 steps (100 ms of brain time) | 0.78 s wall |
| **throughput** | **0.13× realtime** |

This matches the published benchmark for the same model: the PyTorch backend is the slowest of
the six backends measured, roughly 0.10× realtime, while GeNN reaches ~2×.

*(Re-measured after the constant-work below: **0.54 ms/step, 0.19× realtime**, same spike
output. The published comparison stands.)*

### What was actually banked: cheaper constants on the dense step

Before any algorithmic change, three zero-risk changes took the dense step from 0.83 ms to
**0.54 ms (1.5×)** with **bitwise-identical output** (same 52,957 spikes on the benchmark
window):

- **int32 CSR column indices.** torch builds them as int64, cuSPARSE accepts int32, and the
  column array is read for all 15M synapses on every step — halving it halves the step's
  dominant memory traffic.
- **Persistent drive as GPU tensors built once** in `set_drive`, instead of Python lists plus
  two H2D copies per step (~10k steps per second of brain time).
- **In-place zeroing of the per-step drive buffer** instead of a fresh `zeros_like`.

This is the baseline the active-set engine has to beat — and, measured, does not.

## The fix: integrate only what is active

The `fly-brain-minecraft` project runs a larger network (176,422 neurons, 90M synapses) at
**~1.5× realtime on CPU** with the same LIF parameters, by keeping an **active list**:

- A neuron enters the active list when it is away from rest beyond `idleEpsMv`, is refractory,
  has non-zero conductance, or has pending input.
- Each step integrates **only** the active list, then compacts surviving entries in place.
- A neuron that goes idle is dropped back to rest and removed.
- Spikes are queued into a ring buffer `delayBuf[slots][n]` with a `pending[]` counter; spike
  delivery stays on the owner thread for determinism, while integration is parallelised over
  the active list.

Reported cost: **25–37 ms per 50 ms of brain time**, worst case 62 ms, at 15k–100k active
neurons and 2k–40k spikes per tick.

Crucially it uses the **exact per-step linear solution** rather than Euler:

```
v[i] = rest[i] + (v[i]-rest[i])*a + current*(1-a) + g[i]*(a-b)/3
g[i] *= b
a = exp(-dt/20), b = exp(-dt/5)
```

At `dt = 0.1 ms` this reproduces Brian2 to `1e-13 mV`. That is the same integration scheme our
`ConnectomeSim` already uses (see `docs/architecture.md`), so the port is mostly about the
active-set bookkeeping, not about the dynamics.

### The port was built — and measured, it loses on GPU

`ActiveSetSim` implements exactly this: a dense per-step mask finds every neuron that could
deviate from rest (`v` away from rest, conductance above a 1e-30 floor **by magnitude** — the
magnitude test matters: inhibitory conductance is negative, and the first version dropped
neurons holding only inhibition, found by the bitwise equivalence test), delivery copies
per-neuron journal ranges (the synapses reordered by presynaptic index) instead of running
the matmul, and a **pending counter** keeps a resting target in the active set for the full
`delay_steps` between a delivery and its read. On CPU it is **bitwise identical** to the
dense engine; on CUDA likewise; and the brian2 harness passes with it (Jaccard 1.000,
spike-count ratio 1.000).

Measured on the RTX 3070, full connectome, identical drive and identical output:

| Scenario | dense (int32) | active-set |
|---|---|---|
| quiet (no drive, no spikes) | 0.48 s/1000 | **0.42 s/1000** (1.15× faster) |
| single column, 8 neurons driven | **0.46 s/1000** | 0.79 s/1000 (1.7× slower) |
| busy, 256 neurons driven | **0.53 s/1000** | 0.85 s/1000 (1.6× slower) |

Why it loses: the active list and the delivery size are data-dependent, so each step needs
two `nonzero` calls and a length reduction — three device synchronisations that drain the
pipeline the dense engine never drains — plus ~40 small kernel launches. That fixed host
cost (~0.4 ms/step) is on the order of the entire dense step. And the work does not shrink
as much as the synapse count suggests: in the busy regime ~39k neurons sit in the active set
(conductance tails plus pending input), so delivery and integration still touch 28% of the
brain. `fly-brain-minecraft` reaches real time on CPU because a compiled language has no
per-op overhead and delivers on threads; CUDA graphs cannot capture the dynamic shapes.

The engine stays in the tree, opt-in via `FLYBRAIN_ENGINE=active`: it is exact, validated,
and the right foundation if the delivery kernel is ever written in CUDA. It is not the
default anywhere, and nothing depends on it.

## Why not just switch to GeNN

GeNN (PyGeNN 5.4.0, LGPL-2.1) is the fastest measured backend (~2× realtime on a 4070) and has
exactly the right integration surface:

- drive: `additional_input_vars=[('Vstim','scalar',0.0)]` plus `vars['Vstim'].push_to_device()`
- readout: `spike_recording_enabled=True` + `pull_recording_buffers_from_device()` giving
  per-neuron spike identity once per control tick

It remains a strong option — stronger now: the tensor-level active-set port is built,
validated and measured, and its loss is attributable precisely to the per-op host overhead
that code generation eliminates. It adds a CUDA code-generation toolchain and a second build
system, and it was measured on a 4070 rather than a 3070 (**unverified** for our GPU).

Verdict: **GeNN, or a hand-written CUDA step kernel, is now the primary path to real-time.**
The dense engine with int32 indices is the baseline to beat; `ActiveSetSim` is the exactness
reference for whatever replaces it.

## What "real-time" actually needs to mean

It is worth being precise, because full real-time is not required for the thing the user wants.

- The **sensor→light** task needs a decision every few seconds, not every 100 ms. A 2 s control
  tick with a 500 ms brain window is 25% duty cycle — already achievable.
- The **live view** publishes at 20 Hz while advancing a fixed window of brain time per frame,
  and reports both clocks so the decoupling is visible.
- The **readout is trained offline** over recorded episodes, so training does not need to run
  alongside the live loop.

So the active-set port buys headroom and honesty, not the ability to do something otherwise
impossible.

## Idle cost, and adaptive pacing

**The cheapest optimisation available is to not step the brain at all.** Everything above is
about stepping it faster. This section is about stepping it *less*, which requires no new engine.

### Where the energy actually goes

Measured here: one decision is 300 ms of brain time, which costs **~2.3 s of continuous GPU
work** at 0.13× realtime. Flat out that is **~165 W continuously**. The shipped code no longer
does that by default — `LoopConfig.interval_s` defaults to **15 s**, so an unconfigured install
paces itself. Setting `FLYBRAIN_INTERVAL_S=0` puts flat out back, which is what the demo wants
and what the live view was originally built around.

*(This default was changed after the fact, and the change is worth understanding rather than
just noting: the README already described the loop as "paced by a wall-clock interval rather than
run flat out" while the code default was 0, so the documentation and the behaviour had been
disagreeing about what a fresh install costs.)*

`interval_s` gates the *decision*, and the loop genuinely does not advance the brain while it
waits, so average power falls roughly with the duty cycle. The duty cycle is
`2.34 s / interval_s`, and the measured model is `mean ≈ 19 W + duty × 146 W` — checked against
the hardware in [`live-view.md`](live-view.md#why-the-default-is-paced),
which owns the measurements:

| `interval_s` | Duty cycle | Mean power | Energy per day |
|---|---|---|---|
| `0` (flat out) | 100% | ~165 W | **~4.0 kWh** |
| `5` | 47% | ~86 W | ~2.1 kWh |
| `15` | 16% | ~40 W | ~1.0 kWh |
| `60` | 4% | ~25 W | **~0.6 kWh** |

**One line in `.env` is worth roughly 4×.** `FLYBRAIN_INTERVAL_S=15` takes a multi-day recording
from ~4 kWh/day to ~1 kWh/day, and a room does not change faster than that.

It is worth being clear about which knobs are *not* levers, because they look like they should
be. `fps` barely matters: each iteration's `advance()` costs far more than the sleep it replaces,
so the brain is the bottleneck rather than the sleep. `window_ms` moves work around without
removing any. **Only the duty cycle materially reduces energy.**

*(The 165 W figure is from the measurement earlier in this document; the operator's own
observation is nearer 140 W, so the **ratios** in that table are the durable part and the
absolute numbers should be re-measured alongside the network floor from
[`jev.md`](jev.md#api-call-discipline). Everything that follows depends on the ratio only.)*

### The cost of pacing badly

Heavy pacing is not free, but the cost is not the one it looks like. Learning does not need
windows to arrive quickly: a window recorded 60 seconds after the last one carries exactly as
much information as one recorded immediately after it. What degrades is **independence**.

A room changes on the scale of minutes. At a 60-second interval you collect 1,440 windows a day
that are, for most of it, near-duplicates — so the *effective* sample size is closer to a few
dozen independent events. That is the same autocorrelation that makes "split by day, not by
random window" the only honest way to evaluate a readout, and the two problems have the same
answer.

The quantity to maximise is therefore not windows per day. It is **independent, labelled events
per day, per joule**.

### Adaptive pacing: a heartbeat plus a trigger

A fixed interval forces a straight choice between energy and data. Asking *when is a window
worth taking* escapes the trade:

- **Heartbeat.** One window every `heartbeat_s` regardless of activity. This is not padding. The
  brain is never reset, so a heartbeat keeps the reservoir in the regime the readout was fitted
  on — and it produces the **quiet examples** that `house_activity` (roadmap A1) needs. Without
  it an event-driven recorder would have no negatives to learn from, which is how this scheme
  would fail silently.
- **Burst.** When a trigger fires, run at full rate for `burst_s`, so an event is captured in
  detail rather than sampled once.

Power then scales with how interesting the house is, and because the windows taken are *events*
rather than *samples*, the data is **less** redundant rather than more. Both halves are
load-bearing: the heartbeat is what makes the scheme learnable, the trigger is what makes it
cheap.

### What the trigger should be

Not Jev, and not a model. "Did something change?" is a cheap local comparison, and a threshold
beats a model on it every time — the same rule that keeps a model out of the regime probe
([`jev.md`](jev.md#where-jev-must-not-go)).

The better trigger is one the roadmap already describes for another purpose: **`house_novelty`
(A3), the reservoir's own prediction error.** Fit a readout to predict the next window's
reservoir state from the current one; the prediction error is a surprise signal. It is
multivariate and temporal — "the house is behaving unusually" rather than "one sensor crossed a
line" — it is computed locally for nothing, and it needs no network call. That makes A3 the
**scheduler** for pacing rather than merely a notifier, which is a better use of it than the one
it was written for.

Jev's part stays what [`jev.md`](jev.md) says it is: judging the windows the trigger selects,
not selecting them.

### What was actually built

**Status: implemented** in [`flybrain/pacing.py`](../flybrain/pacing.py), which is pure logic over
an injected clock — no torch, no numpy — so the cadence is testable without a GPU, and energy
figures are not measured there because they cannot be.

| Setting | Default | Meaning |
|---|---|---|
| `FLYBRAIN_INTERVAL_S` | `15` | The heartbeat: the **maximum** gap between decisions. `0` is flat out |
| `FLYBRAIN_POLL_S` | `5` | How often to re-read the sensors while waiting. Costs no GPU time |
| `FLYBRAIN_BURST_S` | `10` | How long to run at full rate once something happens |
| `FLYBRAIN_TRIGGER_DELTA` | `0` | How far the primary sensor must move, in its own units, to count as an event. `0` disables the trigger |
| `FLYBRAIN_TRIGGER_COOLDOWN_S` | heartbeat | How often the trigger may *fire*, independently of how often it qualifies. Defaults to the heartbeat — at most one trigger burst per heartbeat window — which keeps `interval_s` the ceiling in every mode. `0` disables the cap |

One field, one meaning: `interval_s` is the ceiling in every mode. A trigger can only make
decisions *sooner*. A second, overlapping "mode" knob would be a way to express the same thing
twice, which is how the `DEAD_STATES` bug happened in the first place.

The v1 trigger is a comparison, and is honest about its limits: the primary sensor by **value**
(in degrees Celsius for the temperature wiring) and motion/contact by **state string**. The state
string is used only for genuinely discrete kinds, because Home Assistant reports a thermometer's
state as its value — treating `"20.0" → "20.1"` as an event would make the trigger fire
constantly and pacing would silently become "always burst", the most expensive possible
misreading of the setting. Illuminance and humidity are *not* watched in v1; the multivariate
answer is A3.

**That failure arrived anyway, from the numeric side, and is now guarded.** Measured live:
the mock room swings 11 °C over 180 s, so a 5 s poll saw ~1.9 °C of movement against a 0.3 °C
delta — the trigger fired on *every* poll, and a heartbeat configured for ~25 W held the GPU at
~85% / 161 W with the pet reporting "startled" 714 of 730 seconds. Two guards went in:

- **Cooldown** (`FLYBRAIN_TRIGGER_COOLDOWN_S`, default = the heartbeat): a trigger can fire at
  most once per cooldown window, however often it qualifies. This keeps the invariant above —
  `interval_s` is the ceiling in every mode — while a genuinely busy house still bursts, once
  per heartbeat at most. An injected A3 trigger is capped the same way.
- **Burst-end baseline refresh**: polling stops during a burst, so the first post-burst poll
  compared against a *pre-burst* snapshot and refired on the drift the burst itself had
  captured. When a burst expires the comparison baseline is dropped, so that poll is a fresh
  start and the chain is broken.

Note the second guard matters even with the first: without the refresh, the first poll after
every cooldown would compare against a minute-old baseline and fire on accumulated drift
regardless of the threshold.

The seam for A3 is the `Trigger` protocol. A trigger is an injected callable, so an A3 novelty
trigger is constructed in `server.py` where the reservoir is in scope, closes over its own fitted
predictor, and replaces `ChangeTrigger` without `pacing.py` learning anything about the brain.

The pacer reports the duty cycle it *actually* achieved, not the one its settings imply, because
once a trigger is in play the cost depends on how interesting the house has been. That is what
the dashboard shows. It is a **running average since the process started** — the honest answer to
"what has this cost me", and deliberately not an answer to "what is it drawing now", which the
process cannot observe about itself. Pausing does not reset it, so the panel labels it an average
rather than leaving a high number beside a paused brain.

### Two traps

**The pacing configuration is part of the training regime.** `AGENTS.md` #2 — train and run in
the same regime — applies here in a way that is easy to miss. Event-weighted windows are a
*different distribution* from fixed-interval ones, so a readout trained on bursts and then run
against a steady heartbeat loses accuracy the same silent way as one trained from rest and run
continuously. The pacing config therefore belongs in the recording's `meta.json` and has to be
reproduced at inference time. It is not a runtime detail. It is now written there automatically
(`server.recording_meta`), and a test asserts the round trip reaches the file.

**Pacing is visible, and that is a real trade.** With `interval_s` set, the 3D view advances in
a burst and then holds still. The view reports that as **waiting**, and as **bursting** while a
burst is running, rather than "stalled" — it was written expecting the first, and the second was
added with this scheme so that "the house just did something" and "the GPU is idle" look
different. Constant motion is a *display* preference, not a modelling requirement — and bursting
on novelty arguably makes the view more interesting than uniform churn, because it moves exactly
when the house does.

### One consequence that is easy to miss

Smoothing is measured in **windows**, not wall clock: `smooth()` steps by `window_ms`
(`flybrain/loop.py`). At `smooth_ms=5000` that is about 16 windows, which at a 15 s heartbeat is
roughly **four minutes** of wall-clock averaging. So a paced loop is slower to react to a real
change than a flat-out one, and under a burst it is faster again — the response time becomes
non-stationary. The *readout* is unaffected (each decision still consumes 300 ms of brain time,
so the per-window input distribution is identical), which is why changing the default does not
invalidate the trained colour readout. But the feel of the light does change, and that trade is
the reason `interval_s` remains a control rather than a constant.

## Gain calibration

**Settled for fidelity: use the published `0.275 mV`.** `recurrent_scale` is `1.0` and the
validation harness reproduces Brian2 on the same network with the same input (Jaccard 1.000 on
both a 4,000-neuron augmented slice and a real 20,000-neuron / 318,232-synapse slice, rate
correlation 1.000 on both). Nothing needs retuning to match the published model.

This section previously told the reader to expect a sweep, citing two independent recalibrations
from `fly-brain-minecraft` (0.179 mV / gain 0.65, and ~0.125 mV / gain ~0.45). Those observations
are about *behavioural* targets in a *different* network — the male CNS edge table, with
antennal-lobe corrections — and they are not evidence of a defect in v783. We treated them as
such for a while, and used them to justify a fudge factor that was in fact compensating for our
own integrator bug. That is the trap worth remembering: **an external calibration number can
make a local implementation bug look like a modelling discrepancy.**

That said, the same project's published integrator independently corroborates the fix. Its
per-step update writes the conductance coefficient as `(a-b)/3`, where `a = exp(-dt/20)` and
`b = exp(-dt/5)`:

```
v[i] = rest[i] + (v[i]-rest[i])*a + current*(1-a) + g[i]*(a-b)/3
```

At `dt = 0.1 ms` that is `0.004938` — within 1% of our `(1-a) = 0.004988`, and nowhere near the
`tau_mem*(1-a) = 0.0998` that the bug was using.

The tooling is in `validation/`: `micro_gain_check.py` pins the integrator against a closed form,
and `validate_against_brian2.py` compares whole networks.

## Known modelling limitations to expect

These are documented by that project from real experience — they are properties of the model
family, not bugs to chase:

- The antennal lobe **saturates at any olfactory receptor rate ≥ 25 Hz** under uniform LIF
  parameters.
- The ON visual pathway is structurally unreachable in a silent LIF because L1 is glutamatergic.
- Kenyon cells need a ~0.25 postsynaptic gain correction.
- Delta synapses run away even at 10% gain.
- Divisive in-degree normalisation silences MN9 and the giant fibre.
- Spike-frequency adaptation kills projection-neuron responses.

For this project the most relevant one is the first: **a home sensor is not a natural sensory
input.** We drive thermosensory neurons with injected current, not with a realistic transduction
cascade, so we should expect saturation effects and tune the encoder gain rather than assume
linearity.
