# Engine decision

The project runs **two** simulation backends behind one interface, deliberately.

| Role | Engine | Why |
|---|---|---|
| Scientific reference | `ConnectomeSim` (`flybrain/sim.py`) — FlyWire v783, exact exponential integrator | Our own, fully inspected, mapped to real thermosensory and antennal populations, and validated against brian2 on identical networks and identical input. `recurrent_scale` is `1.0` and `w_scale_mv` is the published `0.275 mV` — there is no fudge factor. |
| Real-time engine (planned) | Port of the **active-set integrator** from `fly-brain-minecraft` (MIT) | The only approach verified to reach real-time on a whole CNS. |

Neither replaces the other. The reference answers "is this faithful?"; the fast engine answers
"can it keep up?".

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

Why: every 0.1 ms timestep performs a `15,091,983`-element sparse matvec over **all 138,639
neurons**, whether or not they are doing anything. At any moment the overwhelming majority are
at rest.

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

## Why not just switch to GeNN

GeNN (PyGeNN 5.4.0, LGPL-2.1) is the fastest measured backend (~2× realtime on a 4070) and has
exactly the right integration surface:

- drive: `additional_input_vars=[('Vstim','scalar',0.0)]` plus `vars['Vstim'].push_to_device()`
- readout: `spike_recording_enabled=True` + `pull_recording_buffers_from_device()` giving
  per-neuron spike identity once per control tick

It remains a strong option, but it adds a CUDA code-generation toolchain and a second build
system, and it was measured on a 4070 rather than a 3070 (**unverified** for our GPU). The
active-set approach is a contained algorithmic change to code we already control and have
validated, and it helps on CPU as well as GPU.

Verdict: port active-set first; keep GeNN as the fallback if it is not enough.

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
