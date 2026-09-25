# Simulation backends and connectome data sources

Research reference for the engine that steps the FlyWire v783 network, and for the datasets
that supply its connectivity, morphology and annotations. Companion documents:
[asset-inventory.md](asset-inventory.md) (visualisation, Home Assistant, graph and annotation
assets) and [dead-ends.md](dead-ends.md) (traps to avoid). Project-level context is in
[../architecture.md](../architecture.md) and [../licensing.md](../licensing.md).

Target hardware and stack, assumed throughout: RTX 3070 (8 GB), Linux, Python 3.11,
*Drosophila* FlyWire v783 with 138,639 neurons and 15,091,983 synapses.

## Verdict summary

| Engine / source | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| `blendi-remade/fly-brain-minecraft` active-set LIF (pure-Java `com.fruitfly.brain`) | MIT code, CC BY 4.0 data | v0.1.0 released 2026-09-03; bundled FLYB v1 binary (23 MB gzip, 46.6 MB raw) | `setStimulusRate`, `setInjectedCurrent`, `forceSpike`; `spikesThisTick`, `rateHz`, `populationRateHz`, `spikeLog*`, `endTick` | **Best architectural template.** Python port ~300 lines. |
| PyGeNN / GeNN 5.4.0 | LGPL-2.1 (verified: raw `LICENSE` begins "GNU LESSER GENERAL PUBLIC LICENSE Version 2.1") | Documented latest release 5.4.0; not on PyPI | `additional_input_vars=[('Vstim','scalar',0.0)]` + `post_target_var`; `spike_recording_data[b]` | **Recommended GPU engine if moving off PyTorch.** ~20x faster than PyTorch. |
| `eonsystemspbc/fly-brain` benchmark harness | GPL-2.0-or-later (project); `code/paper-phil-drosophila/` separately MIT | Created 2026-03-05, last commit 2026-08-29, 884 stars, no releases | CLI only, no reusable API; `model.py`+`utils.py` are a usable MIT API | Use as **benchmark data + reference model**, not as a library. |
| `philshiu/Drosophila_brain_model` | MIT | Created 2023-02-25, last push 2024-09-14, 337 stars, 69 forks, no releases, not archived | `run_exp(...)` injects into chosen neurons and returns per-neuron spike times | **Use as-is.** Ground-truth model. |
| Brian2 CPU | CeCILL-2.1 (UNVERIFIED which exact file; treat as copyleft) | 2.9.0 (2025-05-14, `>=3.10`) is the numpy-2.x-capable release for 3.11 | Python API, but C++ standalone mode has **no Python-in-the-loop** | **Validation ground truth only**, in `.venv-validation` (numpy 1.26). Not a real-time engine. |
| Brian2CUDA | CeCILL-2.1 (UNVERIFIED) | 1.0a7 (2025-01-10), 1.0b1 (2025-12-05), pins `brian2==2.10.1` | Standalone/CUDA mode is one-shot; no live injection | **Avoid.** |
| NEST GPU | GPL-2.0 (verified from README) | Main branch; CNS 2025 showcase and Zenodo v1.0 (2025-07-02); exact latest tag UNVERIFIED | "Subprocess per trial (cannot reset in-process)" | **Avoid for a live controller.** |
| Brian2GeNN | UNVERIFIED (CeCILL/GPL family) | 1.7.0 pins `Brian2<2.6` | Legacy 4.x CLI, `BRIAN2GENN_GENN_PATH`/`GENN_PATH` | **Avoid**; use direct PyGeNN. |
| `flybrain` PyPI (`alextitonis/fly.ai`) | MIT | 0.1.0 (2026-09-13), `>=3.10` | CPU (numba) / GPU (CuPy); `flybrain/web.py` browser export | **Use as a library**; steal the browser export format. |
| `nftechie/stonkfly` LIF kernel | MIT | Active | `ctypes` C++ kernel; direct `stimulation=[(indices, current)]` path | **Reuse as-is** (C++ kernel + state.py). |
| Our `ConnectomeSim` (PyTorch sparse) | in-project | Working, 0.13x realtime | `inject`/`set_drive`, `spike_counts` | Kept; **not fast enough** for real-time without the active-set trick. |
| Loihi 2 whole-FlyWire (Wang et al., Sandia, arXiv 2508.16792) | Likely restricted (UNVERIFIED) | Hardware demonstration | 992 dedicated spike counters, overcommitted | **Avoid** (no hardware, readout costs the speedup); steal SNN-dCSR. |
| SpiNNaker whole-FlyWire | — | **No verified implementation found — UNVERIFIED / likely non-existent at whole-brain scale** | — | Ignore. |

## Benchmark table (reproduced)

Source: the `eonsystemspbc/fly-brain` rig, `data/results/nature_2026_07/no_io/timings.csv`,
measured on an RTX 4070 under WSL2. `sim_time` is in seconds; `n` is `n_run` (number of
parallel trials). The v783 model: 138,639 neurons, 15,091,983 synapses, dt 0.1 ms.

| Framework | t=1s,n=1 | t=100s,n=1 | t=100s,n=32 | realtime x (t=100,n=1) |
|---|---|---|---|---|
| Brian2 CPU (C++ standalone) | 2.9 | 269-298 | 3977-4804 | 0.35 |
| Brian2CUDA | 10.9-13.3 | 268-399 | 11863-18022 | 0.26-0.37 |
| PyTorch CUDA | 6.5-11.1 | 651-1026 | 18164-19723 | ~0.10 |
| NEST GPU | 0.95-1.28 | 90-144 | 2845-3649 | 0.70-1.11 |
| GeNN (PyGeNN 5.4.0) | 0.47-0.71 | 47.6-56.0 | 1575-6553 | 1.79-2.10 |
| Brian2GeNN | 1.19-1.96 | 123-193 | 3909-6320 | 0.52-0.81 |

Notes on the table:

- GeNN at n=32 has one 6553 s outlier against a 1575-2482 s typical band — **unexplained; treat
  as variance**, not a real regression.
- PyTorch VRAM use was only 1.38-1.65 GB in that rig.
- GeNN is roughly **20x faster than the PyTorch backend** on this model, and the only backend
  that is faster than real time in the single-trial case.
- A 3070 is expected to land near **0.8-1.4x realtime** for GeNN (UNVERIFIED on 3070; the 4070
  measurement is the evidence).

**Why this matters for us:** our own `ConnectomeSim` measured 0.13x realtime on the 3070
(see below), consistent with the PyTorch row. The benchmark says the win is not "more GPU" but
a different execution strategy — PyGeNN's code-generated kernels, or the active-set integrator
below. Any plan that assumes the current PyTorch loop can be tuned into real time is wrong.

## Engine reference

### `blendi-remade/fly-brain-minecraft` — active-set integrator

Repository: <https://github.com/blendi-remade/fly-brain-minecraft>. Version 0.1.0 released
2026-09-03. Code MIT; bundled data CC BY 4.0. Java 21+/Gradle/Fabric for the mod, but the
`com.fruitfly.brain` module has **no Minecraft dependency** and is testable headlessly
(`gradlew brainBench`).

Measured: **25-37 ms per 50 ms of neural time** on 32 cores / 8 workers for the 176k-neuron,
90M-synapse male CNS; worst case 62 ms — about **1.5x real time on CPU**. A Python port is
estimated at ~300 lines.

The core trick is an **active-set integrator**: only neurons away from rest, in refractory, or
with pending input are integrated. Cost scales with activity, not network size. This is the
single most transferable idea in the whole survey.

Integration is an exact per-step linear solution:

```
synCoupling = (tau/(tm-tau)) * (exp(-dt/tm) - exp(-dt/tau))
```

`dt=0.5 ms` in-game; `dt=0.1 ms` reproduces Brian2 to **1e-13 mV**. Parallelised over the
active list with `threads = min(8, cores-2)`; spike delivery happens on the owner thread for
determinism; delayed inputs use a ring buffer `delayBuf[slots][n]` with a `pending[]` counter.

Integration surface (exactly what a Python control loop needs):

| Method | Purpose |
|---|---|
| `setStimulusRate(neuron, Hz)` | Poisson drive |
| `setInjectedCurrent(neuron, mV/ms)` | Graded drive; negative allowed |
| `forceSpike(i)` | Deterministic single spike injection |
| `spikesThisTick(i)` | Per-neuron readout, current tick |
| `rateHz(i)` | Per-neuron rate readout |
| `populationRateHz(pop, windowMs)` | Population rate over a window |
| `spikeLogNeuron` / `spikeLogStep` | Spike log readout |
| `endTick(tickMs)` | Advance one tick |

Population selection is a spec DSL, not hardcoded IDs:
`DNp09/L`, `prefix:ORN_DM1`, `contains:LC10`, `class:gustatory`, `superclass:vnc_motor`,
`subclass:wm`, `nt:gaba`, `nerve:ADMN`, `body:10783`, `all`, with `&` intersection and `,`
union. **Why this matters:** this is the pattern our `RoleResolver` should follow — resolve
roles from annotations at runtime, never hardcode neuron indices.

Data plumbing worth copying:

- `tools/fetch_neuprint.py` — anonymous, resumable, chunked Cypher fetches.
- `tools/build_flyb.py` — the FLYB v1 binary format: CSR with delta-coded + LEB128-varint post
  indices, **3.09 B/edge raw, ~2.08 B/edge gzipped**, against 6.55 B/edge plain CSR and
  18 B/edge int64 bodyId pairs. zstd-19 beats gzip by only ~4%, so **do not add a zstd
  dependency**. A `HashMap<Long,List<Edge>>` for 6M edges costs 500 MB+ and stalls GC.
  Their 15.1M-edge v783 graph would be ~31 MB raw / ~20 MB gzipped in FLYB.

Self-documented model limitations (keep these in mind when interpreting our own runs):

- The antennal lobe **saturates at any ORN rate >= 25 Hz**.
- The ON visual pathway is **structurally unreachable** in a silent LIF because L1 is
  glutamatergic.
- Kenyon cells needed a **0.25 postsynaptic gain**.
- Gain was recalibrated from Shiu's 0.275 mV to **~0.179 mV (gain 0.65)** because the male CNS
  has **1.24x per-neuron synaptic input**.
- **Delta synapses ran away even at 10% gain.**
- **Divisive in-degree normalisation silenced MN9 and the giant fibre.**
- **Spike-frequency adaptation killed projection-neuron responses.**

An independent 300+ run Brian2 cross-check (`gaps/gap-1.md`) recommends **~0.125 mV (gain
~0.45)** plus an antennal-lobe correction.

### PyGeNN / GeNN

Repository: <https://github.com/genn-team/genn>. Docs:
<http://genn-team.github.io/genn/documentation/5/>. Licence **LGPL-2.1** (verified). Documented
latest release **5.4.0**; install from the GitHub tag zip:

```
pip install https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip
```

**PyGeNN is not on PyPI** — the PyPI `genn` 0.7.7 package is an unrelated 2021 project. Native
Windows is supported (Visual Studio 2019+). PyGeNN also needs `pkg-config` and `libffi-dev`.

External drive and readout:

- Declare `additional_input_vars=[('Vstim','scalar',0.0)]` on the LIF neuron model. A
  stimulation synapse sets `stim_syn.post_target_var = 'Vstim'`.
- Runtime drive from Python: write `neurons.vars['Vstim'].current_view[:] = ...` then
  `.push_to_device()`.
- Readout: `neurons.spike_recording_enabled=True`, `model.load(num_recording_timesteps=W)`,
  then per window `model.pull_recording_buffers_from_device()` ->
  `neurons.spike_recording_data[b]` -> `(spike_times, spike_ids)`. Per-neuron counts are a
  `np.bincount`. One device-to-host copy per window gives per-neuron spike identity.
- **CUDA stepping is asynchronous**, so a Python `while model.timestep < N: model.step_time()`
  loop does **not** sync per step.
- `GENN_RECORDING_WINDOW_MAX_SLOTS` default **800000** caps the on-device spike buffer to avoid
  CUDA OOM at large `n_run`.
- Caveat: allocate the recording window so it does not blow the 8 GB on a 3070.

`run_genn.py` in the vendored `eonsystemspbc/fly-brain` tree is a working reference for exactly
this model.

### `eonsystemspbc/fly-brain` benchmark harness

Repository: <https://github.com/eonsystemspbc/fly-brain>. Project licence **GPL-2.0-or-later**,
but `code/paper-phil-drosophila/LICENSE` is separately **MIT** ("Copyright (c) 2023 Philip Shiu
and Nico Spiller") — verified. Created 2026-03-05, last commit 2026-08-29, 884 stars, no GitHub
Releases.

- Six backends share one data/model/output schema: Brian2 C++ CPU, Brian2CUDA, PyTorch CUDA,
  NEST GPU, GeNN/PyGeNN, Brian2GeNN.
- Ships the connectivity + completeness data we already vendored.
- Install: conda `environment.yml` -> env `brain-fly`, `python=3.10`, `numpy=1.26.4`,
  `brian2cuda==1.0a7`, torch cu126. Tested Ubuntu 22.04 under WSL2, RTX 4070, CUDA 12.x.
- NEST GPU needs a from-source build with a custom `user_m1` neuron and a patched
  `nestgpu.py` (files in `scripts/nestgpu_source_files/`).
- Brian2GeNN needs its own conda env because 1.7.0 pins `Brian2<2.6`.
- **No reusable Python API** — it is a benchmark CLI (`main.py` -> `code/benchmark.py` ->
  `run_*.py`), no package, no `__init__.py`.
- But `code/paper-phil-drosophila/model.py` + `utils.py` **is** a usable MIT API:
  `create_model()`, `run_exp()`, `get_spk_trn()`, `get_rate()`.
- Org also hosts `drosophila_brain_model_lif` (a mirror of philshiu's repo, not distinct),
  `flybody` (fork of TuragaLab/flybody), `NEURD-sandbox`, `pathintegrationBPU`.

**Why this matters for us:** use it for its committed connectivity data and its GeNN reference
script; do not vendor its GPL code into our package.

### Brian2

Repository: <https://github.com/brian-team/brian2>. Licence **CeCILL-2.1** (UNVERIFIED which
exact file; treat as copyleft).

| Version | Date | Python | Notes |
|---|---|---|---|
| 2.8.0 | 2024-12-20 | | |
| 2.8.0.1 / 2.8.0.4 | Jan 2025 | | FTBFS on NumPy 2.3 (see below) |
| 2.9.0 | 2025-05-14 | `>=3.10` | Works on 3.11; our validation choice |
| 2.10.0 | 2025-12-04 | `>=3.12` | |
| 2.10.1 | 2025-12-05 | `>=3.12`, `numpy>=2.0.0` | Pinned by brian2cuda 1.0b1 |

Speed on 138k / 15.1M at dt 0.1 ms, `n_run=1`: **~0.35x realtime** (1 s brain ~ 2.9 s wall;
100 s ~ 270-300 s). C++ standalone mode does **not** allow Python-in-the-loop interaction at
all.

**numpy 2.x trap:** brian2 2.8 breaks on numpy 2.4. Debian bug #1114696: brian 2.8.0.4 FTBFS on
NumPy 2.3 with `AttributeError: module 'numpy' has no attribute 'NPY_OWNDATA'`; Debian closed it
as fixed in `brian/2.9.0-1`. So **2.9.0 (2025-05-14, requires_python >=3.10) is the
numpy-2.x-capable release that still works on Python 3.11**; 2.10.x requires 3.12+.

In this project brian2 is used **only as validation ground truth**, in a separate
`.venv-validation` env with numpy 1.26.

### Brian2CUDA

Repository: <https://github.com/brian-team/brian2cuda>. Licence **CeCILL-2.1** (UNVERIFIED).
1.0a7 (2025-01-10), 1.0b1 (2025-12-05), requires `>=3.10`, pins `brian2==2.10.1`. Linux + CUDA
only. Standalone/CUDA mode is one-shot — **no Python-in-the-loop, so not usable for live
injection** — and it is slower than Brian2 CPU at `n_run=1` on this model (10.9-13.3 s vs 2.9 s
for 1 s of sim). **Avoid.**

### NEST GPU

Repository: <https://github.com/nest/nest-gpu>. Docs: <https://nest-gpu.readthedocs.io>.
Licence **GPL-2.0** (verified from README). Main branch; a CNS 2025 showcase and Zenodo v1.0
(2025-07-02) exist; **exact latest tag UNVERIFIED**.

- No Windows path; build from source with CMake + CUDA 12.x.
- Needs a custom `user_m1.{h,cu}` neuron and a patched `nestgpu.py` (weight-array init,
  lines 2225-2227).
- Speed ~0.7-1.1x realtime.
- **Disqualifying**: fly-brain's own architecture table says "Subprocess per trial (cannot reset
  in-process)" — you cannot re-inject drive mid-run in one process.

### Brian2GeNN

Repository: <https://github.com/brian-team/brian2genn>. Licence **UNVERIFIED** (CeCILL/GPL
family). 1.7.0 pins `Brian2<2.6`, which is why it needs a separate conda env from Brian2CUDA
(needs 2.8/2.10). Requires the GeNN 4.x CLI (`genn-buildmodel.sh`) and
`BRIAN2GENN_GENN_PATH`/`GENN_PATH`. Legacy 4.x stack, high friction, no realtime benefit over
direct PyGeNN; speed 0.52-0.81x realtime and **build_time is huge**.

### `philshiu/Drosophila_brain_model` — ground-truth model

Repository: <https://github.com/philshiu/Drosophila_brain_model>. **MIT**. Created 2023-02-25,
last push 2024-09-14, 337 stars, 69 forks, no releases, not archived.

Contents: `model.py` (11,900 B), `utils.py`, `example.ipynb`, `figures.ipynb`,
`Completeness_783.csv` + `Connectivity_783.parquet` and the v630 pair, `sez_neurons.pickle`,
`results/example/*.parquet`, `environment.yml`, `environment_full.yml`. Note the readme is
`Readme.md` (capital R, lowercase m), so a raw `README.md` fetch 404s.

Model verbatim:

```
dv/dt = (v_0 - v + g)/t_mbr
dg/dt = -g/tau
threshold 'v > v_th'
reset     'v=v_rst; w=0; g=0'
method    'linear'
refractory 'rfc'
```

Parameters: `v_0 = v_rst = -52 mV`, `v_th = -45 mV`, `t_mbr = 20 ms`, `tau = 5 ms`,
`t_rfc = 2.2 ms`, `t_dly = 1.8 ms`, `w_syn = 0.275 mV`, Poisson drive `r_poi = 150 Hz`,
scaling `f_poi = 250`.

Synapse: `Synapses(..., 'w : volt', on_pre='g += w', delay=t_dly)` with
`syn.w = 'Excitatory x Connectivity' * w_syn`. Documented **91% prediction accuracy** against
experimental data.

`run_exp(exp_name, neu_exc, path_res, path_comp, path_con, params, neu_slnc, neu_exc2, n_proc,
force_overwrite)` is a ready-made "inject into chosen neurons, get per-neuron spike times out"
API.

### `flybrain` PyPI (`alextitonis/fly.ai`)

Repository: <https://github.com/alextitonis/fly.ai>. **MIT** (LICENSE verified). PyPI `flybrain`
0.1.0 (2026-09-13), requires `>=3.10`. MaleCNS v1.0, 166,700 neurons; CPU (numba) or GPU
(CuPy); `pip install flybrain[gpu]`, `FLY_DEVICE=cuda`. Whole connectome ~**210 MB VRAM**;
**1.4 ms/step on an RTX 4060 laptop**, so comfortable on a 3070.

`flybrain/web.py` (`flybrain export --web <folder>`) exports the connectome **for the browser**:
gzipped CSC weights split <40 MB, `meta.bin` labels/types/classes/sides, `brain.json`, and a
documented binary format. `sshfighter/` ships a live dashboard of every neuron firing; `world/`
is a 3-D fly world. **Why this matters:** this is a ready-made browser transport format for a
139k-neuron live view; see [asset-inventory.md](asset-inventory.md).

### `nftechie/stonkfly` LIF kernel

Repository: <https://github.com/nftechie/stonkfly>. **MIT**. Verified commit
`78ef3e05ab0fa086032098558d893667068944a0` (2026-09-09, single "Initial commit"). It is an
importable library, not a closed CLI: `grep -r coinbase stonkfly/neural/` returns nothing.

Kernel: `stonkfly/neural/kernel.cpp` (96 lines), one `extern "C"` function `memory_advance`
(line 10). Compiled on first use, cached to `data/cache/physiology-v6/libmemory.so` guarded by a
`.json` recording source+binary sha256 (`brain.py:36-61`):

```
subprocess.run(["c++","-O3","-std=c++17","-shared","-fPIC", SOURCE, "-o", temp], check=True)   # brain.py:49-52
```

Loaded by raw `ctypes`, no pybind, no struct; every array is a contiguous numpy buffer passed as
`void*` (`brain.py:80-103, 264-293`). Constants: `dt=0.1 ms` hard-locked (`state.py:8-9` raises
otherwise); `tau_m=20 ms`, `tau_syn=5 ms`; threshold -45 mV; on fire `v=rest, g=0`,
`refractory=rfc`; delay 1.8 ms (`lround(1.8/dt)`, ring of `delay+1` slots); refractory 2.2 ms;
rest -52 mV all cells, **-60 mV for KC** (`brain.py:107-108`); KC adaptation +8 mV jump,
tau 200 ms; weight = `count * sign * 0.275` (`prepare.py:28`); signs from `transmitters.py:6-24`
(ACh +1; GABA/glutamate/histamine -1; ambiguous -> +1).

Core LIF update (`kernel.cpp:29-32`):

```
const float a = exp(-dt*d/20.f), b = exp(-dt*d/5.f);
v[i] = rest[i] + (v[i]-rest[i])*a + current*(1.f-a) + g[i]*(a-b)/3.f;
g[i] *= b;
```

Main loop (`kernel.cpp:41-92`): per tick, evolve active cells, queue spikes into the future
delay slot, dispatch `slot clock % slots`. Cells in `modulation_mask` (dopamine/octopamine/
serotonin) deliver a **decayed trace** instead of fast excitation (`:60-82`); ordinary cells do
`g[j] += weight[e]` (`:83-86`). Event-driven optimisation drops an inactive cell only when `v`,
instantaneous drive **and** asymptotic drive are all sub-threshold (`:55-57`, `:93-95`). At the
end all states are materialised so no threshold can be missed. No graph pruning, no timestep
enlargement.

Data: MaleCNS v1.0, `datasets.json` -> <https://male-cns.janelia.org/download/>, paper
`doi:10.1016/j.cell.2026.08.015`. Three public GCS files under
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`:
`body-annotations-male-cns-v1.0-minconf-0.5.feather` (14,483,314 B),
`body-neurotransmitters-male-cns-v1.0.feather` (43,282,834 B),
`connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1,051,241,946 B). Total ~1.11 GB, no
account/key/token, plain `urllib.request.urlretrieve` (`data.py:77-88`). Data is not
redistributed; upstream is CC BY 4.0 per `THIRD_PARTY.md:4`. Result asserted at exactly
**166,700 neurons / 25,582,938 directed edges / 124,177,617 contacts** (`data.py:35`).

Five-layer checksum/manifest discipline worth adopting: `sources.lock.json` (URL+bytes+sha256);
`data/source.lock.json` that refuses changed sources (`connectome.py:141-150`);
`arrays.lock.json` (per-array sha256 of `graph.npz`); `neurons.lock.json`; kernel build `.json`;
checkpoint provenance (`brain.py:366-380, 424-446`).

Sensory encoding is the mechanism to remember: **no neuron ID is hardcoded anywhere**; every
population is a runtime query on the `cell_type` string column, via `common.annotations(ids)`
(`common.py:25-33`). Prices -> RGB image -> current (never price -> neuron directly).
`display.py:7-35` renders a 320x180 PIL chart; `prepare.py:37-72` infers per-receptor screen
coordinates (`R1-R6`, modal `assignedOlHex1/2` column by summed contact count onto L1/L2/L3,
axial hex->xy, split into overlapping L/R viewports `x in [0,0.6]/[0.4,1.0]`, y flipped);
`sensory.py:6-17` samples `uv`->pixel, sRGB->linear, Rec.709 luminance; `brain.py:226-232`
applies a 10 ms low-pass then a saturating current:

```
self.luminance += (1-exp(-steps*dt/10)) * (clip(light,0,1) - self.luminance)
self.drive[self.lamina] = lamina_bias              # 12.0 mV-equivalent
self.drive[self.retina] = 30*self.luminance/(0.02+self.luminance)
```

Populations: retina = 3,335 mapped R1-R6; R8 = 811; lamina = L1,L2,L3,L5
(`prepare.py:117-119`); sugar = LB3c (`:120`) but it is loaded and seeded active yet **never
driven — effectively dead code**. Counts 3335/811 asserted in `tests/test_neural.py:65`.
Dopamine identities (`circuit.py:12-23`) are pure type-string queries with count assertions:
`kc = types.str.startswith("KC")`; `reward = types.eq("PAM11")` (15);
`aversive = types.eq("PPL101")` (2); `reward_mb = types.eq("MBON07")` (4);
`aversive_mb = types.eq("MBON11")` (2).

Readout is fully hardcoded and hand-picked, zero learned weights (`controller.py:11-46`):
`left/right = DNp20 & sides L/R`, `gate = DNpe017`, `left = mean(counts[left])/seconds`,
`difference = right-left`, `gate = counts[gate].sum()`, `side = HOLD if not gate or
abs(difference) < threshold else BUY if difference>0 else SELL`. `decoder_threshold_hz = 2 Hz`
(`config.py:42`), duration = `neural_ms/1000 = 0.5 s`. `cli.py:173` labels it "Engineered fixed
mapping". `fly-wirehead` reuses the same trick with DNa02 (turn), MN9+DNp09 (motor), PAM11
(reward glow) — `fw:engine.py:30-32`.

Plasticity: two implementations exist; the C++ one is **dead code**. (a) Disabled C++ LTD
(`kernel.cpp:1, 70-79`) — `brain.py:321-329` calls `_neural_step(..., learning=False)` with the
comment "The original LTD update is disabled."; the `modulation[]` trace written at
`kernel.cpp:67-68` is never read. (b) Active rule: `rule.py:27-65`, called from `brain.py:331-343`
every <=10 ms bin into `memory_u`/`memory_w`; `brain.py:344-347` maps back
`weight[edges] = baseline_plastic * (1 + memory_w)`:

```
drive = eta * (kc_hz * (gain.T @ dmid) - (gain.T @ dan_hz) * kmid)
u[:] = old_u*eu + drive*tu*(-expm1(-h/tu))            # tu = 1800 s memory decay
w[:] = w*ew + old_u*c + drive*tu*(-expm1(-h/tw) - c)  # tw = 50 ms filter
np.clip(u, -0.9, 1.0); np.clip(w, -0.9, 1.0)          # 0.1x..2.0x baseline
```

Params (`rule.py:14-24`): `trace_kc 1 s`, `trace_dan 1 s`, `memory_decay 1800 s`,
`weight_filter 0.05 s`, min 0.1, max 2.0, `eta = 0.001`. Model: baseline-centred anti-Hebbian
adaptation of Huang/Luo 2024 eqs 3.2-3.5 (KC activity followed by dopamine depresses; reverse
order potentiates). Plastic edges = all existing KC->MBON07/11 connections = **7,835**; gain =
within-compartment DAN->MBON contact fractions, not receptor kinetics. Reward application
(`controller.py:58-97`) injects `pulse_current = 20 mV-equivalent` for `pulse_ms = 200 ms` into
`circuit["reward"]` (15 PAM11) or `circuit["aversive"]` (2 PPL101); sign is decided in
`reinforcement.py:4-18` from marked-to-bid equity with a +/-0.01 USDC deadband — binary, not
proportional. The plasticity rule consumes the **spike counts** those cells produce, not the
injected current. Repo's own caveat: endogenous dopamine activity changes weights even with
`reinforcement="none"` (`docs/validation.md:15`).

Control loop (`cli.py:198-279`): per external event, one market snapshot -> `market_frame(...)`
RGB -> `controller.observe(frame, kind)` advancing `neural_ms = 500 ms` of neural time
(`config.py:37`) in bins of `neural_bin_ms = 10 ms`; wall observations `>= interval_seconds =
60 s` apart. It does **not** attempt real-time: `cli.py:176` and `docs/model.md:25` call it "a
deliberately compressed market-to-neural clock, not real-time fly physiology". `--fast` skips
the wall sleep (paper only). Checkpointing: `controller.save()` -> `brain.checkpoint()` ->
`np.savez_compressed` of all 20+ state fields + weight + JSON metadata, atomic to two
alternating slots `runs/<mode>/brain-{tick%2}.npz`; restore refuses provenance mismatch.
The ledger/accounting anchor is committed before any trade (`cli.py:232`), and a restart reuses
the same command and run dir (`ledger.py`, `docs/operations.md:47-53`). A STOP file or halted
flag stops decisions; `fcntl.flock` prevents two workers on one run dir.

Reusability, as assessed: reuse as-is (~1,490 lines) `neural/kernel.cpp`, `state.py`,
`transmitters.py`, `sensory.py`, `common.py`, `connectome.py`, `prepare.py`, `datasets.json` +
3 locks, `data.py`, `rule.py`, `circuit.py`, `brain.py`, `visual.py`. Reuse with adaptation:
`neural/controller.py` (103 lines) — `Decoder` is the readout, and `FlyController.observe(rgb,
reinforcement) -> dict` is the clean seam (two inputs, pure numpy). Replace wholesale (~1,230
lines): `actions.py` 47, `broker.py` 288, `market.py` 170, `risk.py` 86, `ledger.py` 177,
`config.py` 99, `display.py` 35, `reinforcement.py` 18, `cli.py` loop 312.

Cost to swap in Home Assistant REST: **low** — keep `FlyController.observe`; replace
`market.snapshot()` with HA `GET /api/states/<entity>`; replace `display.market_frame()` with a
sensor-history renderer (or skip pixels and inject current directly); replace
`StonkflyActions.invoke` with `POST /api/services/<domain>/<service>`. A minimal HA client is
~150-250 lines replacing ~1,230. For a trained readout, `Decoder` is only 36 lines to replace
with a linear layer over `counts[population]`. For non-visual inputs, **skip the pixel
round-trip entirely** and use the documented `stimulation=[(indices, current)]` argument to
`brain.step()` (`brain.py:233-246`).

Stonkfly has **no web UI at all** — no HTTP server, no websocket, no HTML/JS/CSS/Node anywhere.
Its interface is stdout JSON (`cli.py:262-274`), `runs/*/events.jsonl`, `latest.json`,
`latest-input.png`, plus the `status` subcommand reading SQLite.

### `erojasoficial-byte/fly-brain` "Embodied Drosophila"

Repository: <https://github.com/erojasoficial-byte/fly-brain>. DOI
<https://doi.org/10.5281/zenodo.19152238>. **MIT**. 138,639 neurons / 15,091,983 directed
weighted synapses (identical counts to ours) as a LIF network in PyTorch on GPU at a **claimed
5 kHz timestep**, embodied in NeuroMechFly v2 / MuJoCo (87 joints), multi-modal sensory
encoders, DN->drive-rate bridge, Hebbian plasticity. 2026, single-author student preprint, **no
peer review**.

**WARNING — treat as a possibly AI-assisted preprint.** The claimed 5 kHz timestep, the "81% vs
47% escape" individuality claim and "76,034 divergent synapses after 24 hours" are extraordinary
claims from a non-peer-reviewed single-author source. **Its actual code quality is UNVERIFIED**
(file-listing fetches failed). Useful as a parts list only.

### `flyvis`

Repository: <https://github.com/TuragaLab/flyvis>. Docs: <https://turagalab.github.io/flyvis/>.
PyPI `flyvis` 1.2.0, `>=3.9,<3.13`, **MIT**. Connectome- and task-constrained
**differentiable** models of the fly visual system (optic lobe) plus 50+ pretrained models;
tutorials for custom stimuli and training. `pip install flyvis`, then
`flyvis download-pretrained`; `FLYVIS_ROOT_DIR` env var. **Not a spiking whole-brain LIF
simulator** — it is rate-based/differentiable. Steal ideas; not a drop-in engine.

### FlyBrainLab / Neurokernel / GFX

Repository: <https://github.com/FlyBrainLab/FlyBrainLab>. eLife 2021 DOI
<https://doi.org/10.7554/eLife.62362>. **BSD-3-Clause**, last commit 2025-09-29. A JupyterLab
extension (NeuroMynerva) + WAMP backend, **not a standalone app**. **Python 3.11 NOT
supported**: the full-install script pins `PYTHON_VERSION=3.10` with the comment "tested on
python<=3.10"; PyPI 1.1.11 (2024-06-17) deps include `jupyterlab<3.6,>=3.0`,
`graspy<=0.1.1` (abandoned) and `nxt-gem==2.0.1`; full install also needs OrientDB+Java, CUDA,
OpenMPI+mpi4py, torch 1.12.0+cu116. The GFX engine runs circuit models (lamina/medulla/retina),
not the full FlyWire 138k LIF network. Avoid for whole-brain simulation; steal the interactive
circuit-exploration UX.

Neurokernel: <https://github.com/neurokernel/neurokernel>, **BSD-3-Clause** (`LICENSE.rst`;
GitHub's NOASSERTION is a filename artifact). v0.3.1 released 2022-08-02; last commit
2025-09-28 (packaging only). Deps `pycuda>=2020.1`, `mpi4py`, `dill>=0.2.4,<=0.3.3`; README
says `conda create -n nk python=3.7`; docs mention Python 2.7. Pre-FlyWire LPU models, no 3D
display. Avoid.

### Neuromorphic: Loihi 2 and SpiNNaker

Loihi 2 whole-FlyWire (Wang et al., Sandia, arXiv <https://arxiv.org/abs/2508.16792>) — first
biologically realistic whole connectome on neuromorphic hardware: the Shiu FlyWire model (140k
neurons, 15M synapses after merging same-source/target pairs; 50M in the fuller version) mapped
onto **12 Intel Loihi 2 chips** (Kapoho Point, 1440 neurocores). Intermediate layer = Sandia's
STACS simulator with the SNN-dCSR partitioned format; "shared synaptic delivery" (20 chips, 80%
mem util) vs "shared axon routing" (12 chips, 56% mem util, compressed max fan-in from 10,356
to 165). **Readout is the problem**: 992 dedicated spike counters (224 with payload) per
system, overcommitted via payload-carried neuron index; the authors state that requiring
synchronised communication with the embedded CPU "significantly slows down the execution" and
they omitted spike counters when measuring performance. Payload collisions drop <0.1% of spike
indices. Reference Brian2 time: 4419 ms for the sugar experiment. Licence **likely restricted
(UNVERIFIED)**. Avoid (no hardware, and readout costs the speedup); steal the SNN-dCSR
representation and the shared-axon-routing compression idea.

SpiNNaker: **no verified whole-FlyWire implementation found — UNVERIFIED / likely
non-existent at whole-brain scale.**

### Body simulators (closing the motor loop)

`flybody` / NeuroMechFly v2 — `TuragaLab/flybody` (<https://github.com/TuragaLab/flybody>) (MuJoCo
whole-body physics, Vaxenburg et al. *Nature* 643:1312-1320, 2025, DOI
<https://doi.org/10.1038/s41586-025-09029-4>) and NeuroMechFly v2 (Wang-Chen et al. *Nat Methods*
2024, DOI <https://doi.org/10.1038/s41592-024-02497-y>) are the standard body simulators if the
loop is ever closed from decoded motor commands into a physical fly. `eonsystemspbc/flybody`
(<https://github.com/eonsystemspbc/flybody>) is a fork. **Licences UNVERIFIED.**

## Our own measured numbers

`ConnectomeSim` on RTX 3070, full 138,639 neurons / 15,091,983 synapses, dt 0.1 ms:

| Quantity | Measured |
|---|---|
| load + sparse-matrix build | 3.1 s |
| VRAM for the weight matrix | 183.6 MB |
| peak VRAM during stepping | 208 MB |
| throughput | 100 steps (10 ms brain) = 0.10 s wall; 1000 steps (100 ms) = 0.72 s wall |
| realtime factor | **0.13x** (~7.7 s wall per 1 s brain) |

The sparse weight matrix is built as COO -> `coalesce` -> `to_sparse_csr`; CSR `nnz` =
15,091,983.

Validation against brian2 (exact Shiu equations) on a 4,000-neuron subnetwork with 20,000
synapses and an identical prespecified Poisson drive schedule (8,967 input spikes over 200 ms):

> **STALE — superseded, do not cite.** The table below is the pre-fix state. It is kept as the
> record of what the failure looked like. The current result at the same settings is
> brian2 6,868 / ours 6,870 spikes, 322/322 active, Jaccard 1.000, correlation 1.000 — see
> [Resolution](#resolution-the-gain-was-an-integrator-factor-2026-09-22) at the end of this
> document.

| Metric | brian2 | ours (pre-fix) |
|---|---|---|
| total spikes | 6,868 | 8,967 |
| active neurons | 322 | 300 |
| Jaccard | — | 0.932 |
| ratio | — | 1.306 |
| rate correlation | — | 0.862 |

**NOT yet passing.** Two bugs were found and fixed by this harness already: (1) forward Euler
instead of the exact exponential integrator; (2) sensor drive delivered through the synaptic
delay line instead of as an immediate voltage step. The remaining discrepancy and the **lack of
recurrent propagation** (ours shows exactly the 300 driven neurons active, brian2 shows 322) is
an open issue to resolve before the control loop is trusted.

**Why this matters for us:** the validation gap is not a tuning problem; the 300-active-neuron
pattern suggests recurrent spikes are not propagating or are not being counted. Treat the
simulator as untrusted until Jaccard/ratio are explained.

*(Both of those readings turned out to be wrong: the "300 active" was the drive population
being counted instead of the full network, and the gap was the `tau_mem` integrator factor
described at the end of this document.)*

## External drive and readout patterns

The integration surface a live HA control loop needs is: *inject current into arbitrary neuron
indices mid-run*, and *read per-neuron spike identity per window*. Survey results:

| Backend | Mid-run drive | Per-neuron readout | Live-loop viable |
|---|---|---|---|
| `fly-brain-minecraft` | `setInjectedCurrent`, `setStimulusRate`, `forceSpike` | `spikesThisTick`, `rateHz`, `spikeLog*` | **Yes** |
| PyGeNN | `Vstim` `additional_input_vars` + `push_to_device()` | `spike_recording_data[b]` -> `(times, ids)` | **Yes** |
| stonkfly kernel | `stimulation=[(indices, current)]` | `brain.counts` per population | Yes (CPU, 0.5 s windows) |
| Brian2 Python | `PoissonInput` + `SpikeMonitor` | `spike_trains()` per neuron | Yes, but slow |
| Brian2 C++ standalone | none | — | No |
| Brian2CUDA | none | — | No |
| NEST GPU | subprocess per trial | — | No |
| Loihi 2 | counter-gated | costly | No |

The GeNN path is the one to copy for a GPU live loop.

## Connectome data sources

### FlyWire v783 (female, FAFB) — the primary dataset

- Princeton bucket `gs://flywire_v141_m783`: **verified anonymously listable** via the GCS JSON
  API. Top-level prefixes: `16_16_40/ ... 1024_1024_40/` (image pyramid,
  neuroglancer-precomputed), `skeletons_mip_1/`, `skel_with_twigs/`, `mesh_mip_1_err_40/`,
  `archive/`.
- `flywire_v141_m783_pruned` **does not exist (404)**.
- Verified: mesh `.labels` files are ASCII root IDs, e.g.
  `[720575940379373576,720575940379393542,...]`; `mesh_mip_1_err_40/info` =
  `neuroglancer_multilod_draco` sharded (`murmurhash3_x86_128`); `skeletons_mip_1/info` =
  `neuroglancer_skeletons` with radius + `cross_sectional_area`.
- **No auth needed for morphology.** The bucket's voxel segmentation is graphene/
  supervoxel-encoded — cross-section root IDs are **wrong** without the graph server; meshes
  and skeletons are correct by root ID.
- The FlyWire graph server does need auth:
  `prod.flywire-daf.com/segmentation/1.0/flywire_public/info` -> 302 to Google sign-in.
- Licence: **CC BY-NC 4.0** per <https://flywire.ai/guidelines>. Latest public release v783 =
  October 2023 snapshot. See [../licensing.md](../licensing.md).

#### Vendored v783 data (independently verified in this project)

Local paths: `vendor/fly-brain/data/2025_Connectivity_783.parquet` and
`vendor/fly-brain/data/2025_Completeness_783.csv` (see [../../CONTRACT.md](../../CONTRACT.md)).

- `2025_Connectivity_783.parquet` is **exactly 15,091,983 rows**.
- The completeness CSV has **138,640 lines (138,639 neurons + header)**.
- Connectivity columns: `Presynaptic_ID, Postsynaptic_ID, Presynaptic_Index,
  Postsynaptic_Index, Connectivity, Excitatory, Excitatory x Connectivity,
  __index_level_0__`.
- In our run: **9,059,302 excitatory** and **6,032,681 inhibitory** rows, max `|E*C|` = **2405**,
  indices **0..138638**, and **no duplicate `(pre,post)` pairs**.
- Weight convention: `w_mV = 0.275 * Excitatory x Connectivity` (negative => inhibitory), so the
  verified max `|E*C|` = 2405 is the largest per-connection weight input to that scaling.

### Codex public GCS files

Base URL `https://storage.googleapis.com/flywire-data/codex/data/fafb/783/<file>` — **all
verified HTTP 200 anonymous** (no login, despite the Codex web app requiring Google sign-in):

| File | Size | Contents |
|---|---|---|
| `connections.csv.gz` | 50,289,304 B | `pre_root_id,post_root_id,neuropil,syn_count,nt_type` (Buhmann synapses, >=5 convention, one row per pre/post/neuropil) |
| `connections_no_threshold.csv.gz` | 212,093,967 B | all weights |
| `connections_princeton.csv.gz` | ~68.5 MB (size UNVERIFIED, filename verified) | |
| `neurons.csv.gz` | 1,679,884 B | `root_id,group,nt_type,nt_type_score,da_avg,ser_avg,gaba_avg,glut_avg,ach_avg,oct_avg` |
| `classification.csv.gz` | 934,402 B | `root_id,flow,super_class,class,sub_class,hemilineage,side,nerve` |
| `consolidated_cell_types.csv.gz` | 901,707 B | `root_id,primary_type,additional_type(s)` |
| `coordinates.csv.gz` | 5,314,546 B | `root_id,position,supervoxel_id` |
| `fafb_v783_princeton_synapse_table.csv.gz` | 2,695,106,039 B | |
| `labels.csv.gz` | 4.8 MB (UNVERIFIED) | community labels; contains contributor personal names — **do not redistribute** |
| `synapse_coordinates.csv.gz` | ~317 MB (UNVERIFIED) | |

`coordinates.csv.gz` was verified directly: **238,909 rows, 139,255 unique `root_id`, 99,654
duplicate rows** (multiple supervoxels per neuron), **100% coverage** of our 138,639 connectome
neurons after de-duplication on `root_id`. `position` is a string like
`[352484 175164 229040]`; parse with regex `\[\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\]`. Ignore
`supervoxel_id` — the column order in the file means a naive whitespace split yields 4 tokens,
not 3.

**Why this matters for us:** this file is the rendering position source. See the rendering
section of [asset-inventory.md](asset-inventory.md).

### Zenodo mirrors

- DOI [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866) — "FlyWire Whole-brain
  Connectome Connectivity Data" (2024-06-02), licence **cc-by-4.0 per the Zenodo API**
  (conflicts with the NC guidance above; treat as NC). Files: `proofread_connections_783.feather`
  852.0 MB, `flywire_synapses_783.feather` 9.49 GB, `per_neuron_neuropil_count_pre_783.feather`
  16.9 MB, `..._post_783.feather` 233.8 MB, `proofread_root_ids_783.npy` 1.1 MB.
- DOI [10.5281/zenodo.10877326](https://doi.org/10.5281/zenodo.10877326) — Schlegel et al. 2024
  supplementary, **cc-by-4.0**: `sk_lod1_783_healed_ds2.parquet` 5.36 GB (SWC skeletons for all
  FlyWire neurons, readable by `navis.read_parquet` or pyarrow),
  `nblast_flywire_all_right_aba_comp.feather` 809 MB, hemibrain NBLAST x2 (~212/223 MB). All
  root IDs are v783.
- **Synapse-count clarification:** the v783 connectivity dataframe has **15,091,983 rows** =
  unique `(pre,post)` pairs carrying weighted synapse counts. The raw synapse detections are far
  more (`flywire_synapses_783.feather` is 9.5 GB). Shiu's `model.py` uses all 15.1M rows with no
  threshold; fly-brain-minecraft thresholds at **>=5 synapses** on the male CNS (~24% of
  connections, 72% of synapses).

### Male CNS (MCNS) v1.0

Male brain + VNC, 2026. <https://male-cns.janelia.org/>,
<https://neuprint.janelia.org/?dataset=male-cns%3Av1.0>, bulk `gs://flyem-male-cns/v1.0/`
(verified anonymously listable).

- **176,422 neurons; 125,024,863 Neuron->Neuron synapses; 6,287,789 connections at >=5 synapses
  carrying 90,297,299 synapses; 141,781 somata with coordinates.**
- Licence **CC BY 4.0** (paper Berg et al., *Cell* 189(18):5504-5526.e15, 2026-09-03, DOI
  <https://doi.org/10.1016/j.cell.2026.08.015>).
- **Token: NO** — verified anonymous token-less `POST /api/custom/custom` works (GET returns
  401; you must POST with a JSON body; **a bad token is worse than none**). `neuprint-python`
  does **not** work anonymously.
- Bulk files at `v1.0/connectome-data/flat-connectome/`:
  `body-annotations-...-minconf-0.5.feather` 14.5 MB, `body-neurotransmitters-...feather`
  43.3 MB, `connectome-weights-...-traced-only.feather` 508 MB,
  `...-significant-only.feather` 502 MB, `body-stats-...` 778 MB, `syn-partners-...` 6.8 GB,
  `syn-points-...` 13 GB, `tbar-neurotransmitters` 2.65 GB, plus a complete **Neo4j 4.4.16
  dump**, skeletons, meshes, and cross-registered meshes of flywire/banc/manc/hemibrain into
  male-CNS space. Reading Feather needs pyarrow.
- A complete, public, no-auth MaleCNS Neuroglancer state exists:
  <https://storage.googleapis.com/flyem-male-cns/v1.0/male-cns-v1.0.json> (60 KB, 45 layers) —
  see [asset-inventory.md](asset-inventory.md).

**Why this matters for us:** MCNS is CC BY 4.0 (no NC clause), covers brain **and** VNC (real
descending->motor->leg pathways), and is anonymous. It is the commercially clean fallback to
FlyWire v783.

### BANC v888

Female brain + VNC, 2026. Harvard Dataverse
[doi:10.7910/DVN/7WTH1N](https://doi.org/10.7910/DVN/7WTH1N) (v3, licence **CC BY 4.0**, 379
files) — **verified via the male-cns research doc, not by our own fetch**. Files:
`banc_888_edgelist_simple_v3.feather` 359 MB, `banc_888_edgelist_split_v3.feather` 940 MB,
`banc_888_meta.feather` 57.6 MB, `banc_888_metrics.feather`. ~188k neurons (**UNVERIFIED**).
Paper Bates et al., *Nature* 656:957-970 (2026-06-08), DOI
<https://doi.org/10.1038/s41586-026-10735-w>, CC BY 4.0. Also `pip install banc`. Caveat:
different EM modality (GridTape-TEM), so **synapse counts are not directly comparable** to
FIB-SEM.

### MANC v1.2.x

Male VNC. neuPrint `manc:v1.0 / v1.2.1 / v1.2.3`, in the anonymously-available dataset list via
`/api/dbmeta/datasets`. Licence **CC BY 4.0**. **Exact bucket URL and file sizes UNVERIFIED.**
Papers: Takemura et al. *eLife* 13:RP97769 (2024); Marin et al. *eLife* 13:RP97766 (2024);
Cheong et al. *eLife* 13:RP96084.

### hemibrain

Female central brain, ~25k neurons. <https://www.janelia.org/project-team/flyem/hemibrain>,
neuPrint `hemibrain:v1.1 / v1.2.1`, neuroglancer
`gs://flyem-views/hemibrain/v1.2/base.json`.

**Correction to a common assumption: hemibrain is NOT encumbered.** The Janelia page states
verbatim "Hemibrain is licensed under CC-BY". No data use agreement and no account required for
the data; `hemibrain:v1.2.1` appears in the anonymous `/api/dbmeta/datasets` response. Bulk
downloads are anonymous from the `hemibrain` bucket (verified listable: `v1.0/ v1.1/ v1.2/`):

- `http://storage.googleapis.com/hemibrain/v1.0/conn_summary.tgz` = 47,369,620 bytes, HTTP 200
  anonymous.
- v1.2: `https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz`
  (the `storage.cloud.google.com` form requires auth and returns HTML — use
  `storage.googleapis.com`).
- **Beware the decoy bucket: `gs://flyem-hemibrain` returns 401; use `gs://hemibrain`.**

### Pre-computed activity datasets

Shiu et al. Edmond DOI [10.17617/3.CZODIW](https://doi.org/10.17617/3.CZODIW) — **MIT**, no
account (anonymous HTTP 206 verified). `results.zip` = 4.2 GB of parquet, one row per spike,
columns `t` (s), `trial`, `flywire_id`, `exp_name`; filenames carry FlyWire root IDs. FlyWire
v630. **Best pre-computed activity asset**; parse per-experiment rather than unpacking all.
Gap: needs a **v630->v783 remap** (`fafbseg.flywire.update_ids` or Zenodo
`proofread_root_ids_783.npy`).

`eonsystemspbc/fly-brain` spike exports — schema verified as `time_ms, trial, neuron_index,
flywire_id, exp_name`. 600 parquet files; Google Drive folder
`1jiSfb5lNfm9gwP0YyyRz5ATIrDpBAcjs` lists `nature_2026_07.zip` with anonymous HTTP 200, but
actual download without a Google account is **PARTIALLY VERIFIED**. The bundle's own data
licence is **UNVERIFIED**. Direct
`uc?export=download&id=...` returns the normal Google "Virus scan warning" page (=
anonymous access works, big file).

Real whole-brain calcium imaging: CRCNS fly-1 (light-field Ca/voltage <=200 Hz, NIfTI; access
terms **UNVERIFIED**); Dryad [10.5061/dryad.3bk3j9kpb](https://doi.org/10.5061/dryad.3bk3j9kpb)
(CC0, region/PCA `.npy`); BIFROST Dryad
[10.5061/dryad.8pk0p2nx1](https://doi.org/10.5061/dryad.8pk0p2nx1) (CC0). None keyed to FlyWire
root IDs.

**Bottom line: no verified public REAL dataset maps per-neuron activity over time onto FlyWire
root IDs.**

## Version and environment constraints (quick reference)

| Constraint | Detail |
|---|---|
| Python 3.11 target | brian2 2.10.x, scipy 1.18.1, `py-ha-ws-client`, Brian2CUDA 1.0b1, and the `flyvis` upper bound all need care — see [dead-ends.md](dead-ends.md). |
| numpy 2.x | brian2 2.9.0 is the last 3.11-compatible release that handles numpy 2.x; our validation env pins numpy 1.26. |
| GeNN | 5.4.0 from GitHub tag zip, not PyPI; `pkg-config` + `libffi-dev`. |
| NEST GPU | CMake + CUDA 12.x, custom neuron, patched `nestgpu.py`; Linux only. |
| Codex | Anonymous CSV over HTTPS; pyarrow for Feather. |
| MCNS | `POST /api/custom/custom` anonymously; bad token is worse than none. |

## See also

- [asset-inventory.md](asset-inventory.md) — visualisation, UI, Home Assistant, graph and
  annotation assets.
- [dead-ends.md](dead-ends.md) — every trap found in this research, with the alternative.
- [../architecture.md](../architecture.md) — how the simulator is wired into the controller.
- [../licensing.md](../licensing.md) — licence position, including the FlyWire NC conflict.

---

## Open issue: recurrent gain (found 2026-09-21)

Our own validation harness (`validation/validate_against_brian2.py`) compares
`ConnectomeSim` against Brian2 running the published Shiu et al. equations on the same
network with an **identical, prespecified Poisson input schedule**, so any difference is
attributable to the model implementation rather than to differing random draws.

Four silent bugs were found and fixed this way. All of them had the same character: the
simulator ran, produced plausible output, and was wrong.

| # | Bug | Symptom |
|---|---|---|
| 1 | Sparse weight matrix built as `[pre, post]` instead of `[post, pre]` | `torch.sparse.mm(W, spikes)` contracts dim 1, so it returned all zeros and **every synapse in the network was disabled** |
| 2 | Delay line used `torch.roll` over a `delay + 1` slot buffer | A freshly written spike was shifted out of the ring before it was ever read: **complete transmission failure** |
| 3 | Membrane integrator used `(v + g*tau_mem - v_rest)*decay_v - g*tau_mem` | Algebraically different from the correct `v_rest + (v - v_rest)*decay_v + g*tau_mem*(1 - decay_v)`. Excitatory input **hyperpolarised** the cell (-62 mV where -42 mV was correct) |
| 4 | Refractory neurons were allowed to keep integrating | A blocked neuron banked 68.75 mV per step and reached **thousands of mV** |

With those fixed, and with `recurrent_scale = 1/250`, the harness reports:

```
Brian2 spikes total  : 6,813  (active: 311)
ours   spikes total  : 6,762  (active: 300)
active-neuron Jaccard: 0.965
spike-count ratio    : 0.993
rate correlation     : 0.999
RESULT: PASS
```

Driven neurons match **exactly** (identical per-neuron counts across the 300 driven cells).

**The remaining problem:** that agreement is partly hollow. Inspecting the populations
separately shows the driven population matches perfectly while the recurrent one does not:

| Population | brian2 | ours |
|---|---|---|
| driven (0:300) | 6,762 spikes | 6,762 spikes |
| recurrent (300:) | 51 spikes, 11 active neurons | **0 spikes** |

So `recurrent_scale = 1/250` makes recurrent synapses effectively too weak to propagate at
all. With `recurrent_scale = 1.0` (recurrent spikes multiplied by 250, matching the Poisson
drive factor) the network instead runs away: 12,061 spikes, 445 active recurrent neurons,
rate correlation 0.066.

Neither setting reproduces brian2's behaviour. In brian2 a recurrent spike contributes
`w = Excitatory x Connectivity * 0.275 mV` **directly** into `g`, with spike multiplier 1.
Our sparse matrix entries already hold `Excitatory x Connectivity * 0.275`, so
`recurrent_scale` ought to be `1.0` — yet at `1.0` we over-fire by ~2x while brian2 does not,
despite identical weights and identical driven input.

**This is unresolved and is the next thing to fix.** The likely suspects, in order:

1. A residual discrepancy in how a spike is converted into conductance (units or timing).
2. A difference in the refractory/reset interaction on recurrent targets.
3. Conduction of the 250 factor through the reference benchmark implementation, which applies
   `scalePoisson` to recurrent spikes in `run_pytorch.py` — worth re-reading against upstream
   `model.py`, since the two may not be equivalent.

Treat all recurrent-population results with suspicion until this is closed. The driven
population, the sparse connectivity, the delay line and the membrane integration are verified.


---

## Correction: the earlier "PASS" was degenerate (2026-09-21, later)

The validation result reported above (`RESULT: PASS`, correlation 0.999) **did not mean the
model was right.** It passed because `recurrent_scale` was set to `1/250`, which suppresses
recurrent transmission ~250x. Under that setting the driven neurons matched brian2 exactly
*and the recurrent population produced zero spikes* — the network was effectively
disconnected, and the comparison was therefore vacuous.

Testing the physically-correct value (`recurrent_scale = 1.0`, i.e. transmitting
`w = Excitatory x Connectivity * 0.275 mV` straight into `g`, as the reference does) shows the
truth:

| Population | brian2 | ours (`scale=1.0`) |
|---|---|---|
| driven | 6,762 | 6,938 |
| recurrent | 51 spikes, 11 active | 5,123 spikes, 445 active |

So with identical weights and identical driven input, we over-fire recurrently by ~100x. The
cause is **not yet identified** — that remains the open issue.

### The network has no safe middle

A recurrent-gain sweep on the full connectome, driving 256 sensory neurons, shows an
extremely sharp transition:

| `recurrent_scale` | driven | recurrent | active recurrent | recurrent/driven |
|---|---|---|---|---|
| 0.004 | 23,808 | 812 | 75 | 0.03 |
| **0.010** | 23,813 | 12,027 | 701 | **0.51** |
| 0.020 | 23,884 | 32,417 | 1,958 | 1.36 |
| 0.050 | 23,975 | 142,565 | 8,466 | 5.95 |
| 0.300 | 25,872 | 827,515 | 42,003 | 32.0 |
| 1.000 | 27,813 | 1,444,779 | 67,764 | 52.0 |

`0.004` is dead and `0.05` is runaway: there is no wide stable band, which is itself a
finding. A network this close to threshold will be hypersensitive to any calibration change.

`recurrent_scale` is therefore **provisionally set to 0.01** in `ShiuParams`. It is an
interim calibration, not a derivation. It keeps activity in a usable range so the control
loop can be developed, and it must be replaced by a properly calibrated gain before any
result is trusted quantitatively.

### The paper's protocol at the correct scale

Driving with Poisson spikes at 150 Hz and the paper's own event weight (68.75 mV) rather than
a steady current, at `recurrent_scale = 1.0`:

| Population driven | driven spikes | recurrent spikes | active neurons |
|---|---|---|---|
| thermosensory (29) | 746 | 484,279 | **63,910** |
| visual (3,000) | 35,164 | 416,745 | 52,924 |

The published benchmark for a comparable experiment (Sugar GRN, 200 Hz, 100 ms) is **1,558
spikes and 323 active neurons**. Activating 46% of the brain from 29 sensory neurons is not
plausible fly physiology, and it is the clearest evidence that the published `0.275 mV`
weight does not transfer to the v783 edge list unchanged.

This matches the independent recalibrations found in the research: `fly-brain-minecraft`
uses 0.179 mV (gain 0.65) and its own 300+ run brian2 cross-check recommends ~0.125 mV
(gain ~0.45). Those are the numbers to try next, expressed in this codebase as roughly
`recurrent_scale` 0.45-0.65 on top of a reduced `w_scale_mv`.

> **Superseded — do not act on this.** See
> [Resolution](#resolution-the-gain-was-an-integrator-factor-2026-09-22) at the end of this
> document. Every recommendation in this section rests on a bug, and the published weight does
> in fact transfer unchanged.

### A second, independent finding: descending neurons are unreachable

Driving the sensory population hard and measuring where activity lands:

| Population | extra spikes from a strong sensory drive |
|---|---|
| hygrosensory (74) | 9,694 |
| thermosensory (29) | 3,799 |
| ALPN (685) | 697 |
| ALLN (429) | 682 |
| ... everything else | 27 |
| descending (1,299) | **23** |

A steady sensory drive is absorbed almost entirely by the sensory neurons themselves. The
antennal-lobe populations are the only ones that respond usefully within a 300 ms window,
which is why `TemperatureColourLoop` listens to them rather than to the descending/motor
output, and why it drives spike-rate current rather than the paper's 68.75 mV events (which
kick a neuron ~10x past threshold and produce a runaway). **This finding survives the
resolution below** — it was measured with a strong drive and is a property of where a sensory
signal lands, not of the gain.


---

## Resolution: the "gain" was an integrator factor (2026-09-22)

Everything concluded above about the connectome, the v783 edge list, the published weight not
transferring, and the need for a gain sweep was **wrong**. The cause was one missing factor in
the membrane update.

### What was actually wrong

The step function integrated the membrane as

```
v(t+dt) = v_rest + (v(t) − v_rest)·decay_v + g(t)·tau_mem·(1 − decay_v)
```

The exact solution of the coupled linear (v, g) system is

```
v(t+dt) = v_rest + (v(t) − v_rest)·decay_v + g(t)·alpha
alpha   = tau_syn/(tau_mem − tau_syn) · (decay_v − decay_g)
```

At `dt = 0.1 ms`, `tau_mem = 20`, `tau_syn = 5`:

| Coefficient | Value | |
|---|---|---|
| `alpha` (correct) | 0.004938 | |
| `tau_mem·(1 − a)` (what the code did) | 0.09975 | **20.2× too large** |
| `1 − a` (constant-g approximation) | 0.004988 | 1% high, now replaced by `alpha` |

So **every synapse in the connectome was exactly 20× too strong**, and every downstream
conclusion followed from it. The "provisional" `recurrent_scale = 0.01` was not a calibration
at all — it was cancelling a factor of 20.

### Why it took so long to find

Three things conspired:

1. **The symptoms looked like the connectome.** The network was bistable in a way that suggested
   a real dynamical property: 0.004 dead, 0.01 balanced, 0.05 runaway. That is exactly what a
   network sitting near threshold looks like, and it made a gain sweep seem like the right
   experiment to run.
2. **A validation run passed while proving nothing.** The `recurrent_scale = 1/250` "PASS"
   reported correlation 0.999 while the recurrent population produced *zero* spikes. A green
   result with a silently disabled subsystem is worse than a red one.
3. **External numbers made the wrong story fit.** Two independent recalibrations from
   `fly-brain-minecraft` (0.179 mV, ~0.125 mV) appeared to confirm that the published weight
   needed scaling down.

### What actually found it

Not a sweep. A **closed-form micro-test** (`validation/micro_gain_check.py`) compares the peak
membrane deflection caused by a single synaptic event against the analytic double-exponential
peak. For the published time constants that peak is exactly `0.15749·w`:

```
u_max = w · tau_syn/(tau_mem − tau_syn) · (exp(−t*/tau_mem) − exp(−t*/tau_syn))
t*    = tau_mem·tau_syn/(tau_mem − tau_syn) · ln(tau_mem/tau_syn)
```

With `w = 5 mV` the answer is 0.7875 mV. The broken simulator produced 0.1591 mV at the
then-current settings — off by exactly the `tau_mem` factor once `recurrent_scale` was
accounted for. **A two-neuron network localised a bug that a 15-million-synapse comparison had
been attributing to the data.**

The general lesson: when a physical parameter appears to need a ~100× fudge factor, suspect the
units or the arithmetic, not the physics.

### After the fix

| Check | Result |
|---|---|
| micro: 1 event, closed form / ours / brian2 | 0.7875 / 0.7874 / 0.7874 mV |
| micro: 4 cases (sub, supra, chain, inhibition) | spike counts match brian2 in all 4 |
| 4,000-neuron augmented slice, 20,000 synapses | 6,868 vs 6,870 spikes; 322 vs 322 active; Jaccard 1.000; corr 1.000 |
| 20,000-neuron real slice, 318,232 synapses | 5,240 vs 5,241 spikes; 359 vs 359 active; Jaccard 1.000; corr 1.000 |
| 60,000-neuron real slice, 2,807,417 synapses | 6,370 vs 6,418 spikes; 1,373 vs 1,376 active; Jaccard 0.964; corr 0.998 |

The third row was added afterwards, specifically to test the one remaining extrapolation: every
earlier comparison used a slice, and a slice is far sparser than the real brain (in-degree 15.9 at
20k neurons against 108.9 for the whole connectome). At 60k neurons the in-degree reaches 46.8,
nearly 3x denser, and the agreement holds — Jaccard 0.964 rather than 1.000, which is the
expected direction as recurrent coupling rises. The full network is still beyond brian2's reach,
so density above 46.8 is supported by extrapolation, not verified.

`recurrent_scale` is now `1.0` and `w_scale_mv` is the published `0.275 mV`.

### Corrections to the record above

- "the published `0.275 mV` weight does not transfer to the v783 edge list unchanged" — **false**.
  It transfers exactly.
- "driving 29 thermosensory neurons activates 63,910 neurons (46% of the brain)" — an artefact
  of 20× weights. It does not happen at the published weight.
- "there is no wide stable band" — there is no such band *for a network with 20× weights*. The
  gain sweep table is retained above as the record of a wrong turn, **not as guidance**, and
  neither is the "numbers to try next" recommendation, which should not be followed.
- The "published benchmark of 323 active neurons" and our own brian2 reference's 322 active
  neurons are the same number. That agreement was available the whole time; the harness was
  reporting it while the docs described it as unreachable.

### Follow-on corrections in the control loop

Fixing the gain exposed three further problems in the temperature→colour loop. All were
measured (`tools/readout_probe.py`, held-out split) rather than guessed:

| Problem | Measured effect | Fix |
|---|---|---|
| Readout collapsed the population to one rate per colour band, using disjoint random slices | The three features were collinear at r = 0.996–1.000, so the decoder had one effective dimension; held-out ceiling ~155 K | Per-neuron rates, held-out ceiling ~20 K |
| Soft Gaussian band labels + softmax | The label scheme is itself 151 K away from the ideal mapping, so no feature set could do better | Regress the Kelvin value directly |
| Sensory rate ran from 0 Hz at the cold end | At the coldest temperature no current was injected at all; the readout population was completely silent | 20–120 Hz baseline, as real thermoreceptors have |
| `evaluate()` reused the training temperatures | Reported in-sample fit quality as accuracy | Evaluate strictly between training points |

Net effect: **531 K → 28 K** held-out mean error, correlation 0.9992, monotone 8/8.
