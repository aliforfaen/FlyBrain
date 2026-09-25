# Dead ends and traps

A skimmable list of things that look attractive and are not. Each entry states what looks
appealing, why it fails, and what to do instead. Engine and benchmark context is in
[simulation-backends.md](simulation-backends.md); asset context is in
[asset-inventory.md](asset-inventory.md). Licence detail is in [../licensing.md](../licensing.md).

## Simulation engines and speed

- **Brian2 C++ standalone as the live control engine.** *Appealing:* it is the fastest CPU
  backend in the benchmark (0.35x realtime) and matches the reference model exactly. *Fails:*
  C++ standalone mode **does not allow Python-in-the-loop interaction at all** — you cannot
  inject drive between steps, which is the entire job. *Instead:* keep Brian2 as validation
  ground truth only, in `.venv-validation` with `numpy 1.26`, and run the live loop on PyGeNN or
  an active-set kernel.

- **Brian2CUDA for live injection.** *Appealing:* CUDA backends sound like the answer on an
  RTX 3070. *Fails:* its standalone/CUDA mode is one-shot with no Python-in-the-loop, and it is
  actually **slower than Brian2 CPU at `n_run=1`** (10.9-13.3 s vs 2.9 s for 1 s of sim).
  *Instead:* PyGeNN, if moving to GPU. Avoid Brian2CUDA entirely.

- **Tuning the current PyTorch `ConnectomeSim` into real time.** *Appealing:* it already works,
  loads in 3.1 s and uses only ~184 MB of VRAM for weights. *Fails:* it measured **0.13x
  realtime** on the 3070 (1000 steps = 100 ms brain time = 0.72 s wall) — about 7 s wall per 1 s
  of brain time — matching the PyTorch row of the benchmark (0.10x). More tuning will not close
  a 7x gap. *Instead:* adopt the active-set integrator from `blendi-remade/fly-brain-minecraft`
  (only integrate neurons away from rest / in refractory / with pending input) or move to
  PyGeNN.

- **NEST GPU for a live loop.** *Appealing:* 0.70-1.11x realtime, the only other near-realtime
  backend. *Fails:* fly-brain's own architecture table says "Subprocess per trial (cannot reset
  in-process)" — **you cannot re-inject drive mid-run in one process**. It also needs a
  from-source CMake + CUDA 12.x build with a custom `user_m1.{h,cu}` neuron and a patched
  `nestgpu.py` (weight-array init, lines 2225-2227), and has no Windows path. *Instead:* PyGeNN.

- **Brian2GeNN.** *Appealing:* "GeNN speed with a Brian2 front end". *Fails:* 1.7.0 pins
  `Brian2<2.6` while Brian2CUDA needs 2.8/2.10, so it needs its own conda env; it needs the
  legacy GeNN 4.x CLI (`genn-buildmodel.sh`) and `BRIAN2GENN_GENN_PATH`/`GENN_PATH`; speed is
  only 0.52-0.81x realtime and **build_time is huge** with no benefit over direct PyGeNN.
  *Instead:* direct PyGeNN 5.4.0.

- **Installing PyGeNN from PyPI.** *Appealing:* `pip install genn` looks canonical. *Fails:*
  the PyPI `genn` 0.7.7 package is an **unrelated 2021 project**; PyGeNN is not on PyPI.
  *Instead:* `pip install https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip`, plus
  `pkg-config` and `libffi-dev`.

- **Assuming GeNN's benchmark speed transfers exactly to the 3070.** *Appealing:* "1.79-2.10x
  realtime". *Fails:* that is an **RTX 4070 under WSL2**; the 3070 figure of ~0.8-1.4x is
  **UNVERIFIED on 3070**. *Instead:* budget for the slower end and measure before committing the
  control-loop design.

- **Trusting the Loihi 2 whole-FlyWire result as a readout solution.** *Appealing:* first
  biologically realistic whole connectome on neuromorphic hardware, 12 Intel Loihi 2 chips
  (Kapoho Point, 1440 neurocores), "shared axon routing" cutting max fan-in from 10,356 to 165.
  *Fails:* the readout is 992 dedicated spike counters (224 with payload) overcommitted via
  payload-carried neuron index; the authors state that synchronised communication with the
  embedded CPU "significantly slows down the execution" and they **omitted spike counters when
  measuring performance**. Payload collisions drop <0.1% of spike indices. Licence is likely
  restricted (**UNVERIFIED**), and we have no hardware. *Instead:* steal the SNN-dCSR
  representation and the shared-axon-routing compression idea only.

- **Hunting for a SpiNNaker whole-FlyWire implementation.** *Appealing:* it is the obvious
  other neuromorphic platform. *Fails:* **no verified whole-FlyWire implementation was found —
  UNVERIFIED / likely non-existent at whole-brain scale.** *Instead:* ignore.

- **`flyvis` as a whole-brain engine.** *Appealing:* connectome- and task-constrained, 50+
  pretrained models, MIT, `pip install flyvis`. *Fails:* it models the **optic lobe only**, and
  it is **rate-based/differentiable, not a spiking whole-brain LIF simulator**. *Instead:* steal
  ideas for visual-system modelling; do not treat it as a drop-in engine.

- **FlyBrainLab / Neurokernel / GFX on Python 3.11.** *Appealing:* BSD-3, an interactive
  circuit-exploration UX, eLife 2021 (DOI <https://doi.org/10.7554/eLife.62362>). *Fails:*
  Python 3.11 is **NOT supported** — the full-install script pins `PYTHON_VERSION=3.10` with the
  comment "tested on python<=3.10"; PyPI 1.1.11 (2024-06-17) deps include `jupyterlab<3.6,>=3.0`,
  `graspy<=0.1.1` (abandoned) and `nxt-gem==2.0.1`; full install also needs OrientDB+Java, CUDA,
  OpenMPI+mpi4py, torch 1.12.0+cu116. It is a JupyterLab extension, not an app, and GFX runs
  lamina/medulla/retina circuits, not the full 138k network. *Instead:* steal the interactive UX
  ideas; do not build on it.

- **Neurokernel as a FlyWire engine.** *Appealing:* BSD-3-Clause. *Fails:* v0.3.1 was released
  2022-08-02 and the last commit (2025-09-28) was packaging only; deps `pycuda>=2020.1`,
  `mpi4py`, `dill>=0.2.4,<=0.3.3`; README says `conda create -n nk python=3.7` and docs mention
  Python 2.7; it models pre-FlyWire LPUs and has no 3D display. *Instead:* avoid.

- **`erojasoficial-byte/fly-brain` "Embodied Drosophila" as a code base.** *Appealing:* exactly
  our counts (138,639 neurons / 15,091,983 synapses), PyTorch GPU, NeuroMechFly v2 / MuJoCo
  embodiment, MIT, Zenodo DOI <https://doi.org/10.5281/zenodo.19152238>. *Fails:* it is a 2026
  **single-author student preprint with no peer review**, likely AI-assisted; the claimed 5 kHz
  timestep, "81% vs 47% escape" individuality claim and "76,034 divergent synapses after 24
  hours" are extraordinary; **its actual code quality is UNVERIFIED** (file-listing fetches
  failed). *Instead:* treat it as a parts list only.

- **`flybrain` PyPI as a drop-in for *our* connectome.** *Appealing:* MIT, fast (1.4 ms/step on
  an RTX 4060 laptop), ~210 MB VRAM, and a documented browser export format. *Fails:* it ships
  **MaleCNS v1.0 (166,700 neurons)**, not FlyWire v783 (138,639). *Instead:* use it for the
  export format and performance ideas; do not assume its indices or types match ours.

- **Expecting stonkfly to provide a web UI.** *Appealing:* it drives a live brain. *Fails:*
  stonkfly has **no HTTP server, no websocket, no HTML/JS/CSS/Node anywhere**; its interface is
  stdout JSON (`cli.py:262-274`), `runs/*/events.jsonl`, `latest.json`, `latest-input.png`, plus
  the `status` subcommand reading SQLite. Any web UI belongs to fly-wirehead only (and that one
  is unlicensed — see below). *Instead:* build the UI ourselves.

## Numerics and model validity

- **Forward Euler integration.** *Appealing:* trivial to code, looks fine at 0.1 ms. *Fails:*
  it over-delivers drive and inflates firing rates by tens of percent; this was bug (1) found by
  our own validation harness. *Instead:* use the exact exponential integrator (Brian2's
  `linear` method) as in [../architecture.md](../architecture.md).

- **Delivering external drive through the synaptic delay line.** *Appealing:* reuse the synapse
  path. *Fails:* the reference model adds drive to `v` directly (`voltage_stim`); routing it
  through the 1.8 ms delay line was bug (2) found by validation. *Instead:* apply it as an
  immediate voltage step.

- **Trusting a validation "PASS" without checking it is non-empty.** *Appealing:* Jaccard 0.965,
  ratio 0.993 and rate correlation 0.999 on the 4,000-neuron comparison look like agreement.
  *Fails:* that run passed because the network had been silenced by a 250x gain reduction — it
  matched the driven neurons exactly while producing **zero** recurrent spikes against brian2's
  51. *Instead:* always assert that the measured system is actually doing something before
  believing a similarity score, and check sub-populations separately rather than trusting a total.

- **Accepting a parameter that needs a ~100x fudge factor.** *Appealing:* a published weight that
  seems not to transfer, plus two external recalibrations recommending 0.45–0.65x of it, reads as
  a modelling discrepancy to be swept. *Fails:* it was arithmetic. A missing `tau_mem` factor made
  every synapse in the connectome **20.2x** too strong, and the "provisional"
  `recurrent_scale = 0.01` was cancelling it — the whole bistable-gain story (0.004 dead, 0.01
  balanced, 0.05 runaway) was an artefact. With the integrator corrected the published `0.275 mV`
  transfers unchanged and reproduces brian2 exactly (Jaccard 1.000, correlation 1.000).
  *Instead:* when a physical parameter appears to need a double-digit correction, pin the
  arithmetic against a closed form on a two-neuron network **before** touching the data. See
  [simulation-backends.md](simulation-backends.md#resolution-the-gain-was-an-integrator-factor-2026-09-22).

- **Letting an external calibration number explain a local discrepancy.** *Appealing:*
  `fly-brain-minecraft`'s 0.179 mV and ~0.125 mV looked like independent corroboration that the
  published weight needed scaling down. *Fails:* those are *behavioural* gains for a different
  network (male CNS edge table, plus an antennal-lobe correction), not evidence about female v783.
  Treating them as evidence made a wrong story fit and delayed the real fix by a long way.
  *Instead:* use an external number as corroboration only once your own implementation has been
  pinned against something with a known answer.

- **Copying `fly-brain-minecraft`'s stimulus gains unexamined.** *Appealing:* it is a tuned,
  running model. *Fails:* it recalibrated gain from Shiu's 0.275 mV to **~0.179 mV (gain 0.65)**
  because the male CNS has **1.24x per-neuron synaptic input**, and an independent 300+ run
  Brian2 cross-check (`gaps/gap-1.md`) recommends **~0.125 mV (gain ~0.45)** plus an
  antennal-lobe correction. Its own notes record that **delta synapses ran away even at 10%
  gain**, **divisive in-degree normalisation silenced MN9 and the giant fibre**, and
  **spike-frequency adaptation killed projection-neuron responses**. *Instead:* treat gain as a
  calibrated parameter with a documented source. When we finally re-derived it for female v783 the
  answer was the published `0.275 mV` itself — the apparent need to scale it down was our bug, not
  a property of this network.

- **Expecting linear sensory response from the antennal lobe.** *Appealing:* drive ORNs harder
  for a stronger signal. *Fails:* the model **saturates at any ORN rate >= 25 Hz**, and the ON
  visual pathway is **structurally unreachable in a silent LIF because L1 is glutamatergic**.
  Kenyon cells needed a **0.25 postsynaptic gain**. *Instead:* design encoders around the
  saturating regime (as stonkfly does with `30*x/(0.02+x)`) and do not rely on ON-pathway
  readout.

- **Enlarging the timestep or pruning the graph to go faster.** *Appealing:* an easy speedup.
  *Fails:* stonkfly's kernel deliberately does neither; the correct optimisation is the
  event-driven active set. *Instead:* implement the active-set integrator (drop a cell only when
  `v`, instantaneous drive **and** asymptotic drive are all sub-threshold) and materialise all
  states at the end so no threshold is missed.

## Python and dependency versions

- **brian2 2.8 on numpy 2.x.** *Appealing:* 2.8 is the version most tutorials assume. *Fails:*
  it breaks with `AttributeError: module 'numpy' has no attribute 'NPY_OWNDATA'`; Debian bug
  #1114696 (brian 2.8.0.4 FTBFS on NumPy 2.3) was closed as fixed in `brian/2.9.0-1`.
  *Instead:* brian2 **2.9.0 (2025-05-14, `requires_python >=3.10`)** is the numpy-2.x-capable
  release that still works on Python 3.11; our validation env pins numpy 1.26 anyway.

- **brian2 2.10.x / brian2cuda 1.0b1 on Python 3.11.** *Appealing:* newest releases. *Fails:*
  2.10.0 (2025-12-04) and 2.10.1 (2025-12-05) require **Python >=3.12 and numpy>=2.0.0**;
  brian2cuda 1.0b1 pins `brian2==2.10.1`, so it needs Python 3.12+. *Instead:* stay on brian2
  2.9.0 for the 3.11 validation env; do not adopt brian2cuda.

- **`py-ha-ws-client` as the HA WebSocket client.** *Appealing:* Apache-2.0, good API
  (`subscribe_events`, `subscribe_trigger`, `call_service`, auto-reconnect + subscription
  re-registration), released 1.0.0 (2026-08-30). *Fails:* it requires **Python >=3.12**,
  incompatible with our 3.11 target. *Instead:* use `hass-client` 1.3.1 (2026-08-24), which
  requires **Python >=3.11** exactly and has a single dependency, `aiohttp>=3.8.4`.

- **`scipy.sparse` 1.18.1 on Python 3.11.** *Appealing:* latest. *Fails:* 1.18.1 requires
  **Python >=3.12**; on 3.11 you get 1.17.x. *Instead:* accept 1.17.x, which is fine for CSR and
  `csgraph`.

- **Assuming `navis-flybrains` / `cocoa` are safe dependencies.** *Appealing:* they fill real
  gaps (template-brain transforms between hemibrain/FAFB/JRC2018, comparative connectomics
  clustering). *Fails:* **licence and liveness are UNVERIFIED** for both. *Instead:* verify
  before depending; `pip install navis-flybrains` is not sufficient diligence.
  `navis-flybrains`: <https://github.com/navis-org/navis-flybrains>. `cocoa`:
  <https://github.com/flyconnectome/cocoa>.

- **`neuprint-python` as an anonymous MCNS client.** *Appealing:* the standard neuPrint client,
  BSD-3 (**UNVERIFIED**), 0.6.3, `py>=3.9`, <https://github.com/connectome-neuprint/neuprint-python>.
  *Fails:* it requires a token — `Client()` raises
  without one — and does **not** work anonymously against `male-cns:v1.0`. *Instead:* use the
  verified anonymous `POST /api/custom/custom` endpoint.

- **`fafbseg-py` on Windows.** *Appealing:* it has the v630->v783 remap helper
  (`flywire.update_ids`) and mesh/skeleton/connectivity helpers. *Fails:* **Windows is officially
  unsupported** ("I highly recommend you install and use WSL"), it needs a FlyWire API token, and
  it is a **remote-service client, not an offline library**. *Instead:* on Linux, use it for the
  remap only; prefer offline Codex/Zenodo data.

## Data, mirrors and licensing

- **Trusting the Zenodo `cc-by-4.0` tag for FlyWire.** *Appealing:* Zenodo DOI
  [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866) and
  [10.5281/zenodo.10877326](https://doi.org/10.5281/zenodo.10877326) are tagged `cc-by-4.0`,
  which would permit commercial use. *Fails:* <https://flywire.ai/guidelines> states verbatim
  "FlyWire's public release data is made available under license CC BY-NC 4.0" — a genuine
  conflict between two official-looking sources. *Instead:* treat as **CC BY-NC 4.0**. If a
  commercially clean path is ever needed, use MaleCNS v1.0, BANC v888, MANC v1.2.x or hemibrain
  v1.2.1, all CC BY.

- **Vendoring `eonsystemspbc/fly-brain` code.** *Appealing:* it ships the connectivity parquet
  and completeness CSV we want, plus a working GeNN script. *Fails:* the project is
  **GPL-2.0-or-later**. Its `code/paper-phil-drosophila/LICENSE` is separately **MIT**
  ("Copyright (c) 2023 Philip Shiu and Nico Spiller"), which protects only that subtree.
  *Instead:* keep `vendor/fly-brain/` as a **data directory** (as
  [../licensing.md](../licensing.md) records) or replace it with the upstream MIT repository;
  port ideas, not GPL code.

- **Forking `mattyhempstead/fly-wirehead`.** *Appealing:* it is the closest existing analogue to
  the whole project — 166,700 neurons / 25.6M connections, Python 3.11 + C++17 kernel via
  ctypes, Three.js scene, live overlay, checkpointing, loopback server with session token.
  *Fails:* there is **no root LICENSE**, `LICENSE`/`LICENSE.md`/`LICENSE.txt`/`COPYING` all 404
  on `main` and `master`, `pyproject.toml` has no `license` field, `git ls-files` (61 tracked
  files) shows only `licenses/stonkfly-MIT.txt` and `dist/vendor/THREE-LICENSE.txt`, and no
  licence was ever deleted (`git log --diff-filter=D`). Its `THIRD_PARTY.md` claims "The 3D
  scene is original to this repository" while granting no licence for it. **Default copyright
  applies: all rights reserved.** *Instead:* read it for architecture; reimplement. The MIT
  parts are only `flywirehead/neural/**` + `data.py` (inherited from stonkfly) and vendored
  Three.js.

- **Forking `cnqso/infinite-sugar`.** *Appealing:* browser whole-brain emulation, TS + three.js
  + cannon-es. *Fails:* **NO LICENSE file -> unlicensed.** *Instead:* steal ideas only. (Also
  note its neural map is a sampled point/connection canvas; the README says "Lines join
  representative positions and do not trace the shapes of neurons".)

- **Redistributing Codex `labels.csv.gz`.** *Appealing:* community cell labels are convenient.
  *Fails:* the file (4.8 MB, size **UNVERIFIED**) **contains contributor personal names**.
  *Instead:* do not redistribute; derive what you need locally.

- **Redistributing `flyconnectome/flywire_annotations` as a product.** *Appealing:* it is the
  best annotation table and is fetched raw with no token. *Fails:* the repository has **NO
  LICENSE file** (GitHub licence field is null). *Instead:* attribute the underlying papers
  (Dorkenwald et al. 2024, Schlegel et al. 2024, Matsliah et al. 2024) and do not ship the table
  standalone.

- **Treating `flyconnectome/flywire_annotations` and Codex as interchangeable.** *Appealing:*
  both give cell types. *Fails:* the annotations repo explicitly supersedes Codex — "Codex
  presents a mix of annotations from different sources which likely diverge". *Instead:* use
  flywire_annotations for cell typing, and Codex only for the specific per-neuron tables it
  uniquely provides (neurotransmitter averages, coordinates).

- **Assuming hemibrain is encumbered.** *Appealing:* caution about data use agreements.
  *Fails:* the Janelia page states verbatim "Hemibrain is licensed under CC-BY"; no data use
  agreement and no account are required, and `hemibrain:v1.2.1` appears in the anonymous
  `/api/dbmeta/datasets` response. *Instead:* use it freely if a ~25k-neuron central brain
  helps — just do not expect whole-brain coverage.

- **Fetching hemibrain from the wrong bucket or URL form.** *Appealing:* the names look
  plausible. *Fails:* **`gs://flyem-hemibrain` returns 401 — the decoy bucket**; use
  `gs://hemibrain`. And `storage.cloud.google.com` requires auth and returns HTML — use
  `storage.googleapis.com`. *Instead:* verified anonymous:
  `http://storage.googleapis.com/hemibrain/v1.0/conn_summary.tgz` (47,369,620 bytes, HTTP 200)
  and `https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz`.

- **Using the FlyWire graph server without auth.** *Appealing:* the bucket is anonymous, so the
  graph server might be too. *Fails:* `prod.flywire-daf.com/segmentation/1.0/flywire_public/info`
  returns **302 to Google sign-in**. *Instead:* use meshes and skeletons from the bucket, which
  are anonymous and correct by root ID.

- **Reading FlyWire voxel segmentation root IDs.** *Appealing:* cross-section IDs look like
  neuron IDs. *Fails:* the bucket's voxel segmentation is **graphene/supervoxel-encoded**, so
  cross-section root IDs are **wrong** without the graph server. *Instead:* use meshes and
  skeletons (correct by root ID) or the Codex coordinate table.

- **Looking for `flywire_v141_m783_pruned`.** *Appealing:* a pruned variant would be smaller.
  *Fails:* it **does not exist (404)**. *Instead:* prune locally from the connectivity table.

- **Assuming a Google account is needed for the spike bundle.** *Appealing:* Drive links
  usually require sign-in. *Fails:* the folder `1jiSfb5lNfm9gwP0YyyRz5ATIrDpBAcjs` lists
  `nature_2026_07.zip` and the embeddedfolderview HTML works anonymously; direct
  `uc?export=download&id=...` returns the normal Google "Virus scan warning" page, which means
  anonymous access works for a big file. *But* the actual download without a Google account is
  only **PARTIALLY VERIFIED**, and the bundle's own data licence is **UNVERIFIED**. *Instead:*
  prefer the Shiu Edmond corpus, which is anonymously verified (HTTP 206).

- **Using the Shiu activity corpus without remapping IDs.** *Appealing:* 4.2 GB of per-spike
  parquet with FlyWire IDs, MIT, anonymous. *Fails:* it is **FlyWire v630**, and our network is
  **v783** — the IDs differ. *Instead:* remap with `fafbseg.flywire.update_ids` or Zenodo
  `proofread_root_ids_783.npy` before joining.

- **Fetching `README.md` from `philshiu/Drosophila_brain_model`.** *Appealing:* the normal
  filename. *Fails:* the file is `Readme.md` (capital R, lowercase m), so a raw `README.md`
  fetch 404s. *Instead:* fetch `Readme.md`.

- **Using raw synapse detections when you want connections.** *Appealing:* "more data is
  better". *Fails:* `flywire_synapses_783.feather` is **9.49 GB** of raw detections; the
  connectivity dataframe's **15,091,983 rows** are unique `(pre,post)` pairs with weighted
  counts. Shiu's `model.py` uses all 15.1M rows with no threshold; fly-brain-minecraft
  thresholds at **>=5 synapses** (~24% of connections, 72% of synapses on the male CNS).
  *Instead:* decide the threshold deliberately and document it.

- **Comparing BANC synapse counts with FlyWire/MaleCNS counts.** *Appealing:* both are fly
  connectomes. *Fails:* BANC uses a **different EM modality (GridTape-TEM)**, so counts are not
  directly comparable to FIB-SEM. *Instead:* compare within a modality, or normalise.

- **Sending a bad token to the MCNS API.** *Appealing:* supply a token to be safe. *Fails:*
  GET returns 401; the verified anonymous path is `POST /api/custom/custom` with a JSON body,
  and **a bad token is worse than none**. *Instead:* omit the token entirely.

- **Waiting for a FlyWire Codex live-query API.** *Appealing:* the Codex web app is rich.
  *Fails:* it provides **read-only static snapshots** and explicitly "intentionally does not
  provide a general programmatic live-query API"; bulk access is static gzipped CSV downloads.
  It also **cannot display our activity**. *Instead:* use Codex as a ground-truth data source and
  UI inspiration.

- **Hunting for a real activity dataset keyed to FlyWire root IDs.** *Appealing:* it would let
  us validate the whole network directly. *Fails:* CRCNS fly-1 (access terms **UNVERIFIED**),
  Dryad [10.5061/dryad.3bk3j9kpb](https://doi.org/10.5061/dryad.3bk3j9kpb) (CC0) and BIFROST
  Dryad [10.5061/dryad.8pk0p2nx1](https://doi.org/10.5061/dryad.8pk0p2nx1) (CC0) are **not keyed
  to FlyWire root IDs**; everything that is keyed by root ID is simulated (Shiu, fly-brain).
  *Instead:* accept that ground truth is simulated, and validate simulator-vs-simulator.

## Data parsing gotchas

- **De-duplicating the annotation table on `supervoxel_id`.** *Appealing:* it is the first
  column. *Fails:* `root_id` is the neuron; 139,248 rows cover only 138,639 neurons, and a
  neuron can appear more than once. Many rows also have empty `cell_class`/`cell_sub_class`/
  `supertype`/`cell_type`. *Instead:* de-duplicate on `root_id` (we verified **100.0% coverage**
  of our 138,639 neurons that way) and memoise the join to
  `data/annotations/flywire_annotations_supl1.indexed.parquet`.

- **Whitespace-splitting Codex `coordinates.csv.gz`.** *Appealing:* `position` looks like three
  integers in brackets. *Fails:* the column order means a naive whitespace split yields **4
  tokens, not 3**; `supervoxel_id` is present and should be ignored. *Instead:* parse with regex
  `\[\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\]`.

- **Assuming one coordinate row per neuron.** *Appealing:* a per-neuron table. *Fails:*
  `coordinates.csv.gz` has **238,909 rows / 139,255 unique root_id = 99,654 duplicate rows**
  (multiple supervoxels per neuron). *Instead:* de-duplicate on `root_id`; we verified 100%
  coverage of our 138,639 neurons after doing so.

- **Looking for `seung-lab/meshparty` or `seung-lab/nglui`.** *Appealing:* those are the
  historically cited URLs. *Fails:* both 404. *Instead:* the live repositories are
  **`CAVEconnectome/MeshParty`** (Apache-2.0, 2.0.3, 2025-07-10, `py>=3.10`) and
  **`CAVEconnectome/nglui`** (MIT, 4.8.0, 2026-09-11, `py>=3.10`).

- **Looking for `FlyBrainLab/neu3d`.** *Appealing:* the FlyBrainLab namespace. *Fails:*
  `FlyBrainLab/neu3d` 404s; the live repo is **`fruitflybrain/neu3d`**
  (<https://github.com/fruitflybrain/neu3d>) (ISC per `package.json`, **no LICENSE file**), npm
  1.1.4 (2023-08-18), last commit 2024-12-30.

- **Looking for `navis-org/fafbseg-py` docs at the old org.** *Appealing:* consistency with the
  `flyconnectome` org. *Fails:* the org partially migrated — `fafbseg-py` docs now list source
  as <https://github.com/navis-org/fafbseg-py> and navis lives at
  <https://github.com/navis-org/navis>; `flyconnectome` still holds `flywire_annotations` and
  `cocoa`. *Instead:* use the navis-org URLs for navis/fafbseg-py.

- **Expecting `eonsystemspbc/drosophila_brain_model_lif` to be a distinct project.**
  *Appealing:* a second implementation to compare against. *Fails:* it is a **mirror of
  philshiu's repo, not distinct**. *Instead:* compare against `philshiu/Drosophila_brain_model`
  itself.

- **Expecting a reusable Python API from `eonsystemspbc/fly-brain`.** *Appealing:* it is a
  benchmark suite with six backends. *Fails:* it is a **CLI only** (`main.py` ->
  `code/benchmark.py` -> `run_*.py`), no package, no `__init__.py`. *Instead:* use its committed
  data and `run_genn.py`; call `code/paper-phil-drosophila/model.py` + `utils.py` (MIT) when an
  API is needed.

## Browser, rendering and UI

- **Drawing the 15M edges.** *Appealing:* seeing the whole wiring diagram is compelling.
  *Fails:* there is **no authoritative, reproducible in-browser benchmark at 139k nodes /
  15M directed edges**; all "millions of edges in a browser" numbers are vendor-positioned.
  sigma.js's own docs say smooth at "tens of thousands" and tuning matters at "hundreds of
  thousands" of edges. *Instead:* subset aggressively — ego/neighbourhood graphs, supernode
  aggregation to cell types/neuropils/hemilineages with summed synapse weights, top-K by weight
  plus a minimum-synapse threshold, and LOD/stochastic edge sampling.

- **Trusting Graphistry's "20M+ edges" claim.** *Appealing:* it is the headline number for
  browser graph rendering. *Fails:* it is a **vendor claim (July 2026), not independently
  benchmarked**, and the rendering server is **proprietary** with self-host downloads gated
  behind their support portal (`pygraphistry` client is BSD-3 only). *Instead:* plan for the
  subsetting above.

- **Pushing 139k values per frame through a Python model-sync channel.** *Appealing:* Panel,
  Bokeh server, Dash and Solara all make it easy to mutate Python state from Python. *Fails:*
  serialising 139k values at 30-60 Hz is the choke point, not the GPU. *Instead:* per-point
  colouring in a GPU shader from a compact quantised intensity array, over a raw binary
  WebSocket, with the Python framework only for layout/forms/charts at **<=1-2 Hz**.

- **Shipping JSON for per-frame neuron data.** *Appealing:* simple to produce and debug.
  *Fails:* ~1 MB/frame plus a 139k-number `JSON.parse`. *Instead:* binary. For reference:

  | Encoding | Size per frame |
  |---|---|
  | 1-channel `Uint8`/`Float16` intensity | 139k B = **139 KB** |
  | 3-component float colour (139k x 3 x 4 B) | **1.7 MB** |
  | Unconverted `Float32` intensity at 30 Hz | ~**17 MB/s** on loopback — fine |

- **Building a 3-component float colour buffer.** *Appealing:* direct RGB control. *Fails:*
  1.7 MB/frame versus 139 KB for a single intensity channel. *Instead:* one `Uint8`/`Float16`
  intensity attribute plus a colormap uniform in a custom `ShaderMaterial`.

- **Changing `BufferAttribute` usage flags after first draw.** *Appealing:* set them when
  convenient. *Fails:* three.js **usage cannot be changed after first use**. *Instead:* call
  `setUsage(DynamicDrawUsage)` at creation and use `addUpdateRange(start,count)` for partial
  uploads.

- **Streaming 139k neuron states through Neuroglancer `segment_colors`.** *Appealing:* state is
  a `Trackable`/`Map`, so Python mutations push live to the browser. *Fails:* **every change
  resends the ENTIRE state document**, and there is **no native "color segments by a data
  property" field**. *Instead:* stream a few hundred highlighted neurons at **1-5 Hz**, or serve
  activity from your own HTTP server as a precomputed annotation/`segment_properties` source with
  cache-busting, or use an `AnnotationLayer` with
  `annotation_properties=[AnnotationPropertySpec(id='rate', type='float32')]` +
  `linked_segmentation_layer` + a GLSL shader reading `prop_rate()`.

- **Using sigma.js for the full connectome.** *Appealing:* fast for graph views, MIT, a v4 with
  data textures. *Fails:* its own docs top out at "hundreds of thousands of edges". *Instead:*
  use it for subsets only; register only `pathLine` (4 vertices) and not the default `pathLoop`
  (66 vertices); set `antialiasEdges:false` (gives "up to x5"); leave `enableEdgeEvents` off
  because edge picking redraws all edges into a picking buffer.

- **Regenerating deck.gl buffers per frame.** *Appealing:* just set new data. *Fails:* buffer
  regeneration is the expensive operation and GPU buffer regen **crashes around 10M-100M items**.
  *Instead:* use `updateTriggers` to invalidate only specific attributes and keep data object
  identity stable.

- **Using Potree for live per-neuron colour.** *Appealing:* BSD-2-Clause, demos to 18 billion
  points. *Fails:* it is **static out-of-core octree point clouds** with classifications, not
  live per-point colour, and it needs the PotreeConverter format. *Instead:* steal the LOD ideas;
  render live activity with three.js or cosmos.gl.

- **Expecting Neu3D to animate per-neuron activity in shipped FlyBrainLab.** *Appealing:* it
  has an `animateActivity` API. *Fails:* the activity API is **opacity, not colour**; it is not
  in its `commandDispatcher`; it has zero networking; and **no caller of `animateActivity`
  exists** in FlyBrainLab/FBLClient/NeuGFX. *Instead:* use it as a three.js SWC viewer / steal
  ideas; the WAMP seam (`ffbo.nk.launch.<session>` returning
  `{'output': {rid: {'spike_time': {'data': [...]}}}}`) is the interesting part.

- **Expecting fly-wirehead to have solved the 3D live view.** *Appealing:* it looks like a live
  brain UI. *Fails:* it ships **no neuron positions, meshes or `.obj`/`.glb`/`.gltf`/`.swc`/
  `.npy`/`.npz` at all**; the fly is a hand-built low-poly model with hardcoded coordinates in
  `dist/scene.js:62-157`. Its per-neuron raster is accumulated into `state["raster"]` and
  published in `/api/status`, but **no frontend code consumes it** (grep of `dist/` gives zero
  hits), and `docs/model.md:30` confirms the network plot and 96-cell raster were removed from
  the display. *Instead:* use our own Codex positions
  (`data/codex/positions_normalized.npy`); do not look to fly-wirehead for this.

- **Treating FLYBOX as a 3D morphology renderer.** *Appealing:* "Open-source connectome
  sandbox... inspect live activity in 3D", Apache-2.0, FastAPI + React + WebSocket. *Fails:*
  `frontend/src/BrainView.tsx` renders a **2D canvas point cloud** (`getContext("2d")`,
  `points:[x,y,z,type]`), not real morphology; its README lists "full neuron morphology /
  skeleton rendering" as an open TODO; and it uses **MaleCNS, not FlyWire FAFB**. *Instead:*
  steal the FastAPI+WS+React live-frame architecture (and note `frame_payload()` shape); write
  the renderer ourselves.

- **Streamlit for the live view.** *Appealing:* fastest way to a Python dashboard. *Fails:* it
  is a **rerun model, not a push model**; `st.fragment(run_every=...)` reruns are sequential.
  *Instead:* FastAPI + a static page owning its own `requestAnimationFrame` loop.

- **Dash `set_props` at 60 Hz.** *Appealing:* Dash 4.2 added WebSocket callbacks, so it looks
  push-capable. *Fails:* each `set_props` is still a **Python-built prop tree**, so 139k points
  per frame is hopeless. *Instead:* steal the WS callback pattern; do not use Dash for the
  canvas.

- **NiceGUI `socket.io` for the frame path.** *Appealing:* NiceGUI is the best single-page
  shell (FastAPI + Vue/Quasar + three.js `ui.scene`, arbitrary custom elements). *Fails:*
  socket.io plus outbox batching is the wrong transport for 139k values per frame. *Instead:*
  keep NiceGUI for shell/layout and put heavy data on a **raw WS bypath**.

- **Picking a UI framework that owns the render loop.** *Appealing:* framework-managed canvas
  widgets. *Fails:* the canvas must pull frames on its own schedule. *Instead:* ranking for a
  custom WebGL view is **raw FastAPI static page > NiceGUI custom element/`ui.scene` > Panel ESM
  component > Dash custom component > Bokeh custom extension**.

- **Using WebSockets without backpressure handling.** *Appealing:* WS is the natural choice.
  *Fails:* MDN notes the stable WebSocket API has **no backpressure** ("fill up... memory... or
  100% CPU"). SSE is one-way and over HTTP/1.1 is capped at **~6 connections per browser**
  (HTTP/2 raises it to 100). *Instead:* WS binary for spikes with an explicit drop/coalesce
  policy; SSE for low-rate panels; polling <=1 Hz.

## Python libraries

- **`networkx` on the 15.1M-edge graph.** *Appealing:* universal, familiar API. *Fails:*
  pure-Python dict-of-dicts at **~200+ bytes/edge** means **~3-10 GB+ RAM** and minutes-to-hours
  per traversal. **It will appear to work on a 100k-edge subgraph and then die.** *Instead:*
  scipy.sparse / torch.sparse for the matvec, rustworkx or igraph for algorithms; use networkx
  only on type-collapsed/aggregated graphs.

- **Installing `cugraph` from PyPI.** *Appealing:* `pip install cugraph` looks right. *Fails:*
  the plain PyPI **`cugraph` 0.6.1.post1 is a stale, unrelated package**. *Instead:*
  `pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com` (26.8.0, requires
  `>=3.11`, Linux/WSL only) — and note our 121 MB CSR graph is small enough that cuGraph
  overhead may not pay off.

- **Installing `python-igraph`.** *Appealing:* the familiar package name. *Fails:* it is the
  **legacy name, superseded by `igraph`** (1.0.0, `>=3.9`, GPL verified). *Instead:* use
  `igraph`.

- **`graph-tool` for a cross-platform build.** *Appealing:* fast, GPL-3 verified, 2.11.
  *Fails:* no Windows support and conda-only in practice. *Instead:* rustworkx or igraph if
  Windows matters.

- **`navis` without noting the licence.** *Appealing:* GPL-3.0 (verified PyPI), 1.12.0,
  `py>=3.10,<4.0`, `pip install navis`, loads meshes and skeletons, `read_parquet`, `read_swc`,
  `downsample_neuron`, NBLAST, `plot2d`/`plot3d`. *Fails:* **GPL-3.0 is copyleft** — a real
  consideration if the project is ever distributed. *Instead:* use it as a library with that
  in mind; `navis.plot3d(m, color_by=<per-vertex array>, palette="viridis")` and
  `plot2d(nl, color_by=<labels>, palette="tab10")` were verified to work headless with
  matplotlib+plotly.

## Home Assistant

- **Streaming the fly brain through MQTT.** *Appealing:* `ha-mqtt-discoverable` makes entity
  creation easy. *Fails:* state is **only published if it changed** compared to its previous
  state unless `force_update=True`, and even then 10-60 Hz entities will flood the system.
  *Instead:* publish only curated low-rate entities, and keep the high-rate stream on your own
  WebSocket.

- **Publishing high-rate entities to HA without excluding them from Recorder.** *Appealing:* it
  just works. *Fails:* Recorder writes **EVERY state change**; `commit_interval` defaults to
  **5 s** and `purge_keep_days` to **10**, so a 10-60 Hz entity grows the database fast.
  *Instead:* use Recorder `include`/`exclude` (`domains`, `entity_globs`, `entities`) to keep
  simulator entities out, or do not publish them at all.

- **Using `panel_iframe`.** *Appealing:* it embeds a URL in the sidebar. *Fails:* it is a
  **REMOVED integration**; the docs URL redirects to `/more-info/removed-integration`.
  *Instead:* `panel_custom` with a tiny custom element that creates an `<iframe>` and
  `embed_iframe: true`, with the page stored in `<config>/www` (served at `/local`). Note
  `panel_custom` takes a **JS module URL, not an arbitrary iframe src**.

- **Planning an Ingress app on HA Container or HA Core.** *Appealing:* Ingress handles auth and
  the gateway supports WebSockets and streaming. *Fails:* it **REQUIRES HA OS or Supervised** —
  HA's own install docs state Home Assistant Container "don't have access to apps" (the Apps row
  is marked only for HA OS), and HA Core has no Supervisor. *Instead:* for Container/Core use
  `panel_custom` with a custom element that iframes your URL, or simply link to the page.
  (Ingress specifics if available: `ingress: true`, default port **8099** or `ingress_port`,
  allow only **172.30.32.2**, header `X-Ingress-Path`.)

- **Relying on `ha-mqtt-discoverable-cli`'s licence metadata.** *Appealing:* PyPI field says
  Apache-2.0. *Fails:* the classifier says "Other/Proprietary License" — a **conflict**; the
  PyPI release 0.25.2 (2026-05-23) requires `Py >=3.10,<4.0` and only offers
  `hmd create binary sensor` / `hmd create device`. *Instead:* verify the LICENSE file before
  relying on it; prefer the library `ha-mqtt-discoverable` if MQTT is needed.

- **Publishing simulator state via `POST /api/states/<entity_id>` at high rate.** *Appealing:*
  the REST API will create/update any state. *Fails:* it is a full state write and feeds
  Recorder. *Instead:* use `POST /api/services/<domain>/<service>` for actions and reserve state
  writes for deliberate, low-rate entities.

- **Confusing the two HA integrations.** *Appealing:* "integrate with HA" sounds singular.
  *Fails:* (a) **embedding your page inside HA** (`panel_custom` or an ingress app) and
  (b) **publishing simulator state as entities** (`ha-mqtt-discoverable` / REST POST) are
  different problems with different rate profiles. *Instead:* decide which one is being built;
  the existing `flybrain/ha.py` (`MockHomeAssistant`, `RestHomeAssistant` over
  `HA_BASE_URL`/`HA_TOKEN`/`HA_MODE`) covers (b) for actions.

- **React inside a Lovelace custom card.** *Appealing:* reuse a React component. *Fails:* React
  is **problematic inside custom elements**. *Instead:* use a plain custom element, host the
  canvas directly, and subscribe to HA states through the context-request event
  (`context='states'`, `subscribe=true`, `callback`).

## Observability and dashboards

- **Prometheus Pushgateway for live telemetry.** *Appealing:* push metrics without running a
  scrape target. *Fails:* the project says verbatim "We only recommend using the Pushgateway in
  certain limited cases"; it **"never forgets series"** pushed to it unless manually deleted;
  you **"lose Prometheus's automatic instance health monitoring via the `up` metric"**; and
  "the only valid use case... is for capturing the outcome of a service-level batch job".
  *Instead:* use Grafana Live (`POST /api/live/push/:streamId` with InfluxDB line protocol) for
  soft-real-time, or Netdata (Agent GPLv3+, UI closed-source NCUL1) for host/GPU metrics.

- **Expecting hard real-time from Grafana Live.** *Appealing:* persistent WS pub/sub, default
  max **100 connections, ~50 KB/conn**. *Fails:* their own caveat: "soft real-time... delay can
  be up to several hundred ms or higher"; WS payloads must be JSON. *Instead:* use it for
  operator dashboards, not for the spike path; steal the Node Graph pattern (node colour,
  `arc__*` activity rings, `mainstat`, `nodeRadius`, edge colour/thickness, **200 visible nodes
  by default**, Force layout for 500+).

- **Grafana Infinity for large frames.** *Appealing:* poll any REST/JSON/CSV/GraphQL source.
  *Fails:* docs say it is "not designed for handling large amounts of data... inline snippets
  <1MB". *Instead:* tables only; use Grafana Live or a direct WebSocket for streams.

- **`sortingview` for private spike viewing.** *Appealing:* SpikeInterface's web raster backend.
  *Fails:* it **uploads data to a public cloud bucket / kachery-cloud** — not local/private.
  *Instead:* steal SpikeInterface's raster abstraction (`plot_rasters()`, `plot_traces()`,
  `plot_unit_*`) and render locally; its own licence is **UNVERIFIED**.

- **`phy` as a web asset.** *Appealing:* 2.1.0 (Jul 2026) is active, a mature curation GUI for
  large ephys. *Fails:* **2.1.0 explicitly replaced its web GUI component with Qt-native**, the
  licence is **UNVERIFIED** (LICENSE 404), and it is a desktop app. *Instead:* avoid;
  `ephyviewer` (Qt, licence **UNVERIFIED**) and `neo` (licence **UNVERIFIED**) are likewise not
  browser assets.

- **`bionet.ee.columbia.edu`.** *Appealing:* cited in older FlyBrainLab material. *Fails:*
  **DEAD** (404/GitHub Pages). *Instead:* `fruitflybrain.org` is live; public WAMP routers live
  at `128.59.65.19`, including a FlyWire NeuroNLP endpoint.

## Miscellaneous verification gaps

These are not yet traps, but they are **UNVERIFIED** and should not be treated as settled:

- `navis-flybrains` (<https://github.com/navis-org/navis-flybrains>) — licence and liveness
  **UNVERIFIED**.
- `cocoa` (`flyconnectome/cocoa`, <https://github.com/flyconnectome/cocoa>) — **UNVERIFIED**.
- `neuprint-python` (<https://github.com/connectome-neuprint/neuprint-python>) — BSD-3
  **UNVERIFIED**; needs a token.
- HACS — **UNVERIFIED** (covers Integrations, Dashboard plugins/cards, AppDaemon apps, Python
  scripts, Templates, Themes).
- NeuroPulse (`tareqrwk/neuropulse`) — **UNVERIFIED** ("Interactive web-based neural activity
  visualization simulator", search result only).
- SNUB (JOSS <https://doi.org/10.21105/joss.06187>) — **UNVERIFIED**.
- `flybody` / NeuroMechFly v2 licences — **UNVERIFIED**.
- `bionet`-adjacent portals `braincircuits.io` and `fafb-flywire.catmaid.org` — details
  **UNVERIFIED**.
- NEST GPU exact latest tag — **UNVERIFIED**; BANC ~188k neurons — **UNVERIFIED**; MANC exact
  bucket URL and file sizes — **UNVERIFIED**; Codex `connections_princeton.csv.gz` size
  (~68.5 MB), `labels.csv.gz` (4.8 MB) and `synapse_coordinates.csv.gz` (~317 MB) —
  **UNVERIFIED**; PyGeNN-on-3070 speed — **UNVERIFIED**.
- The `genn` PyPI 0.7.7 package being unrelated is the reason to avoid it, but the exact
  provenance of that package is itself only described as "an unrelated 2021 project".

## See also

- [simulation-backends.md](simulation-backends.md) — engines, benchmark table, data sources.
- [asset-inventory.md](asset-inventory.md) — the assets these traps refer to.
- [../architecture.md](../architecture.md) — the integration decisions and validation harness.
- [../licensing.md](../licensing.md) — the authoritative licence position.
