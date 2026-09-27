# Research index

Every claim in these documents was checked against a live source unless explicitly marked
**UNVERIFIED**. Where a scout could not confirm something, it is flagged rather than smoothed
over — several of the most useful findings are corrections to things that are widely assumed.

## Documents

| Document | Contents |
|---|---|
| [asset-inventory.md](asset-inventory.md) | Visualisation/UI assets, Home Assistant integration assets, graph libraries, annotation tables, live brain viewers, telemetry dashboards. |
| [simulation-backends.md](simulation-backends.md) | Simulation engines (Brian2, GeNN, NEST GPU, PyTorch), connectome data releases, activity datasets, and the benchmark table. |
| [dead-ends.md](dead-ends.md) | Things that look promising and are not — with the reason and the alternative. |
| [../licensing.md](../licensing.md) | Licences, the CC BY-NC trap, and the unlicensed-UI trap. |
| [../data.md](../data.md) | Data schemas, provenance, and the coordinate-parsing gotchas. |
| [../engine.md](../engine.md) | Which simulation engine, why, and the calibration warnings. |
| [../live-view.md](../live-view.md) | The dashboard: wire protocol, rendering strategy, settings. |
| [../architecture.md](../architecture.md) | How the whole system fits together. |

## The findings that changed the design

1. **A whole existing project does this pattern.** [`nftechie/stonkfly`](https://github.com/nftechie/stonkfly)
   (MIT) runs a fly connectome and uses a neural readout to place real Coinbase orders, with a
   dopamine learning rule. Its neural package has **zero** trading imports, so it is importable.
   "When this sensor goes, turn on this light" is the same shape as "when price moves, buy".
2. **The Python→browser bridge is the bottleneck, not the GPU.** Any dashboard that serialises
   139k values per frame through a framework's model-sync will crawl. Hence the binary
   WebSocket + shader-coloured point cloud in `../live-view.md`.
3. **Neuron morphology is public and token-free.** `gs://flywire_v141_m783` exposes meshes and
   skeletons keyed by root ID; the FlyWire graph server needs auth but the bucket does not.
4. **3D soma coordinates were the one genuine gap** — neither fly project ships them. We closed
   it with the public Codex `coordinates.csv.gz` (100% coverage after de-duplication).
5. **The active-set integrator is the route to real-time.** See `../engine.md`.
6. **FlyWire v783 is CC BY-NC 4.0**, despite Zenodo mirrors tagged CC BY 4.0. Non-commercial.
7. **Two popular projects are unlicensed.** `fly-wirehead` and `infinite-sugar` have no licence
   file at all, which means all rights reserved. Read them; do not copy them.
8. **The published synapse weight does transfer to this dataset, unchanged.** This finding was
   originally recorded the other way round — that the published `0.275 mV` could not be used and
   needed scaling to 0.45–0.65×. That was wrong, and the way it was wrong is the most useful
   thing in this research folder. A missing `tau_mem` factor in the membrane update made every
   synapse 20.2× too strong; the resulting instability looked exactly like a property of
   the connectome, and two external recalibration numbers made the wrong story fit. See
   [simulation-backends.md](simulation-backends.md#resolution-the-gain-was-an-integrator-factor-2026-09-22).
9. **The descending/motor output is nearly unreachable from a sensory drive** (~23 extra spikes
   across all 1,299 descending neurons), so the control loop reads antennal-lobe populations
   instead. A measured compromise, not the biologically ideal choice. This one survived the
   integrator fix.
10. **Training and running must use the same regime.** A readout fitted on windows that start
    from a resting brain loses accuracy when run against a continuously-running one; a linear
    probe separates the two states perfectly. Training now sweeps the brain continuously, which
    is both more accurate *and* cheaper (~50 s vs ~96 s). See `../live-view.md`.
11. **The readout's feature shape dominated its accuracy, not the brain.** Feeding a linear
    readout the firing rate of every neuron in a population rather than one collapsed scalar per
    output took held-out colour error from ~155 K to ~20 K; regressing the quantity actually
    being measured (Kelvin) instead of soft labels over three colour bands removed a further
    151 K of built-in error. See `../live-view.md`.

## What went wrong on the way

Worth reading before trusting any number in this repository.

Four **silent** simulator bugs were caught only by the brian2 comparison harness: the sparse
weight matrix was transposed, the delay line dropped every spike, the membrane integrator was
algebraically wrong so excitatory input *hyperpolarised* cells, and refractory neurons
accumulated unbounded voltage. All four produced plausible-looking output.

Worse, one reported "PASS" was **degenerate**: the comparison passed because the network had
been silenced by a 250× gain reduction, so it matched the driven neurons exactly while
producing no recurrent activity at all. The lesson is recorded in full in
[simulation-backends.md](simulation-backends.md): *a validation that agrees by being empty is
not a validation.* Always check that the thing being measured is doing something.

And a fifth bug — the `tau_mem` factor on the conductance term — survived all of that, hid
behind a fudge factor, and was finally found **not** by the 15-million-synapse comparison but by
a two-neuron test with a closed-form answer (`validation/micro_gain_check.py`). Two transferable
lessons:

- **When a physical parameter seems to need a ~100× fudge factor, suspect the arithmetic, not
  the physics.** A fudge factor that size is a bug report.
- **An external calibration number can make a local bug look like a modelling discrepancy.**
  Two published gains from another project were read as confirmation that the weight did not
  transfer; they were about a different network and different behavioural targets.

### Four defects found by an external review (2026-09-25)

A second model reviewed the codebase cold. All four findings were verified against the code before
anything was changed — none were false positives — and two of them had a cause more interesting
than the symptom.

1. **An unavailable sensor could drive a real light.** `HAClient._parse_state` substitutes a `0.0`
   fallback for an offline entity, and `LiveLoop.drive_channels` checked only that the temperature
   entity was *present* — while the extra-channel path a few lines below checked the state. An
   unavailable thermometer was therefore encoded as a genuine 0 °C reading and decoded like any
   other. The docstring three lines above the bug stated the correct principle.
2. **A failed fetch left the loop acting on stale input.** `get_signals()` returns `[]` on failure;
   with no signals the drive was never cleared and `_active_source_c` was never reset, so the
   simulator kept running on the previous window's input and `decide()` kept decoding from it.
3. **The recorder could silently misalign windows against feature rows.** `record()` appends the
   feature row and *then* the window line, so a crash between the two leaves the files one apart;
   and because the writer reopened in append mode, a partial row from a torn write ended up in the
   *middle* of the file, shifting every later row by a constant offset while the row count still
   looked correct.
4. **A malformed settings message could kill the dashboard's activity stream.** The WebSocket
   handler cast and assigned in one loop with no `try`; a bad value raised out of the receive loop
   into a blanket handler that closed the connection, so one out-of-range number froze the view.

Two mechanisms from these are worth carrying forward, because both are general:

- **A duplicated constant is an invitation to an inconsistent check.** Finding 1 existed because
  `{"unavailable", "unknown", "none", ""}` was defined *three* times — in `ha.py`, `wiring.py` and
  `loop.py`. Two of the three call sites remembered to check it; the third did not. It now lives
  once, in `types.py`, behind a shared helper.
- **A "validated" path can still apply half of a patch.** Finding 4's server-side version cast each
  field and assigned it inside the same loop, so `{"fps": 30, "gain": "abc"}` applied the frame rate
  and *then* raised: a partial update the caller was told had been rejected. Coerce everything, then
  commit everything. `LoopConfig.apply` had the same shape, and additionally read the string
  `"false"` as `True` — `bool("false") is True` — which is how "keep the real light read-only"
  becomes a live service call.

The lesson is the same one already recorded above: **the expensive bugs here are the quiet ones.**
None of these four produced an error, a log line, or a failed test. Finding 1 required the loop's
own docstring to contradict its code; finding 3 required a row count that still looked plausible
after every row had shifted. All four are now covered by tests that were confirmed to **fail**
against the pre-fix source — because a regression test that passes either way is the degenerate
validation described above, wearing a different hat.

## Corrections to commonly repeated claims

| Claim | Reality |
|---|---|
| "hemibrain is behind a data use agreement" | **False.** Janelia states hemibrain is **CC-BY**, and its neuPrint dataset is anonymously readable. |
| "FlyWire is CC BY" | **False** for the data: CC BY-**NC** 4.0. The Zenodo mirrors tagged CC BY 4.0 conflict with this. |
| "FlyWire has ~5M connections" | The v783 dataframe has **15,091,983** (pre, post) rows. ~5M usually refers to a thresholded subset. |
| "brian2 works fine on numpy 2" | **2.8.x breaks** on numpy ≥2.3 (`numpy.NPY_OWNDATA`). Use 2.9.0 on Python 3.11, or 2.10.x on 3.12+. |
| "FlyBrainLab is a usable platform" | It pins Python ≤ 3.10, needs OrientDB+Java+CUDA+OpenMPI, and its engine runs circuit models, not the whole connectome. |
| `seung-lab/meshparty`, `seung-lab/nglui` | **404.** The live repositories are `CAVEconnectome/MeshParty` and `CAVEconnectome/nglui`. |
| `seung-lab` hemibrain bucket | `gs://flyem-hemibrain` returns **401**. Use `gs://hemibrain`. |
| `storage.cloud.google.com/...` | Returns an auth HTML page. Use `storage.googleapis.com/...`. |

## Method

Three parallel research passes, each required to fetch and verify rather than recall:

1. Connectome visualisation and 3D assets.
2. Simulation backends, connectome data releases, and activity datasets.
3. Live-view UI stacks and Home Assistant integration.

Plus a read-only code archaeology pass over `stonkfly` and `fly-wirehead` clones, and our own
direct verification of the data (row counts, annotation coverage, coordinate parsing, and the
GPU throughput measurements in `docs/engine.md`).

GitHub's REST API rate-limited during the passes, so several repository metadata points come
from raw file fetches, PyPI JSON, or npm registry data instead — those are marked.
