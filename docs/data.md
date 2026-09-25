# Data

## Getting the data

**None of the large tables are committed to git.** The repository is code, tests, docs and one
small trained readout. The four source tables below are downloaded from their public upstreams on
first run, anonymously — no account, no token:

```bash
.venv/bin/python tools/fetch_data.py            # download whatever is missing
.venv/bin/python tools/fetch_data.py --check    # report status, download nothing
.venv/bin/python tools/fetch_data.py --force    # re-download everything
```

Each file is verified against its expected byte count and only moved into place once the count
matches, so an interrupted transfer cannot leave a truncated parquet behind that fails much later
inside the simulator with a confusing error.

### Downloaded

| Path | Size | What it is |
|---|---|---|
| `vendor/fly-brain/data/2025_Connectivity_783.parquet` | 100.8 MB | The FlyWire v783 connectivity: **15,091,983** directed (pre, post) pairs with synapse counts. |
| `vendor/fly-brain/data/2025_Completeness_783.csv` | 3.5 MB | The neuron list: **138,639** FlyWire root IDs. Row index == connectome index. |
| `data/annotations/flywire_annotations_supl1.tsv` | 31.7 MB | Cell typing: 139,248 rows × 31 columns. After de-duplication it covers **138,625 of the
138,639** connectome neurons — 99.99%, with 14 unlabelled (see `flybrain/mapping.py`). |
| `data/codex/coordinates.csv.gz` | 5.3 MB | Neuron soma positions from Codex. Public GCS, no token. |

### Derived (built locally, never committed)

| Path | Size | What it is |
|---|---|---|
| `data/annotations/flywire_annotations_supl1.indexed.parquet` | — | Memoised join of the TSV onto connectome indices (built on first `ConnectomeSim.load()`). |
| `data/codex/positions_normalized.npy` | 1.7 MB | `138639 × 3` float32 positions, converted to microns and normalised to ≈[-1, 1] for the 3D view. Built by `tools/build_positions.py`, which `fetch_data.py` runs for you. |
| `data/validation/*.npz` | small | brian2 reference and our simulator's spike counts for the validation harness. |

### Committed on purpose

| Path | What it is |
|---|---|
| `data/experiments/colour_readout.npz`, `colour_meta.json` | The trained colour readout, a few KB, so the demo runs without retraining. |
| `data/codex/positions_meta.json` | Provenance for the normalised position buffer (shape, scale, voxel size, source). |

## Provenance and licences

- **Connectivity + completeness**: FlyWire v783 (female adult fly brain, FAFB), October 2023
  snapshot. Committed to git inside
  [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain), which is why no Codex
  account is needed; `fetch_data.py` downloads just those two files from its `main` branch. **That
  repository is GPL-2.0 but none of its code is fetched or used** — only its copy of the data.
  The data itself is **CC BY-NC 4.0** — see [licensing.md](licensing.md).
- **Annotations**: [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations),
  `supplemental_files/Supplemental_file1_neuron_annotations.tsv`. No licence file.
- **Coordinates**: `https://storage.googleapis.com/flywire-data/codex/data/fafb/783/coordinates.csv.gz`,
  anonymous. The Codex *web app* needs Google sign-in; these GCS files do not.

## Connectivity schema

```
Presynaptic_ID            int64   FlyWire root id of the presynaptic neuron
Postsynaptic_ID           int64   FlyWire root id of the postsynaptic neuron
Presynaptic_Index         int64   0..138638, row index into the completeness table
Postsynaptic_Index        int64   0..138638
Connectivity              int64   raw synapse count (always > 0)
Excitatory                int64   +1 or -1
Excitatory x Connectivity int64   signed synapse count
```

Verified properties:

- 15,091,983 rows; **9,059,302 excitatory** and **6,032,681 inhibitory**.
- Indices span exactly `0..138638`; the completeness CSV has 138,639 neurons.
- **No duplicate (pre, post) pairs** — so the table is already a graph edge list, not a
  multi-edge list. This is why the sparse matrix build is a straightforward `coalesce()`.
- `max |Excitatory × Connectivity| = 2405`.

Weight used by the simulator:

```
w_mV = 0.275 × (Excitatory × Connectivity)
```

Recurrent spikes transmit this weight **as-is** (the parameter that scales them,
`recurrent_scale`, is `1.0`). The separate `poisson_scale = 250` applies only to the
*external* Poisson drive, where the reference model injects `0.275 x 250 = 68.75 mV` into `v`
per event. Note the sign convention is carried entirely by `Excitatory`; `Connectivity` is
always positive.

## Annotations

The join in `flybrain/sim.py::_load_annotations` de-duplicates on `root_id` (the TSV has
multiple rows per neuron because `supervoxel_id` is the leading column) and reindexes onto the
connectome's 138,639 root IDs **in file order**, so that `annotation_table.loc[i]` describes
neuron index `i`. That ordering is essential — every role in `flybrain/mapping.py` depends on it.

Useful columns and the population sizes we verified:

| Column | Notable values |
|---|---|
| `super_class` | `optic` 77,530 · `central` 32,379 · `sensory` 16,352 · `visual_projection` 8,038 · `ascending` 1,736 · `descending` 1,299 · `motor` 110 |
| `cell_class` | `Kenyon_Cell` 5,177 · `mechanosensory` 2,656 · `olfactory` 2,279 · `gustatory` 408 · `brain_motor_neuron` 105 · `hygrosensory` 74 · `thermosensory` 29 |
| `flow` | `intrinsic` 118,480 · `afferent` 18,664 · `efferent` 1,481 |

The thermosensory population is real and small — 29 cells, subtypes `TRN_VP1m` (13),
`TRN_VP2` (7), `TRN_VP3a` (7), `TRN_VP3b` (2) — which is exactly why `RoleResolver.pool()`
exists: a 29-neuron population cannot carry an arbitrary number of distinct rates, so small
roles are pooled and cycled deliberately rather than silently truncated.

## Coordinates

`coordinates.csv.gz` needs care in two ways:

1. It has **238,909 rows for 139,255 unique neurons** — multiple supervoxels per neuron. The
   column order in the file means a naive whitespace split yields **four** tokens, not three.
   Parse with a regex:
   `\[\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\]`
   and de-duplicate on `root_id`.
2. After de-duplication it covers **100% of all 138,639 connectome neurons**.

Conversion used for the 3D view:

```
voxel size = [4, 4, 40] nm  →  nanometres  →  microns  →  centre  →  divide by max |coord|
```

Giving a bounding extent of roughly **3261 × 1566 × 11139 µm** — z is the long
anterior–posterior axis, which is expected for a fly brain.

## Other data you can reach without an account

All verified as publicly readable:

| Source | Content |
|---|---|
| `gs://flywire_v141_m783` | v783 image pyramid, `mesh_mip_1_err_40/` meshes, `skeletons_mip_1/` skeletons — keyed by root ID, anonymously listable. |
| Codex `.../codex/data/fafb/783/` | `classification.csv.gz`, `consolidated_cell_types.csv.gz`, `neurons.csv.gz` (neurotransmitters), `connections*.csv.gz`. |
| `gs://flyem-male-cns/v1.0/` | Male CNS v1.0, CC BY 4.0, including a ready-made 45-layer Neuroglancer state. |
| Zenodo `10.5281/zenodo.10877326` | 5.36 GB parquet of SWC skeletons for all v783 neurons. |

The FlyWire **graph server** (`prod.flywire-daf.com`) does require authentication — but meshes
and skeletons in the bucket above do not, so morphology work needs no token.

## Reproducing the derived files

```bash
# everything the project needs, downloaded and verified
.venv/bin/python tools/fetch_data.py

# positions for the 3D view only (writes data/codex/positions_normalized.npy)
.venv/bin/python tools/build_positions.py

# annotation join + role table (also built lazily on first simulator load)
.venv/bin/python -c "from flybrain.sim import ConnectomeSim; ConnectomeSim().load()"
```
