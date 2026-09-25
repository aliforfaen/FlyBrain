# Licensing

**Read this before shipping anything.** The data and the code carry different licences, and one
of them has a non-commercial clause that is easy to miss.

## Short version

| Thing | Licence | Consequence |
|---|---|---|
| This project's own code | **MIT** | Do anything, with attribution. See [`LICENSE`](../LICENSE). |
| FlyWire v783 connectome **data** | **CC BY-NC 4.0** | Non-commercial only. Personal/homelab use is fine. Do not monetise. |
| `philshiu/Drosophila_brain_model` (the model) | **MIT** | Do anything, with attribution. |
| `eonsystemspbc/fly-brain` (data source we cloned) | **GPL-2.0-or-later** | We use only its **data**, not its code. Do not vendor the harness. |
| `nftechie/stonkfly` (LIF kernel, plasticity rule) | **MIT** | Reusable. |
| `blendi-remade/fly-brain-minecraft` (active-set integrator) | **MIT** code + **CC BY 4.0** data | Reusable including commercially. |
| `flyconnectome/flywire_annotations` (cell types) | **no licence file** | Ambiguous. Attribute the papers. |
| three.js | **MIT** | Vendored at `web/vendor/three/`. |
| `mattyhempstead/fly-wirehead` (Three.js live brain UI) | **NO LICENCE — all rights reserved** | **Do not copy its code.** See below. |

## FlyWire is CC BY-NC 4.0, despite what the mirrors say

The authoritative statement is on [flywire.ai/guidelines](https://flywire.ai/guidelines):

> FlyWire's public release data is made available under license CC BY-NC 4.0

This matters because **the Zenodo mirrors of the same data are tagged `cc-by-4.0`**, which
would permit commercial use. That is a genuine conflict between two official-looking sources.
The conservative reading — and the one to act on — is **CC BY-NC 4.0**: non-commercial.

So: this project, built on FlyWire v783, is fine for personal use and must not be sold or used
commercially as-is.

If you ever want a commercially clean path, the alternatives are all CC BY 4.0 and are
anonymously downloadable:

| Dataset | Licence | Notes |
|---|---|---|
| Male CNS (MCNS) v1.0 | CC BY 4.0 | Male brain **and** ventral nerve cord: 176,422 neurons, and real descending→motor→leg pathways. Berg et al., *Cell* 2026. |
| BANC v888 | CC BY 4.0 | Female brain + VNC. Bates et al., *Nature* 2026. |
| MANC v1.2.x | CC BY 4.0 | Male VNC. |
| hemibrain v1.2.1 | **CC-BY** | ~25k neurons. Contrary to a common assumption, this is **not** encumbered by a data use agreement. |

## The `fly-wirehead` trap

`fly-wirehead` is the closest existing project to our live brain view, and it is tempting to
copy. **Do not.** A thorough check found:

- No `LICENSE`, `LICENSE.md`, `LICENSE.txt` or `COPYING` anywhere in the repository — checked
  on both `main` and `master` (HTTP 404), across all 61 tracked files, with no deleted licence
  in the git history, and no `license` field in `pyproject.toml`.
- The only licence files present are `licenses/stonkfly-MIT.txt` and
  `dist/vendor/THREE-LICENSE.txt` — i.e. for **third-party** components.
- Therefore: its neural backend is MIT (inherited from stonkfly), three.js is MIT, and
  **its own UI code is unlicensed, which defaults to all rights reserved.**

Its `THIRD_PARTY.md` claims "The 3D scene is original to this repository" while granting no
licence for it. Reading it for ideas is fine; copying code is not.

The same applies to `cnqso/infinite-sugar` (browser whole-brain emulation, **no licence file**).

## Why `eonsystemspbc/fly-brain` is GPL but we're fine

That repository matters for exactly one reason: **it commits the v783 connectivity parquet and
completeness CSV to git**, so neither needs a Codex account. `tools/fetch_data.py` downloads those
two files over HTTPS into `vendor/fly-brain/data/`, which is gitignored.

**None of that repository's code is cloned, fetched or used**, so its GPL-2.0-or-later licence
does not attach to anything here. It is treated purely as a distribution channel for FlyWire data,
and the data's own licence — CC BY-NC 4.0 — is the one that applies.

(For completeness: the Shiu et al. materials inside that repository,
`code/paper-phil-drosophila/`, carry their **own separate MIT licence** — verified:
`code/paper-phil-drosophila/LICENSE` is MIT, "Copyright (c) 2023 Philip Shiu and Nico Spiller".)

Our own simulator in `flybrain/sim.py` is an independent implementation of the published
equations, written from the paper and the MIT reference, not copied from any GPL harness.

Note also that the FlyWire data itself is NC, so the GPL-vs-MIT question is secondary here.

## Annotation table

`data/annotations/flywire_annotations_supl1.tsv` comes from
[flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations),
which has **no licence file at all** (GitHub's licence field is null). That is a real ambiguity.
The practical position: attribute the underlying papers (Dorkenwald et al. 2024, Schlegel et al.
2024, Matsliah et al. 2024) and do not redistribute the table as a standalone product.

It is worth using anyway because the repository explicitly supersedes Codex's cell typing:

> Codex presents a mix of annotations from different sources which likely diverge

## Attribution

If you publish anything based on this, cite:

- Dorkenwald, S. et al. "Neuronal wiring diagram of an adult brain." *Nature* (2024). — the FlyWire connectome.
- Shiu, P. K. et al. "A leaky integrate-and-fire computational model based on the connectome of the entire adult *Drosophila* brain reveals insights into sensorimotor processing." *Nature* (2024). — the model.
- Schlegel, P. et al. (2024) — systematic annotation.
