# Asset inventory: visualisation, Home Assistant and supporting libraries

Inventory of everything that is *not* the simulation engine or a raw connectome source. The
engine and the data sources live in [simulation-backends.md](simulation-backends.md); the traps
live in [dead-ends.md](dead-ends.md).

Target stack, assumed throughout: RTX 3070 (8 GB), Linux, Python 3.11, FlyWire v783 with
138,639 neurons and 15,091,983 synapses, FastAPI + a three.js dashboard in `web/`.

## The one architectural fact

**The bottleneck is not the browser; it is the Python->JS per-frame bridge.**

Pushing ~139k neuron values at 30-60 Hz through a Python-serialised channel (Bokeh/Panel model
sync, Streamlit reruns, Dash props, Solara/ipywidgets, NiceGUI socket.io outbox) is the choke
point, not the GPU. Requirements that follow:

- Per-point colouring must happen **in a GPU shader**, from a compact quantised intensity array.
- Transport must be a **raw binary WebSocket**.
- The Python framework is used for layout, forms and charts at **<=1-2 Hz**, never for the
  per-frame data path.
- The canvas owns its own `requestAnimationFrame` loop and **pulls** binary frames.

**Why this matters for us:** this justifies the existing `flybrain/server.py` design (binary
WebSocket + static dashboard) over any framework that owns the render loop.

## Python web frameworks

| Asset | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| FastAPI + WebSockets | MIT | Active; native SSE in 0.135+ | Raw WS; `send_bytes`/`receive_bytes`/`iter_bytes` first-class; 30-60 Hz fine | **Use as the backbone.** Serve a static page; JS owns its own rAF loop. |
| Panel / HoloViz | BSD-3 (PyPI JSON) | 1.9.4, `Py>=3.10`, depends on Bokeh | Bokeh-server model sync over WS; canvas good <10k markers, WebGL >25k; ESM custom components and a DeckGL/pydeck pane exist | **Use as a library / steal widgets.** |
| Bokeh server | BSD-3 | 3.10.0 | Same model-sync bottleneck | Use as a library (Panel wraps it). |
| Plotly Dash | MIT | Dash 4.2 added WebSocket callbacks (`websocket_callbacks=True` + `persistent=True` + `set_props`, requires FastAPI or Quart); `dcc.Interval` is client polling | Even pushed, each `set_props` is a Python-built prop tree | **Steal the WS callback pattern; avoid for 60 Hz.** |
| Streamlit | Apache-2.0 | | Rerun model, not push; `st.fragment(run_every=...)` reruns are sequential | **Avoid.** |
| NiceGUI | MIT (PyPI `license_expression`) | 3.17.1, `Py>=3.10`, FastAPI + Vue/Quasar + socket.io, "timer... even every 10 ms" | socket.io + outbox batching; `ui.scene` is three.js (bundles three 0.180.0); `ui.echart`/`ui.plotly`; arbitrary custom elements | **Best single-page shell**, but keep heavy data on a raw WS bypath. |
| Solara | MIT (LICENSE fetched) | | Pure-Python React on ipywidgets; reactive re-render, no high-frequency push story | **Steal ideas only.** |

Ranking for a custom WebGL view: **raw FastAPI static page > NiceGUI custom element / `ui.scene`
> Panel ESM component > Dash custom component > Bokeh custom extension.** Whatever is picked,
the canvas must own its own `requestAnimationFrame` loop and pull binary frames.

## JS / WebGL libraries

| Asset | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| three.js | MIT | 0.186.0 | 100k+ points trivially; 139k `THREE.Points` at 60 fps is nothing on a 3070. Per-point colour via `BufferAttribute`: mutate the `Float32Array` in place -> `needsUpdate = true`; `setUsage(DynamicDrawUsage)`; `addUpdateRange(start,count)` for partial uploads. **Usage cannot be changed after first use.** | **Use as the 3D renderer.** |
| deck.gl | MIT | 9.4.0 | Vendor docs: 60 FPS to ~1M items for `ScatterplotLayer` on a 2015 MBP; 10-20 FPS near 10M; GPU buffer regen crashes 10M-100M. Regeneration is the expensive op; use `updateTriggers` and keep data object identity stable | **Use as library / perf playbook.** |
| regl | MIT | 5,581 stars, pushed 2026-09-08 | Your shaders, your limits | Use when a bespoke GL path is needed. |
| PixiJS | MIT | 8.21.0 | Huge 2D sprite/particle counts (`ParticleContainer`) | **2D only, not 3D.** |
| sigma.js v3 | MIT | npm 3.0.3, v4 alpha | Own words: "graphs of thousands of nodes"; v4 docs: smooth at "tens of thousands", tuning matters at "hundreds of thousands of edges"; v4 uses data textures | **Use for graph subsets**, never the full 15M edges. |
| graphology | MIT | 0.26.0 | Data structure only | Pair with sigma.js. |
| cosmos.gl / `@cosmos.gl/graph` | MIT | 3.4.2, luma.gl WebGL2, `node>=22` | README: "real-time simulation of network graphs consisting of hundreds of thousands of points and links"; `setPointPositions`/`setPointColors` with `Float32Array` and GPU transitions | **Use as library for big graph views.** |
| Potree | BSD-2-Clause (LICENSE fetched; GitHub API said NOASSERTION) | 5,613 stars, pushed 2026-01-08, demos to 18 billion points | Static out-of-core octree point clouds; classifications, not live per-point; needs PotreeConverter format | **Steal LOD ideas; avoid for live colour.** |
| Neuroglancer (JS) | Apache-2.0 | 1,525 stars, pushed 2026-09-17 (npm 2.41.2) | See the dedicated section below | **Use as-is (viewer) + use as library (state API).** |
| Neu3D | ISC per `package.json` (no LICENSE file) | live repo `fruitflybrain/neu3d` (`FlyBrainLab/neu3d` 404s); npm 1.1.4 (2023-08-18); last commit 2024-12-30; three.js ^0.151.3 | Activity API is **opacity, not colour**: `animateActivity({neuronLabel:[values]})` -> `meshDict[key].updateOpacity(value)`; colours via `setColor(id,color)`. Not in its `commandDispatcher`, zero networking, and **no caller of `animateActivity` exists** in FlyBrainLab/FBLClient/NeuGFX | **Use as a library (three.js SWC viewer) / steal ideas.** |

### Neuroglancer in detail

- PyPI `neuroglancer` 2.41.2 (2025-09-23) requires `>=3.10`; prebuilt **abi3 manylinux wheel**
  so it installs on py3.11 with **no compiler and no Node**.
- `viewer.state.layers[i].segment_colors` is a `typed_map(uint64 -> hex str)`; state is a
  `Trackable`/`Map` so Python mutations push to the browser live — **but every change resends
  the ENTIRE state document**. Stream a few hundred highlighted neurons at 1-5 Hz, **not 139k
  per frame**.
- There is **no native "color segments by a data property" field**; colouring is state-driven
  (`segmentColors`/`colorSeed`/`segmentDefaultColor`) or GLSL-driven on annotation layers via
  `prop_<name>()`.
- Richer options: an `AnnotationLayer` with
  `annotation_properties=[AnnotationPropertySpec(id='rate', type='float32')]` +
  `linked_segmentation_layer` + a custom GLSL shader reading `prop_rate()`; or serve activity
  from your own HTTP server as a precomputed annotation/`segment_properties` source and
  cache-bust the URL.
- Also `LocalVolume`, `skeleton_shader`, `segment_default_color`, `color_seed` with a Python
  port in `neuroglancer/segment_colors.py::hex_string_from_segment_id`.

## Efficient per-point colour (concrete recipe)

- Store a **1-channel `Uint8`/`Float16` intensity attribute** (139k B = **139 KB/frame**) and
  colourize in a custom `ShaderMaterial` from a colormap uniform.
- Do **not** ship a 3-component float colour buffer (139k x 3 x 4 B = **1.7 MB/frame**).
- At 30 Hz even unconverted `Float32` intensity is ~17 MB/s on loopback — fine. **JSON is the
  thing to avoid** (~1 MB/frame plus a 139k-number `JSON.parse`).

## Home Assistant integration assets

Two distinct integrations, **do not conflate**:

1. **Embedding your page inside HA** — `panel_custom` custom element, or an ingress app.
2. **Publishing simulator state into HA as entities** — `ha-mqtt-discoverable` / REST POST, at
   low rate, recorder-excluded.

| Asset | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| HA WebSocket API | HA core Apache-2.0 | Verified in full | Handshake `auth_required` -> `auth(access_token)` -> `auth_ok`/`auth_invalid`; `subscribe_events` (optional `event_type` e.g. `state_changed`), `subscribe_trigger`, `unsubscribe_events`, `get_states`, `get_config`, `call_service`, `fire_event`, `ping`/`pong`, `validate_config`, `supported_features{coalesce_messages}`; also `config/entity_registry/list_for_display` | Docs: <https://developers.home-assistant.io/docs/api/websocket/>. **Use.** |
| HA REST API | HA core Apache-2.0 | Verified | `Authorization: Bearer <Long-Lived Access Token>`, default port 8123; `GET/POST /api/states`, `/api/states/<entity_id>` (POST creates/updates any state), `/api/services/<domain>/<service>`, `/api/events/<event_type>`, `/api/history/period` (+`minimal_response`, `no_attributes`, `significant_changes_only`) | Docs: <https://developers.home-assistant.io/docs/api/rest/>. **Use.** |
| `home-assistant-js-websocket` | Apache-2.0 (MIT only <=0.6.0) | npm 9.7.0 (2026-09-15) | Official JS client | Use if a browser-side HA client is needed. |
| `hass-client` | Apache-2.0 | PyPI 1.3.1 (2026-08-24), requires **Python >=3.11 (exact match)**, sole dep `aiohttp>=3.8.4` | "Connects to Home Assistant over websockets and REST"; author Marcel van der Veldt (marcelveldt, musicassistant org); `project_urls` is null so repo/stars **UNVERIFIED** | **A maintained Python HA WS client does exist; use as library.** <https://pypi.org/project/hass-client/> |
| `py-ha-ws-client` | Apache-2.0 | PyPI 1.0.0 (2026-08-30); requires **Python >=3.12** | Good API (`subscribe_events`, `subscribe_trigger`, `call_service`, auto-reconnect + subscription re-registration) | **Do not adopt** — incompatible with our 3.11 target. <https://pypi.org/project/py-ha-ws-client/> |
| `ha-mqtt-discoverable` | Apache-2.0 | 169 stars, pushed 2026-09-16; PyPI 0.25.2 (2026-05-22), `Py>=3.10` | Creates HA-discoverable MQTT entities. Entity types: Binary sensor, Button, Camera, Cover, Device, Device trigger, Image, Light, Lock, Number, Select, Sensor, Switch, Text, Valve | **TRAP: state is only published if it changed** compared to its previous state unless `force_update=True` — do not stream the fly brain through MQTT; publish only curated low-rate entities. <https://github.com/unixorn/ha-mqtt-discoverable> |
| `ha-mqtt-discoverable-cli` | Metadata conflict: PyPI field Apache-2.0 vs classifier "Other/Proprietary License" | PyPI 0.25.2 (2026-05-23), `Py >=3.10,<4.0`; only `hmd create binary sensor` / `hmd create device` | | **Verify the LICENSE file before relying on it.** |
| `panel_custom` | Apache-2.0 | Alive (`@2026.9.3`) | Config: `name`, `sidebar_title`, `sidebar_icon`, `url_path`, `js_url`, `module_url`, `config`, `require_admin`, `embed_iframe`, `trust_external_script`. Takes a **JS module URL, not an arbitrary iframe src**; to embed an external-origin page ship a tiny element that creates `<iframe src="http://host:port">` and set `embed_iframe: true`. Store files in `<config>/www` -> served at `/local` | **Use.** <https://home-assistant.io/integrations/panel_custom/> |
| `panel_iframe` | — | **REMOVED integration** | Docs URL redirects to `/more-info/removed-integration` | **Do not use.** |
| Add-ons / "Apps" + Ingress | Apache-2.0 | Alive | `ingress: true`, default server port **8099** (or `ingress_port`), only **172.30.32.2** should be allowed, auth handled by HA, header `X-Ingress-Path`. Gateway supports HTTP/1.x, streaming content, WebSockets. **REQUIRES HA OS or Supervised** | Docs: <https://developers.home-assistant.io/docs/apps/presentation/>. HA's own install docs state Home Assistant Container "don't have access to apps" (the Apps row is marked only for HA OS), and HA Core has no Supervisor. **Fallback: `panel_custom` with a tiny custom element that iframes the URL, or simply link to the page.** |
| Lovelace custom card | Apache-2.0 | Alive | Custom element, `setConfig(config)`, `hass` property; recommended data path is a context-request custom event (`context='states'`, `subscribe=true`, `callback`); registered as a module from `/local/*.js`. Can host canvas/WebGL and subscribe live to HA states. **React is problematic inside custom elements.** | Docs: <https://developers.home-assistant.io/docs/frontend/custom-ui/custom-card/>. **Use for in-dashboard panels.** |
| HACS | **UNVERIFIED** | | Installs Integrations, Dashboard plugins/cards, AppDaemon apps, Python scripts, Templates, Themes | <https://hacs.xyz>. Distribution channel only. |
| Recorder | Apache-2.0 | Alive | Writes **EVERY** state change; `commit_interval` default **5 s**; `purge_keep_days` default **10**; supports `include`/`exclude` (domains, `entity_globs`, entities) | **Do not publish 10-60 Hz entities without excluding them.** <https://home-assistant.io/integrations/recorder/> |

### HA mapping for this project

The existing `flybrain/ha.py` (`MockHomeAssistant`, `RestHomeAssistant`) already follows the
REST shape above via `HA_BASE_URL`, `HA_TOKEN`, `HA_MODE` (`mock`|`rest`), matching
[../../CONTRACT.md](../../CONTRACT.md). For the dashboard, the low-friction path is `panel_custom`
pointing at a `<config>/www` module, with an ingress app only if the deployment is HA OS.

## Annotation tables

| Asset | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| `flyconnectome/flywire_annotations` -> `supplemental_files/Supplemental_file1_neuron_annotations.tsv` | **NO LICENSE file** (GitHub licence field null) | Verified raw fetch, no token; actively updated, has tagged releases | Plain raw.githubusercontent.com fetch, no token. Explicitly supersedes Codex's mixed annotations ("Codex presents a mix of annotations from different sources which likely diverge") | **Use as-is for the annotation table.** <https://github.com/flyconnectome/flywire_annotations> |
| Local copy `data/annotations/flywire_annotations_supl1.tsv` | inherits above | Schema-identical to upstream; **139,248 rows, 31 columns** | Columns: `supervoxel_id, root_id, pos_x, pos_y, pos_z, soma_x, soma_y, soma_z, nucleus_id, flow, super_class, cell_class, cell_sub_class, supertype, cell_type, hemibrain_type, ito_lee_hemilineage, hartenstein_hemilineage, top_nt, top_nt_conf, known_nt, known_nt_source, side, nerve, vfb_id, fbbt_id, status, dimorphism, matching_notes, fru_dsx, synonyms` | **Don't rebuild it.** |
| Codex `classification.csv.gz` | Codex public GCS | 934,402 B, anonymous HTTP 200 | `root_id, flow, super_class, class, sub_class, hemilineage, side, nerve` | Good for coarse grouping. |
| Codex `consolidated_cell_types.csv.gz` | Codex public GCS | 901,707 B, anonymous HTTP 200 | `root_id, primary_type, additional_type(s)` | Simplest one-type-per-neuron table. |
| Codex `neurons.csv.gz` | Codex public GCS | 1,679,884 B, anonymous HTTP 200 | `root_id, group, nt_type, nt_type_score, da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg` | Per-neuron neurotransmitter table for Dale's-law sign assignment. |
| Hemibrain metadata | as upstream | | `flywire_annotations/supplemental_files/Supplemental_file5_hemibrain_meta.csv`; also `Supplemental_file2_non_neuron_annotations.tsv`, `..._3_hemilineages_clustering.csv`, `..._4_summary_with_ngl_links.csv` | Use as needed. |

**Coverage caveats (verified):** 139,248 rows vs 138,639 neurons, and a neuron can appear more
than once (`supervoxel_id` is the first column; `root_id` is the neuron) — **de-duplicate on
`root_id`**. Many rows have empty `cell_class`/`cell_sub_class`/`supertype`/`cell_type`. After
de-duplicating on `root_id` we verified **100.0% coverage** of our 138,639 connectome neurons;
our loader memoises the join to `data/annotations/flywire_annotations_supl1.indexed.parquet`.

**Transmitter sign recipe** (from the Minecraft mod, used for Dale's-law sign assignment):
ACh +1; GABA/glutamate/histamine -1; dopamine/octopamine/serotonin/unclear +1; fall back
`consensusNt` -> `predictedNt` if conf >= 0.5 -> `celltypePredictedNt`.

## Live-view / 3D assets

| Asset | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| `gs://flywire_v141_m783` meshes + skeletons | FlyWire v783 **CC BY-NC 4.0** | Publicly listable and readable; keyed by FlyWire root ID | `.labels` = ASCII root IDs; `mesh_mip_1_err_40/info` = `neuroglancer_multilod_draco` sharded; `skeletons_mip_1/info` = `neuroglancer_skeletons` with radius + `cross_sectional_area`; no auth for morphology | **Use for real morphology.** The graph server needs auth: `prod.flywire-daf.com/segmentation/1.0/flywire_public/info` -> 302 to Google sign-in. |
| MaleCNS Neuroglancer state | MaleCNS **CC BY 4.0** | Public, no auth: <https://storage.googleapis.com/flyem-male-cns/v1.0/male-cns-v1.0.json> (60 KB, **45 layers**) | Layers include `cns-seg` = `precomputed://gs://flyem-male-cns/v1.0/segmentation` with subsources `mesh` (`meshes-malecns/single-res-meshes`), `numeric_properties`, `type_property`, `tags_property`; `cns-mirror` with `skeletons-malecns-mirrored/skeletons-precomputed`; a **soma-points annotation layer with a production GLSL shader using `prop_*()` + `#uicontrol` + `discard`** (a perfect template for activity colouring); presyn/postsyn annotation layers with shaders + `linkedSegmentationLayer`; `flywire-meshes` = `precomputed://gs://flyem-male-cns/flywire2mcns_meshes/783/` (FlyWire v783 meshes in MaleCNS space); ~30 ROI/neuropil layers. `comprehensive_properties/info` verified as `neuroglancer_segment_properties`: **165,122 ids x 9 properties** incl. int32/float32 numerics. Also `v1.0/male-cns-meshes-transformed-to-fafb-flywire/` = a precomputed mesh layer in FAFB/FlyWire coordinates | **Prime reference for a live 3D view.** |
| `cloud-volume` | BSD-3-Clause | 12.15.0 (2026-09-18), pure-py wheel | Cloud storage volume access | Use. |
| MeshParty | Apache-2.0 | live repo is **`CAVEconnectome/MeshParty`** (<https://github.com/CAVEconnectome/MeshParty>) (`seung-lab/meshparty` 404s); 2.0.3 (2025-07-10), `py>=3.10` | `trimesh_io.download_meshes(seg_ids, target_dir, cv_path, mesh_endpoint=...)` | Use. |
| `nglui` | MIT | live repo is **`CAVEconnectome/nglui`** (<https://github.com/CAVEconnectome/nglui>) (`seung-lab/nglui` 404s); 4.8.0 (2026-09-11), `py>=3.10` | `statebuilder.ViewerState().add_segmentation_layer(...)` | Use for building Neuroglancer states. |
| `caveclient` | **no license declared on PyPI** | 8.2.1 (2026-07-10), `py>=3.9` | CAVE materialisation client | Use with the licence caveat. |
| `flybrain` (`alextitonis/fly.ai`, <https://github.com/alextitonis/fly.ai>) | MIT (LICENSE verified) | PyPI 0.1.0 (2026-09-13), `>=3.10` | `flybrain/web.py` (`flybrain export --web <folder>`) exports the connectome **for the browser**: gzipped CSC weights split <40 MB, `meta.bin` labels/types/classes/sides, `brain.json`, documented binary format. `sshfighter/` ships a live dashboard of every neuron firing; `world/` is a 3-D fly world. Whole connectome ~210 MB VRAM; 1.4 ms/step on an RTX 4060 laptop | **Use as library / steal the browser export format.** |
| FLYBOX (`gauravvvvvvvvvv/flybox`, <https://github.com/gauravvvvvvvvvv/flybox>) | Apache-2.0 | "Open-source connectome sandbox... inspect live activity in 3D" | FastAPI + `flybrain==0.1.0` backend, React/Vite/TS frontend, `@app.websocket("/ws")`, one-origin Dockerfile. `frame_payload()` = `{"type":"frame","t","running","speed","mock","flies","world","events","challenge",...}`. **BUT** `frontend/src/BrainView.tsx` renders a 2D canvas point cloud (`getContext("2d")`, `points:[x,y,z,type]`), not real morphology, despite the claim; README lists "full neuron morphology / skeleton rendering" as an open TODO. Uses **MaleCNS, not FlyWire FAFB** | **Steal ideas** (the FastAPI+WS+React live-frame architecture is right; the renderer is the weak part). |
| DesktopFly (`DenisSergeevitch/desktop-fly`, <https://github.com/DenisSergeevitch/desktop-fly>) | Code MIT; data/ **CC BY-NC 4.0** with a `DATA_LICENSE.md` citing Dorkenwald & Schlegel | 761 stars; default branch `master` (not main); changelog 1.1.0 (2026-09-05); macOS 13+/Swift 5.9+; a Windows/Linux Electron + three.js port lives in `windows/` | "Live brain window: 23,210 real neuron soma positions from FlyWire v783, with live spikes flashing at real neuron locations." Live 1 kHz LIF sim of a **668-neuron / 18,968-connection** FlyWire v783 female circuit (LC4 104 + LPLC2 210 looming detectors, DNp01/GF 2, DNa01/DNa02 4 steering, DNp09 2 forward, DNg11 6 grooming, MDN 4 backward, DNp02/04/11 6 escape-wing, plus their 330 strongest partners), rendered over 23,210 real soma positions. A second extract: 1,045-neuron MaleCNS v1.0 locomotor circuit (16 DN, 622 VNC IN, 220 leg MNs, 153 leg sensory, 34 AN; 17,224 edges = 708,689 synapses) driving articulated legs with joint/contact feedback. Runs at 120 Hz sensory/body loop. Behaviour mapping: cursor approach -> LC4/LPLC2 -> GF spike -> takeoff; DNp09 rate -> walk speed; DNa01-DNa02 -> steering; DNg11 -> grooming; MDN -> backward | **Closest existing thing to this goal -> steal ideas + reuse the circuit selections.** |
| Infinite Sugar (`cnqso/infinite-sugar`, <https://github.com/cnqso/infinite-sugar>) | **NO LICENSE file -> unlicensed** | Browser whole-brain emulation, TS + three.js + cannon-es | Its neural map is a sampled point/connection canvas — README says "Lines join representative positions and do not trace the shapes of neurons" | **Steal ideas only.** |
| `neuVid` (<https://github.com/connectome-neuprint/neuVid>) | as upstream | canonical repo `connectome-neuprint/neuVid` (master) | Python -> JSON -> Blender/VVDViewer **offline videos**, not interactive | Use for offline renders only. |
| `neuPrintExplorer` | as upstream | Anonymous read access to `male-cns:v1.0` verified | Interactive neuPrint browser for hemibrain/neuPrint datasets | Use for exploration. |

### Rendering position data (verified in this project)

Codex `coordinates.csv.gz` (5,314,546 bytes, public GCS, no token) has **238,909 rows /
139,255 unique `root_id`** and gives **100% coverage** of our 138,639 connectome neurons after
de-duplication. Parsed with regex `\[\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\]` into 3 floats,
multiplied by the FlyWire voxel size `[4,4,40]` nm, converted to microns, centred and
normalised to about `[-1,1]`. Extent is roughly **3252 x 1566 x 11139 micrometres** (z is the
long anterior-posterior axis). Saved as a `138639x3` float32 buffer
(`data/codex/positions_normalized.npy`).

**Why this matters for us:** positions are solved — do not spend time hunting morphology to get
a 3D view working.

## Existing brain / spike viewers

| Asset | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| `mattyhempstead/fly-wirehead` (<https://github.com/mattyhempstead/fly-wirehead>) | **NO top-level LICENSE found** (LICENSE, LICENSE.md, LICENSE.txt, COPYING all 404). `THIRD_PARTY.md`: neural backend adapted from stonkfly MIT; Three.js MIT; MaleCNS v1.0 data CC BY 4.0; "3D scene is original to this repository" | Active | 166,700 neurons / 25.6M connections, Python 3.11 + C++17 kernel via ctypes, Three.js browser scene, live neural overlay + a full-width PAM11 firing-rate chart, JSONL telemetry, checkpointing, FastAPI-ish local server on loopback with session token, 0.1 ms integration steps and 50 ms neural time/frame | **Steal architecture aggressively / fork only if the author clarifies licensing.** See below for the definitive licence analysis. |
| `nftechie/stonkfly` (<https://github.com/nftechie/stonkfly>) | MIT (LICENSE fetched) | Active | Same 166,700-neuron MaleCNS LIF kernel + R1-R6/R8 visual projection + KC->MBON plasticity; upstream of fly-wirehead | **Use as library (MIT).** Kernel details in [simulation-backends.md](simulation-backends.md). |
| NeuroPulse (`tareqrwk/neuropulse`, <https://github.com/tareqrwk/neuropulse>) | **UNVERIFIED** | "Interactive web-based neural activity visualization simulator" (search only) | | Investigate. |
| SpikeInterface widgets | **UNVERIFIED licence** | | Backends: matplotlib, ipywidgets, sortingview (web, but **uploads data to a public cloud bucket / kachery-cloud**, so not local/private), ephyviewer (Qt). Has `plot_rasters()`, `plot_traces()`, `plot_unit_*`. Operates on its own Recording/Sorting objects | **Steal the raster abstraction; avoid sortingview for a private LAN app.** |
| `phy` | **UNVERIFIED** (LICENSE 404) | 2.1.0 (Jul 2026), active | Desktop Qt curation GUI for large ephys; 2.1.0 explicitly replaced its web GUI component with Qt-native | **Avoid as a web asset.** <https://github.com/cortex-lab/phy> |
| `ephyviewer` | **UNVERIFIED** | | pyqtgraph/Qt desktop; buildable from numpy (`TraceViewer.from_numpy`) | **Avoid (not browser).** <https://github.com/NeuralEnsemble/ephyviewer> |
| `neo` | **UNVERIFIED** | | Data model + file IO | **Avoid.** <https://github.com/NeuralEnsemble/python-neo> |
| SNUB (JOSS <https://doi.org/10.21105/joss.06187>) | **UNVERIFIED** | | Systems Neuro Browser | Investigate. |
| FlyWire Codex (`codex.flywire.ai`) | **UNVERIFIED** | | Read-only static snapshots (FAFB v783, BANC v888, MANC, MAOL, MCNS). Its 3D viewer is Neuroglancer; Connectivity/Pathways/Motifs produce subset graphs; explicitly "Codex intentionally does not provide a general programmatic live-query API"; bulk access = static gzipped CSV downloads | **Cannot display our activity; use as ground-truth data source + UI inspiration.** The Connectivity tool already models the subset UX we need. |

### `fly-wirehead` — definitive licence analysis

- No `LICENSE`, `LICENSE.md`, `LICENSE.txt` or `COPYING` anywhere in the repository — `main`
  and `master` both HTTP 404; `git ls-files` (all 61 tracked files) shows only
  `licenses/stonkfly-MIT.txt` and `dist/vendor/THREE-LICENSE.txt`;
  `find . -iname '*licen*'` returns exactly those two plus the `licenses/` directory (nothing in
  `docs/`); `pyproject.toml` has no `license` field; `git log --diff-filter=D` shows no deleted
  licence.
- Therefore: `flywirehead/neural/**` + `data.py` are **MIT** (nftechie/DOOMFLY,
  `licenses/stonkfly-MIT.txt`, pinned at stonkfly `78ef3e0` via `upstream.json`);
  `dist/vendor/three.*` is **MIT** (Three.js Authors); and the **original** fly-wirehead work —
  `dist/scene.js`, `main.js`, `video-feed.js`, `garden.js`, `motion.js`, `swipe.js`,
  `simulation.js`, `dopamine-plot.js`, `backend.js`, `style.css`, `index.html`, `server.py`,
  `engine.py`, `cli.py`, `scripts/download_videos.py` — carries **no licence grant. Default
  copyright applies: all rights reserved.**

Pinned commits: stonkfly `78ef3e05ab0fa086032098558d893667068944a0` (2026-09-09, single "Initial
commit", MIT); fly-wirehead `fcefe9441f80e25aab713411ebced53f5e5ea172` (2026-09-11, main).
fly-wirehead pins exactly that stonkfly commit and copies `flywirehead/neural/` nearly verbatim
(only `common.py` differs: env var rename; `controller.py` omitted).

Architecture worth stealing (ideas only, not code):

- **Vanilla ES modules + vendored Three.js 0.180.0** (`dist/vendor/three.module.js`, importmap
  at `dist/index.html:13`), served by Python `http.server`. **No bundler, no framework, no
  build step.**
- File sizes: `dist/scene.js` (249), `main.js` (165), `video-feed.js` (87), `garden.js` (85),
  `motion.js` (20), `swipe.js` (39), `simulation.js` (26), `dopamine-plot.js` (17),
  `backend.js` (48), `style.css` (10), `index.html` (56).
- Transport is **HTTP polling, NOT websocket**: `ThreadingHTTPServer` bound loopback-only
  `127.0.0.1:4173` (`server.py:255`). Endpoints: `GET /api/session` -> ephemeral token;
  `GET /api/status` -> JSON state; `POST /api/frame` (raw `application/octet-stream` RGBA);
  `POST /api/control` (`server.py:202-243`). Auth = `X-Fly-Token` (`server.py:215`) plus a
  Host/Origin check (`:187-192`).
- Poll cadence 350 ms (2000 ms on error) (`backend.js:27,45`), gated on
  `phase==='ready' && !paused && !busy && !document.hidden`. Frame = fixed 90x160 RGBA,
  bottom-up, flipped to RGB server-side (`engine.py:11,15-21`). Each accepted frame advances
  **50 ms neural time** (`cli.py:19`; `engine.py:45-76`).
- Per-neuron activity to the browser: `engine.py:69` emits
  `bins = [{"end_ms","duration_ms","counts": spikes[self.sample].tolist()}]` for a **96-cell
  fixed sample** (DAN + 24 retina + 8 R8 + MBON + motor + 32 KC, padded by `linspace` if short;
  `engine.py:30-38`). `server.py:146,154` accumulates 120 bins into `state["raster"]` and
  publishes it in `/api/status` — but **no frontend code consumes `raster`** (grep of `dist/`
  gives zero hits) and `docs/model.md:30` says "The old network plot and 96-cell raster are
  removed from the display; their underlying telemetry remains available through the API." **So
  there is no per-neuron browser visualisation at all; only aggregate telemetry reaches the
  UI.**
- **3D positions: NONE shipped.** No neuron-position dataset, mesh, or `.obj`/`.glb`/`.gltf`/
  `.swc`/`.npy`/`.npz` exists anywhere in the repo. The fly is a hand-built low-poly model with
  literal hardcoded coordinates in `dist/scene.js:62-157`, plus a hand-modelled phone and
  tether. `prepare.py` computes only 2D retinal UV coordinates, never rendered.
- Overlay/chart (`index.html:36-45`, `main.js:52-129`): two numbers — "DOPAMINE ACTIVITY" =
  `telemetry.pam11_hz` and "FLY SPIKES" = `telemetry.total_spikes` — plus one full-width
  2D-canvas line chart of the last 120 `{sim_ms, pam11_hz}` samples with auto y-scale and
  labelled bounds (`dopamine-plot.js:2-17`, `main.js:106-129`).
- Movement is artistic amplification (`motion.js:6-18`): `motorHz` -> wing/leg flutter, `turnHz`
  (DNa02 R-L) -> head turn, `pam11Hz` -> electrode glow, 100 ms attack / 700 ms release. Also a
  WebMCP `control_fly_experiment` tool (`main.js:156-165`).

**Why this matters for us:** fly-wirehead proves the browser side is easy and the protocol
(350 ms `/api/status` JSON poll + 90x160 RGBA POST, token auth) is the reusable idea — but it
ships **no 3D positions and no per-neuron view**, and its UI code cannot be copied. We already
have positions and a binary WebSocket, so our design is ahead of it.

## Graph libraries (138k nodes / 15.1M weighted sparse directed edges)

| Library | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| `scipy.sparse` | BSD-3 | **1.18.1 requires Python >=3.12** (on 3.11 you get 1.17.x, fine) | CSR/CSC, `csr_matrix @ v`, `sum_duplicates`, `eliminate_zeros`, `csgraph.shortest_path`/`connected_components`. 15.1M nnz is **~121 MB** (int32 idx + float32 data) | **Use.** |
| `torch.sparse` / `torch.sparse.mm` | BSD-3 | | GPU sparse matvec; the whole network state is one sparse matvec per step | **Use.** |
| `rustworkx` | Apache-2.0 (**UNVERIFIED**) | 0.18.1, `>=3.10`, prebuilt wheels | Rust-backed, orders of magnitude faster than networkx | **Use for graph algorithms.** |
| `igraph` (PyPI `igraph`) | GPL (verified) | 1.0.0, `>=3.9` | Full graph algorithm suite | **Use if GPL acceptable.** |
| `python-igraph` | GPL | 1.0.0 (`>=3.6`) | Legacy package name, superseded by `igraph` | **Avoid.** |
| `graph-tool` | GPL-3 (verified) | 2.11 | No Windows, conda-only in practice | **Avoid on Windows.** |
| `networkx` | BSD-3 | 3.6.1, `>=3.11,!=3.14.1` | Pure-Python dict-of-dicts, **~200+ bytes/edge**, so **~3-10 GB+ RAM** and minutes-to-hours per traversal. It will appear to work on a 100k-edge subgraph and then die | **AVOID for the 15M-edge graph.** Use only on type-collapsed/aggregated graphs. |
| `cugraph` (RAPIDS) | Apache-2.0 | `cugraph-cu12` 26.8.0, requires `>=3.11`, Linux/WSL only | `pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com`. The plain PyPI `cugraph` 0.6.1.post1 is a **stale, unrelated package — do not install it**. Our graph (121 MB CSR) is small enough that cuGraph overhead may not pay off | Use only if GPU graph algorithms become the bottleneck. |

## Telemetry dashboards

| Asset | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| Grafana | AGPL-3.0 (LICENSE fetched) | 13.2.2 (2026-09-15) | Grafana Live pub/sub over persistent WS (default max **100 connections, ~50 KB/conn**). `POST /api/live/push/:streamId` accepts InfluxDB line protocol (ns timestamps) so no Prometheus/Pushgateway needed. Their own caveat: "soft real-time... delay can be up to several hundred ms or higher"; WS payloads must be JSON. Node Graph panel: nodes/edges data frames; node colour, `arc__*` (activity rings), `mainstat`, `nodeRadius`; edge colour/thickness; **200 visible nodes by default**; Force layout for 500+ | **Steal the Node Graph + Live pattern; not the 3D view.** |
| Grafana MQTT datasource | plugin | v1.3.7, Grafana >=11 | Subscribes to MQTT topics and streams into panels in real time; **no history; QoS 0; MQTT v3.1.x only** | Strong if a broker already exists. |
| Grafana Infinity datasource | plugin | v4.0.0, Grafana >=11.6 | Polling REST/JSON/CSV/GraphQL. Docs: "not designed for handling large amounts of data... inline snippets <1MB" | **Tables only.** |
| Prometheus Pushgateway | Apache-2.0 | | Verified verbatim: "We only recommend using the Pushgateway in certain limited cases"; it "never forgets series" pushed to it unless manually deleted; you "lose Prometheus's automatic instance health monitoring via the `up` metric"; "the only valid use case... is for capturing the outcome of a service-level batch job" | **Avoid.** |
| Netdata | Agent GPLv3+, **UI closed-source free (NCUL1)** | | Per-second collection; custom apps via StatsD/OpenMetrics; UI at `:19999` | **Use-as-is for host/GPU metrics.** |
| FastAPI+WS dashboard demos | licences **partially unverified** | | `fedesanchez/fastapi-realtime-dashboard`, `permitio/fastapi_websocket_pubsub`, `silverstar33/monitrix`, `oemercemtabar/iot-monitoring-fastapi`, `JOravetz/stock-market-dashboard` | **Steal structure.** |

### Transport choice: SSE vs WS vs polling

- **MDN:** SSE is one-way and over HTTP/1.1 capped at **~6 connections per browser** (HTTP/2 ->
  100).
- **MDN:** the stable WebSocket API has **no backpressure** ("fill up... memory... or 100%
  CPU").
- **FastAPI 0.135+** gives native SSE via `EventSourceResponse`.
- **Verdict: WS binary for spikes; SSE for low-rate panels; polling <=1 Hz.**

## Connectome as a graph: rendering ceilings

**Do NOT try to draw 15M edges.** Verified ceilings and tactics:

- sigma.js own docs: smooth at **tens of thousands** of edges; tuning matters at **hundreds of
  thousands**. Frames are fragment/overdraw-bound, not vertex-bound; `antialiasEdges:false` gave
  "up to x5"; the default edge path is `pathLoop` (66 vertices) vs `pathLine` (4) — register only
  `pathLine`; edge picking redraws all edges into a picking buffer (leave `enableEdgeEvents`
  off).
- Graphistry vendor claim (July 2026): 20M+ edges, and "typically 100k+ edges" is where they say
  generic browser libs stop — **vendor claim, not independently benchmarked**; `pygraphistry`
  client is BSD-3 but the rendering server is proprietary and self-host downloads are gated
  behind their support portal.
- Gephi: dual **CDDL-1.0/GPL-3.0**, Java desktop (OpenGL), latest 0.11.3 (2026-09-06), README
  claims up to "a million elements"; **not a browser engine** — use offline for layout or to
  export coordinates.
- cosmos.gl (MIT): hundreds of thousands of points+links, GPU force layout.
- **NO authoritative, reproducible in-browser benchmark at 139k nodes / 15M directed edges
  exists — treat all "millions of edges in a browser" numbers as vendor-positioned.**
- Realistic subsetting: ego/neighbourhood graphs; **supernode aggregation** (collapse to cell
  types/neuropils/hemilineages, edge weight = summed synapses, coloured by live activity); top-K
  by weight + minimum-synapse threshold; LOD/stochastic edge sampling. FlyWire Codex's
  Connectivity tool already models exactly this subset UX.

## Real activity datasets (for display and validation)

| Asset | Licence | Status | External drive / readout | Verdict |
|---|---|---|---|---|
| Shiu et al. pre-computed spikes | MIT | Edmond DOI [10.17617/3.CZODIW](https://doi.org/10.17617/3.CZODIW), no account (anonymous HTTP 206 verified) | `results.zip` = 4.2 GB of parquet, one row per spike, columns `t` (s), `trial`, `flywire_id`, `exp_name`; **FlyWire v630**, so needs a v630->v783 remap | **Best pre-computed activity asset.** |
| `eonsystemspbc/fly-brain` spike exports | bundle data licence **UNVERIFIED** | Schema verified as `time_ms, trial, neuron_index, flywire_id, exp_name`. 600 parquet files; Drive folder `1jiSfb5lNfm9gwP0YyyRz5ATIrDpBAcjs` lists `nature_2026_07.zip` with anonymous HTTP 200, but actual download without a Google account is **PARTIALLY VERIFIED** | | Usable if the download is confirmed. |
| CRCNS fly-1 | access terms **UNVERIFIED** | Light-field Ca/voltage <=200 Hz, NIfTI | | Not keyed to FlyWire root IDs. |
| Dryad [10.5061/dryad.3bk3j9kpb](https://doi.org/10.5061/dryad.3bk3j9kpb) | CC0 | region/PCA `.npy` | | Not keyed to FlyWire root IDs. |
| BIFROST Dryad [10.5061/dryad.8pk0p2nx1](https://doi.org/10.5061/dryad.8pk0p2nx1) | CC0 | | | Not keyed to FlyWire root IDs. |

**Bottom line: no verified public REAL dataset maps per-neuron activity over time onto FlyWire
root IDs.** Everything keyed by root ID is simulated (Shiu, fly-brain).

## Dead servers and portals

- `bionet.ee.columbia.edu` **DEAD** (404/GitHub Pages). `fruitflybrain.org` live.
- Public WAMP routers live at `128.59.65.19`, including a FlyWire NeuroNLP endpoint;
  `FlyBrainLab/datasets` (<https://github.com/FlyBrainLab/datasets>) lists FlyWire snapshot
  **783 (CC BY-NC 4.0)**.
- Further portals: `braincircuits.io` and `fafb-flywire.catmaid.org` (details **UNVERIFIED**).

## See also

- [simulation-backends.md](simulation-backends.md) — engines, benchmark table, connectome data
  sources.
- [dead-ends.md](dead-ends.md) — the traps in this inventory, with alternatives.
- [../architecture.md](../architecture.md) — module map and the live-view design.
- [../licensing.md](../licensing.md) — licence position and attribution.
