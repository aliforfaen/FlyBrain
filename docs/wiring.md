# Wiring: which sensor drives which part of the brain

`loop.py` wires one thermometer to one role and produces one light colour. That is a
demonstration. This document describes **the architecture**: a declarative map from Home Assistant
entities onto the fly's real sensory populations.

The guiding decision is that the connectome is **not uniform** and should not be treated as if it
were:

| Role | Neurons | Sense |
|---|---|---|
| `visual` | 10,855 | photoreceptors + motion pathway |
| `kenyon_cells` | 5,177 | mushroom body (memory, not an input) |
| `mechanosensory` | 2,656 | touch, vibration, hearing |
| `olfactory` | 2,279 | odour |
| `thermosensory` | 29 | temperature |

Driving colour from a thermometer uses **0.27%** of the brain's sensory investment. Wiring a
camera into a random slice throws away the 10,855 neurons that evolved for seeing. So each entity
is matched to the population that would actually process it.

## The pathway table

| Home Assistant | Fly pathway | Why |
|---|---|---|
| camera motion / person / pet | `visual` | a moving body is what the visual motion pathway detects (T4/T5, lobula plate) |
| illuminance / lux | `visual` | photoreceptors R1-6/R7/R8 |
| temperature | `thermosensory` | TRN_VP1m/VP2/VP3 |
| humidity | `hygrosensory` | HRN_VP1d/VP1l/VP4/VP5 |
| VOC / CO₂ / PM2.5 | `olfactory` | the fly's largest sensory investment |
| contact / door / window / vibration | `mechanosensory` | touch |
| **acoustic events** (bark, meow, glass break, sound) | `mechanosensory` | the fly *hears* through mechanosensory structures — a meow is not motion |

**The ordering of the rules is load-bearing, not cosmetic.** A camera vendors an acoustic event as
a binary sensor whose `device_class` is `motion`. Classifying by device class alone would wire a
bark into the visual system — the wrong pathway, silently, with no error anywhere. Sound is
therefore matched **before** motion, and there is a test for exactly that
(`test_sound_is_matched_before_motion`).

## The matching bug that discovery found

The first implementation matched keywords as raw substrings. Run against the real house it
produced:

```
sensor.backup_last_attempted_automatic_backup   ->  thermosensory
sensor.rack_gputemperature                     ->  thermosensory
```

The first matched **"temp" inside "attempted"**. The second is a GPU die temperature being wired
to the fly's thermosensory neurons as if it were the room. Both would have looked entirely
plausible in a summary table and quietly corrupted the input.

Matching is now on **whole words**: the entity id is split on `.` and `_` and a keyword must be a
complete token. `attempted` no longer matches `temp`, and `gputemperature` is not `temperature`.
Both regressions are pinned by tests.

## Discovery

`flybrain/wiring.py` reads a live instance and proposes the wiring:

```bash
set -a; source .env; set +a
.venv/bin/python -m flybrain.wiring                 # table
.venv/bin/python -m flybrain.wiring --json          # machine-readable, can be pinned with load_wiring()
.venv/bin/python -m flybrain.wiring --channels 64   # also build encoder channels (needs connectome)
```

It separates three outcomes, and the distinction matters:

- **live pathways** — matched a role and currently reporting a value
- **dormant** — matched a role but the entity reports nothing *right now*. Kept visible: a camera
  whose detectors are dark is a wiring problem to fix, not a wiring decision to hide.
- **unreachable roles** — a known sensory role with no sensor at all. Stated explicitly, because
  "no olfactory pathway" and "olfactory pathway that is silently broken" look identical from the
  brain's side.

### What it says about this house

```
live pathways: 3  across 2 role(s)
  sensor.hallway_temperature           thermosensory   temperature
  binary_sensor.hallway_motion         visual          motion
  sensor.hallway_illuminance           visual          illuminance

dormant (6) — wired to a role, but reporting nothing:
  binary_sensor.door_camera_motion_alarm          visual   unavailable
  binary_sensor.door_camera_person_detection      visual   unavailable
  binary_sensor.door_camera_cell_motion_detection visual   unavailable
  ... (_2 variants)

no sensor at all for: hygrosensory, mechanosensory, olfactory
```

## Channels: disjoint pools, deliberately

`Wiring.to_channels()` splits each role's population between the pathways on that role and
subsamples to a common width.

**Disjointness is the point.** If two motion sensors drove overlapping neurons, the brain would
receive their *sum* and no readout could ever tell which one fired. Each role's population is
divided evenly, so two visual pathways get two separate pools from the 10,855 visual neurons.
Selection is seeded from a hash of the entity id, so a restart re-creates the identical wiring
rather than silently reshuffling the brain's inputs.

A role smaller than the requested width is used whole — thermosensory has 29 neurons, and padding
it would be inventing cells that do not exist.

An unknown role in a wiring file costs that channel and logs a warning; it does not stop the loop.

## Current status

| Piece | State |
|---|---|
| Pathway rules + matching | **built**, 28 tests |
| Discovery CLI against the real house | **built**, run and verified |
| Channel construction with disjoint pools | **built** |
| Wiring pinned to disk (`load_wiring`) | **built** |
| Driving the live loop from *multiple* channels | **built** — `LiveLoop.drive_channels`, 15 tests, verified live |

`LiveLoop.drive_channels()` applies the trained temperature path plus every configured channel in a
**single** `set_drive` call. That is not a style choice: `set_drive` *replaces* the whole drive map,
so driving temperature and then the extras separately would silently leave only the last one
applied — and it would look like "the extra sensors do nothing" rather than like an error.

### What it does to the colour readout (measured)

Enabling extra channels is opt-in (`FLYBRAIN_CHANNELS=auto`) because the colour readout was fitted
with **temperature alone** driving the brain, and a readout is a function of the whole reservoir
state. Driving motion and illuminance as well changes that state.

Measured against the live house, same configuration, 3 samples each:

| Channels | decoded − ideal |
|---|---|
| off (temperature only) | −77 to −141 K |
| on (motion + illuminance) | −65 to −112 K |

**No measurable degradation — but this is weaker evidence than it looks.** Both extra channels sit
at their *floor* in this test: the room is empty (motion `off` → 20 Hz) and dark (6 lx → 21 Hz), so
the added drive is nearly identical to the temperature path's own floor. The test does **not**
exercise motion while detected or a brightly lit room. Treat the readout as unvalidated under real
motion until it is re-fitted on recorded data.

### A bug this measurement exposed

The dashboard's error readout was wrong whenever the sensitivity range was stretched. `snapshot()`
computed the "ideal" colour from the **raw room reading**, while the loop maps that reading onto
the trained range first — so a correctly-behaving loop reported an error of ~1500 K. It now
evaluates the ideal at the temperature the *brain* was given (`_active_brain_c`), which is what the
loop actually targets. The numbers above are only meaningful after that fix.

## What this house does not have

Three roles have no sensor, and each is a genuine gap rather than a wiring choice:

- **`olfactory`** — no VOC, CO₂ or PM2.5 entity. This is the most biologically apt pathway in the
  whole project and it needs hardware.
- **`mechanosensory`** — no contact sensor, and no acoustic event sensor. The Tapo C120 exposes
  `select.door_camera_bark_detection`, `_meow_detection` and `_glass_break_detection`, but those
  are **sensitivity settings**, not event sensors: there is no binary sensor that fires when the
  camera hears a bark. So the camera currently produces no acoustic events at all. The rule is
  ready for a sensor; this hardware does not have one.
- **`hygrosensory`** — no humidity sensor anywhere.
