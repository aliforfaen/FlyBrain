# Where this could go

A use plan for the fly brain in a real house. Written for a homelabber, not a
neuroscientist, and deliberately honest about what the brain is and is not good for.

The colour demo is a **proof of concept**: one sensor, one actuator, one learned mapping. This
document is about what a second, third and tenth thing would look like — and which of them are
actually worth building.

---

## 1. What the brain actually is

Strip away the biology for a second and the architecture is this:

```
many HA sensors ──► encoder ──► FROZEN connectome (recurrent, spiking) ──► spike rates
                                              │
                                              └──► small TRAINED linear readout ──► outputs
```

A big fixed recurrent network with a small trainable layer on the end has a name: a
**reservoir**. It is the same shape as an echo-state network or a liquid state machine. The
connectome is a very elaborate, biologically real reservoir — but it is a reservoir, and that
tells you exactly what it is good for.

What a reservoir gives you that a plain `if temperature > 21` does not:

- **Memory of the recent past.** The recurrent connections hold a fading trace of what just
  happened. A threshold sees the present; a reservoir sees a smeared-out history.
- **Nonlinear feature expansion.** A linear readout on top of a rich nonlinear state can express
  relationships a linear rule cannot — "warm *and* humid *and* the house has been quiet for ten
  minutes".
- **Many outputs for the price of one.** The expensive part is *stepping the connectome*. Once it
  is stepping, an extra readout is a matrix multiply. Ten different questions cost almost nothing
  more than one.

That third point is the whole economic argument for this project, and it is the thing to build
around.

### What it is not

- **It does not reason, plan, or understand.** There is no language, no goals, no model of your
  house. It is a dynamical system with a linear map bolted on.
- **It will not beat a purpose-built model on a single narrow task.** For "is music playing", a
  small audio classifier wins. Every time.
- **It cannot be told a rule.** You cannot write "when the sensor goes above X turn on the light"
  into it. It can only be *shown examples* and have a readout fitted. Everything it knows, it
  learned from data you had to record and label.

So the honest pitch is not "an AI brain for your house". It is: **one shared, never-retrained
feature extractor with temporal memory, that can answer many questions at once.** Build for that
shape and it is genuinely useful and genuinely fun. Build for "it will figure out my house" and
you will be disappointed.

---

## 2. Division of labour: HA does reflexes, the fly does context

The hard constraint is speed. The simulator runs at **0.13× realtime** — a 300 ms window of brain
time takes about **2.3 seconds** of wall clock. So:

| Layer | Latency | Who does it |
|---|---|---|
| Reflex | milliseconds | **Home Assistant automations.** Motion → light on. Doorbell → announcement. |
| Context | seconds to minutes | **The fly.** "The house is winding down." "Something is unusual." |

This is not a compromise, it is a good architecture, and it is how the brain works too — the fly
has fast reflex circuits *and* slow modulatory state. Let the fly be the slow layer.

Concretely, the fly should not *control* your lights directly. It should publish a small number of
slow, meaningful state variables — `house_activity`, `house_novelty`, `house_mood` — as Home
Assistant sensors, and your existing automations should use them as **conditions**:

```yaml
# your automation stays in charge of the light; the fly biases it
- alias: Evening wind-down lighting
  trigger: [ ...whatever you already have... ]
  condition:
    - condition: numeric_state
      entity_id: sensor.flybrain_activity
      below: 0.2
```

That single design choice turns a toy into something you would actually leave running.

---

## 3. The recorder — built

**This was the prerequisite for everything else, and it now exists** (`flybrain/recorder.py`,
verified against the live house: 289 entities captured per window, zero writes). Companion
material: [`live-view.md`](live-view.md#recording-windows).

The live loop used to run and forget. To teach the brain a second thing you need training data,
and training data means: for every window, the 512-dimensional reservoir spike vector, the HA
sensor values at that moment, and a label.

```
data/recordings/<name>/
  spikes.npy     # (n_windows, 512) reservoir state per window
  sensors.parquet# everything HA knew at that window
  labels.jsonl   # timestamped labels you applied by hand
  meta.json      # roles, window_ms, encoder config, git revision
```

Two things this buys:

1. **Offline fitting.** Re-fit a readout from the log on the CPU without the GPU running live, in
   seconds.
2. **Honest accuracy measurement.** You can only claim a readout works if you have held-out data
   from a *different* session than the one you trained on.

Record continuously from day one, even before you know what you want to predict. Disk is cheap and
you cannot go back and record last week.

---

## 4. Sensing: map your devices onto the fly's actual senses

This is the difference between a gimmick and using the asset. The connectome has real sensory
pathways, and the encoder can already target them by role. Wire each HA device to the modality it
most resembles, not to a random slice of neurons.

Populations available today (from `RoleResolver.summary()`):

| Role | Neurons | Wire this to |
|---|---|---|
| `visual` | **10,855** | Cameras — motion energy / frame difference |
| `kenyon_cells` | **5,177** | (not an input) mushroom body: associative memory, novelty |
| `mechanosensory` | **2,656** | Speakers, hifi, vibration — audio envelope |
| `olfactory` | **2,279** | Air quality: VOC, CO₂, PM2.5, eCO₂ sensors |
| `descending` | 1,299 | (output) brain → ventral nerve cord |
| `antennial_projection` | 685 | Second-order odour output |
| `antennial_local` | 429 | Antennal-lobe local interneurons |
| `gustatory` | 408 | Taste — probably nothing in a house |
| `motor` | 110 | (output) the fly's motor neurons |
| `mushroom_body_output` | 96 | (output) learned-odour associations |
| `hygrosensory` | 74 | Humidity — **no humidity sensor exists in this house** |
| `thermosensory` | 29 | Temperature — what the demo uses |

Two things fall straight out of this table:

- **The fly is a visual animal.** 10,855 visual neurons against 29 thermosensory. Driving colour
  from a temperature sensor is using 0.27% of the brain's sensory investment. Cameras are where
  the connectome is actually deep — and the fly's famous superpower is *motion* detection. Feed
  frame-differences, not raw pixels.
- **Humidity is *not* free — there is no humidity sensor.** `hygrosensory` is resolved and in the
  encoder's role list, which makes it look like a one-line win, but the live instance reports no
  humidity entity at all (`docs/ha-inventory.md`). What the house *does* have is a motion sensor
  and an ambient light reading, both from the same 4-in-1 device as the thermometer — and
  `illuminance` maps to `visual`, the fly's largest population.

### Resolved: the circadian clock role was unreachable

`ROLE_SPECS["clock"]` filtered `cell_class == "clock"`, which matches **nothing** — so the role
silently resolved to zero neurons while looking perfectly reasonable. It is now fixed and resolves
to **48** cells (`l-LNv`, `s-LNv`, `LNd`, `DN1`), listed by explicit `cell_type`.

The trap is worth recording, because the first attempt at fixing it was also wrong. The obvious
move is to match the `lLN*` names, which read like "large lateral neuron". They are not clock
cells: their `cell_class` is `ALLN`, antennal-lobe **local** neurons — olfactory interneurons. A
prefix match pulls in **158** of them (3× the real clock) while still missing most of the actual
clock. This is why the role is an explicit list of ten cell types, and why
`tests/test_mapping.py` asserts that no clock value starts with `lLN`.

`docs/live-view.md` lists "a time-of-day prior" as a cheap idea. The role is now reachable; wiring
an actual time-of-day input into it is a separate change, and needs the readout re-fitted because
it alters the input distribution.

---

## 5. Applications, tiered

Ranked by *actually useful* × *honest use of the reservoir* × *works at 0.13× realtime*.

### Tier A — the context layer (best value, plays to the reservoir's strengths)

**A1. `house_activity` — one slow scalar for "how alive is the house".**
Inputs: motion sensors, `device_tracker` for your phone, `media_player` states, any sound level.
Output: a single 0–1 number, smoothed over minutes. Use it to bias lighting scenes, route
notifications (only ping the phone when the house is quiet), set hifi volume, trigger night mode.
*Why a reservoir:* it remembers the *texture* of the last few minutes, so it can tell "someone
walked past the sensor" from "the house is properly busy" — which a threshold cannot.

**A2. Settling vs passing through.**
Same inputs, different question: is this motion the start of sustained presence, or a passing body?
*Why a reservoir:* this is a question about the shape of the recent past. It is what recurrent
state is for. Use it to avoid killing lights on someone who just went to the kitchen.

**A3. `house_novelty` — surprise.**
Train a readout to predict the *next* window's reservoir state from the current one. The prediction
error is a surprise signal. High surprise → "something is unusual" → notify your phone.
*Why it fits:* the mushroom body (5,177 Kenyon cells → 96 MBONs) is the fly's novelty and
associative-learning circuit; there is a real analogue here, not just a metaphor. And the honest
advantage over a per-sensor threshold is that this is **multivariate and temporal** — it notices
"the house is behaving wrongly", not "this one sensor crossed a line".

**A4. Soft sensors.**
Infer something you do not measure from things you do: humidity in a room with no sensor, or
"the kitchen is in use" from temperature + humidity + motion + sound. This is genuine nonlinear
fusion and it is where a linear readout on a rich state genuinely beats a hand-written rule.

### Tier B — modality-matched (currently blocked by missing hardware)

Worth stating plainly: **three of the four Tier B ideas cannot be built today.** Not because the
code is missing, but because the sensors are. This is what moved Tier A from "nice" to "the plan".

**B1. Cameras → `visual` (10,855 neurons) — half-unblocked.** The fly's motion pathway is its most
famous circuit. Output a graded motion confidence plus per-hour novelty, so you are alerted when
motion is *unusual for this time of day* rather than every time a cat walks past. Caveat: give the
readout a longer window than 300 ms — motion is fast, and at 0.13× realtime the brain's own memory
is short (see §6).

**State as of the camera integration coming online:** the *streams* are up
(`camera.door_camera_hd_stream_direct` and `_sd_stream` are `idle`), but every motion/person
**event** sensor is still `unavailable`. So there are two routes, and they are very different
amounts of work:

- **Cheap:** wait for `binary_sensor.door_camera_motion_alarm` / `_person_detection` to start
  reporting. Then the camera is just another pathway (see [`wiring.md`](wiring.md)) and the fly
  gets a real visual-motion drive for nothing.
- **Expensive but far better:** decode the stream itself. `GET /api/camera_proxy/<entity>` is
  read-only, so frames can be pulled, differenced, and fed in as *actual* motion energy into the
  visual population. That is the fly doing the seeing rather than reading someone else's verdict —
  and it is the one idea here that uses the 10,855 visual neurons as more than a big number.

Note also that the camera exposes **no acoustic event sensors**; its bark/meow/glass-break
entities are sensitivity *settings*, not events. See
[`wiring.md`](wiring.md#what-this-house-does-not-have).

**B2. Speakers / hifi → `mechanosensory` (2,656 neurons) — available.** Audio amplitude envelope.
Output music-vs-speech-vs-quiet, then duck the hifi when speech is detected.
Honest caveat: a small audio classifier beats this on accuracy. The win is that it is the *same*
engine and the *same* readout machinery — one more tap, not one more service.

**B3. Air quality → `olfactory` (2,279 neurons) — blocked, no sensor.** Olfaction is the fly's
largest sensory investment, so this is the most biologically apt mapping in the list. Output
"cooking is happening" or an air-quality trend, for the extractor fan or a "open a window" nudge.
The house reports **no** VOC, CO₂, PM2.5 or eCO₂ entity. This needs a sensor purchased first — and
it is the single highest-value purchase for this project, because it is the one modality the fly
is genuinely built around.

**B4. ~~Humidity → `hygrosensory`~~ — not available.** There is no humidity sensor in the house
(`docs/ha-inventory.md`), so this cannot be built. It stays listed only so it is not re-proposed
as a "one-line change"; the encoder genuinely does support it, the data does not exist.

### Tier C — actuation beyond one bulb

**C1. Multi-output scenes.** `codec.py` already carries a `BrainCommand` with multiple
`action_keys`, so the readout can output a vector. Temperature + activity + audio → a whole scene:
light colour *and* speaker volume *and* phone DND *and* a notification. The current
temperature→Kelvin path is a deliberately narrow specialisation of machinery that is already
general.

**C2. Region taps — several "organs" from one simulation.** Because stepping the connectome is the
only real cost, you can hang several readouts off the same run: one on the visual population, one
on the olfactory, one on the mushroom body. That is much closer to how a brain is actually
organised than one monolithic readout, and it is nearly free.

**C3. A "house vibe" number in the dashboard.** One slow scalar summarising the reservoir's state,
shown as a single big number. Cheap to build, and it is the visible hook for the whole context
layer — the thing you glance at.

---

## 6. Two honest constraints

**The fly's memory is short in brain-time.** Neural time constants here are tens of milliseconds.
Even discounting for the slow simulation, that is well under a second of wall clock. House-relevant
memory — minutes — therefore **cannot come from the connectome**. It has to come from either
input smoothing / stacking several windows (the existing `smooth_ms=5000` is exactly this), or from
deliberately running the brain in *subjective slow motion*: hold each sensor sample for many brain
steps so that brain-time maps onto house-time. That second option is a free and rather elegant
knob, and it makes the fly's intrinsic memory span the timescales you actually care about. Decide
the mapping on purpose; do not inherit it by accident.

**Training a new sense is the expensive part.** Stepping the connectome is cheap per window; getting
*labelled examples* is not. The colour readout needed a continuous sweep. A "cooking" readout needs
several real cooking sessions, recorded and labelled. Budget for the data, not the compute. This is
also why §3 comes first, and why **teaching by demonstration** — an HA button that timestamps "this
is what I mean", plus a nightly refit over the recorded log — is the right UX. It is honest about
how the thing learns, and it is the only workflow that scales past one or two readouts.

---

## 7. Phases

| Phase | Work | Why now |
|---|---|---|
| **0** | ~~Recorder~~ — **built** (§3). Logs reservoir state + HA sensors + labels per window, with `labelled()` replay for offline fitting. | Prerequisite for every new sense. |
| **0b** | ~~Fix the `clock` role~~ — **done.** Now resolves to 48 cells (§4). Wiring an actual time-of-day input into it is still open. | Was a silent zero-neuron role. |
| **1** | **Multi-readout plumbing.** Generalise `loop.py` into a registry of readouts over one `ConnectomeSim`. It currently hardcodes temperature→Kelvin. | Makes sense #2 cheap instead of a rewrite. |
| **2** | **Label button + nightly refit.** The CLI label button exists (`python -m flybrain.recorder --label busy`); what is missing is binding it to the Hue dimmer buttons and a job that re-fits readouts from the log. | The teaching workflow. |
| **3** | **First non-light sense.** Recommend `house_activity` (A1) — most useful, and it stress-tests multi-sensor input + multi-output publish. **Now unblocked: the recorder can collect the training data.** | Proves the context layer end to end. |
| **4** | **Scene output** (C1) and publish state to HA as sensors (§2). | Turns it from a viewer into something your automations consume. |
| **5** | **Engine: active-set integrator** (`engine.md`). Only if you want reactivity. | Not needed for any Tier A/B idea. Needed for motion-triggered anything. |

Phases 0–3 are the ones that change what the project *is*. 4 is polish with high leverage. 5 is a
separate project and should not block anything above it.

**Also worth noting for Phase 3:** `house_activity` needs examples of the house being busy and
quiet, and the only source of those labels is you pressing something when they happen. So the
practical order is: turn recording on now (it costs disk and nothing else), live in the house
normally, label occasionally, and train once there is something to train on. Recording cannot be
done retroactively.

---

## 8. Open questions — answered

The live instance was read on 2026-09-23; full inventory in [`ha-inventory.md`](ha-inventory.md).

- **Which entity IDs exist?** 289 of them, HA `2026.7.1`. The environmental sensing is *one* 4-in-1
  device (`hallway_sensor`: motion + temperature + illuminance + battery). Everything else is
  phones, watches, TVs, speakers and lights.
- **Air-quality sensors?** **No.** B3 needs hardware bought.
- **Camera motion events?** The integration supports motion and person detection, but all three
  cameras are `unavailable` right now, so nothing can be built or tested against them.
- **A sound level anywhere?** No — only `media_player` on/off states, which is what B2 would have to
  work from.
- **The light?** `light.hall_lamp`: 2000–6535 K *and* `xy`-capable, so the output can be colour
  rather than only white.
- **The thermometer?** `sensor.hallway_temperature` — **not** the `sensor.living_room_temperature`
  that the loop defaults to.

The last two answers change the immediate work more than the rest combined: real-HA mode needs the
entity set explicitly, and the warm end of the colour range can go 700 K lower than the software
currently allows.
