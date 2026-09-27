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
- **It is not a fly.** The connectome is a **wiring graph inferred from microscopy**, not a
  complete animal. The simulator supplies generic LIF dynamics rather than the fly's actual
  membrane biophysics, and the mapping from a Home Assistant sensor onto sensory neurons is an
  encoding *we* invented. The honest description of the result is a **fly-inspired house pet**,
  not a digital fly. That does not make it less interesting — it is a fascinating reservoir — but
  it is the difference between a claim about biology and a claim about our own construction, and
  only one of those is true.

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

**This posture is taken further in [§9](#9-the-house-pet-direction)**, which reaches the same
conclusion — observation only, publish context, never own the policy — from the "house pet" framing
rather than from the latency budget.

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

**It has a second use that is arguably better than the first.** Because it answers "is anything
unusual happening *right now*", the same signal is the natural **trigger for adaptive pacing**:
burst at full rate when surprise is high, idle when it is not. That turns A3 from a notifier into
the scheduler for the whole system, and it is the piece that makes multi-day recording affordable.
See [`engine.md`](engine.md#what-the-trigger-should-be).

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

## 6. Three honest constraints

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

**Evaluation must split by day, not by random window.** Windows from one continuous session are
autocorrelated: a room changes over minutes, so neighbouring windows are near-duplicates of each
other. Splitting them randomly puts a window's own twin in the test set and reports an accuracy the
readout cannot reproduce tomorrow. Any claim that a readout — or the reservoir underneath it —
earned its place has to be measured on **whole held-out days or sessions**, against at least one
cheap baseline: a plain HA rule, and a linear fit on smoothed raw sensors. If the reservoir cannot
beat smoothing-plus-a-line on held-out *days*, then it is contributing presentation rather than
prediction, and saying so is more useful than a flattering number from a random split.

This is the same autocorrelation that makes heavy pacing cheap in *effective* sample terms — see
[`engine.md`](engine.md#the-cost-of-pacing-badly). One cause, two places it bites.

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
| **6** | **Adaptive pacing — built** ([`pacing.py`](../flybrain/pacing.py)). A heartbeat plus a burst on a change trigger, so the brain runs hard when the house is interesting and idles when it is not; the pacing config is recorded in each session's `meta.json`. The zero-code version is still available: `FLYBRAIN_INTERVAL_S` in `.env`. See [`engine.md`](engine.md#idle-cost-and-adaptive-pacing). | Performance is the binding constraint on multi-day recording. This is the only lever that materially moves energy — `fps` and `window_ms` do not. **Still open:** the trigger watches the primary sensor and discrete sensors only; the multivariate version is the A3 novelty score, which drops into the same seam. |
| **J1** | **Jev client + credential surface — built** ([`flybrain/jev.py`](../flybrain/jev.py)). First-party at `jevtypesafeai.com` — **not** the `api.typesafe.ai` the public quickstart names, which refused a perfectly good key with HTTP 401 ([`jev.md`](jev.md#the-endpoint-was-wrong-not-the-key)). Typed questions, the three call disciplines as code, a **measured network floor** (median ~49 ms warm here; the 154 ms usually quoted was the first request this machine ever made to the old host — see [`jev.md`](jev.md#the-endpoint-was-wrong-not-the-key)), and confidence routing calibrated on this project's own states. **Live against the real endpoint** (~$0.005 spent). **No placement has a UI yet** — J2–J4 are all still to build. | Zero risk, and every step below depends on it. |
| **J2** | **Decision inspector** (Jev placement A) — clickable points in *Colour chosen*, each judged into one of seven documented failure modes. Built offline against a recording first. | The highest-value placement, and it needs no live-loop changes. |
| **J3** | **Label proposal** (Jev placement B) — Jev proposes a label, you confirm or correct; written as `source="jev"`, with an agreement metric. | A lower-friction version of Phase 2, and it measures its own trustworthiness. |
| **J4** | **Attention director** (Jev placement C) — a slow timer picks which region the dashboard emphasises. | Highest visual payoff; build last, once the vocabularies are proven. |

Phases 0–3 are the ones that change what the project *is*. 4 is polish with high leverage. 5 is a
separate project and should not block anything above it.

**J1–J4 are a parallel track, not a competing one.** They do not block 1–5 and are not blocked by
them, but **J3 supersedes Phase 2** — once the client exists there is no reason to build a
type-the-label button when a confirm-or-correct one is available. The design, the closed
vocabularies, the measured API facts, and the four places Jev must *not* be used all live in
[`jev.md`](jev.md). The framing there is deliberately narrow: **Jev is a teacher and an auditor,
never a narrator and never in the frame path**, and it is justified by typed decisions plus
confidence — *not* by accuracy, which measured as a tie with a general chat model.

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

---

## 9. The house pet direction

A direction rather than a feature, recorded because it is the clearest statement of what all of the
above is *for* — and because it agrees with §2 while having arrived there from somewhere else, which
is the most useful kind of agreement.

### Observable personality, honest provenance

The pitch: **let the reservoir produce slow state, and let explicit software decide how that state is
presented.** The pet may appear curious, startled or settling. It must not *claim* to feel hunger,
joy or intent — and the code should make that distinction structural rather than a matter of
wording.

Two rules follow:

- **A small, closed vocabulary of states**, each derived from measured quantities — activity,
  novelty, house signals — and never from a hand-written mood table. `resting`, `curious`,
  `startled`, `settling` is a workable starting set.
- **Every label shows its contributors.** If the pet says "curious", the panel shows which sensor
  history and which region activity produced it. A label whose inputs are visible is a measurement;
  a label whose inputs are hidden is a claim.

That is the same discipline as [`jev.md`](jev.md#where-jev-goes): a closed vocabulary, with the
evidence beside the answer on screen.

### Observation only, to start

**Built, in the dashboard.** The five pet panels — state with contributors, pacing, the three
layers, the memory trail and the trust badges — are described in
[`live-view.md`](live-view.md#the-pet-panels-what-each-one-is-for). They watch and show; they
control nothing. The state vocabulary is `resting` / `curious` / `startled` / `settling`, derived
in [`flybrain/pet.py`](../flybrain/pet.py) from measured activity, sensor movement and the pacing
trigger, and every label carries the numbers behind it.

What is deliberately still missing: marking a trail entry *"yes, that fits"* or *"no, just passing
through"*. That is the interactive half, and it writes to the training set, so it waits for the
same confidence discipline placement B needs ([`jev.md`](jev.md)).

The first version **watches and never controls anything essential**. It may read motion, illuminance,
temperature and time of day, and express itself through a small accent light, a desk animation or a
notification. It does not own the room's lighting policy.

That is §2's conclusion reached from another direction, and it is the right one. If the fly ever does
influence the house, it should publish slow context sensors **with confidence and freshness**
attached and let existing automations decide what to do about them — never a whole-room policy from a
reservoir.

### A daily journal

One short daily line — *"more active than usual at 18:20"* — with the sensor and spike trace that
produced it, replayable. This is where the pet framing earns its keep: it makes behaviour legible
over *days*, which is the timescale the reservoir's own memory cannot reach (§6). It is also the
honest counterweight to a dashboard that only ever shows *now*.

### The first real experiment

Record several days with output disabled, and label three things: `quiet`, `passing through`, and
`settled activity`. Then compare, on **whole held-out days** (§6):

1. simple Home Assistant rules,
2. a linear readout on smoothed **raw sensors**,
3. a linear readout on **reservoir features**.

If (3) wins consistently, the fly is contributing prediction rather than presentation and there is
something worth writing down. If it does not, the honest conclusion is that this is a beautiful thing
to watch — a legitimate result, and better to know than to assume.

### Promoting a behaviour

Nothing becomes pet behaviour on the strength of one good-looking session. The gate: it beat those
baselines on days it had never seen, **and** the pacing configuration it was trained under was
recorded and reproduced (Phase 6 — event-weighted windows are a different distribution from
fixed-interval ones). Until then it stays a panel, not a behaviour.
