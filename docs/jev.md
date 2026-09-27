# Jev: a judgment layer for the dashboard

**Status: design only. Nothing in this document is built.** This is the design record for
adding [Jev](https://docs.typesafe.ai) — TypeSafe's "System One" model — to the live view, plus
the evaluation that says where it earns its place and where it does not.

Jev is not a chat model. It reads a **state** and a set of **typed questions**, and returns
**decisions drawn from answers you supplied**. TypeSafe calls it a System One model; the
practical description is a classifier with a language model inside.

Two decisions shape everything below:

1. **Jev is called first-party, never through a reseller gateway.** The measurement that forced
   this is in [Why hosted, and why first-party](#why-hosted-and-why-first-party). The host is
   `jevtypesafeai.com` — **not** the `api.typesafe.ai` the public quickstart names, which refused
   a perfectly good key with HTTP 401. See
   [The endpoint was wrong, not the key](#the-endpoint-was-wrong-not-the-key).
2. **It is not self-hosted, and it never enters the frame path.**

How those calls are made is not an implementation detail. The disciplines in
[API call discipline](#api-call-discipline) are what keep the bill, the latency and the error
rate down, and they constrain the UI design rather than following it.

---

## The one rule

> **Jev is a teacher and an auditor. It is never a narrator, and it is never in the hot path.**

Two consequences that are easy to get wrong:

1. **It cannot write prose.** It picks from a menu. Every use below is therefore defined by a
   *closed vocabulary*, and the quality of the feature is entirely the quality of that menu.
   "Explain what the brain is doing" is not a Jev feature; "classify this decision as one of
   seven documented outcomes" is.
2. **Its output is a judgment, not a fact.** A bounded output interface does not stop a model
   from misreading the input. `format valid` and `answer correct` are different claims, and Jev
   only guarantees the first.

Two requirements follow, and they apply to every vocabulary in this document:

- **Always include an `unknown` or `other` option.** If every option is wrong, one still wins.
  A confidence of 0.8 *among the candidates you supplied* is not an 80% real-world success rate.
- **Surface `abstain`.** A refusal is a UI state, not an error, and it is the signal that this
  window is worth a human's attention.

---

## What Jev actually returns

Three question kinds. They map almost one-to-one onto UI widgets, and that mapping *is* the
design:

| Question kind | Returns | `confidence`? | UI widget | Used for here |
|---|---|---|---|---|
| `choice` | `choice`, `probabilities`, **`confidence`** | **yes** | **highlight / select** | which region to show, which failure mode |
| `score` | `score`, `legend`, `probabilities`, **`confidence`** | **yes** | **a number on a dial** | graded judgments |
| `noul` | `noul` only, `0..1` | **no** | **indicator / traffic light** | clean yes/no |

**Only `choice` and `score` carry a `confidence`.** It is a statistic computed from the
`probabilities` distribution — concentrated means certain, spread means uncertain — returned so
you do not have to do the math yourself. A `noul` returns the bare probability and nothing else.

That asymmetry is a design constraint, not a footnote: **any judgment that will be routed on
confidence must be a `choice` or a `score`.** A `noul` gives you only distance from `0.5`.

### Response shapes that are easy to get wrong

These were taken from the vendor's **generated wire schema** (``typesafe_sdk/_schemas/models.py``
in ``typesafe-sdk==0.7.2``, which mirrors their OpenAPI document) rather than inferred, and the
HTTP surface was confirmed by live probe. Two of them contradicted an earlier draft of this
document, which is why they are spelled out:

- **The envelope is ``{"model", "answers", "usage"}``.** Answers are keyed by the question ids you
  chose; ``usage`` is ``{"input_tokens", "output_tokens"}``; ``model`` is the versioned id that
  actually answered and *"may differ from the alias supplied in the request"*.
- **A ``choice`` question is written with ``criteria``, a mapping of label → description.** It is
  not ``options`` and it is not a list. The labels you supply are what come back.
- **A ``score`` question takes ``criteria`` as an ordered sequence** of level descriptions, one per
  level from zero. The levels are positional and unnamed, which is why the response keys them
  ``"0"``, ``"1"``, … and why ``legend`` exists to map them back.
- **``choice`` and ``score`` key their ``probabilities`` differently**, and this is the easiest
  thing here to get wrong:
  - ``choice`` → keyed by **choice label** (``{"calm": 0.8, "busy": 0.2}``)
  - ``score`` → keyed by **level, as a string** (``{"0": 0.1, "2": 0.8}``)
- **``score`` is not a ``0..1`` float.** It returns the probability-weighted mean of the rubric
  levels, so a four-level rubric can return ``2.68`` — "between High and Urgent, leaning Urgent".
  Recomputing it from the returned ``probabilities`` reproduces it to within ±0.02, because those
  are rounded to two decimals while ``score`` is computed from full precision. **That small
  mismatch is not a bug**, and the client asserts it live so a real disagreement is caught.
- **``noul`` ships no confidence field at all.** The entire answer is
  ``{"type": "noul", "noul": 0.99}``. If you need to gate on it, distance from ``0.5`` is all you
  have — which is why the client refuses to route one and says to ask a two-option ``choice``.
- **Errors arrive as ``{"detail": {"error_type", "message"}}``**, verified against the live
  endpoint. ``error_type`` is ``"authentication_error"`` for both credential failures.

### The measured facts

| Fact | Measured | Consequence for this project |
|---|---|---|
| Latency, raw wall clock | 458–563 ms median | Too slow for 20 fps (50 ms), fine for a 2.3 s control tick |
| Network floor alone | **198.8 ms** median (their machine) | "Anyone publishing a Jev latency number without measuring the floor is publishing their own geography" |
| Latency, head-to-head median | 352 ms | Consistent with the raw range |
| Cost per decision | **$0.0000153–$0.0000226** | ~20× *below* the $0.0004 figure in circulation |
| Batching | 5 questions ≈ same cost and latency as 1 (70 ms vs 74 ms server time) | **Ask everything in one call** |
| Accuracy | 27/27, tying `mistral-small-3.2-24b` | Jev does **not** win on accuracy |
| Route matters more than the model | first-party p50 **313 ms** / p90 423 ms — OpenRouter p50 734 ms / p90 **1739 ms** | **We call first-party.** See below |

Measured **on this machine**, by `flybrain.jev.measure_network_floor_ms()`, which defaults to the
configured host. The floor belongs to the *path*, not to the vendor, so both hosts are listed:

| Measurement | Value |
|---|---|
| **Working host**, `jevtypesafeai.com`, 9 warm samples | **45.4–68.4 ms, median 49.0 ms** (first sample 55.5 ms) |
| Abandoned host, `api.typesafe.ai`, 9 warm samples | 44.6–51.6 ms, median 46.6 ms |
| Abandoned host, the first request ever made to it | **154.4 ms** |
| Abandoned host: unauthenticated `POST /v1/systemone` | **HTTP 403** — `"Must supply an API key!"` |
| Abandoned host: `GET /v1/models` with a rejected key | **HTTP 401** — `"Cannot authenticate with the server"` |

**The cold/warm gap is real, but it is a first-contact effect and not a per-process one.** The
154.4 ms sample was the first request this machine ever made to that host — DNS plus a full TLS
handshake. Re-measured against the working host, the first sample was 55.5 ms against a 49.0 ms
median, only ~1.1×. So the honest rule is still *measure repeatedly, take the median, and say how it
was measured* — a single number is a claim about when it was taken as much as about the network.

**Two corrections this project should not repeat.**

First: the ~440 ms figure used in earlier discussion is in range, but quoting it bare would be
quoting our own network path. The measured network floor was ~199 ms of the ~460–560 ms. A
latency claim without a floor measurement says more about geography than about Jev.

Second, and more important: **"Jev is more accurate" and "typed output means no schema errors"
are both unsupported.** On 27 labelled tickets Jev tied a general chat model, and the type
reliability advantage disappeared entirely once the chat model was called with
`response_format: {"type": "json_schema", "strict": true}`. The real justifications for Jev
here are narrower and should be stated as such:

1. **A typed decision is the interface, not an afterthought** — no prompt engineering to get a
   usable answer.
2. **It returns calibrated-ish confidence**, which the chat models do not return at all. That is
   what makes the label-proposal workflow below possible.
3. **Cost is ~20× below the quoted figure**, which makes generous labeling affordable.

That is enough to justify the experiment. It is not enough to justify a claim that Jev is
**better** than the alternatives, and this document should not pretend otherwise.

---

### What the confidences actually look like

The margin quoted at the top of this document (0.979 clear / 0.841 ambiguous) came from someone
else's ticket-triage data. **Measured here, on this project's own questions and states**, the
picture is different and more useful. Six calls, ~$0.0013:

| The state… | Confidence |
|---|---|
| determines the answer (fresh reading → tracks, stale → holding, drifted → drifting) | **0.940 · 0.980 · 0.990** |
| genuinely under-determines it (age 45 s but still choosing, small delta) | **0.340** |
| partially determines it (small delta, ambiguous drift) | 0.880 |

So the separation is **wider** than the borrowed figures suggested — a state that fixes the answer
lands at 0.94–0.99, and one that does not collapses to 0.34. That is good news for routing.

**And a caveat that constrains it.** A byte-identical request sent six times:

| | min | median | max | spread |
|---|---|---|---|---|
| `choice` confidence | 0.490 | 0.590 | 0.660 | **0.170** |
| `score` | 2.46 | 2.485 | 2.53 | 0.070 |
| `score` confidence | 0.440 | 0.470 | 0.490 | 0.050 |
| latency | 694 ms | 726 ms | 977 ms | 283 ms |

The answer never changed; the confidence moved by **0.17**. Three consequences, and the third is
the one that matters:

1. **A threshold must not sit on a knife-edge.** A router that treats 0.945 and 0.955 as different
   outcomes is reading noise.
2. **Re-calibration is not a one-off.** Any threshold is a statement about a distribution, not a
   number, and it should be re-measured rather than inherited.
3. **The conservative default is doing real work.** `JEV_LABEL_ACT=0.95` against a 0.17 spread and
   a 0.94 floor means an under-determined answer *cannot* write a label, which is the outcome the
   whole risk ladder exists to produce. The default looks over-cautious only if you assume the
   confidences are stable, and they are not.

**The methodological catch worth recording.** The first two cases I tried to measure inverts the
labelling: the state I called "clear" (a healthy window whose chosen colour sat 30 K from the
ideal) scored **0.33**, and the state I called "ambiguous" (literally `stale: true`) scored **1.0**.
Confidence tracks how well *the state determines the answer to the question asked* — not how hard
the situation feels to the person writing the probe. My "clear" case contained a contradiction
(a 30 K gap is evidence for `drifting`), so the model was right to be unsure and my label was
wrong. Any calibration harness has to be built from states whose answer is fixed by construction,
which is what the table above is.

## Why hosted, and why first-party

### Why hosted at all

Self-hosting was evaluated and rejected for this project. The findings, recorded so they do not
have to be re-derived:

- **The latency is mostly round-trip, not inference.** Jev's latency barely moves between a small
  request and a 1024-token / 32-candidate one (~295 ms vs ~301 ms), which means fixed overhead
  dominates. Self-hosting would delete the round trip but not the compute.
- **The local models that fit this machine are a different accuracy class.** The box has an
  RTX 3070 with **8 GB**. Open-Jev-9B needs 16 GB in bf16; Open-Jev-2B fits but scores 150/231
  on JevBench against Jev's ~200/231. There is no local Jev.
- **A 400M encoder is genuinely fast and genuinely worse.** The Verdict-class models run at
  ~1 ms on CPU with a conformal abstain guarantee, but they are weak zero-shot (~0.59 on
  Banking77 against Jev's ~0.80–0.87) and need ~16 labels per class to reach 0.86.
- **Therefore:** use the accurate hosted model for the small, high-value job (bootstrapping
  labels and auditing), and let the fly's own readout be the fast tier.

### Why first-party, not a gateway

Two reasons, both measured, and the second is the one that decides it:

1. **A gateway roughly doubles the latency.** Interleaved against the same network in the same
   minute — so both routes saw identical conditions — first-party measured **p50 313 ms /
   p90 423 ms**, the gateway **p50 734 ms / p90 1739 ms**. The p90 is the number that matters
   for a UI: a 1.7 s tail on a dashboard verdict is the difference between a feature and a
   nuisance.
2. **The gateway is not a transparent pass-through for this model.** Jev's modality is
   `text->decisions`, not `text->text`, so on OpenRouter it does **not** appear in the model
   catalogue and `/api/v1/chat/completions` rejects it — searching a gateway for "jev" can
   return nothing while the model is live and serving traffic. It has to be addressed by exact
   id through a non-standard route. That is a layer of accidental complexity between us and the
   thing we are trying to measure, and it is the kind of complexity that fails silently later.

A single credential was the only argument for a gateway, and it does not outweigh either point.

**What hosting costs us:**

| Cost | Detail |
|---|---|
| House data leaves the machine | A prompt contains Home Assistant entity values. `data/recordings/` is the inside of a real house over time. TypeSafe states that **Jev is not trained on customer requests or responses** (ZDR is an enterprise option), which softens the concern but does not remove it |
| A network dependency | The project otherwise runs entirely local and offline |
| A dependency on a hosted version | The model can change under us. **Mitigated by pinning the versioned id** — see [Configuration](#configuration) |
| A second licence | The connectome is already CC BY-NC; Jev's terms are separate |

**Rule:** use it for **judgment**, never for **house state we could compute ourselves**. Do not
send the raw 289-entity window when four summary numbers will do. See
[API call discipline](#api-call-discipline).

---

## The latency rule

| Loop | Budget | Jev fits? |
|---|---|---|
| Render frame | 50 ms (20 fps) | **No** — 9× too slow |
| Slow attention tick | ~15 s | Yes, comfortably |
| Control loop decision | ~2.3 s | Yes, but see below |
| Offline labeling | unbounded | Yes, and latency is irrelevant |

**Jev never blocks a frame, a WebSocket message, or the control loop.** The wiring rules:

| Rule | Why |
|---|---|
| On-demand button, or a slow timer | Honest about the latency instead of hiding it behind a spinner |
| Cache the verdict against the frame `seq` | A verdict is about a window. Judging the same window twice is wasted money |
| Display the verdict's **age** | `paintStaleness()` in [`web/app.js`](../web/app.js) already does exactly this for the brain tag — reuse the pattern |
| Show confidence and `abstain` | Otherwise a confident wrong answer looks like a fact |
| **Batch every question into one call** | Five questions cost the same as one |

Backend shape: a single `POST /api/jev` that assembles the state, asks **all** applicable
questions in one request, and returns typed answers. `httpx` is already a dependency, so this
adds nothing to `pyproject.toml`. The server holds the key; the browser never sees it.

---

## API call discipline

Three rules. All three are cheap to design in and expensive to retrofit, and they constrain the
placements below rather than following from them.

### 1. Trim the state

State is the only thing billed — **$0.042 per Mtok of input, output free** — and the main thing
that moves latency. Trimming is not an optimization on top of the cost model; it *is* the cost
model.

For scale: the first window of `data/recordings/session-20260924-135739` carries **289 Home
Assistant entities**. A naive `entity: value` dump of that is thousands of tokens, and most of it
is noise — `update.*`, `button.*`, `conversation.*`, backup-manager state. None of it bears on
any judgment in this document.

| Rule | Why |
|---|---|
| Send an **object with named fields**, not a flat entity dump | The docs recommend an object for most requests: names keep the relationships between parts of the state clear |
| Send only what the question needs | A `choice` about antennal-lobe saturation does not need the backup manager's state |
| **Send only what changed** since the previous window | "What is new" is the honest representation, and the reservoir already holds the history |
| Round numbers | `21.4` is one token; a full float is five |
| Send the reservoir *summary*, not the 512 raw rates | `region_usage()` already collapses 138,639 neurons to 12 rows |

A target state for placement A — a few hundred tokens instead of a few thousand, with every field
earning its place as an input to a documented failure mode:

```json
{
  "window":   {"sim_ms": 300, "active_neurons": 11204, "spikes": 38412, "mean_rate_hz": 0.92},
  "decision": {"sensor_c": 21.4, "chosen_k": 4120, "ideal_k": 4090, "delta_k": -30},
  "sensing":  {"entity": "sensor.hallway_temperature", "age_s": 1.2, "stale": false},
  "regions":  [{"class": "ALPN", "spikes": 812, "rate_hz": 41.2}],
  "settings": {"deadband_k": 25, "interval_s": 15, "smoothing_s": 5}
}
```

The vendor's own framing is the right test: state is *"the material you would present to a panel
of experts before asking them to make a judgment."* A dump of every entity in the house is not
that — it is a database export.

### 2. Batch the questions

Every question in one request shares one state, is evaluated **independently and in parallel**,
and comes back under the id you chose. Five questions cost and take the same as one (70 ms vs
74 ms server time, measured). Three separate calls pay for the state three times.

- **Ask a whole placement's questions in one request** — placement A should not make one call per
  field it wants to know.
- **When A, B and C fire in the same moment they should share one request.** They read the same
  window, so they can. That is the difference between one call per tick and three.

**But batching is not the same as asking one broad question.** The docs are explicit that System
One models want *one snap judgment per question* — the judgment a knowledgeable person makes in a
second. "Analyze this and decide what is wrong" is the wrong shape; it is a signal to split the
task and compose the answers in code:

- Ask `which_failure_mode` (choice) and `is_the_population_responding` (noul) **in the same
  request**, then combine them with ordinary code — the docs call this composite scoring, and the
  practical benefit is that priorities live in your weights rather than in a prompt.
- **The model never sees your question ids.** They are for your code. Write the complete question
  in `instructions`, even when the id looks self-explanatory.

Limits are 64k tokens per request, of which 32k covers the `state` plus the single longest
question. We are nowhere near either — which is a reason to trim for **cost**, not to fit.

### 3. Route by confidence

This is what makes the feature safe rather than merely working. `confidence` comes back on every
`choice` and `score` and tells you **whether to act on the answer**, which is a different question
from what the answer is.

| Band | Behaviour here |
|---|---|
| **High** | Act — paint the verdict, or write the label |
| **Medium** | Propose and ask — the confirm / correct / skip UI |
| **Low** | Do not act — mark the window `needs_human` and leave it. This is the `abstain` state |

**The threshold is not one number; it scales with what the judgment is allowed to affect.** This
project has a natural risk ladder and the thresholds should follow it:

| If the judgment can… | Risk | Threshold |
|---|---|---|
| Paint a verdict on screen | lowest — a wrong verdict costs a glance | low floor |
| Write a label into `labels.jsonl` | **higher — a wrong label silently corrupts a training set** | high |
| Gate a Home Assistant action | highest | highest |

The middle row is the one to get right. A wrong on-screen verdict is a curiosity. A wrong label
corrupts the only training data the project has, silently, and this repo has a documented history
of the silent failures being the expensive ones.

Two measured caveats:

- **A single answer's confidence is noisy even when the state is fixed.** Six byte-identical
  requests moved a `choice` confidence by **0.170** (0.49–0.66). The separation *between*
  determined and under-determined states is wide (0.94–0.99 vs 0.34), but the number attached to
  any one call is not precise to two decimals. So a threshold must sit **with margin**, never on a
  measured value — and a decision that turns on a 0.02 difference is not one this data supports.
- **`noul` returns no confidence.** Any judgment that will be gated must therefore be a `choice`
  or a `score`. A `noul` is fine as a *composed fact* — a piece of ordinary logic — but the moment
  its answer needs to gate a write on its own certainty, it has to become a two-option `choice`.

---

## Where Jev goes

Three placements, in the order they should be built. Each one names the panel it lives in, the
question kind, and the vocabulary — because the vocabulary is the actual work.

### A. The decision inspector — *build this first*

**Panel:** the existing **Colour chosen** scatter. Make each point clickable; the verdict appears
beside it.

**This is already in the repo as an unbuilt idea** — [`live-view.md`](live-view.md) lists *"A
'why' trace for one decision... Turns the dashboard from a readout into a debugger."* Jev is the
thing that can populate it, because the verdict is a classification rather than a sentence.

**Question:** `choice`. **Input:** that decision's temperature, chosen Kelvin, ideal Kelvin,
active neurons, spikes, window ms, top-5 regions, sensor staleness, and the deadband/interval
settings — all of which are already available from `/api/loop`, `/api/regions` and `/api/settings`.

**Vocabulary — every entry is a failure mode this project already documents:**

| Label | Meaning | Source |
|---|---|---|
| `healthy` | Tracks the ideal mapping within tolerance | — |
| `saturated_sensory` | Antennal lobe at its documented ceiling | [`engine.md`](engine.md) — saturates at receptor rate ≥ 25 Hz |
| `regime_mismatch` | Window began from rest, not the trained regime | [`AGENTS.md`](../AGENTS.md) #2 |
| `too_few_spikes` | The readout population barely responded | [`architecture.md`](architecture.md) |
| `sensor_stale` | The reading is not updating | `paintStaleness()` |
| `throttled` | The deadband is gating the service call | [`live-view.md`](live-view.md) |
| `unknown` | None of the above | required |

**Why this one first:**

- It needs **no new UI concept** — the scatter chart exists, and the labels come from docs we
  already wrote.
- It is the single most useful thing for someone who does not know fly brains: it answers *"why
  did it do that?"* instead of showing 11,000 active neurons and expecting interpretation.
- **It can be built and tested entirely offline** against `data/recordings/`, before any live
  endpoint exists.
- It is an auditor role, which is the job Jev was designed for.

**The panel should read in three layers, in this order:**

| Layer | Says | Where it comes from |
|---|---|---|
| **The house said** | the sensor readings that drove this window | `windows.jsonl` / `last_action` |
| **The brain did** | active neurons, top regions, spike counts | `/api/regions`, `/api/frame` |
| **We mapped it to** | the decoded Kelvin, the ideal, and Jev's verdict | the readout and this call |

That ordering is deliberate. It preserves the magic — the middle layer is genuinely the fly — while
making it unmistakable that the third layer is *ours*. A dashboard that shows only the verdict
invites the reader to think the brain reached it.

### B. A label button that proposes instead of asking

**Panel:** a small addition beside the recorder controls in **Light connection**.

Today labeling is `python -m flybrain.recorder --label busy`: the operator must *recall* to
label and then *type* the word. Invert it. Show Jev's proposal with its confidence, and offer
confirm / correct / skip:

```
Jev thinks:   busy            (0.81)
              [ confirm ]  [ correct ▾ ]  [ skip ]
```

**Question:** `choice` over the label vocabulary (open-ended and user-defined; seed with
`busy`, `quiet`, `settling`, `passing`, `cooking`), plus the confidence. Writes to
`labels.jsonl` with **`source="jev"`** — a field that already exists on `Label`, so machine
labels are first-class and separately auditable from the start.

**Why it earns its place:**

- It produces the training data, which [`roadmap.md`](roadmap.md) identifies as the real
  bottleneck: *"Budget for the data, not the compute."*
- **It is measurable.** Because `source` distinguishes `manual` from `jev`, the dashboard can
  report *"Jev agreed with you 84% of the time across 60 labels."* That is the audit trail,
  and it costs nothing extra to collect.
- **Reacting beats recalling.** Confirming a proposal is far lower activation energy than
  generating a label from nothing. A blank text field goes unlabelled; a suggestion gets a
  yes/no. The `abstain` case is an invitation, not a chore.
- **The disagreement is the interesting data.** Windows where Jev abstains or is corrected are
  exactly the windows worth a human's attention.

### C. The attention director

**Panels:** **The brain itself** (highlight in the point cloud) and **Which regions are busy**
(pin the chosen region to the top).

**Question:** `choice`, on a slow ~15–30 s timer. **Input:** the output of `region_usage()`,
plus active-neuron count and mean rate. **Output:** which cell class to emphasise, and a reason
from a fixed set.

This is the highest visual payoff — the screen visibly changes, immediately, in a way that can be
connected to a decision. It also addresses a real problem: 138,639 points is more than any human
can scan, and the current emphasis rule is a static threshold in `MemoryBrain.quantize()`
(`gain`, `saturation`, `gamma`).

**Ranked third only because it is the most likely to feel like a toy until the vocabulary is
good.** Build A, learn which menus work, then reuse them here.

---

## Where Jev must not go

This section matters more than the three above. Each of these looks reasonable and is wrong.

**1. Do not replace `plainBrainWord()` or `paintStaleness()`.** The **Brain activity** panel
already produces *"the brain is busy"* from a three-line threshold over a firing rate. A 460 ms
network call to reproduce `if rateHz < 0.05` is worse on every axis: cost, latency, determinism,
debuggability, and offline operation. [`roadmap.md`](roadmap.md) states the principle directly —
*"It will not beat a purpose-built model on a single narrow task. Every time."*

**2. Do not use it for regime detection.** This is the tempting one, because the failure is
*silent*. But [`AGENTS.md`](../AGENTS.md) #2 already contains the answer: **"a linear probe
separates them perfectly."** `tools/loop_regime_probe.py` does exactly this, locally, for free.
Jev would be a slower, lossier, paid version of a solved problem.

**3. Do not let it near the integrator's active set.** That is a numerical-correctness question
with a Brian2 validation behind it ([`engine.md`](engine.md)). A heuristic or model has no
business there, and the one time a validation "passed while proving nothing" is recorded in
[`research/`](research/README.md).

**4. Do not send the raw 289-entity window.** Cost and latency scale with state length, because
input tokens are the only thing billed. Send what changed and what the reservoir did.

The pattern: **Jev earns its place where the answer needs judgment over text, comes from a
closed set, and nothing cheaper already solves it.** Everywhere else it is a downgrade.

---

## Configuration

Jev is optional and off by default. With no key present, the dashboard runs exactly as it does
today and any Jev surface reports itself unavailable — the project must keep working against the
mock home with no configuration at all.

Credentials live in `.env`, which is already gitignored, and are loaded by
[`flybrain/env.py`](../flybrain/env.py) at startup. Real environment variables still win, so a
systemd unit or container env block works unchanged. See [`.env.example`](../.env.example).

| Variable | Meaning |
|---|---|
| `TYPESAFE_API_KEY` | The credential, from the [TypeSafe console](https://console.typesafe.ai/keys). Absent ⇒ the whole feature is off |
| `JEV_MODEL` | **`jev-1.13.0`** — the versioned id, deliberately, *not* the `jev-latest` alias. See below |
| `JEV_API_KEY` | The credential, from the provider console. **Absent ⇒ the whole feature is off.** The older `TYPESAFE_API_KEY` spelling is still read, so an existing `.env` is not silently ignored |
| `JEV_BASE_URL` | `https://jevtypesafeai.com/api/v1/decide`; the bare host also works |
| `JEV_TIMEOUT_S` | Request timeout; default `30` |
| `JEV_NETWORK_FLOOR_MS` | The measured floor, recorded beside every latency figure. Unset ⇒ reported as "not measured", never as `0` |
| `JEV_STRICT_MODEL` | Default `1`. Refuse an answer from a version other than the pinned one |
| `JEV_PAINT_ACT` / `_CONFIRM` | Thresholds for a judgment that only paints a verdict on screen |
| `JEV_LABEL_ACT` / `_CONFIRM` | The ones that matter: a wrong label silently corrupts the training set. Default act `0.95` — above the under-determined case (0.34) so it can never write, and below the top of the determined range (0.99) so a maximally determined answer can |
| `JEV_HA_ACT` / `_CONFIRM` | Thresholds for a judgment that gates a Home Assistant action. Default act `0.98` |

An unusable threshold — non-numeric, out of range, or inverted so that `confirm_at` exceeds
`act_at` — falls back to its default and logs. A configuration typo must never *loosen* a
threshold: the failure that matters is `JEV_LABEL_ACT` being mistyped into "write the label".

### The API surface

Verified against [the official docs](https://docs.typesafe.ai/introduction/quickstart) rather
than inferred from a third party:

- **One endpoint:** `POST https://jevtypesafeai.com/api/v1/decide`, with
  `Authorization: Bearer $JEV_API_KEY` and `Content-Type: application/json`.
- **Body:** `{"state": ..., "model": "jev-1.13.0", "questions": {...}}`.
- **Official Python SDK:** `typesafe_sdk` — `TypeSafeClient`, `client.system_one(...)`, and typed
  `Choice` / `Score` / `Noul` objects. It **retries with backoff by default and honours
  `retry-after`**, which is most of the reason to prefer it over hand-rolled `httpx`.
- **Rate limits:** 250,000 tokens/sec and 1,200 requests/min; a breach returns `429`. The vendor
  notes these are adjusting dynamically, so do not treat them as fixed capacity.
- **Price:** $42/Btok = **$0.042 per Mtok of input**. Output tokens are free — the bill is
  entirely a function of state length, which is why
  [trimming the state](#1-trim-the-state) is the first discipline.
- **Data handling:** TypeSafe states Jev is **not trained on customer requests or responses**;
  zero data retention is an enterprise option.
- Jev accepts **text only** — a string, a JSON object, or an array of text values. English is the
  primary training language; other languages are accepted but less accurate, so watch
  `confidence` if the state is ever non-English.
- **There is no free endpoint to probe on this host.** `GET /api/v1/models` answers
  `{"error": "Unknown endpoint /api/v1/models."}`, so the availability check asks one *trivial
  question* instead. That costs a fraction of a cent rather than nothing, which is why the result
  is cached and `refresh` is explicit. The other host did have such an endpoint; this one does not,
  and the client says which situation it is in rather than pretending the check is free.

### The endpoint was wrong, not the key

Recorded because it cost real time and the symptom was actively misleading.

This project was built against `POST https://api.typesafe.ai/v1/systemone`, the host the vendor's
public quickstart names. Every request was refused:

```
HTTP 401  {"detail": {"error_type": "authentication_error",
                      "message": "Cannot authenticate with the server..."}}
```

That reads as a bad credential. The key was well-formed (`jv_live_…`, 51 characters, no
whitespace), so the reasonable conclusions were "revoked", "mistyped", or "from another
environment" — and all three were wrong. The key was fine. **The host was not the one that serves
this account**, and the working endpoint (`https://jevtypesafeai.com/api/v1/decide`) was found by
the operator, not by reading the docs harder.

Three lessons, all of which are already this repository's stated ones:

- **A 401 is a statement about the pair (key, endpoint), not about the key.** Nothing in the
  response distinguished "this key is bad" from "this host does not know this key".
- **Two hosts under one vendor can differ in more than the hostname.** These two disagree on the
  error envelope (`{"error": ...}` vs `{"detail": {...}}`), on whether a cheap probe endpoint
  exists, and on whether the cost is reported. Every one of those was assumed from the wrong host
  and had to be corrected.
- **The client now reports the distinction the operator needs.** `available()` separates
  `no_key`, `unauthorized`, and `unreachable`, and because a *missing* key and an *invalid* key
  both come back as **401** on this host, the message is read rather than the status code trusted:
  `"Missing API key…"` and `"Invalid or revoked API key."` send a person to different places.

### We use `httpx`, not the SDK — and why

This reverses an earlier recommendation in this document, so the reasoning is recorded rather
than the conclusion.

The SDK is real, maintained and better-built than a hand-rolled client: it retries with backoff,
honours both `retry-after` spellings, and its `typesafe_sdk/_schemas/models.py` **is** the vendor's
OpenAPI schema, which is what this document's wire shapes were checked against. Nothing here is a
criticism of it.

It was still not taken, for three reasons:

1. **It is a second HTTP stack.** `typesafe-sdk==0.7.2` depends on `httpx2` + `httpcore2` +
   `truststore` + `tenacity`, alongside the `httpx` this project already uses for Home Assistant.
   (`httpx2` is legitimate — it is the Pydantic team's next-generation client, not a typosquat;
   that was checked, because a dependency nobody recognises deserves checking.)
2. **The feature is off by default.** Four new packages for a layer that is disabled unless an
   API key is set is a poor trade at this size, where the alternative is ~30 lines of retry logic
   with tests.
3. **The vendor's SDK defaults to `jev-latest`.** Adopting it wholesale would adopt the alias by
   default, which is the one thing [below](#pin-the-version-not-the-alias) says must not happen.

So `flybrain/jev.py` is the only file that would change if this decision is reversed, and its
tests pin the wire shapes either way.

### Pin the version, not the alias

`jev-latest` is an alias. It resolves to `jev-1.13.0` today and **moves when a new release ships**.
The docs state the consequence directly: *"If you have tuned confidence thresholds against a
specific version, pin that version's ID instead of the alias and move to the new one on your own
schedule."*

Since [routing by confidence](#3-route-by-confidence) is the whole plan, **`JEV_MODEL` must be
`jev-1.13.0`.** The alias would let a vendor release silently invalidate every threshold we
calibrated — a change under us that announces itself as nothing. That is the exact failure shape
this repository has been burned by before.

Log the `model` field from every response: it reports the versioned id that actually answered, so
drift becomes visible in the logs rather than inferred from behaviour that has quietly got worse.

### Cost accounting

**Corrected:** the API *does* report the cost. Every response carries
`usage.cost_usd` and `usage.credits_remaining_usd`, which the SDK's published schema does not
describe and the first draft of this document therefore denied:

```json
"usage": {"input_tokens": 458, "output_tokens": 50,
          "cost_usd": 0.000193, "credits_remaining_usd": 4.997265}
```

The client prefers the vendor's figure and falls back to `input_tokens × $0.042 / 1e6` only when
the field is absent, recording which it used (`cost_reported`). Recomputing our own number when the
billed one is right there would drift from it silently the day the price or the rounding changes.

The credit balance is surfaced in the dashboard, because **a balance reaching zero is how a feature
stops working without anyone noticing.** Measured spend: a three-question call with a small state
cost $0.000233; a single-question call $0.00014–$0.00019. A dashboard that probes availability
every five minutes costs roughly **$0.03 a day** if left open, which is worth knowing before
leaving it open for a month.

---

## TODO

**J0, J1 and J1a are built** ([`flybrain/jev.py`](../flybrain/jev.py), tested in
`tests/test_jev.py` and `tests/test_jev_live.py`). Nothing has a UI yet, and the placements below
still do not exist. The one thing that is *blocked* rather than unbuilt is called out below.

**J0 — credential surface — done**
- [x] Document `JEV_API_KEY` / `JEV_*` in `.env.example` and `.env`
- [x] A `JevConfig` reader following the `os.environ.get(...)` pattern, with tests
- [x] Default `JEV_MODEL` to the **versioned id `jev-1.13.0`**, never the alias, and refuse to
      proceed if a response's `model` field differs from the pinned id
- [x] An `available()` check that reports *why* it is off, with **three distinct** failure
      reasons rather than two: `no_key` (403), `unauthorized` (401 — a key was sent and rejected),
      and `unreachable`. The third state was not in the original plan; the live endpoint produced
      it immediately, and it is the difference between "you forgot to configure this" and "your
      key is being refused", which send an operator to different places.

**J1 — the client, with no UI — done**
- [x] `httpx` rather than the official SDK — the reasoning is
      [above](#we-use-httpx-not-the-sdk--and-why), and it is a reversal recorded rather than
      quietly dropped
- [x] `flybrain/jev.py`: one call to `POST /v1/systemone`; typed `choice` / `score` / `noul`
      helpers; `legend` and per-kind `probabilities` parsing
- [x] **Measured the network floor on this machine** — and found that the *first* sample is
      ~3× the warm median, so the measurement is repeated and the method is recorded. Median
      **49.2 ms** over 5 warm samples to the configured host
- [x] Reconcile `score` against `probabilities` to ±0.02
- [x] A test that runs the whole client against fixtures with no network — and one live test that
      asserts the client's *diagnosis* is truthful, so it is useful even while the credential is
      rejected

**J1a — the three disciplines, as code rather than convention — done**
- [x] **Trim:** `build_state()` emits named fields, only changed values, rounded numbers, and the
      busiest few regions — with a budget test. The budget is a character proxy for now, because
      no successful response could be captured to read a real `usage.input_tokens` from.
- [x] **Batch:** one request per judgment, asserted by counting calls
- [x] **Route:** `route(answer, risk) -> act | confirm | needs_human` over three named risk tiers,
      tested at both band boundaries, and monotone: a confidence that paints a verdict may only
      propose a label, and may not even confirm a house action. Every tier is overridable from the
      environment (`JEV_LABEL_ACT` and friends) rather than fixed in source, because these numbers
      have to be calibrated on this project's own data and an unusable override falls back rather
      than loosening the gate.
- [x] A cost log line per call: computed cost, latency, the measured floor beside it, and the
      `model` id that answered

**No longer blocked.** The credential works: `available()` reports `ok` against
`https://jevtypesafeai.com/api/v1/decide`, and a real three-question call costs $0.000233. What
remains for J2 is the vocabulary and the UI, not access.

**J2 — placement A, offline first**
- [ ] Define the failure-mode vocabulary in code, including `unknown`
- [ ] Build the state from a *recorded* window via `build_state()`, and run it over
      `data/recordings/session-20260924-135739` (37 windows, no labels) as a pipeline test
- [ ] `POST /api/jev` on the server, backed by the client and keyed by frame `seq`
- [ ] Clickable points in the **Colour chosen** chart; verdict beside the point, with age and
      confidence shown

**J3 — placement B**
- [ ] Proposal panel beside the recorder controls; confirm / correct / skip
- [ ] **Every label write goes through `route()`** — medium confidence proposes, only high
      confidence writes without confirmation
- [ ] Write `source="jev"` labels via the existing `Recorder.label()`
- [ ] Agreement metric: Jev-versus-manual, surfaced in the UI

**J4 — placement C**
- [ ] Slow timer (~15–30 s) director over `region_usage()`
- [ ] Highlight the chosen cell class in the point cloud and pin it in **Which regions are busy**

**J5 — the honest accounting**
- [ ] A sensors-only baseline beside every reservoir result, so "the brain earned this" is
      measurable rather than assumed
- [ ] **Split by held-out day or session, never by random window.** Windows from one continuous
      session are autocorrelated, so a random split puts a window's own near-twin in the test set
      and reports an accuracy the readout cannot reproduce tomorrow. See
      [`roadmap.md`](roadmap.md#6-three-honest-constraints)
- [ ] Cost and latency logged per call, next to the floor measurement from J1
- [ ] Record in [`roadmap.md`](roadmap.md) whether the feature paid for itself

---

## Open questions

Recorded rather than smoothed over:

1. **Does Jev add anything a threshold does not?** Placement A is the test. If it returns
   `healthy` for everything, or if its verdicts never change anyone's behaviour, it has not
   earned its dependency.
2. **What are the right confidence thresholds, and how stable are they?** Answered, with a
   caveat that matters more than the answer. Measured here: a state that determines the answer
   scores 0.94–0.99 and one that does not drops to 0.34 — a wider separation than the borrowed
   figures implied. But a byte-identical request varies by **0.17** in `choice` confidence, so the
   thresholds must sit with margin and be re-measured rather than inherited. The defaults are
   deliberately conservative *because of* that spread, not despite it. See
   [What the confidences actually look like](#what-the-confidences-actually-look-like).
3. **What happens when the label vocabulary is wrong?** If every option is wrong, one still
   wins. The `unknown` option is the mitigation, not a solution.
4. **Does Jev's judgment drift?** Partly mitigated by pinning `jev-1.13.0` — a vendor release can
   no longer move under us. Not mitigated: a pinned model can still be re-served or re-weighted
   server-side, and we cannot verify weights we do not host. Logging the returned `model` id
   catches the first case and not the second.
5. ~~**Is the recorder's window↔feature pairing sound?**~~ **Resolved — it was not sound.** An
   external review of this checkout claimed it was not; the claim was verified against the code
   rather than taken on trust: the writer reopened in append
   mode and wrote *after* a partial row left by a crash, so from that point on every row sat at a
   constant offset from `windows.jsonl` while the row count still looked correct; and a short
   feature file made `labelled()` index past its end. Both are fixed — `Recorder` now repairs a
   partial row before opening for append, and `Recording` exposes only rows that are certainly
   paired — with tests that fail against the old code. This was a real blocker for placement B,
   not a hypothetical one.
