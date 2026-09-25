# Vision: feeding a camera into the fly

Design notes for driving the visual population from a camera. **Not built** — this is the record of
a design conversation, written down so it does not have to be re-derived. Everything here was
derived from the actual code and the actual hardware in this house; see
[`ha-inventory.md`](ha-inventory.md) and [`wiring.md`](wiring.md).

Today the live loop drives **one** channel (temperature). The visual population is wired
(`wiring.md` maps camera motion/person to `visual`) but nothing feeds it yet.

---

## What a fly actually sees

Worth being precise, because it sets the ceiling on what is worth building.

The fly's optic lobe — **10,855 neurons** here, its largest sensory population — is a motion
machine. It has:

- **Direction-selective motion detectors** (T4/T5 in the lobula plate), the circuit the fly is
  famous for.
- **Loom-sensitive neurons**, for an object expanding in the visual field: an escape trigger.
- **Small-target motion**, for tracking something tiny and moving.
- **Optic flow** for navigation.

It does **not** recognise objects. There is no "person" or "cat" representation to read out.

> **A fly knows what things *are* by smell, not sight.** Identity in the fly lives in the mushroom
> body — 5,177 Kenyon cells, an olfactory learning centre. Vision answers *where and how is it
> moving*; olfaction answers *what is it*.

That is the honest argument for the project's biggest hardware gap: `olfactory` is the largest
sensory pathway available and this house has **no** VOC/CO₂ sensor, while it has three cameras.
The camera is the accessible input; olfaction is the *appropriate* one for identity.

## The constraint that decides the whole design

`ConnectomeSim.set_drive()` applies a **constant, persistent current per neuron, held for the whole
window** (`sim.py`). There is no way to express "frame A, then frame B" *within* a window. The
drive is a static spatial pattern.

So a frame sequence cannot be fed as a time series inside one window. There are exactly two routes:

### Route A — spatial multiplexing (recommended)

Present K frames **simultaneously** on K disjoint sub-populations. The recurrence then does the
temporal work, because each frame's neurons feed the same downstream circuits.

```
8 frames × 32×32 pixels  = 8,192 neurons   (of 10,855 visual)
window                   = 300 ms
frame spacing            = 300 / 8 = ~37 ms of brain time
fly tau_mem              = 20 ms
```

37 ms against a 20 ms membrane constant is the happy part: consecutive frames are ~2 time constants
apart, so the recurrent state genuinely persists between them and the network sees **motion**
rather than unrelated stills. This route needs **no engine change** and **no window change**.

Resolution is cheap to trade for frames: 16×16 × 16 frames = 4,096 neurons, same idea. What matters
is `frames × pixels ≤ ~10,000` and frame spacing near `tau_mem`.

### Route B — shorten the window

Set `window_ms` to ~50 ms and feed one frame per window, letting the recurrent state carry motion
across windows.

Attractive because it *also* raises the decision rate: 50 ms of brain time is ~385 ms of wall clock
at 0.13×, so ~2.6 decisions/s instead of ~0.43. The costs: rate estimates per window get much
noisier (fewer spikes), and the trained readout is invalid — it was fitted on 300 ms windows.

Route A is the better first move: no retraining of the window regime, no engine work.

## Frame acquisition

| Route | Rate | Notes |
|---|---|---|
| `GET /api/camera_proxy/<entity>` | **~1 fps** | Simplest. Already authenticated with the HA token, and read-only. Too slow for 8 frames per 2.3 s window (~2 frames), but fine for prototyping. |
| **RTSP substream** | any | The Tapo C120 exposes a substream (~640×360) as `stream2`. Needs a **camera account** created in the Tapo app — separate from the TP-Link cloud login. Pulling RTSP does not change camera state. |

```bash
# 8 fps, 32x32 grayscale, raw bytes on stdout -> ~1 KB/frame
ffmpeg -i "rtsp://user:pass@HOST:554/stream2" \
       -vf fps=8,scale=32:32,format=gray -f rawvideo -
```

Decoding 640×360 H.264 at 8 fps is negligible CPU. The 32×32 grayscale conversion is microseconds.
Throughput is ~8 KB/s.

## Hardware required

**None beyond what is already in the house.** The camera is networked, and the video path is
almost free:

- No GPU for video — the downscale is CPU work measured in microseconds.
- No capture card; RTSP over the existing network.
- Storage is irrelevant: frames are consumed, not kept.

The bottleneck is **not** the video path. It is the simulator.

## The realtime bound (and why a bigger GPU does not fix it)

The simulator is **memory-bandwidth bound**, not compute bound:

```
weights read per step : ~184 MB
step time             : ~0.78 ms
implied bandwidth     : ~236 GB/s
RTX 3070 peak         : ~448 GB/s
```

Already at roughly half of peak bandwidth, reading the whole synapse matrix every step. A 4090
(~1 TB/s) buys about **2.2×** — not the **~7.7×** needed for 1:1 realtime. So:

- **Hardware does not solve realtime here.** The fix is algorithmic: the **active-set integrator**
  (only integrate neurons that can reach threshold), documented in [`engine.md`](engine.md).
- **For the context layer, none of this matters.** At ~2.3 s per decision the brain is perfectly
  adequate for "the house is winding down", and the video path costs nothing extra. Realtime only
  matters if you want *reflexes* — and reflexes belong to Home Assistant automations anyway
  ([`roadmap.md`](roadmap.md#2-division-of-labour-ha-does-reflexes-the-fly-does-context)).

## What this would and would not learn

A linear readout on spike rates decodes what is **linearly present** in the reservoir's response.

**Plausible:** motion energy, direction of travel, change, "something moved", coarse occupancy of
the field, brightness change.

**Not plausible:** objects, people, faces, "is that a cat". That needs far more than a linear
readout, and it is not what the connectome is for.

The honest framing is that this gives the fly **a retina**, not an understanding. Its value is that
the fly then does the seeing — rather than the loop reading the camera vendor's motion verdict and
forwarding it.

## Status of this house's camera

The Tapo C120's **streams are up** (`camera.door_camera_hd_stream_direct` and `_sd_stream` are
`idle`), but every motion/person **event** sensor is `unavailable`
([`ha-inventory.md`](ha-inventory.md#5-the-cameras-streams-online-events-still-dark)). Away mode is
meant to enable listening for configured events; as of the last check the entities had not come
alive.

Two consequences:

- **Nothing here is verified against real frames yet.** Every number above is design arithmetic.
- **The event route is the cheap one.** If the camera's own motion/person sensors start reporting,
  they are just another pathway (`wiring.md`) and the fly gets a real visual-motion drive for
  free. Only reach for RTSP if you want the fly doing the seeing itself.

Also note: this camera exposes **no acoustic event sensors**. Its bark/meow/glass-break entities are
sensitivity *settings*, not events, so there is nothing to wire to the auditory pathway.

## Summary of the recommendation

1. **Now:** use the sensors that reliably work — the dedicated motion sensor, the thermometer, the
   illuminance reading. The camera is a bonus, not a dependency.
2. **When the camera reports:** wire its motion/person sensors through `wiring.py` (already
   mapped). Cheap, immediate, no video decoding.
3. **If you want real vision:** Route A — 8 × 32×32 grayscale frames over the RTSP substream,
   spatially multiplexed onto 8,192 visual neurons at ~37 ms spacing.
4. **If you want realtime:** that is the active-set integrator, not a new GPU.
