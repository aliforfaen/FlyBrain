# The reference Home Assistant instance

What a real house actually looks like when the loop is pointed at it, discovered by reading a
live instance. Everything below was obtained with **`GET` requests only** — no service was
called, no entity was changed.

| | |
|---|---|
| URL | `http://homeassistant.local:8123` |
| Version | Home Assistant `2026.7.1` |
| Entities | 289 (91 `sensor`, 45 `update`, 23 `switch`, 22 `binary_sensor`, 11 `light`, …) |
| Read on | 2026-09-23 |

> **Entity ids in this document are genericised.** The names below are placeholders for a
> working installation, not any particular house. Real ids, the real URL and the real token
> belong in `.env`, which is gitignored — never in source or documentation. The *findings* are
> the point of this file and are reproduced exactly: they are what kills several plausible ideas.

This document is the ground truth for the roadmap. Where the roadmap or `live-view.md` assumes
something that is not here, **this file wins**, and those documents have been corrected.

---

## Findings that change the plan

### 1. There is no humidity sensor anywhere

No entity id contains `humid`, and no `sensor.*` reports `%` except batteries, storage and phone
state. The only two `device_class: temperature` entities are `sensor.hallway_temperature`
(real) and `sensor.rack_gputemperature` (`unavailable`, a GPU).

This kills two documented ideas outright:

- `docs/live-view.md` offers "**A second sensor.** Humidity already drives real hygrosensory
  neurons and is wired into the encoder" as a cheap next step. It cannot be built — the data does
  not exist. (The *encoder* is still fine; there is simply nothing to feed it.)
- `docs/roadmap.md` **B4** (`hygrosensory`) is unavailable for the same reason.

There is also **no second temperature sensor**, so multi-room temperature work is out until a
sensor is added.

### 2. The temperature entity the loop defaults to does not exist

`flybrain/loop.py` defaults `temperature_entity` to `sensor.living_room_temperature`. That is the
**mock home's** entity name; it is not in this instance. The real one is:

```
sensor.hallway_temperature     # 23.6 °C
```

Real-HA mode must set this explicitly, or the loop will read nothing and fail. This is a genuine
trap: the mock default makes real mode look configured when it is not.

### 3. Most lights cannot accept the command the loop sends

The loop sends `light.turn_on` with `color_temp_kelvin`. Only *some* of the lights support colour
temperature at all. `light.ceiling_lamp` — the obvious living-room target — does **not**:

| Light | Colour modes | Kelvin range |
|---|---|---|
| `light.sofa_lamp_2` | `color_temp`, `xy` | **2000 – 6535 K** |
| `light.hall_lamp` | `color_temp`, `xy` | **2000 – 6535 K** |
| `light.corridor_lamp` | `color_temp` | 2202 – 6535 K |
| `light.bedroom_lamp` | `color_temp` | 2202 – 6535 K |
| `light.bathroom_lamp` | `color_temp` | 2202 – 6535 K (currently `unavailable`) |
| `light.all_lights` (group) | `color_temp`, `xy` | 2202 – 6535 K |
| `light.home_group` (group) | `color_temp`, `xy` | 2202 – 6535 K |
| `light.ceiling_lamp` | **`onoff`** | — |
| `light.reading_lamp` | **`onoff`** | — |
| `light.night_lamp` | **`onoff`** | — |
| `light.camera_floodlight_timed` | **`onoff`** | — (`unavailable`) |

**The project's light is `light.hall_lamp`** (the owner's choice). It is the best candidate in the
house for two reasons:

1. **Widest range** — 2000–6535 K. The readout is currently constrained to 2700–6500 K, so the
   software's warm floor is *more conservative than the lamp*. Reaching 2000 K is **not** free,
   though: `LiveLoop` clamps the emitted colour to the trained band range, so the trained range
   itself has to be widened (`COLOUR_BANDS`) and the readout re-fitted — about a minute of GPU
   time, but not a config change.
2. **It advertises `xy` as well as `color_temp`** — so it is not limited to white. This makes the
   oldest entry in `docs/live-view.md`'s "ideas not built" list (*"Colour temperature only spans
   white — amber to daylight. A hue light … would make the brain's behaviour far more legible"*)
   buildable on real hardware rather than hypothetical.

The current loop sends `color_temp_kelvin` only. Driving `xy` would use the same readout with a
different action payload, and would make the brain's output legible as *colour* rather than as a
subtle shift in white.

### 4. One motion sensor, and it is also the temperature sensor

`binary_sensor.hallway_motion` is the only live motion sensor. It is the *same physical
device* as the temperature and light readings:

| Entity | State now |
|---|---|
| `binary_sensor.hallway_motion` | `off` |
| `sensor.hallway_illuminance` | `6 lx` |
| `sensor.hallway_temperature` | `23.6 °C` |
| `sensor.hallway_battery` | `100 %` |
| `switch.hallway_motion_sensor_enabled` | `on` |
| `switch.hallway_light_sensor_enabled` | `on` |

One 4-in-1 sensor (motion + temperature + illuminance + battery) is the entire environmental
sensing in this house. That is worth stating plainly: **`house_activity` will be built on one
motion sensor, one light-level reading and one thermometer.** Motion-only activity detection is
exactly the case where a threshold looks adequate and a reservoir's memory is the only thing that
adds anything (is this the start of sustained presence, or someone walking past?).

Note that `sensor.hallway_illuminance` (6 lx, dark) is a genuine ambient-light input, and
`light_level` maps to the fly's `visual` population — the connectome's largest sensory investment.

### 5. The cameras: streams online, events still dark

**Updated.** A third-party camera integration is now installed
(`switch.camera_integration_pre_release`, `update.camera_integration_update`) and the streams
came up — `camera.door_camera_hd_stream_direct` and `camera.door_camera_sd_stream` went from
`unavailable` to **`idle`**. The camera is a **TP-Link Tapo C120**, firmware `1.9.3`, with
`motion_detection: "on"` in its attributes.

**But the event sensors are still dark.** Every detection binary sensor remains `unavailable`:

```
binary_sensor.door_camera_motion_alarm             unavailable   device_class=motion
binary_sensor.door_camera_person_detection         unavailable   device_class=motion
binary_sensor.door_camera_cell_motion_detection    unavailable   device_class=motion
binary_sensor.door_camera_motion_alarm_2           unavailable
binary_sensor.door_camera_person_detection_2       unavailable
binary_sensor.door_camera_cell_motion_detection_2  unavailable
```

So the split is: **video is available, detection events are not.** The `_2` variants suggest two
channels (HD and SD) of the same device.

**A correction worth recording**, because it is easy to get wrong from the entity names alone:
there is **no** `binary_sensor.door_camera_bark_detection`, `_meow_detection` or
`_glass_break_detection`. Those names exist only as `select.*` **sensitivity settings**
(`high`/`normal`/`low`/`off`). The camera can be *configured* to detect a bark; it does not
*publish* an entity when it hears one. Any wiring that assumed acoustic event sensors from this
camera was wrong. See [`wiring.md`](wiring.md#what-this-house-does-not-have).

### The camera's actuator surface

The integration exposes far more than video, and it is the richest actuation source in the house
after the lights. All of it is currently **out of scope** while the read-only constraint stands,
but it is what a future "the fly reacts" would drive:

| Entity | What it does |
|---|---|
| `siren.door_camera_siren` | full siren, plus `number.*_siren_volume`, `_siren_duration`, `select.*_siren_type` |
| `light.camera_floodlight_timed` | floodlight, with `number.*_spotlight_intensity` and `select.*_spotlight_on_off_for` |
| `number.door_camera_speaker_volume`, `_microphone_volume` | two-way audio |
| `switch.door_camera_privacy`, `_privacy_zones` | privacy mode |
| `switch.door_camera_notifications`, `_rich_notifications` | push notifications |
| `select.door_camera_night_vision` | Infrared / Smart / Full Colour |
| `switch.door_camera_trigger_alarm_on_*` | per-detector alarm triggers (motion, people, glass **on**; sound, bark, meow, line-crossing **off**) |
| `button.door_camera_manual_alarm_start` / `_stop` | manual alarm |
| `switch.door_camera_media_sync`, `_record_audio`, `_indicator_led`, `_lens_distortion_correction` | device settings |

Note `switch.door_camera_trigger_alarm_on_glass_detection` is **on** while
`_trigger_alarm_on_sound_detection` is **off** — so the camera will raise an alarm for breaking
glass but not for sound generally.

### 6. There are no air-quality sensors

No VOC, CO₂, PM2.5 or eCO₂ entity exists, so `roadmap.md` **B3** (olfaction / cooking detection)
is unavailable. This was the most biologically apt idea in the roadmap and it needs new hardware.
Confirmed again after the camera integration came online.

---

## What *is* available

### Environmental
| Entity | Notes |
|---|---|
| `sensor.hallway_temperature` | the thermometer — replaces the mock default |
| `sensor.hallway_illuminance` | ambient light, `lx` |
| `binary_sensor.hallway_motion` | the only motion sensor |
| `sun.sun` + `sensor.sun_next_dawn/dusk/rising/setting/noon/midnight` | free time-of-day context |
| `weather.forecast_home` | `cloudy` |

### Presence and "is a human here"
| Entity | Notes |
|---|---|
| `person.owner`, `person.guest` | `not_home` / `unknown` |
| `device_tracker.phone` | `not_home` |
| `binary_sensor.phone_presence` | presence |
| `input_boolean.owner_home` | helper, currently `on` |
| `binary_sensor.watch_on_body_sensor` | watch worn |
| `sensor.watch_heart_rate`, `sensor.watch_daily_steps` | strong "a body is awake and moving" |

### Phone and media activity
| Entity | Notes |
|---|---|
| `sensor.phone_detected_activity` | `still` / walking / … |
| `sensor.phone_media_session` | `Playing` |
| `binary_sensor.phone_music_active` | boolean |
| `sensor.phone_active_notification_count`, `sensor.phone_sleep_confidence` | |
| `media_player.living_room_speaker` | the smart speaker |
| `media_player.living_room_tv`, `media_player.tv_box`, `media_player.media_centre` | TVs and players |
| `media_player.kitchen_display`, `media_player.spotify` | |

### Physical input — free label buttons
Eight `event` entities already exist as **Hue dimmer buttons**, which is exactly the "press a
button to label this moment" affordance the roadmap's Phase 2 wants, using hardware that is
already on the wall:

```
event.dimmer_a_button_1..4          event_types: initial_press, repeat,
event.dimmer_b_button_1..4                       short_release, long_press, long_release
```

Reading these is a WebSocket subscription to `state_changed`, which is still strictly read-only —
so a label can be recorded without writing anything to Home Assistant.

---

## Read-only policy

The instance was inspected with `GET /api/config` and `GET /api/states` only. No `POST` was issued;
no service was called; no entity changed. `HA_DRY_RUN=1` remains the default for real mode.

Publishing anything back — including the `house_activity` sensor the roadmap proposes — is a
**write** (`POST /api/states/<entity_id>`). That is out of scope while the read-only constraint
stands, so the readout is computed and shown locally and the publish path stays disabled until
it is explicitly turned on.
