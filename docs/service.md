# Running FlyBrain as a service

The README's quick start runs the brain in a foreground terminal:

```bash
.venv/bin/python -m flybrain.server     # → http://127.0.0.1:8765/
```

Close the terminal and it is gone. Leave the terminal and you cannot do anything else with it.
Stop it and you are matching a process id out of `ps`, because `pkill flybrain` does not match
`python -m flybrain.server` reliably and the wrong match is somebody else's Python.

That is fine for a demo and wrong for a thing that controls a room. This document is the other
way: a **systemd user service** that starts the brain in the background, restarts it when it
falls over, keeps it running while you are logged out, and stops it by name.

Everything here is user-scope. Nothing needs `root` except one optional `loginctl` line, and
nothing installs into `/etc`.

## Install

```bash
uv sync                                # once, so .venv/bin/python exists
cp .env.example .env                   # once, if you want the real house
deploy/install-service.sh --now
```

The installer is idempotent and never touches `.env`. What it does:

| Step | Why |
|---|---|
| Renders `deploy/flybrain.service` to `~/.config/systemd/user/flybrain.service` | A unit file cannot know where you cloned the repository, so the shipped one assumes `~/FlyBrain` and the installer rewrites that path to the checkout it is running from |
| `systemctl --user daemon-reload` | Makes systemd re-read the unit |
| `systemctl --user enable flybrain.service` | Starts it at login, via `default.target` |
| `systemctl --user restart flybrain.service` (only with `--now`) | Starts it now, and restarts rather than reporting "already active" if it was already running |
| Checks linger, `.env`, `FLYBRAIN_ALWAYS_ON`, `HA_MODE`/`HA_DRY_RUN` | Warns about the four ways a correctly-running service still does nothing |

Run it without `--now` to install and enable but leave the GPU alone:

```bash
deploy/install-service.sh
systemctl --user start flybrain.service
```

<details>
<summary>Installing by hand instead</summary>

```bash
mkdir -p ~/.config/systemd/user
cp deploy/flybrain.service ~/.config/systemd/user/
# edit the two %h/FlyBrain paths inside if you did not clone to ~/FlyBrain
systemctl --user daemon-reload
systemctl --user enable --now flybrain.service
```

</details>

## Day to day

| Want | Command |
|---|---|
| Start / stop | `systemctl --user start flybrain` · `systemctl --user stop flybrain` |
| Restart | `systemctl --user restart flybrain` |
| Is it up? | `systemctl --user status flybrain` |
| Follow the logs | `journalctl --user -u flybrain -f` |
| Last 200 lines | `journalctl --user -u flybrain -n 200` |
| Since a point in time | `journalctl --user -u flybrain --since "1 hour ago"` |
| Boot behaviour | `systemctl --user enable flybrain` · `disable` |
| Stop it coming back | `systemctl --user disable --now flybrain` |
| Uninstall | `deploy/install-service.sh --uninstall` |

`stop` is a **SIGTERM to the whole cgroup**, which `uvicorn` handles, so the process tree goes
away cleanly — no orphaned worker, no `pkill`. That is the point of the service.

### Logs

There is no log file to rotate. The process logs to stderr, systemd ships it to the journal, and
`journald` enforces its own size limit. `StandardOutput=journal` plus `PYTHONUNBUFFERED=1` in the
unit is why a log line shows up as it happens rather than when a buffer fills.

The startup lines worth seeing are the connectome load and the decision cadence:

```
flybrain.server: loading connectome ...
flybrain.sim: connectome loaded: 138639 neurons, 15091983 synapses, device=cuda
flybrain.server: ready: 138639 neurons, 15091983 synapses
```

If you see `ready` and then nothing for a while, that is **correct** — see *pacing* below.

## Configuration

`.env` is loaded by the process itself (`flybrain/env.py`), so there is deliberately **no
`EnvironmentFile=` in the unit**: one loader, one set of quoting rules, one place to look. The
unit comment says so, because an absent `EnvironmentFile=` looks like an oversight until you
know it is a decision.

Precedence, highest first:

1. a systemd `Environment=` — in the unit or, better, in a drop-in
2. a real exported environment variable
3. `.env`
4. the code's defaults

Use a drop-in for machine-specific overrides so a `git pull` and a re-install cannot clobber
them:

```bash
systemctl --user edit flybrain      # opens an override.conf in $EDITOR
```

```ini
[Service]
Environment=FLYBRAIN_INTERVAL_S=60
```

Then `systemctl --user daemon-reload && systemctl --user restart flybrain`. A drop-in beats
`.env` — which is exactly the behaviour `flybrain/env.py` documents, and the reason a systemd
unit keeps working unchanged.

**Every flag is documented in one place:** [`live-view.md`](live-view.md#configuration). The
`.env.example` next to it is the annotated, copy-pasteable version.

### The one flag an unattended service needs

`FLYBRAIN_ALWAYS_ON` defaults to **0**: with the defaults the control loop only advances while a
dashboard is connected, because the simulator is expensive and a demo is better off spending the
GPU while somebody is watching. A service has nobody watching, so:

```bash
FLYBRAIN_ALWAYS_ON=1
```

Without it the service is healthy, serves the dashboard, and **never makes a decision** — a
state that looks exactly like working. The installer warns if it is unset.

### Pacing, and why silence in the log is fine

`FLYBRAIN_INTERVAL_S` defaults to **15 s**, so a decision every 15 seconds is the designed
behaviour, not a stall: the GPU is genuinely idle in between. Set it to `0` for flat out. The
measured cost model is `mean ≈ 19 W + duty × 146 W` — about **40 W** at the default and **165 W**
flat out. The dashboard's *Light connection* panel shows the current number, and
[`engine.md`](engine.md#idle-cost-and-adaptive-pacing) and
[`live-view.md`](live-view.md#what-it-costs-to-leave-running) own the measurements.

**Set the interval before you leave it running for a week.** It is the only knob that materially
moves energy; `fps` and `window_ms` do not.

## Linger: running while you are logged out

A user service lives and dies with the user's systemd manager. By default that manager starts at
login and is torn down at logout, so without one more step your "always-on" brain stops when you
close your session:

```bash
sudo loginctl enable-linger "$USER"      # once
loginctl show-user "$USER" -p Linger     # → Linger=yes
```

This is what makes an unattended brain survive a logout. A desktop that is always logged in does
not strictly need it, but enabling it is free and removes a whole class of "why did it stop at
6 pm" questions.

To undo it later: `sudo loginctl disable-linger "$USER"`. That stops the service too, so
`systemctl --user disable --now flybrain` first if you want a clean removal.

## GPU access

A user service gets the same devices as a shell in the same session. The requirement is group
membership, not privilege: membership in **`video`** (and `render` on distributions that use it)
to open `/dev/nvidia*` and `/dev/dri/*`.

```bash
id -nG | tr ' ' '\n' | grep -E '^(video|render)$'
nvidia-smi
```

If the group is missing, `systemctl --user start` succeeds and the connectome load fails in the
journal with a CUDA error — the process is up, the brain is not. Add the group, then log out and
back in (or `newgrp`), then restart the service.

No display or `DISPLAY` is needed: the server and the 3D dashboard are both HTTP.

## Traps

These are the things that bite *this* program specifically, not generic systemd advice.

- **Do not set `ProtectHome=` or `ProtectSystem=strict`.** The trained readout, the neuron
  positions and every recording are written **inside the checkout under `$HOME`**. Making home
  read-only makes recording fail — and it fails at the first completed window, not at startup, so
  the service looks healthy for a minute first. The shipped unit deliberately omits them.
- **`WorkingDirectory=` is load-bearing.** `flybrain/recorder.py` writes to the *relative* path
  `data/recordings`, so a unit without a working directory drops recordings somewhere in `$HOME`
  instead of in the repository. `.env` does **not** depend on it — `flybrain/env.py` finds the
  repository from its own location — so the failure is silent: the loop runs, the dashboard works,
  and the recordings are in the wrong place. The shipped unit sets it.
- **Pause is not stop.** `POST /api/pause` (and the dashboard button, and the space bar) drops the
  card to ~19 W but keeps the connectome resident: VRAM stays around **1.8 GB**. `systemctl --user
  stop flybrain` releases the GPU. Use pause for "leave the house alone tonight"; use stop for "I
  need the GPU". See [`live-view.md`](live-view.md#pause).
- **One instance, one fixed port.** The bind address is hardcoded to `127.0.0.1:8765` in
  `flybrain/server.py`; there is no host/port environment variable. A second instance fails with
  *address already in use*, and `Restart=on-failure` will retry it until `StartLimitBurst` (5 in
  300 s) stops the loop. `systemctl --user status` then says `failed`; the journal says why.
- **`Restart=on-failure`, not `always`.** A clean exit — Ctrl-C in a foreground run, or a
  deliberate shutdown — stays down. Only a crash or a signal restarts, which is what you want from
  something that should not fight you when you stop it.
- **Do not train while the service is streaming.** `uv sync`-style training and the live loop share
  the GPU: `flybrain.experiment` goes from ~50 s to 215 s for identical output. Stop the service
  first:
  ```bash
  systemctl --user stop flybrain
  .venv/bin/python -m flybrain.experiment
  systemctl --user start flybrain
  ```
- **Private `/tmp`.** The unit sets `PrivateTmp=true`; the process gets its own `/tmp`, which is
  fine here because nothing in the loop exchanges files through it. `/dev/shm` is unaffected.

## Upgrading and changing the unit

```bash
git pull
uv sync                                  # only if dependencies changed
deploy/install-service.sh --now          # re-renders the unit and restarts
```

If you changed `deploy/flybrain.service`, a plain `systemctl --user restart` is **not** enough — the
installed copy in `~/.config/systemd/user/` is what systemd reads. Re-run the installer (or copy
the file and `daemon-reload`) first. Editing with `systemctl --user edit` avoids that step for
environment overrides, but not for unit changes.

`data/experiments/colour_readout.npz` is trained output, not source; a `git pull` will not
retrain it. Re-run `flybrain.experiment` after a change that affects the readout's regime.

## Uninstall

```bash
deploy/install-service.sh --uninstall
```

That disables it, stops it, removes `~/.config/systemd/user/flybrain.service` and reloads the
manager. It leaves `.env`, `data/recordings/`, `data/experiments/` and the rest of the checkout
exactly where they were, on purpose: the recordings are the only irreplaceable thing here (they
describe a real house over time and cannot be recollected).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `status` says `failed`, journal shows `ModuleNotFoundError: flybrain` | `WorkingDirectory` or `ExecStart` points at the wrong checkout | Re-run `deploy/install-service.sh --now` |
| Journal shows a CUDA/`no CUDA GPUs` error | Not in `video`/`render` | `id -nG`, add the group, log back in, restart |
| Service is `active` but *Colour chosen* never fills | `FLYBRAIN_ALWAYS_ON` is 0, or `HA_MODE=rest` with entity ids that do not exist | Set `FLYBRAIN_ALWAYS_ON=1`; in `HA_MODE=rest`, set the real entity ids — a missing thermometer leaves the loop idle while looking configured |
| Dashboard loads, nothing moves | Pacing: a decision every 15 s is the default | Set `FLYBRAIN_INTERVAL_S=0` for the demo, or watch for a burst when the trigger fires |
| `address already in use` | A second instance, or a foreground run still holding 8765 | `systemctl --user status flybrain`; kill the other with `systemctl --user stop`, not `pkill` |
| Recordings not in `data/recordings/` | `WorkingDirectory` was overridden | Fix the unit or the drop-in; the path is relative on purpose |
| Restarts every 10 s | A crash loop, usually a bad value that slips past parsing or a missing connectome | `journalctl --user -u flybrain -n 100`; the first traceback is the real one |
| It stops when you log out | Linger is off | `sudo loginctl enable-linger "$USER"` |

`systemctl --user reset-failed flybrain` clears a unit that has hit `StartLimitBurst` so it can be
started again without waiting out the interval.

## What this deliberately does not do

- **No system scope.** A system service would start before login and run without a session, but it
  would need `User=`, its own environment plumbing, and root to install — for a program whose whole
  job is to read one house. User scope is the smaller, safer thing.
- **No log file, no rotation.** The journal owns that.
- **No backups, no watchdog, no HA-side failover.** If the process dies, systemd restarts it; if the
  house or Home Assistant is unreachable, the loop logs and waits (see
  [`live-view.md`](live-view.md#when-the-sensor-stops-reporting-the-loop-stops-acting)). Nothing
  here pretends to be orchestration.
- **No reverse proxy or authentication.** The server binds loopback only, and the dashboard — which
  can pause and drive the brain — has no auth. Do not expose the port.
