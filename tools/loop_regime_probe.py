"""The measurement that moved training from from-rest windows to a continuous sweep.

**Historical.** This is the probe whose result changed ``experiment.py``; it is kept because that
finding is why *train and run in the same regime* is a rule here (``AGENTS.md`` #2), not because it
still describes how the shipped readout is fitted.

At the time, ``experiment.py`` trained on windows that always started from a **resting** brain:
``sample()`` reset the simulator, drove at a fixed temperature for 300 ms, and read the spikes from
that window. A live control loop does not work that way — the brain keeps running, the temperature
drifts, and each decision reads the *most recent* 300 ms of an already-active network. This tool
measured the difference directly, and the cross-over was bad enough that ``fit()`` now trains on
``sweep()``, which is continuous by construction (see ``experiment.py``).

It is still useful as a *contrast*. For each temperature it records two windows:

* ``rest`` -- reset, drive for 300 ms, read that window (the old training regime)
* ``live`` -- keep going another 300 ms, read that window (what a live loop would see)

then fits ridge readouts and reports held-out error for rest->rest, live->live, and both
cross-overs, plus a linear probe that tries to tell the two regimes apart. When that probe can
separate them perfectly, pooling the two silently costs accuracy — which is what it found.

Usage::

    .venv/bin/python tools/loop_regime_probe.py            # collect + analyse
    .venv/bin/python tools/loop_regime_probe.py --reuse    # analyse the cached windows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.experiment import COLOUR_BANDS, TemperatureColourLoop

CACHE = Path("/tmp/loop_regime.npz")
N_TEMPS = 26


def collect() -> dict:
    loop = TemperatureColourLoop()
    sim = loop.sim
    cfg = loop.config
    dt = sim.params.dt_ms
    steps = max(1, round(cfg.window_ms / dt))
    window_s = cfg.window_ms / 1000.0

    temps = np.linspace(cfg.temp_min_c, cfg.temp_max_c, N_TEMPS)
    ideal = np.array([loop.ideal_kelvin(float(t)) for t in temps])
    n = loop.readout_indices.size
    rest = np.zeros((temps.size, n))
    live = np.zeros((temps.size, n))

    print(f"readout pool {n}, window {cfg.window_ms:g} ms ({steps} steps)")
    for i, t in enumerate(temps):
        rate_hz = loop.rate_for_temperature(float(t))
        current = rate_hz * dt / 1000.0 * cfg.current_per_spike_mv
        sim.reset()
        loop.brain.clear_input_drive()
        loop.brain.set_input_drive(loop.input_indices, current)
        sim.step(steps)
        rest[i] = sim.spike_counts(reset=True)[loop.readout_indices] / window_s
        sim.step(steps)                       # keep running: the brain is not reset
        live[i] = sim.spike_counts(reset=True)[loop.readout_indices] / window_s
        if i % 5 == 0 or i == temps.size - 1:
            print(f"  {t:5.1f}C  rest {rest[i].sum():9.1f} Hz   live {live[i].sum():9.1f} Hz")

    np.savez(CACHE, temps=temps, ideal=ideal, rest=rest, live=live)
    print(f"saved {CACHE}")
    return {"temps": temps, "ideal": ideal, "rest": rest, "live": live}


def ridge(Xtr, ytr, l2=10.0):
    Xb = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
    d = Xb.shape[1]
    return np.linalg.solve(Xb.T @ Xb + np.diag([l2] * (d - 1) + [0.0]), Xb.T @ ytr)


def score(Xtr, ytr, Xte, yte, l2=10.0) -> tuple[float, float]:
    W = ridge(Xtr, ytr, l2)
    pred = np.hstack([Xte, np.ones((Xte.shape[0], 1))]) @ W
    lo, hi = COLOUR_BANDS[0][1], COLOUR_BANDS[-1][1]
    pred = np.clip(pred, lo, hi)
    return float(np.mean(np.abs(pred - yte))), float(np.corrcoef(pred, yte)[0, 1])


def analyse(d: dict) -> int:
    temps, ideal, rest, live = d["temps"], d["ideal"], d["rest"], d["live"]
    train = np.zeros(temps.size, bool)
    train[::2] = True
    test = ~train

    print("\npopulation rate, rest vs live window")
    print(f"  rest: {rest.sum(axis=1).min():.0f} - {rest.sum(axis=1).max():.0f} Hz")
    print(f"  live: {live.sum(axis=1).min():.0f} - {live.sum(axis=1).max():.0f} Hz")
    d_rate = live.sum(axis=1) - rest.sum(axis=1)
    print(f"  live - rest: {d_rate.min():+.0f} to {d_rate.max():+.0f} Hz "
          f"(mean {d_rate.mean():+.0f})")

    print(f"\nheld-out accuracy ({train.sum()} train / {test.sum()} test)")
    print(f"  {'train -> test':<22} {'mean |err| K':>13} {'corr':>8}")
    for label, Xtr, Xte in (
        ("rest -> rest", rest[train], rest[test]),
        ("live -> live", live[train], live[test]),
        ("rest -> live (transfer)", rest[train], live[test]),
        ("live -> rest (transfer)", live[train], rest[test]),
    ):
        err, corr = score(Xtr, ideal[train], Xte, ideal[test])
        print(f"  {label:<22} {err:>13.1f} {corr:>8.4f}")

    # How separable are the two regimes? If a probe cannot tell them apart the
    # question is academic; if it can trivially, they really are different states.
    both = np.vstack([rest, live])
    labels = np.r_[np.zeros(rest.shape[0]), np.ones(live.shape[0])]
    both_train = np.r_[train, train]
    both_test = np.r_[test, test]
    W = ridge(both[both_train], labels[both_train].reshape(-1, 1), l2=10.0)
    pred = (np.hstack([both[both_test], np.ones((both_test.sum(), 1))]) @ W).ravel()
    acc = float(np.mean((pred > 0.5) == (labels[both_test] > 0.5)))
    print(f"\n  can a linear probe tell rest from live? accuracy {acc:.2f} "
          f"({'indistinguishable' if acc < 0.7 else 'clearly different regimes'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare the from-rest and continuous training regimes (historical probe)."
    )
    ap.add_argument(
        "--reuse",
        action="store_true",
        help=f"analyse the cached windows in {CACHE} instead of re-simulating",
    )
    args = ap.parse_args(argv)

    if CACHE.exists() and args.reuse:
        with np.load(CACHE) as z:
            d = {k: z[k] for k in ("temps", "ideal", "rest", "live")}
    else:
        d = collect()
    return analyse(d)


if __name__ == "__main__":
    raise SystemExit(main())
