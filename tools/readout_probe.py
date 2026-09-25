"""Probe the readout population: does the brain state actually encode temperature?

The control loop collapsed the readout population into ``n_bands`` scalar rates, which
throws away nearly all of the brain state. This tool measures what is actually
recoverable *before* we change the readout, so the change is driven by a measurement
rather than a guess.

Collection simulates the brain across a fine temperature sweep and caches the raw
per-neuron rate vectors to ``/tmp/readout_probe.npz``; analysis is then free to re-run.
Pass ``--reuse`` to skip collection.

The generalisation numbers use a **held-out split** (train on even sweep indices, test on
odd), with any feature selection done using training data only. That matters: with 512
features and only a few dozen windows, an in-sample or selection-leaked number would look
far better than the model really is.

Usage::

    .venv/bin/python tools/readout_probe.py            # collect + analyse
    .venv/bin/python tools/readout_probe.py --reuse    # analyse cached data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.experiment import COLOUR_BANDS, TemperatureColourLoop

CACHE = Path("/tmp/readout_probe.npz")
N_TEMPS = 51


def ridge_fit(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Ridge fit with an unregularized bias column; returns the weight vector."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    d = Xb.shape[1]
    reg = np.diag([l2] * (d - 1) + [0.0])
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)


def collect() -> dict:
    loop = TemperatureColourLoop()
    n_neurons = loop.readout_indices.size
    print(f"readout pool: {n_neurons} neurons from {loop.config.readout_roles}")
    print(f"readout features per sample: {loop.readout.n_features} (per-neuron rates + bias)")
    print(f"sensor rate range: {loop.config.min_rate_hz:g}-{loop.config.max_rate_hz:g} Hz")

    temps = np.linspace(loop.config.temp_min_c, loop.config.temp_max_c, N_TEMPS)
    ideal = np.interp(
        temps,
        [loop.config.temp_min_c, loop.config.temp_max_c],
        [COLOUR_BANDS[0][1], COLOUR_BANDS[-1][1]],
    )
    X = np.zeros((temps.size, n_neurons), dtype=np.float64)
    for i, t in enumerate(temps):
        _, rates = loop.sample(float(t))
        X[i] = rates
        if i % 10 == 0 or i == temps.size - 1:
            print(f"  {t:5.1f}C  total rate {rates.sum():10.1f} Hz  "
                  f"active {int((rates > 0).sum()):4d}/{n_neurons}")
    np.savez(CACHE, temps=temps, X=X, ideal=ideal)
    print(f"saved {CACHE}")
    return {"temps": temps, "X": X, "ideal": ideal}


def analyse(data: dict) -> int:
    temps, X, ideal = data["temps"], data["X"], data["ideal"]
    n_neurons = X.shape[1]

    print("\ndynamic range")
    totals = X.sum(axis=1)
    print(f"  total population rate: {totals.min():.0f} Hz at {temps[totals.argmin()]:.1f}C "
          f"-> {totals.max():.0f} Hz at {temps[totals.argmax()]:.1f}C")
    print(f"  silent temperatures: {int((totals <= 0).sum())}/{temps.size}")

    corr = np.nan_to_num(np.array([
        np.corrcoef(X[:, k], temps)[0, 1] if X[:, k].std() > 0 else 0.0
        for k in range(n_neurons)
    ]))
    print("\nper-neuron tuning to temperature (all data, for description only)")
    print(f"  neurons with |r| > 0.9: {int((np.abs(corr) > 0.9).sum())} / {n_neurons}")
    print(f"  neurons with |r| > 0.7: {int((np.abs(corr) > 0.7).sum())} / {n_neurons}")
    print(f"  max |r|: {np.abs(corr).max():.3f}")

    n_bands = len(COLOUR_BANDS)
    bands = np.stack([X[:, sl].mean(axis=1)
                      for sl in np.array_split(np.arange(n_neurons), n_bands)], axis=1)
    bc = np.corrcoef(bands.T)
    print(f"\ncollinearity of the {n_bands} band features (the old design)")
    for row in bc:
        print("   ", " ".join(f"{v:+.3f}" for v in row))

    train = np.zeros(temps.size, dtype=bool)
    train[::2] = True
    test = ~train
    print(f"\nheld-out split: {train.sum()} train / {test.sum()} test windows")

    # Feature selection uses TRAIN data only, so the test score stays honest.
    train_corr = np.nan_to_num(np.array([
        np.corrcoef(X[train, k], temps[train])[0, 1] if X[train, k].std() > 0 else 0.0
        for k in range(n_neurons)
    ]))
    order = np.argsort(-np.abs(train_corr))

    subsets: list[tuple[str, np.ndarray]] = [("band rates (old)", bands)]
    for k in (8, 16, 32, 64, 128, 512):
        subsets.append((f"top-{k} tuned", X[:, order[:k]]))

    print(f"\n  {'features':<20} " + " ".join(f"{f'l2={v:g}':>10}"
                                               for v in (0.01, 0.1, 1.0, 10.0, 100.0)))
    best = (float("inf"), "", 0.0)
    for label, feats in subsets:
        cells = []
        for l2 in (0.01, 0.1, 1.0, 10.0, 100.0):
            W = ridge_fit(feats[train], ideal[train], l2)
            pred = np.hstack([feats[test], np.ones((test.sum(), 1))]) @ W
            err = float(np.mean(np.abs(pred - ideal[test])))
            cells.append(f"{err:10.1f}")
            if err < best[0]:
                best = (err, label, l2)
        print(f"  {label:<20} " + " ".join(cells))

    print(f"\n  best held-out: {best[1]} at l2={best[2]:g} -> {best[0]:.1f} K")

    # What the 3-band soft-label target can represent at best, i.e. the ceiling imposed
    # by the label scheme rather than by the features.
    idx = np.arange(n_bands, dtype=np.float64)
    for width in (0.6, 0.4, 0.3, 0.2):
        u = (temps - temps.min()) / (temps.max() - temps.min())
        tk = u * (n_bands - 1)
        w = np.exp(-0.5 * ((idx[None, :] - tk[:, None]) / width) ** 2)
        w /= w.sum(axis=1, keepdims=True)
        soft_k = w @ np.array([c for _, c in COLOUR_BANDS])
        print(f"  soft-label target, width {width:g}: mean |err| vs ideal "
              f"{np.mean(np.abs(soft_k - ideal)):6.1f} K")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe the readout population.")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the cached rate vectors instead of re-simulating")
    args = ap.parse_args()

    if args.reuse and CACHE.exists():
        with np.load(CACHE) as d:
            data = {"temps": d["temps"], "X": d["X"], "ideal": d["ideal"]}
    else:
        data = collect()
    return analyse(data)


if __name__ == "__main__":
    raise SystemExit(main())
