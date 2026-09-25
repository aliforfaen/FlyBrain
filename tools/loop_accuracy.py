"""Measure the live control loop end to end, exactly as the dashboard runs it.

``flybrain.experiment`` reports a held-out error, but that is still an *offline* number:
it sweeps the brain itself and never touches Home Assistant. This tool runs the real
:class:`~flybrain.loop.LiveLoop` against the real mock sensor, with the brain running
continuously and the simulated room drifting on the same sine the dashboard uses, and
reports the colour the loop actually emitted versus the ideal mapping.

If this disagrees with the offline held-out figure, the offline figure is the one that is
wrong, because this is the thing that ships.

Usage::

    .venv/bin/python tools/loop_accuracy.py [--decisions 48] [--period 180]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.ha import Scenario
from flybrain.loop import LoopConfig, build_loop
from flybrain.sim import ConnectomeSim


async def run(decisions: int, period_s: float) -> int:
    sim = ConnectomeSim().load()
    loop = build_loop(sim, LoopConfig(mode="mock"))
    if loop is None:
        print("no trained readout - run `.venv/bin/python -m flybrain.experiment` first")
        return 1

    scenario = Scenario(
        temperature_start=22.5, temperature_swing_c=11.0, temperature_period_s=period_s
    )
    loop.ha.scenario = scenario
    loop.ha.advance(0.0)

    steps = max(1, round(loop.config.window_ms / sim.params.dt_ms))
    mock_dt = period_s / decisions
    print(
        f"live loop: {decisions} decisions over one {period_s:.0f}s cycle "
        f"(room moves {mock_dt:.1f}s per decision), {loop.config.window_ms:.0f} ms window"
    )
    print(f"  {'temp':>7}  {'colour':>7}  {'ideal':>7}  {'err':>6}  band")
    temps, kels, ideals = [], [], []
    for _ in range(decisions):
        temperature = await loop.read_temperature()
        loop.drive_temperature(temperature)
        sim.step(steps)
        counts = sim.spike_counts(reset=True)
        await loop.decide(counts, loop.config.window_ms)
        snap = loop.snapshot()
        temps.append(snap["temperature_c"])
        kels.append(snap["kelvin"])
        ideals.append(snap["ideal_kelvin"])
        print(
            f"  {snap['temperature_c']:6.2f}C  {snap['kelvin']:5d}K  {snap['ideal_kelvin']:5d}K  "
            f"{snap['error_k']:+5d}K  {snap['band']}"
        )
        loop.ha.advance(mock_dt)

    kels_a = np.array(kels, dtype=float)
    ideals_a = np.array(ideals, dtype=float)
    err = np.abs(kels_a - ideals_a)
    print(
        f"\n  decisions {len(kels_a)}   mean |err| {err.mean():.0f} K   max {err.max():.0f} K   "
        f"correlation {np.corrcoef(kels_a, ideals_a)[0, 1]:.4f}"
    )
    print(f"  mock service calls sent: {len(loop.ha.calls)}")
    if loop.ha.calls:
        print(f"  last call: {loop.ha.calls[-1]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the live loop end to end.")
    ap.add_argument("--decisions", type=int, default=48)
    ap.add_argument("--period", type=float, default=180.0)
    args = ap.parse_args()
    return asyncio.run(run(args.decisions, args.period))


if __name__ == "__main__":
    raise SystemExit(main())
