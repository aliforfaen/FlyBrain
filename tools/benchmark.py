"""Reproduce the numbers in ``docs/engine.md``.

The engine chapter publishes load time, weight VRAM, milliseconds per step, throughput as a
multiple of realtime, and a dense-vs-active comparison — and until this script existed, nothing
in the repository produced any of them. They were honest measurements with no harness behind
them, which meant they could silently rot (and one of them did: ``flybrain/pacing.py`` still
asserted 0.13x realtime after the step was measured at 0.19x).

This is that harness. It loads the **full** connectome, drives it exactly the way the live loop
drives it (the thermosensory input encoding from ``flybrain.experiment``, not an invented
current), and reports per-scenario cost. It exists to be re-run whenever ``sim.py`` changes, and
its output is the thing ``docs/engine.md`` should quote.

What it measures, and what it deliberately does not:

* **Wall time per step** includes the host-side launch overhead the dense engine actually pays,
  because it times ``sim.step(n)`` as the loop calls it. That overhead is the reason the
  active-set engine loses, so smoothing it away would hide the finding.
* **Spike counts are kept** so the dense and active engines can be compared *bitwise* on the
  same drive. A speed number without the identity check is worthless — an engine that is fast
  because it is wrong is not fast.
* It is **not** a power measurement. Power is measured with ``nvidia-smi`` on a real desktop and
  is documented in ``docs/live-view.md``; a process cannot measure its own draw.

Usage::

    .venv/bin/python tools/benchmark.py                     # dense + active, all scenarios
    .venv/bin/python tools/benchmark.py --engine dense       # just the reference
    .venv/bin/python tools/benchmark.py --markdown           # a table to paste into the docs
    .venv/bin/python tools/benchmark.py --scenario busy --steps 3000 --json /tmp/bench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.experiment import (
    ExperimentConfig,
    drive_current_for_rate,
    rate_for_temperature,
)
from flybrain.sim import ShiuParams, make_sim

SCENARIOS = ("quiet", "column", "busy")

#: The roles the live loop actually drives. A benchmark that heats an arbitrary slice of the
#: brain is measuring a different machine from the one that ships.
INPUT_ROLES = ("thermosensory", "hygrosensory")


@dataclass
class Result:
    """One engine, one scenario."""

    engine: str
    scenario: str
    drive_neurons: int
    steps: int
    wall_s: float
    ms_per_step: float
    steps_per_s: float
    realtime_x: float
    spikes: int
    #: Wall seconds for one 300 ms control window at this step cost — the number pacing uses.
    decision_s: float


@dataclass
class EngineReport:
    engine: str
    load_s: float
    #: The synapse weights alone (15M float32) — the number people mean by "the weights".
    values_mib: float
    #: crow + col + values, i.e. the resident CSR structure. The int32 column-index change
    #: halved this, which is why the old "~184 MB for weights" is no longer true.
    csr_mib: float
    peak_mib: float
    n_neurons: int
    n_synapses: int
    device: str
    results: list[Result]


def _torch():
    import torch

    return torch


def _drive_pool(sim, size: int, seed: int, prefer_sensory: bool) -> np.ndarray:
    """Pick ``size`` distinct neuron indices for the drive.

    ``prefer_sensory`` draws from the thermosensory/hygrosensory pool the live loop uses, which
    is what "a single column" should mean. It falls back to the whole brain when the real
    population is smaller than the request, so ``busy`` really does drive the requested count —
    ``RoleResolver.pool`` cycles with repetition to reach a *slot* count, and silently driving
    30 distinct neurons while claiming 256 would be exactly the kind of unmeasured claim this
    script is meant to prevent.
    """
    base: np.ndarray | None = None
    if prefer_sensory and sim.annotation_table is not None:
        try:
            from flybrain.mapping import RoleResolver

            resolver = RoleResolver.from_sim(sim)
            pools = [resolver.resolve(role).indices for role in INPUT_ROLES]
            pools = [p for p in pools if p.size]
            if pools:
                base = np.unique(np.concatenate(pools))
        except Exception:  # noqa: BLE001 - the fallback below is always valid
            base = None
    if base is None or base.size < size:
        base = np.arange(sim.n_neurons, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(base, size=min(size, base.size), replace=False)).astype(np.int64)


def _timed_steps(sim, steps: int, warmup: int) -> float:
    """Advance ``sim`` by ``steps`` and return wall seconds, warm caches excluded."""
    torch = _torch()
    on_cuda = str(sim.device).startswith("cuda")
    sync = torch.cuda.synchronize if on_cuda else (lambda: None)

    if warmup > 0:
        sim.step(warmup)
    # Zero the spike counters *after* warmup, so the reported count describes the timed steps
    # and not the ones spent filling caches.
    sim.spike_counts(reset=True)
    sync()
    started = time.perf_counter()
    sim.step(steps)
    sync()
    return time.perf_counter() - started


def run_engine(
    engine: str,
    scenarios: list[str],
    args: argparse.Namespace,
) -> tuple[EngineReport, dict[str, np.ndarray]]:
    torch = _torch()
    on_cuda = args.device != "cpu"
    if on_cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    sim = make_sim(engine, **({"device": args.device} if args.device else {}))
    load_started = time.perf_counter()
    sim.load()
    load_s = time.perf_counter() - load_started

    def _mib(t) -> float:
        return float(t.numel() * t.element_size() / 2**20)

    values_mib = _mib(sim._W.values())
    csr_mib = values_mib + _mib(sim._W.col_indices()) + _mib(sim._W.crow_indices())
    peak_mib = (
        float(torch.cuda.max_memory_allocated() / 2**20)
        if str(sim.device).startswith("cuda")
        else float("nan")
    )

    cfg = ExperimentConfig()
    current = drive_current_for_rate(
        rate_for_temperature(args.temperature, cfg), sim.params.dt_ms, cfg
    )
    # 300 ms of brain time is one control decision; at 0.1 ms/step that is 3000 steps.
    steps_per_decision = round(cfg.window_ms / sim.params.dt_ms)

    results: list[Result] = []
    counts: dict[str, np.ndarray] = {}
    for scenario in scenarios:
        sim.reset()
        if scenario == "quiet":
            sim.clear_drive()
            driven = np.zeros(0, dtype=np.int64)
        else:
            size = args.column_size if scenario == "column" else args.busy_size
            driven = _drive_pool(sim, size, args.seed, prefer_sensory=scenario == "column")
            sim.set_drive(driven, current)

        wall = _timed_steps(sim, args.steps, args.warmup)
        spike_counts = sim.spike_counts(reset=True)
        counts[scenario] = spike_counts
        ms = wall * 1000.0 / args.steps
        results.append(
            Result(
                engine=engine,
                scenario=scenario,
                drive_neurons=int(driven.size),
                steps=args.steps,
                wall_s=wall,
                ms_per_step=ms,
                steps_per_s=args.steps / wall,
                realtime_x=(args.steps * sim.params.dt_ms / 1000.0) / wall,
                spikes=int(spike_counts.sum()),
                decision_s=ms * steps_per_decision / 1000.0,
            )
        )

    report = EngineReport(
        engine=engine,
        load_s=load_s,
        values_mib=values_mib,
        csr_mib=csr_mib,
        peak_mib=peak_mib,
        n_neurons=int(sim.n_neurons),
        n_synapses=int(sim._W.values().numel()),
        device=str(sim.device),
        results=results,
    )
    del sim
    if on_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report, counts


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard over the *sets* of neurons that fired, matching the validation harness."""
    sa, sb = a > 0, b > 0
    union = int(np.logical_or(sa, sb).sum())
    return float(np.logical_and(sa, sb).sum() / union) if union else 1.0


def print_plain(reports: list[EngineReport], steps_per_decision: int) -> None:
    width = 78
    print("=" * width)
    print("FlyBrain engine benchmark")
    print("=" * width)
    for report in reports:
        print()
        print(f"engine={report.engine}  device={report.device}")
        print(
            f"  load {report.load_s:.2f} s   values {report.values_mib:.1f} MiB   "
            f"CSR {report.csr_mib:.1f} MiB   peak {report.peak_mib:.1f} MiB   "
            f"{report.n_neurons:,} neurons   {report.n_synapses:,} synapses"
        )
        print(
            f"  {'scenario':<9} {'driven':>7} {'wall s':>8} {'ms/step':>8} "
            f"{'steps/s':>9} {'xreal':>7} {'spikes':>9} {'decision s':>11}"
        )
        for r in report.results:
            print(
                f"  {r.scenario:<9} {r.drive_neurons:>7} {r.wall_s:>8.3f} {r.ms_per_step:>8.3f} "
                f"{r.steps_per_s:>9.0f} {r.realtime_x:>6.2f}x {r.spikes:>9,} "
                f"{r.decision_s:>11.2f}"
            )
    print()
    print(f"(decision s = {steps_per_decision} steps, i.e. the 300 ms window one decision consumes)")


def print_markdown(reports: list[EngineReport]) -> None:
    print()
    print("<!-- generated by tools/benchmark.py -->")
    for report in reports:
        print()
        print(
            f"**{report.engine}** — load {report.load_s:.1f} s, values "
            f"{report.values_mib:.0f} MiB, CSR {report.csr_mib:.0f} MiB, peak "
            f"{report.peak_mib:.0f} MiB, {report.n_neurons:,} neurons / "
            f"{report.n_synapses:,} synapses on {report.device}"
        )
        print()
        print("| Scenario | Driven | Wall s | ms/step | ×realtime | Spikes |")
        print("|---|---|---|---|---|---|")
        for r in report.results:
            print(
                f"| {r.scenario} | {r.drive_neurons} | {r.wall_s:.2f} | {r.ms_per_step:.3f} "
                f"| {r.realtime_x:.2f}× | {r.spikes:,} |"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the engine numbers in docs/engine.md on the full connectome.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--engine", choices=["dense", "active", "both"], default="both")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--steps", type=int, default=1000, help="timed steps per scenario")
    parser.add_argument("--warmup", type=int, default=100, help="untimed steps before timing")
    parser.add_argument(
        "--temperature",
        type=float,
        default=30.0,
        help="sensor temperature whose live encoding sets the drive current",
    )
    parser.add_argument("--column-size", type=int, default=8, help="neurons driven in 'column'")
    parser.add_argument("--busy-size", type=int, default=256, help="neurons driven in 'busy'")
    parser.add_argument("--device", default=None, help="cuda, cpu, or unset for auto-detect")
    parser.add_argument("--seed", type=int, default=0, help="drive-index selection seed")
    parser.add_argument("--json", default=None, help="also write the raw results here")
    parser.add_argument("--markdown", action="store_true", help="emit docs-ready tables")
    args = parser.parse_args(argv)

    if args.steps <= 0:
        parser.error("--steps must be positive")

    engines = ["dense", "active"] if args.engine == "both" else [args.engine]
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    cfg = ExperimentConfig()
    dt_ms = ShiuParams().dt_ms
    steps_per_decision = round(cfg.window_ms / dt_ms)

    print(
        f"python {platform.python_version()}  "
        f"steps={args.steps} warmup={args.warmup}  T={args.temperature}C  "
        f"drive={drive_current_for_rate(rate_for_temperature(args.temperature, cfg), dt_ms, cfg):.3f} mV/step"
    )
    reports: list[EngineReport] = []
    counts: dict[str, dict[str, np.ndarray]] = {}
    for engine in engines:
        report, engine_counts = run_engine(engine, scenarios, args)
        reports.append(report)
        counts[engine] = engine_counts
        print(f"  {engine}: done")

    print_plain(reports, steps_per_decision)

    if len(reports) == 2:
        dense, active = reports
        print()
        print("dense vs active (same drive, same steps)")
        by_scenario = {r.scenario: r for r in active.results}
        print(f"  {'scenario':<9} {'dense ms':>9} {'active ms':>10} {'speedup':>8} {'spikes equal':>13} {'Jaccard':>8}")
        for r in dense.results:
            a = by_scenario.get(r.scenario)
            if a is None:
                continue
            identical = np.array_equal(counts["dense"][r.scenario], counts["active"][r.scenario])
            print(
                f"  {r.scenario:<9} {r.ms_per_step:>9.3f} {a.ms_per_step:>10.3f} "
                f"{r.ms_per_step / a.ms_per_step:>7.2f}x {identical!s:>13} "
                f"{_jaccard(counts['dense'][r.scenario], counts['active'][r.scenario]):>8.3f}"
            )

    if args.markdown:
        print_markdown(reports)

    if args.json:
        payload = {
            "python": platform.python_version(),
            "steps": args.steps,
            "warmup": args.warmup,
            "temperature_c": args.temperature,
            "engines": [asdict(r) for r in reports],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
