"""Validate ``ConnectomeSim`` against the published model.

Ground truth is Brian2 running the exact Shiu et al. (2024) equations, as in the
reference implementation vendored under ``vendor/fly-brain``. To make the comparison
rigorous rather than merely statistical, **both simulators receive the identical,
prespecified Poisson input schedule** (same driven neurons, same spike times, same
weights). Any difference in the output is therefore attributable to the model
implementation, not to differing random draws.

Two interpreters are required because Brian2 does not support NumPy 2.x, while this
project targets NumPy 2.x:

    .venv-validation/bin/python validation/validate_against_brian2.py --mode ref
    .venv/bin/python            validation/validate_against_brian2.py --mode ours
    .venv/bin/python            validation/validate_against_brian2.py --mode compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.sim import (
    DEFAULT_COMPLETENESS,
    DEFAULT_CONNECTIVITY,
    ShiuParams,
)

REF = ShiuParams()
DEFAULT_REF_OUT = ROOT / "data/validation/brian2_ref.npz"
DEFAULT_OURS_OUT = ROOT / "data/validation/torch_out.npz"


# --------------------------------------------------------------------- network


def build_subnetwork(
    n_neurons: int = 4000, drive_neurons: int = 300, seed: int = 0, augment: bool = True
):
    """Extract a dense, self-contained subnetwork of the real connectome.

    Takes the first ``n_neurons`` neurons as nodes and keeps every synapse whose two
    endpoints are inside that set, then stochastically adds excitatory edges so the
    isolated slice still carries enough recurrent drive to spike.
    """
    comp = pd.read_csv(DEFAULT_COMPLETENESS, index_col=0)
    fly_ids = comp.index.to_numpy(np.int64)[:n_neurons]

    conn = pd.read_parquet(
        DEFAULT_CONNECTIVITY,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    inside = (conn["Presynaptic_Index"] < n_neurons) & (conn["Postsynaptic_Index"] < n_neurons)
    sub = conn[inside].reset_index(drop=True)
    print(f"real intra-slice synapses: {len(sub):,}")

    rng = np.random.default_rng(seed)
    if not augment:
        print(f"using the pure real slice: {len(sub):,} synapses")
    if augment and len(sub) < 20_000:
        extra = 20_000 - len(sub)
        pre = rng.integers(0, n_neurons, extra)
        post = rng.integers(0, n_neurons, extra)
        counts = sub["Excitatory x Connectivity"].to_numpy()
        w = rng.choice(counts, extra) if len(counts) else rng.integers(1, 10, extra)
        sub = pd.concat(
            [
                sub,
                pd.DataFrame(
                    {
                        "Presynaptic_Index": pre,
                        "Postsynaptic_Index": post,
                        "Excitatory x Connectivity": w,
                    }
                ),
            ],
            ignore_index=True,
        )
        print(f"augmented with {extra:,} random excitatory edges -> {len(sub):,}")

    n = int(max(sub["Presynaptic_Index"].max(), sub["Postsynaptic_Index"].max()) + 1)
    n = max(n, n_neurons, int(drive_neurons))
    return fly_ids, sub, np.arange(drive_neurons), n


def make_drive_schedule(n_drive: int, real_neurons: int, duration_ms: float, seed: int):
    """Prespecified Poisson input, identical for both simulators.

    The drive is expressed as ``n_drive`` *virtual* neurons appended after the real
    population, each with its own Poisson spike train. Giving every spike its own source
    neuron means duplicates never collapse into a single synapse, so both simulators see
    exactly the same events. Returns ``(source_idx, target_idx, times_ms, weight_mv)``.
    """
    rng = np.random.default_rng(seed)
    dt = REF.dt_ms
    n_steps = round(duration_ms / dt)
    p_spike = 150.0 * dt / 1000.0
    src, tgt, times = [], [], []
    for step in range(n_steps):
        fired = np.flatnonzero(rng.random(n_drive) < p_spike)
        if fired.size:
            src.append(fired + real_neurons)
            tgt.append(fired)
            times.append(np.full(fired.size, step * dt, dtype=np.float64))
    if not src:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0), REF.w_scale_mv * REF.poisson_scale
    return (
        np.concatenate(src).astype(np.int64),
        np.concatenate(tgt).astype(np.int64),
        np.concatenate(times),
        REF.w_scale_mv * REF.poisson_scale,
    )


# -------------------------------------------------------------------- brian2


def run_brian2(sub, drive, real_neurons: int, n_drive: int, duration_ms: float):
    """Reference simulation on the published equations, driven by a fixed schedule."""
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        defaultclock,
        ms,
        mV,
    )

    defaultclock.dt = REF.dt_ms * ms
    src, tgt, times, weight = drive
    n_neurons = real_neurons + n_drive

    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau              : volt (unless refractory)
    rfc                           : second
    """
    neu = NeuronGroup(
        N=n_neurons,
        model=eqs,
        method="linear",
        threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV",
        refractory="rfc",
        namespace={
            "v_0": REF.v_rest_mv * mV,
            "v_rst": REF.v_reset_mv * mV,
            "v_th": REF.v_thresh_mv * mV,
            "t_mbr": REF.tau_mem_ms * ms,
            "tau": REF.tau_syn_ms * ms,
        },
    )
    neu.v = REF.v_rest_mv * mV
    neu.g = 0 * mV
    neu.rfc = REF.t_refrac_ms * ms
    # Virtual drive neurons must be free to spike, so give them no refractory period.
    neu.rfc[real_neurons:] = 0 * ms

    int_syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=REF.t_delay_ms * ms)
    int_syn.connect(
        i=sub["Presynaptic_Index"].to_numpy(), j=sub["Postsynaptic_Index"].to_numpy()
    )
    int_syn.w = sub["Excitatory x Connectivity"].to_numpy() * REF.w_scale_mv * mV

    # Fixed input schedule: an explicit spike source, so no RNG is involved.
    objects = [neu, int_syn]
    if src.size:
        drive_group = SpikeGeneratorGroup(n_neurons, src, times * ms)
        # The drive is an immediate voltage step at the spike time, exactly as the
        # reference model's ``voltage_stim``. It must NOT go through a synaptic delay:
        # delaying it lets a neuron accumulate two inputs before its threshold check.
        in_syn = Synapses(drive_group, neu, "w : volt", on_pre="v += w")
        in_syn.connect(i=src, j=tgt)
        in_syn.w = weight * mV
        objects += [drive_group, in_syn]

    monitor = SpikeMonitor(neu)
    objects.append(monitor)
    Network(*objects).run(duration_ms * ms)

    counts = np.zeros(n_neurons, dtype=np.int64)
    for neuron_index, spike_times in monitor.spike_trains().items():
        counts[neuron_index] = len(spike_times)
    return counts[:real_neurons]


# --------------------------------------------------------------------- torch


def run_torch(sub, drive, real_neurons: int, n_drive: int, duration_ms: float, engine: str = "dense"):
    """Our simulator on the identical network and identical input schedule."""
    import torch

    from flybrain.sim import make_sim

    _src, tgt, times, weight = drive
    n_neurons = real_neurons + n_drive

    sim = make_sim(engine, params=REF, device="cpu")
    sim.n_neurons = n_neurons
    sim._torch = torch
    sim._W = (
        torch.sparse_coo_tensor(
            torch.from_numpy(
                np.stack(
                    [
                        sub["Postsynaptic_Index"].to_numpy(np.int64),
                        sub["Presynaptic_Index"].to_numpy(np.int64),
                    ]
                ).copy()
            ),
            torch.from_numpy(
                (sub["Excitatory x Connectivity"].to_numpy(np.float32) * REF.w_scale_mv).copy()
            ),
            (n_neurons, n_neurons),
        )
        .coalesce()
        .to_sparse_csr()
    )
    sim.reset()
    if engine == "active":
        # The subclass builds its delivery journal inside load(); this harness assembles the
        # network by hand, so the journal has to be built explicitly.
        sim._build_journal()

    dt = REF.dt_ms
    n_steps = round(duration_ms / dt)
    per_step: dict[int, list[int]] = {}
    for target, t in zip(tgt.tolist(), times.tolist()):
        per_step.setdefault(round(t / dt), []).append(target)

    for step in range(n_steps):
        hit = per_step.get(step)
        if hit:
            # Immediate voltage step (no axonal delay), matching the reference model.
            sim.inject(np.array(hit, dtype=np.int64), float(weight))
        sim.step(1)
    return sim.spike_counts()[:real_neurons].astype(np.int64)


# -------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate ConnectomeSim against Brian2.")
    ap.add_argument("--mode", choices=["ref", "ours", "compare"], default="compare")
    ap.add_argument("--ref-out", default=str(DEFAULT_REF_OUT))
    ap.add_argument(
        "--ours-out",
        default=None,
        help="where to write our spike counts; defaults to "
        "data/validation/torch_out_<engine>.npz so a dense and an active run cannot clobber "
        "each other",
    )
    ap.add_argument("--neurons", type=int, default=4000)
    ap.add_argument("--drive", type=int, default=300)
    ap.add_argument("--duration", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--engine",
        choices=["dense", "active"],
        default="dense",
        help="which simulator engine to validate in --mode ours (write one file per engine, then "
        "run --mode compare against each)",
    )
    ap.add_argument(
        "--no-augment",
        action="store_false",
        dest="augment",
        help="use only the real intra-slice synapses (no random edges)",
    )
    args = ap.parse_args()

    # Per-engine by default. A single fixed path meant an `--engine active` run silently
    # overwrote the dense counts, and `compare` could then only ever see whichever ran last —
    # which is the opposite of "both must pass".
    ours_out = (
        Path(args.ours_out)
        if args.ours_out
        else DEFAULT_OURS_OUT.with_name(f"torch_out_{args.engine}.npz")
    )

    _fly_ids, sub, drive_idx, real_neurons = build_subnetwork(
        args.neurons, args.drive, args.seed, augment=args.augment
    )
    n_drive = int(drive_idx.size)
    schedule = make_drive_schedule(n_drive, real_neurons, args.duration, args.seed + 1)
    Path(args.ref_out).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"network: {real_neurons} real + {n_drive} virtual drive neurons, "
        f"{len(sub):,} synapses, {schedule[0].size:,} input spikes over {args.duration} ms"
    )

    if args.mode == "ref":
        counts = run_brian2(sub, schedule, real_neurons, n_drive, args.duration)
        np.savez(args.ref_out, counts=counts)
        print(f"[brian2] total spikes: {int(counts.sum()):,} -> {args.ref_out}")
        return 0

    if args.mode == "ours":
        counts = run_torch(sub, schedule, real_neurons, n_drive, args.duration, args.engine)
        np.savez(ours_out, counts=counts)
        print(f"[ours:{args.engine}] total spikes: {int(counts.sum()):,} -> {ours_out}")
        return 0

    ref = np.load(args.ref_out)["counts"]
    ours = np.load(ours_out)["counts"]
    n = min(ref.size, ours.size)
    if ref.size != ours.size:
        print(f"note: sizes differ (brian2={ref.size}, ours={ours.size}); comparing first {n}")
    ref, ours = ref[:n], ours[:n]

    ref_total, our_total = int(ref.sum()), int(ours.sum())
    both = (ref > 0) & (ours > 0)
    union = (ref > 0) | (ours > 0)
    jaccard = both.sum() / max(union.sum(), 1)
    ratio = our_total / ref_total if ref_total else float("nan")
    corr = np.corrcoef(ref[both], ours[both])[0, 1] if both.sum() > 10 else float("nan")

    print(f"\nBrian2 spikes total : {ref_total:,}  (active: {(ref > 0).sum():,})")
    print(f"ours   spikes total : {our_total:,}  (active: {(ours > 0).sum():,})")
    print(f"active-neuron Jaccard: {jaccard:.3f}")
    print(f"spike-count ratio    : {ratio:.3f}")
    print(f"rate correlation     : {corr:.3f}")

    ok = (
        ref_total > 0
        and 0.8 <= ratio <= 1.25
        and jaccard > 0.7
        and (np.isnan(corr) or corr > 0.9)
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
