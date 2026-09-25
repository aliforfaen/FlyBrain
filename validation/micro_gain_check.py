"""Micro-validation of the LIF integrator against Brian2.

The whole-brain harness (``validate_against_brian2.py``) compares spike *counts* on a
large network, which conflates three separate things: the integrator, the connectome
topology, and the network's operating regime. This script removes the first two as
variables by using a tiny, fully hand-checkable network where the analytic answer is
known.

The analytical result being checked
-----------------------------------
The reference model is::

    dv/dt = (v0 - v + g) / tau_m
    dg/dt = -g / tau_s

For a step in which ``g`` is treated as constant (which is what Brian2's ``method='linear'``
does -- it integrates the linear system exactly), the closed-form update is

    v(t+dt) = v0 + g*(1 - exp(-dt/tau_m)) + (v(t) - v0)*exp(-dt/tau_m)

Note the conductance coefficient is ``(1 - decay_v)``, which is ~0.005 for the published
constants. It is *not* ``tau_m * (1 - decay_v)`` (~0.0998), which is a forward-Euler
lookalike and over-weights every synapse by exactly ``tau_m`` = 20x.

A single synaptic event of weight ``w`` produces a membrane deflection whose peak is
(exact solution of the two-timescale linear system, from rest)::

    u(t) = w * tau_s/(tau_m - tau_s) * (exp(-t/tau_m) - exp(-t/tau_s))

which peaks at ``t* = tau_m*tau_s/(tau_m-tau_s) * ln(tau_m/tau_s)``, giving

    u_max = w * 0.15749   for tau_m=20 ms, tau_s=5 ms

So ``u_max`` is roughly ``w/6``. That is the number the integrator has to reproduce: if
the code multiplies the conductance term by an extra ``tau_m``, every measurement comes
out exactly 20x too large.

Usage (two interpreters, as with the whole-brain harness)::

    .venv-validation/bin/python validation/micro_gain_check.py --mode ref
    .venv/bin/python            validation/micro_gain_check.py --mode ours
    .venv/bin/python            validation/micro_gain_check.py --mode compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.sim import ConnectomeSim, ShiuParams

REF = ShiuParams()
DEFAULT_REF_OUT = ROOT / "data/validation/micro_ref.npz"
DEFAULT_OURS_OUT = ROOT / "data/validation/micro_ours.npz"

DURATION_MS = 60.0
DRIVE_STEP = 100  # fires the virtual driver at t = 10 ms


# --------------------------------------------------------------------- cases
#
# Each case is (n_real, edges, description). ``edges`` are (pre, post, weight_mv).
# Index ``n_real`` is the virtual driver; it fires exactly once at DRIVE_STEP.

CASES: dict[str, tuple[int, list[tuple[int, int, float]], str]] = {
    # One subthreshold synapse onto neuron 0. The ONLY thing this measures is the
    # integrator's conductance gain. Correct answer: a bump of ~0.63*w mV.
    "subthreshold": (
        1,
        [(1, 0, 5.0)],
        "single subthreshold synapse, w=5.0 mV; must stay subthreshold",
    ),
    # One suprathreshold synapse: neuron 0 must fire exactly once, shortly after the
    # event arrives (t_dly = 1.8 ms) plus the rise time to threshold.
    "suprathreshold": (
        1,
        [(1, 0, 50.0)],
        "single suprathreshold synapse, w=50.0 mV; neuron 0 fires once",
    ),
    # A three-hop chain. Tests the delay line, the reset, and refractory handling.
    "chain": (
        3,
        [(3, 0, 50.0), (0, 1, 50.0), (1, 2, 50.0)],
        "3-hop chain, w=50.0 mV each; 4 spikes spaced by > t_dly",
    ),
    # Inhibition must be able to veto: neuron 0 fires from a strong excitatory input,
    # but a simultaneous strong inhibitory input must cancel it.
    "inhibition": (
        1,
        [(1, 0, 50.0), (1, 0, -200.0)],
        "excitatory + inhibitory from one source; net negative, no spike",
    ),
}


def analytic_peak(weight_mv: float) -> float:
    """Closed-form peak membrane deflection for one synaptic event of ``weight_mv``."""
    tm, ts = REF.tau_mem_ms, REF.tau_syn_ms
    t_star = tm * ts / (tm - ts) * np.log(tm / ts)
    return float(weight_mv * ts / (tm - ts) * (np.exp(-t_star / tm) - np.exp(-t_star / ts)))


# -------------------------------------------------------------------- brian2


def run_brian2(case: str) -> dict:
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        StateMonitor,
        Synapses,
        defaultclock,
        ms,
        mV,
    )

    n_real, edges, _ = CASES[case]
    n = n_real + 1
    driver = n_real
    defaultclock.dt = REF.dt_ms * ms

    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau              : volt (unless refractory)
    rfc                           : second
    """
    neu = NeuronGroup(
        N=n,
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
    neu.rfc[driver] = 0 * ms

    pre = [e[0] for e in edges]
    post = [e[1] for e in edges]
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=REF.t_delay_ms * ms)
    syn.connect(i=pre, j=post)
    syn.w = np.array([e[2] for e in edges]) * mV

    # The virtual driver emits exactly one spike, at DRIVE_STEP.
    gen = SpikeGeneratorGroup(n, [driver], [DRIVE_STEP * REF.dt_ms] * ms)
    drive_syn = Synapses(gen, neu, "w : volt", on_pre="v += w")
    drive_syn.connect(i=[driver], j=[driver])
    drive_syn.w = [REF.w_scale_mv * REF.poisson_scale] * mV

    mon = SpikeMonitor(neu)
    state = StateMonitor(neu, ["v", "g"], record=list(range(n_real)))
    Network(neu, syn, gen, drive_syn, mon, state).run(DURATION_MS * ms)

    return {
        # brian2 quantities are in SI base units; convert volts -> mV explicitly, or
        # every comparison is silently offset by the 1000x unit factor.
        "v": np.asarray(state.v[:n_real] / mV).T,  # (n_steps, n_real) in mV
        "g": np.asarray(state.g[:n_real] / mV).T,
        "counts": np.array([len(mon.spike_trains().get(i, [])) for i in range(n)], dtype=np.int64),
        "times": np.array(
            [[float(t / ms) for t in mon.spike_trains().get(i, [])] for i in range(n)],
            dtype=object,
        ),
        "dt": REF.dt_ms,
    }


# --------------------------------------------------------------------- ours


def run_ours(case: str) -> dict:
    import torch

    n_real, edges, _ = CASES[case]
    n = n_real + 1
    driver = n_real

    sim = ConnectomeSim(REF, device="cpu", batch_size=1)
    sim.n_neurons = n
    sim._torch = torch
    sim._W = (
        torch.sparse_coo_tensor(
            torch.from_numpy(
                np.stack([[e[1] for e in edges], [e[0] for e in edges]]).astype(np.int64).copy()
            ),
            torch.from_numpy(np.array([e[2] for e in edges], dtype=np.float32).copy()),
            (n, n),
        )
        .coalesce()
        .to_sparse_csr()
    )
    sim.reset()

    n_steps = round(DURATION_MS / REF.dt_ms)
    v_rows, g_rows = [], []
    for step in range(n_steps):
        if step == DRIVE_STEP:
            # Same "immediate voltage step" the reference Poisson input performs.
            sim.inject(np.array([driver]), float(REF.w_scale_mv * REF.poisson_scale))
        sim.step(1)
        v_rows.append(sim._v[0, :n_real].numpy().copy())
        g_rows.append(sim._g[0, :n_real].numpy().copy())

    # Read the simulator's own spike tally. Detecting spikes by looking for
    # "v == v_reset and g == 0" is wrong: that is also the initial condition, so every
    # silent neuron is counted as having fired once.
    counts = sim._spike_counts_gpu[:n].detach().numpy().astype(np.int64)

    return {
        "v": np.asarray(v_rows),
        "g": np.asarray(g_rows),
        "counts": counts,
        "dt": REF.dt_ms,
    }


# ------------------------------------------------------------------ compare


def compare(ref: dict, ours: dict) -> bool:
    ok = True
    print(f"{'case':<16} {'quantity':<22} {'brian2':>12} {'ours':>12} {'verdict':>10}")
    print("-" * 78)
    for case, (_n_real, edges, desc) in CASES.items():
        rc = ref[case]
        oc = ours[case]

        # Analog gain: peak deflection of neuron 0 above rest.
        r_v = rc["v"][:, 0]
        o_v = oc["v"][:, 0]
        r_peak = float(r_v.max() - REF.v_rest_mv)
        o_peak = float(o_v.max() - REF.v_rest_mv)
        gain_ratio = o_peak / r_peak if r_peak > 1e-9 else float("nan")

        print(f"{case:<16} {'peak v - v_rest (mV)':<22} {r_peak:>12.4f} {o_peak:>12.4f} "
              f"{f'ratio {gain_ratio:.2f}':>10}")
        print(f"{'':<16} {'spike counts':<22} "
              f"{list(rc['counts'])!s:>12} {list(oc['counts'])!s:>12} "
              f"{'MATCH' if np.array_equal(rc['counts'], oc['counts']) else 'DIFFER':>10}")
        print(f"{'':<16} {'expected u_max (mV)':<22} "
              f"{analytic_peak(edges[0][2]):>12.4f}")
        print(f"{'':<16} {desc}")

        if not np.array_equal(rc["counts"], oc["counts"]):
            ok = False
        if r_peak > 1e-9 and abs(gain_ratio - 1.0) > 0.02:
            ok = False
        print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Micro-validation of the LIF integrator.")
    ap.add_argument("--mode", choices=["ref", "ours", "compare"], default="compare")
    ap.add_argument("--ref-out", default=str(DEFAULT_REF_OUT))
    ap.add_argument("--ours-out", default=str(DEFAULT_OURS_OUT))
    args = ap.parse_args()

    if args.mode == "ref":
        out = {c: run_brian2(c) for c in CASES}
        np.savez(args.ref_out, **{f"{c}__{k}": v for c, d in out.items() for k, v in d.items()},
                 cases=json.dumps(list(CASES)))
        print(f"wrote {args.ref_out}")
        return 0

    if args.mode == "ours":
        out = {c: run_ours(c) for c in CASES}
        np.savez(args.ours_out, **{f"{c}__{k}": v for c, d in out.items() for k, v in d.items()},
                 cases=json.dumps(list(CASES)))
        print(f"wrote {args.ours_out}")
        return 0

    ref_raw = np.load(args.ref_out, allow_pickle=True)
    ours_raw = np.load(args.ours_out, allow_pickle=True)
    cases = json.loads(str(ref_raw["cases"]))
    keys = ("v", "g", "counts", "dt")
    ref = {c: {k: ref_raw[f"{c}__{k}"] for k in keys if f"{c}__{k}" in ref_raw} for c in cases}
    ours = {c: {k: ours_raw[f"{c}__{k}"] for k in keys if f"{c}__{k}" in ours_raw} for c in cases}
    ok = compare(ref, ours)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
