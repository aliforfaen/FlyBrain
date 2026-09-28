"""Connectome simulator: FlyWire v783 as a frozen LIF spiking network.

Neuron and synapse dynamics reproduce the model of Shiu et al., *Nature* (2024),
"A leaky integrate-and-fire computational model based on the connectome of the entire
adult Drosophila brain", as implemented by the reference project vendored at
``vendor/fly-brain``:

    dv/dt = (v0 - v + g) / tau_mem
    dg/dt = -g / tau_syn
    spike when v > v_th, then v = v_reset, g = 0, refractory for t_refrac

An alpha-function synapse is realised as an exponential conductance ``g`` whose input is
delayed by ``t_delay``. Every spike contributes ``w_scale * Excitatory x Connectivity``
(negative for inhibitory neurons), scaled by ``poisson_scale`` exactly as the reference
model scales its Poisson drive.

The network is *frozen*: this module never learns. It exists to turn external drive into
spike counts that a small readout can decode.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONNECTIVITY = ROOT / "vendor/fly-brain/data/2025_Connectivity_783.parquet"
DEFAULT_COMPLETENESS = ROOT / "vendor/fly-brain/data/2025_Completeness_783.csv"
DEFAULT_ANNOTATIONS = ROOT / "data/annotations/flywire_annotations_supl1.tsv"


@dataclass(frozen=True)
class ShiuParams:
    """Model constants from Shiu et al. (2024), matching the reference implementation."""

    dt_ms: float = 0.1
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    t_delay_ms: float = 1.8
    t_refrac_ms: float = 2.2
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_thresh_mv: float = -45.0
    w_scale_mv: float = 0.275
    #: scaling applied to the injected Poisson drive (the reference model's `f_poi`).
    poisson_scale: float = 250.0
    #: scaling applied to spikes travelling through the connectome. The reference
    #: implementation transmits recurrent spikes into `g` directly (`g += w`, with
    #: `w = synapse_count * 0.275 mV`), i.e. WITHOUT the Poisson factor. `poisson_scale`
    #: applies only to the external Poisson drive.
    #:
    #: NOTE: this is 1.0, i.e. the published model, with no fudge factor. It was
    #: previously 0.01 as a *compensation* for an integrator bug: the membrane update
    #: multiplied the conductance by an extra `tau_mem` (20x), so every synapse was 20
    #: times too strong and a scale of ~1/20 was needed to cancel it. With the update
    #: corrected the physical value is both correct and stable - see
    #: validation/micro_gain_check.py for the closed-form check and
    #: docs/research/simulation-backends.md for the history.
    recurrent_scale: float = 1.0
    #: added to every neuron every step; the reference injects Poisson noise into a
    #: designated population, we keep a small global floor for numerical liveliness.
    background_mv: float = 0.0

    @property
    def alpha(self) -> float:
        """Conductance decay factor per step."""
        return self.dt_ms / self.tau_syn_ms

    @property
    def mem_factor(self) -> float:
        """Membrane integration factor per step."""
        return self.dt_ms / self.tau_mem_ms

    @property
    def delay_steps(self) -> int:
        return int(self.t_delay_ms / self.dt_ms)


class ConnectomeSim:
    """The frozen whole-brain LIF network.

    Supports batching so that many trials (or many sensor conditions) simulate in
    parallel on the GPU. Batch element 0 is the "live" stream used for control.

    Parameters
    ----------
    params:
        Neuron/synapse constants. Defaults reproduce the published model.
    device:
        ``"cuda"``, ``"cpu"`` or ``None`` for auto-detection.
    batch_size:
        Number of parallel state copies.
    """

    def __init__(
        self,
        params: ShiuParams | None = None,
        device: str | None = None,
        batch_size: int = 1,
    ) -> None:
        self.params = params or ShiuParams()
        self.batch_size = batch_size
        self._torch = None
        self.n_neurons: int = 0
        self._n_steps = 0
        self._device = device
        self._W = None
        self._v = None
        self._g = None
        self._refrac = None
        self._delay_buf = None
        self._head = 0
        self._drive = None
        # The persistent drive as GPU tensors, built once in set_drive rather than per step:
        # the drive is re-applied on every step of every window, and rebuilding Python lists
        # plus two host->device copies for a few hundred indices 10,000 times per second of
        # brain time was pure host overhead in the hot loop.
        self._drive_idx = None
        self._drive_vals = None
        self._spike_counts: np.ndarray | None = None
        self._spike_counts_gpu = None
        self.annotation_table = None
        self.flywire_ids: np.ndarray | None = None
        self._id_to_index: dict[int, int] = {}

    # ------------------------------------------------------------------ setup

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def sim_time_ms(self) -> float:
        """Total simulated time advanced so far."""
        return self._n_steps * self.params.dt_ms

    def load(
        self,
        connectivity_path: str | Path = DEFAULT_CONNECTIVITY,
        completeness_path: str | Path = DEFAULT_COMPLETENESS,
        annotations_path: str | Path | None = DEFAULT_ANNOTATIONS,
        index_cache: str | Path | None = None,
    ) -> ConnectomeSim:
        """Load the connectome and build the sparse weight matrix.

        Weights are stored as ``w_scale * Excitatory x Connectivity``; the extra
        ``poisson_scale`` factor is applied to spike values at simulation time, matching
        the reference model's Poisson drive scaling.
        """
        import pandas as pd
        import torch

        self._torch = torch

        completeness = pd.read_csv(completeness_path, index_col=0)
        self.flywire_ids = completeness.index.to_numpy(np.int64)
        self.n_neurons = len(self.flywire_ids)
        self._id_to_index = {int(fid): i for i, fid in enumerate(self.flywire_ids)}

        conn = pd.read_parquet(
            connectivity_path,
            columns=[
                "Presynaptic_Index",
                "Postsynaptic_Index",
                "Excitatory x Connectivity",
            ],
        )
        pre = conn["Presynaptic_Index"].to_numpy(np.int64)
        post = conn["Postsynaptic_Index"].to_numpy(np.int64)
        weights = (
            conn["Excitatory x Connectivity"].to_numpy(np.float32) * self.params.w_scale_mv
        )
        # Index order is [POSTSYNAPTIC, PRESYNAPTIC]. `torch.sparse.mm(W, spikes)`
        # contracts over dim 1, so dim 1 must be the presynaptic axis for a spike
        # vector to be scattered to its targets. Getting this backwards yields an
        # all-zero result and silently disables every synapse in the network.
        indices = torch.from_numpy(
            np.stack([post, pre]).astype(np.int64).copy()
        )
        values = torch.from_numpy(weights.copy())
        coo = torch.sparse_coo_tensor(indices, values, (self.n_neurons, self.n_neurons))
        csr = coo.coalesce().to_sparse_csr()
        # 32-bit CSR indices: torch builds them as int64, but cuSPARSE accepts int32, and the
        # column indices are read for every synapse on every step. Halving that array halves
        # the dominant memory traffic of the step (measured here: 0.83 -> 0.77 ms/step), with
        # no numerical difference - the indices are the same integers.
        self._W = torch.sparse_csr_tensor(
            csr.crow_indices().to(torch.int32),
            csr.col_indices().to(torch.int32),
            csr.values(),
            csr.size(),
        ).to(self.device)
        logger.info(
            "connectome loaded: %d neurons, %d synapses, device=%s",
            self.n_neurons,
            values.numel(),
            self.device,
        )

        if annotations_path is not None and Path(annotations_path).exists():
            self.annotation_table = self._load_annotations(annotations_path, index_cache)

        self.reset()
        return self

    def _load_annotations(self, path: str | Path, cache: str | Path | None):
        """Join the public FlyWire annotation table onto connectome indices.

        Returns a DataFrame indexed by connectome index, so ``sim.annotation_table.loc[i]``
        describes neuron ``i``. The join is memoised to parquet because the TSV is 31 MB.
        """
        import pandas as pd

        cache_path = Path(cache) if cache else Path(path).with_suffix(".indexed.parquet")
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        ann = pd.read_csv(
            path,
            sep="\t",
            usecols=[
                "root_id",
                "flow",
                "super_class",
                "cell_class",
                "cell_sub_class",
                "supertype",
                "cell_type",
                "top_nt",
                "side",
            ],
        )
        ann["root_id"] = ann["root_id"].astype(np.int64)
        ann = ann.drop_duplicates(subset="root_id").set_index("root_id")
        table = ann.reindex(self.flywire_ids)
        table.index.name = "connectome_index"
        table = table.reset_index()
        try:
            table.to_parquet(cache_path, index=False)
        except Exception as exc:  # noqa: BLE001 - cache is an optimisation  # pragma: no cover
            logger.warning("could not cache annotations: %s", exc)
        return table.set_index("connectome_index")

    # ------------------------------------------------------------------ state

    def reset(self, batch_size: int | None = None) -> None:
        """Reset all neurons to rest and clear spike counts."""
        torch = self._torch
        if batch_size is not None:
            self.batch_size = batch_size
        p = self.params
        n, b = self.n_neurons, self.batch_size
        self._v = torch.full((b, n), p.v_rest_mv, device=self.device)
        self._g = torch.zeros((b, n), device=self.device)
        self._refrac = torch.zeros((b, n), device=self.device)
        # Exactly `delay_steps` slots: read at head, then overwrite that slot.
        self._delay_buf = torch.zeros((b, max(1, p.delay_steps), n), device=self.device)
        self._head = 0
        self._drive = torch.zeros((b, n), device=self.device)
        self._spike_counts_gpu = torch.zeros(n, dtype=torch.int64, device=self.device)
        self._spike_counts = np.zeros(n, dtype=np.int64)
        self._n_steps = 0

    def index_of_flywire_id(self, flywire_id: int) -> int:
        """Map a FlyWire root id to its connectome row index."""
        return self._id_to_index[int(flywire_id)]

    def indices_of_flywire_ids(self, ids) -> np.ndarray:
        return np.array([self._id_to_index[int(i)] for i in ids], dtype=np.int32)

    # ------------------------------------------------------------------ drive

    def set_drive(self, indices, current_mv: float | np.ndarray) -> None:
        """Set a persistent subthreshold current (mV) into ``indices``.

        The drive is applied every step until cleared with :meth:`clear_drive`. Like the
        reference model's ``voltage_stim``, it acts on the membrane potential immediately
        (not through the axonal delay line).
        """
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return
        torch = self._torch
        vals = np.broadcast_to(np.asarray(current_mv, dtype=np.float32), idx.shape)
        self._drive_idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        self._drive_vals = torch.as_tensor(
            np.ascontiguousarray(vals), dtype=torch.float32, device=self.device
        )

    def clear_drive(self) -> None:
        """Remove all persistent drive."""
        self._drive_idx = None
        self._drive_vals = None

    def inject(self, indices, current_mv: float | np.ndarray, batch: int = 0) -> None:
        """Add an immediate (undelayed) current for the next step only."""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return
        torch = self._torch
        vals = torch.as_tensor(
            np.broadcast_to(np.asarray(current_mv, dtype=np.float32), idx.shape),
            device=self.device,
        )
        self._drive[batch, torch.as_tensor(idx, device=self.device)] += vals

    def _schedule_persistent(self) -> None:
        """Queue the persistent drive to enter the delay line this step."""
        if self._drive_idx is None:
            return
        self._drive[:, self._drive_idx] += self._drive_vals

    # -------------------------------------------------------------- stepping

    def step(self, n_steps: int = 1) -> None:
        """Advance the network by ``n_steps`` timesteps.

        Uses the **exact** solution of the linear membrane/synapse system rather than a
        forward-Euler step, matching Brian2's ``linear`` method. Over one step with a
        constant conductance ``g``:

            g(t+dt) = g(t) * exp(-dt/tau_syn)
            v(t+dt) = v_rest + (v(t) - v_rest)*exp(-dt/tau_mem)
                            + g(t)*(1 - exp(-dt/tau_mem))

        The conductance coefficient is ``(1 - decay_v)`` (~0.005 for the published
        constants). Writing it as ``tau_mem * (1 - decay_v)`` (~0.0998) is the tempting
        mistake: it resembles a forward-Euler term scaled up, and it inflates **every
        synapse in the connectome by exactly tau_mem = 20x**. That single factor was the
        root cause of the long-standing "recurrent gain is provisional" problem - the
        network appeared to require a fudge factor of ~0.01, when in reality the
        integrator was inflating each synapse 20-fold and the fudge was cancelling it.
        It is pinned down by ``validation/micro_gain_check.py``, which compares the peak
        deflection produced by one synaptic event against the closed form
        ``u_max = 0.15749 * w`` for the published time constants.

        Refractoriness is an explicit countdown in milliseconds, as in Brian2: a neuron
        that fires cannot fire again until ``t_refrac`` has elapsed. Spiking also clears
        the neuron's conductance, as in the reference model.
        """
        import math

        torch = self._torch
        p = self.params
        decay_g = math.exp(-p.dt_ms / p.tau_syn_ms)
        decay_v = math.exp(-p.dt_ms / p.tau_mem_ms)
        refractory_ms = float(p.t_refrac_ms)
        # Exact conductance coefficient for the coupled 2-D (v, g) system over one step.
        # Solving dv/dt = (v_rest - v + g)/tau_mem together with dg/dt = -g/tau_syn gives
        # the double-exponential (alpha) kernel, whose per-step weight is
        #     tau_syn/(tau_mem - tau_syn) * (exp(-dt/tau_mem) - exp(-dt/tau_syn))
        # = 0.004938 mV per mV of g at dt=0.1 ms. `(1 - decay_v)` = 0.004988 is the
        # constant-g approximation of the same quantity: only 1% larger, but 1% is
        # measurable, so we use the exact form. `fly-brain-minecraft` writes this same
        # coefficient as `(a-b)/3` with a=exp(-dt/20), b=exp(-dt/5).
        alpha = p.tau_syn_ms / (p.tau_mem_ms - p.tau_syn_ms) * (decay_v - decay_g)

        for _ in range(n_steps):
            # External drive reaches the membrane directly, mirroring the reference
            # model's ``voltage_stim`` term. It is not subject to the axonal delay.
            self._schedule_persistent()
            voltage_stim = self._drive

            # Conductance: exponential decay plus the input that traversed the delay line.
            # Input arriving while a neuron is refractory is suppressed, matching the
            # reference model's `delay_buffer * refractory` term.
            g = self._g * decay_g + self._delay_buf[:, self._head, :] * (self._refrac <= 0.0)

            # Exact membrane integration, then threshold handling.
            #
            #   v(t+dt) = v_rest + (v(t) + voltage_stim - v_rest)*exp(-dt/tau_mem)
            #                   + g(t)*alpha
            #
            # The trap here is the conductance coefficient. Writing it as
            # `tau_mem * (1 - decay_v)` looks like a scaled forward-Euler term and makes
            # every synapse in the connectome exactly 20x too strong - that single factor
            # was the real cause of the long-standing "recurrent gain" problem, and the
            # bogus `recurrent_scale` of 0.01 was cancelling it.
            v = (
                p.v_rest_mv
                + (self._v + voltage_stim - p.v_rest_mv) * decay_v
                + g * alpha
            )
            # A neuron is refractory for `t_refrac` ms after firing. brian2's `unless
            # refractory` clause freezes the membrane, so a blocked neuron is returned to
            # rest instead of banking voltage it can never discharge.
            blocked = self._refrac > 0.0
            above = v > p.v_thresh_mv
            spikes = (above & ~blocked).to(torch.float32)
            self._v = torch.where(above, p.v_reset_mv, v)
            # A spike clears the neuron's synaptic conductance (reference behaviour).
            self._g = g - g * spikes

            # Refractory countdown in ms, reloaded by a spike.
            self._refrac = torch.clamp(self._refrac - p.dt_ms, min=0.0)
            self._refrac = torch.maximum(self._refrac, spikes * refractory_ms)

            # Emitted spikes are written into the ring slot we just read, so a value
            # written on step t is read on step t + delay_steps. `torch.roll` cannot be
            # used here: rolling a `delay + 1` slot buffer shifts a freshly written
            # spike out of the ring before it is ever read, silently disabling every
            # synapse in the network.
            self._delay_buf[:, self._head, :] = torch.sparse.mm(
                self._W, (spikes * p.recurrent_scale).t()
            ).t()
            self._head = (self._head + 1) % self._delay_buf.shape[1]

            self._spike_counts_gpu += spikes.sum(dim=0).to(torch.int64)
            # Reused in place. `zeros_like` here allocated a fresh 550 KB every step and made
            # the caching allocator churn; the drive is always fully re-scheduled or zero, so
            # zeroing in place is equivalent and allocation-free.
            self._drive.zero_()
            self._n_steps += 1

    def run(self, duration_ms: float) -> None:
        """Advance the network by ``duration_ms`` of simulated time."""
        self.step(max(1, round(duration_ms / self.params.dt_ms)))

    # ------------------------------------------------------------- read-out

    def spike_counts(self, indices=None, reset: bool = True) -> np.ndarray:
        """Cumulative spike counts since the last call, optionally restricted to ``indices``."""
        counts = self._spike_counts_gpu
        if counts is None:
            return np.zeros(0, dtype=np.int64)
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        total = counts.detach().to("cpu").numpy()
        if reset:
            self._spike_counts_gpu = self._torch.zeros(
                self.n_neurons, dtype=self._torch.int64, device=self.device
            )
        if indices is None:
            return total
        idx = np.asarray(indices, dtype=np.int64)
        return total[idx]

    def rates_hz(self, indices=None, reset: bool = True) -> np.ndarray:
        """Mean firing rate (Hz) since the last call over the elapsed simulated time."""
        elapsed = self.sim_time_ms if self._n_steps else self.params.dt_ms
        counts = self.spike_counts(indices=indices, reset=False)
        rates = counts / (elapsed / 1000.0)
        if reset:
            self.spike_counts(reset=True)
        return rates

    def spike_counts_dict(self, indices) -> dict[int, int]:
        """Spike counts as a ``{connectome_index: count}`` mapping for the decoder."""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return {}
        counts = self.spike_counts(indices=idx, reset=False)
        return {int(i): int(c) for i, c in zip(idx, counts) if c > 0}

    def active_neurons(self, threshold: int = 1) -> int:
        """Number of neurons that spiked at least ``threshold`` times this window."""
        counts = self._spike_counts_gpu
        if counts is None:
            return 0
        return int((counts >= threshold).sum().item())

    # ------------------------------------------------------------- utilities

    def summary(self) -> dict:
        return {
            "n_neurons": self.n_neurons,
            "n_synapses": int(self._W.values().numel()) if self._W is not None else 0,
            "device": self.device,
            "engine": type(self).__name__,
            "batch_size": self.batch_size,
            "sim_time_ms": self.sim_time_ms,
            "annotations": self.annotation_table is not None,
        }

    def save_index_map(self, path: str | Path) -> None:
        """Persist the flywire-id -> index mapping for other tools."""
        Path(path).write_text(json.dumps(self._id_to_index))


class ActiveSetSim(ConnectomeSim):
    """The same network and the same exact integrator, run only on what is awake.

    The dense engine steps all 138,639 neurons and scans all ~15M synapses every 0.1 ms,
    whether or not anything is happening. At any moment the overwhelming majority of the
    brain is exactly at rest with zero conductance and no input - and for those neurons the
    step is the identity: ``v`` stays ``v_rest``, ``g`` stays 0, no spike is possible. This
    class makes that identity explicit. It is the design the ``fly-brain-minecraft`` project
    runs at real time on CPU (see ``docs/engine.md``), adapted to the tensor operations we
    already have.

    A neuron is **active** on a step when it could deviate from rest: its membrane is away
    from ``v_rest``, its conductance is above a vanishing floor, it is refractory, it is
    receiving drive, or it has **pending** delayed input - conductance written into its delay
    slot that has not been read yet. The pending counter is what keeps a neuron in the active
    set for the full ``delay_steps`` between a presynaptic spike and the input landing;
    without it, a resting target would be dropped and the spike silently lost.

    Exactness, stated precisely rather than assumed:

    * For a neuron at rest with ``g = 0``, no drive and no pending input, the update
      computes ``v_rest + 0 * decay + 0 * alpha = v_rest`` - skipping it changes nothing.
    * Delivery sums only the contributions of spiking presynaptic neurons. The dense
      matmul sums the same contributions plus exact zeros (`0.0 * w`), and adding a zero
      is exact in floating point, so the sums agree - up to the *order* of addition.
      On CPU the order matches the matmul's and results are bitwise identical; on CUDA
      ``index_add_`` uses atomics, so sums can differ by ~1 ULP and a threshold crossing
      sitting exactly on that knife edge can flip. The Brian2 harness result for each
      engine is recorded separately for exactly this reason.
    * The conductance floor ``G_FLOOR`` bounds the error of dropping a neuron: below it,
      ``g * alpha`` is ~1e-33 mV, forty orders of magnitude under one ULP of ``v``. Without
      a floor, ``g`` decays asymptotically and never reaches exactly zero, which would pin
      every neuron that ever received a spike into the active set for hundreds of steps.

    Cost per step is proportional to the number of *active* neurons rather than to the
    network, plus one dense 138k-element mask pass and two device synchronisations (the
    active list and the delivery size are data-dependent). **Measured on the RTX 3070 at
    full connectome, that trade loses**: the syncs and ~40 small kernel launches per step
    cost a fixed ~0.4 ms/step of host time, which is most of what the dense engine spends
    on the matmul it eliminates. Quiet (zero drive) it is ~1.15x faster than dense; under
    real drive it is ~1.6x slower, with spike output bitwise identical. The win here is
    exactness plus the bookkeeping pattern, not speed: getting to real time on GPU needs
    compiled per-step code (CUDA graphs cannot capture the dynamic shapes; GeNN can), which
    is the fallback docs/engine.md already names. Kept opt-in via ``FLYBRAIN_ENGINE=active``
    so the measured numbers stay reproducible; ``dense`` remains the default everywhere.

    Batch size must be 1: the live loop runs one stream, and delivery is per-batch by
    construction. Offline training that wants batches keeps using :class:`ConnectomeSim`.
    """

    #: Conductances below this are treated as zero. See the exactness note above.
    G_FLOOR = 1e-30

    def __init__(self, params: ShiuParams | None = None, device: str | None = None) -> None:
        super().__init__(params=params, device=device, batch_size=1)
        self._j_pre = None
        self._j_post = None
        self._j_w = None
        self._j_ptr = None
        self._pending = None

    # ------------------------------------------------------------------ setup

    def load(self, *args, **kwargs) -> ActiveSetSim:
        """Load as usual, then build the presynaptic (CSC-order) delivery journal.

        The dense engine's weight matrix is CSR with rows indexed by *postsynaptic* neuron,
        which is the wrong axis for delivery: a spike needs the *out-edges* of the spiking
        neuron. The journal is the same synapses reordered by presynaptic index with an
        offset pointer per neuron, so delivering a spike means copying one contiguous range
        per spiking neuron instead of scanning all 15M rows.
        """
        super().load(*args, **kwargs)
        self._build_journal()
        return self

    def _build_journal(self) -> None:
        import torch

        n = self.n_neurons
        device = self.device
        csr = self._W.cpu()
        crow = csr.crow_indices().to(torch.int64)
        col = csr.col_indices().to(torch.int64)
        # Expand the CSR row pointer into a post id per synapse, then reorder everything by
        # presynaptic id. `stable=True` keeps each pre-neuron's edges in post order, which
        # keeps delivery order deterministic.
        post = torch.repeat_interleave(
            torch.arange(n, dtype=torch.int64), crow[1:] - crow[:-1]
        )
        order = torch.sort(col, stable=True).indices
        self._j_pre = col[order].to(torch.int32).to(device)
        self._j_post = post[order].to(torch.int32).to(device)
        self._j_w = csr.values()[order].to(device)
        counts = torch.bincount(col, minlength=n)
        self._j_ptr = torch.zeros(n + 1, dtype=torch.int64, device=device)
        self._j_ptr[1:] = counts.cumsum(0)
        self._max_fanout = int(counts.max().item())
        import logging

        logging.getLogger(__name__).info(
            "delivery journal built: %d synapses, max fanout %d",
            int(csr.values().numel()),
            self._max_fanout,
        )

    def reset(self, batch_size: int | None = None) -> None:
        import torch

        if batch_size is not None and batch_size != 1:
            raise ValueError("ActiveSetSim runs a single stream; use ConnectomeSim for batches")
        super().reset(batch_size=1)
        self._pending = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)

    # -------------------------------------------------------------- stepping

    def step(self, n_steps: int = 1) -> None:
        """Advance by ``n_steps`` timesteps, integrating and delivering only active neurons.

        The per-step math is the same exact double-exponential solution as the dense engine
        (see :meth:`ConnectomeSim.step` for the derivation and the traps). The differences
        are bookkeeping: a dense mask finds the active set, delivery copies journal ranges
        for spiking neurons, and the pending counter carries "input is in flight" between a
        delivery and its read `delay_steps` later.
        """
        import math

        import torch

        p = self.params
        device = self.device
        decay_g = math.exp(-p.dt_ms / p.tau_syn_ms)
        decay_v = math.exp(-p.dt_ms / p.tau_mem_ms)
        alpha = p.tau_syn_ms / (p.tau_mem_ms - p.tau_syn_ms) * (decay_v - decay_g)
        refractory_ms = float(p.t_refrac_ms)

        for _ in range(n_steps):
            self._schedule_persistent()

            # ---- the active set: anyone who could deviate from rest this step
            # Conductance is signed (inhibitory is negative), so the floor test is on its
            # magnitude - `g > floor` would drop every neuron holding only inhibition, and
            # the dropped g would stop decaying: divergence found by the equivalence test.
            mask = (
                (self._v[0] != p.v_rest_mv)
                | (self._g[0].abs() > self.G_FLOOR)
                | (self._refrac[0] > 0.0)
                | (self._drive[0] != 0.0)
                | (self._pending != 0.0)
            )
            act = mask.nonzero().flatten()  # data-dependent; one device sync
            v_a = self._v[0].index_select(0, act)
            g_a = self._g[0].index_select(0, act)
            r_a = self._refrac[0].index_select(0, act)
            d_a = self._drive[0].index_select(0, act)
            in_a = self._delay_buf[0, self._head].index_select(0, act)
            # The slot is consumed whether or not the neuron is refractory (a refractory
            # neuron's input is suppressed below, exactly like the reference model's
            # `delay_buffer * refractory` term - and exactly as destroyed).
            self._pending.index_add_(0, act, -(in_a != 0.0).to(torch.float32))

            # ---- the same exact update, restricted to the active rows
            g_new = g_a * decay_g + in_a * (r_a <= 0.0).to(torch.float32)
            v_new = p.v_rest_mv + (v_a + d_a - p.v_rest_mv) * decay_v + g_new * alpha
            above = v_new > p.v_thresh_mv
            blocked = r_a > 0.0
            spikes = (above & ~blocked).to(torch.float32)
            self._v[0, act] = torch.where(above, p.v_reset_mv, v_new)
            self._g[0, act] = g_new - g_new * spikes
            r_new = torch.clamp(r_a - p.dt_ms, min=0.0)
            self._refrac[0, act] = torch.maximum(r_new, spikes * refractory_ms)

            self._spike_counts_gpu.index_add_(0, act, spikes.to(torch.int64))

            # ---- delivery: copy each spiking neuron's journal range into the slot
            self._delay_buf[0, self._head].zero_()
            spiking = (spikes > 0).nonzero().flatten()  # data-dependent; one device sync
            k = spiking.numel()
            if k:
                spike_pre = act.index_select(0, spiking)
                spike_val = spikes.index_select(0, spiking) * p.recurrent_scale
                lengths = self._j_ptr[spike_pre.to(torch.int64) + 1] - self._j_ptr[
                    spike_pre.to(torch.int64)
                ]
                total = int(lengths.sum().item())  # data-dependent; one device sync
                if total:
                    which = torch.repeat_interleave(
                        torch.arange(k, device=device), lengths, output_size=total
                    )
                    starts = self._j_ptr.index_select(0, spike_pre.to(torch.int64))
                    base = torch.repeat_interleave(starts, lengths, output_size=total)
                    within = torch.arange(total, device=device) - torch.repeat_interleave(
                        lengths.cumsum(0) - lengths, lengths, output_size=total
                    )
                    syn = base + within
                    tgt = self._j_post[syn].to(torch.int64)
                    w = self._j_w[syn] * spike_val.index_select(0, which)
                    slot = self._delay_buf[0, self._head]
                    slot.index_add_(0, tgt, w)
                    # Input is now in flight toward these neurons; they stay active until the
                    # slot is read, however many delay steps that takes.
                    self._pending += (slot != 0.0).to(torch.float32)

            self._drive.zero_()
            self._head = (self._head + 1) % self._delay_buf.shape[1]
            self._n_steps += 1


def make_sim(engine: str = "dense", **kwargs) -> ConnectomeSim:
    """Build a simulator engine by name: ``dense`` (reference) or ``active`` (real-time path).

    Both run the same model and the same integrator; see :class:`ActiveSetSim` for what
    differs. ``dense`` remains the default everywhere so that the validated reference
    behaviour is what runs unless the operator asks for speed.
    """
    name = (engine or "dense").strip().lower()
    if name == "dense":
        return ConnectomeSim(**kwargs)
    if name == "active":
        return ActiveSetSim(**kwargs)
    raise ValueError(f"unknown engine {engine!r} (expected 'dense' or 'active')")
