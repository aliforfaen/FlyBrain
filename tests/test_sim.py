"""Regression tests for the connectome simulator.

Every test here corresponds to a bug that was found the hard way. All of them were
*silent* failures: the simulator ran, produced plausible-looking output, and was simply
wrong. They are cheap to check and expensive to rediscover.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flybrain.sim import ConnectomeSim, ShiuParams

pytestmark = pytest.mark.filterwarnings("ignore")


def tiny_chain(weights=(100.0, 100.0), n_neurons: int = 3) -> ConnectomeSim:
    """A minimal 0 -> 1 -> 2 chain with explicit weights."""
    sim = ConnectomeSim(ShiuParams(), device="cpu", batch_size=1)
    sim.n_neurons = n_neurons
    sim._torch = torch
    pre = torch.tensor([0, 1][: len(weights)])
    post = torch.tensor([1, 2][: len(weights)])
    # Index order is [POSTSYNAPTIC, PRESYNAPTIC]: W[post, pre] is the synapse pre -> post.
    sim._W = (
        torch.sparse_coo_tensor(
            torch.stack([post, pre]), torch.tensor(list(weights), dtype=torch.float32),
            (n_neurons, n_neurons),
        )
        .coalesce()
        .to_sparse_csr()
    )
    sim.reset()
    return sim


def test_sparse_matrix_orientation_scatters_to_targets():
    """`torch.sparse.mm(W, spikes)` contracts dim 1, so dim 1 must be the presynaptic axis.

    With the order reversed the multiplication returns all zeros and every synapse in the
    network is silently disabled.
    """
    sim = tiny_chain(weights=(100.0,))
    spikes = torch.zeros(3)
    spikes[0] = 1.0
    out = torch.sparse.mm(sim._W, spikes.unsqueeze(1)).squeeze(1)
    assert out[1].item() == pytest.approx(100.0)
    assert out[0].item() == 0.0


def test_spikes_propagate_across_the_chain():
    """A driven neuron must be able to drive its targets. This failed completely when the
    delay line dropped freshly written spikes.

    Weights are chosen relative to `recurrent_scale`: with the scale at 1/250 a single
    strong synapse still needs to deliver enough conductance to cross threshold."""
    syn = 100.0 / ShiuParams().recurrent_scale
    sim = tiny_chain(weights=(syn, syn))
    for _ in range(120):
        sim.inject(np.array([0]), 68.75)
        sim.step(1)
    counts = sim.spike_counts(reset=False)
    assert counts[1] > 0, "neuron 1 never fired: synaptic transmission is broken"
    assert counts[2] > 0, "neuron 2 never fired: transmission is more than one hop broken"


def test_delay_line_delivers_after_delay_steps():
    """The ring buffer must deliver input exactly `delay_steps` after the spike."""
    sim = tiny_chain(weights=(100.0,))
    sim.inject(np.array([0]), 68.75)
    sim.step(1)
    # The spike is recorded in the slot at the current head, then the head advances.
    assert sim._delay_buf[0, sim._head - 1].abs().sum().item() > 0

    # Over the next `delay_steps - 1` steps the conductance must stay at rest, because
    # the input has not arrived yet; then it must fire.
    resting = True
    for step in range(1, ShiuParams().delay_steps):
        sim.step(1)
        if sim._g[0, 1].item() > 0:
            resting = False
    assert resting, "input arrived before the axonal delay elapsed"


def test_membrane_voltage_stays_in_a_physiological_range():
    """A refractory neuron must not integrate without bound.

    Before the fix, a blocked neuron accumulated 68.75 mV per step and reached thousands
    of millivolts.
    """
    sim = tiny_chain(weights=(0.275,))
    for _ in range(200):
        sim.inject(np.array([0]), 68.75)
        sim.step(1)
    peak = float(sim._v.max())
    assert peak <= ShiuParams().v_thresh_mv + 1e-3, f"voltage escaped to {peak} mV"


def test_single_synapse_does_not_trigger_a_spike():
    """The exact integrator must scale the conductance term correctly.

    When the coefficients were missing the `tau_mem` factor, a single 0.275 mV synapse
    contributed ~200x too little; with the wrong orientation it contributed everything.
    A single weak synapse should depolarise slightly but not fire.
    """
    sim = tiny_chain(weights=(0.275,))
    # Deliver exactly one presynaptic spike and let it traverse the delay.
    sim.inject(np.array([0]), 68.75)
    sim.step(1)
    for _ in range(ShiuParams().delay_steps + 2):
        sim.step(1)
    assert float(sim._v[0, 1]) > ShiuParams().v_rest_mv, "synapse had no effect at all"
    assert float(sim._v[0, 1]) < ShiuParams().v_thresh_mv, "one weak synapse caused a spike"


def test_recurrent_scale_is_not_the_poisson_scale():
    """`poisson_scale` applies to the injected drive, not to recurrent transmission.

    Using the Poisson factor for recurrent spikes inflates every synapse 250x, which
    makes the network run away; the reference transmits `w` directly into `g`.
    """
    params = ShiuParams()
    assert params.recurrent_scale != params.poisson_scale
    assert params.poisson_scale == 250.0


def test_reset_clears_ring_head_and_counts():
    sim = tiny_chain()
    sim.inject(np.array([0]), 68.75)
    sim.step(3)
    sim.reset()
    assert sim._head == 0
    assert sim._delay_buf.abs().sum().item() == 0
    assert sim._g.abs().sum().item() == 0
    assert sim.spike_counts(reset=False).sum() == 0
    assert sim.sim_time_ms == 0


class TestActivityDriver:
    """Regressions for the live-view driver.

    Both bugs here were silent: the drive appeared to clear while the network kept firing,
    and the usage panel appeared to work while returning an empty list nearly every time.
    """

    def _driver(self):
        from flybrain.activity import ActivitySettings, MemoryBrain

        sim = tiny_chain(weights=(0.0,))
        sim.n_neurons = 3
        sim.annotation_table = None
        return MemoryBrain(sim, ActivitySettings(window_ms=10.0, background_drive_mv=0.0))

    def test_clearing_the_drive_stops_the_network(self):
        """`clear_input_drive` must reach the simulator, not just the local driver."""
        driver = self._driver()
        driver.set_input_drive(np.array([0]), 20.0)
        assert driver.advance().total_spikes > 0

        driver.clear_input_drive()
        assert driver.sim._persistent_drive == {}, "simulator still holds a drive"
        # The first window may still contain spikes queued before the clear.
        for _ in range(3):
            driver.advance()
        assert driver.advance().total_spikes == 0, "network kept firing after clearing"

    def test_setting_an_empty_population_clears_instead(self):
        driver = self._driver()
        driver.set_input_drive(np.array([0]), 20.0)
        driver.advance()
        driver.set_input_drive(np.array([], dtype=np.int64), 20.0)
        assert driver.sim._persistent_drive == {}

    def test_region_usage_reads_the_cached_window(self):
        """It must not read the simulator tensor that the advance loop has already reset."""
        driver = self._driver()
        # A single cell class over three neurons, so a spiking cell shows up.
        import pandas as pd

        driver.sim.annotation_table = pd.DataFrame(
            {"cell_class": ["visual", "visual", "unclassified"]}
        )
        assert driver.region_usage() == [], "no window yet should yield nothing"

        driver.set_input_drive(np.array([0]), 20.0)
        driver.advance()
        regions = driver.region_usage()
        assert regions, "region_usage returned nothing after a spiking window"
        assert sum(r["spikes"] for r in regions) > 0
        assert all("rate_hz" in r for r in regions)
