"""Unit tests for the temperature -> light-colour loop.

These avoid loading the 138,639-neuron connectome: they cover the encoding curve, the
ideal mapping, the readout arithmetic and the train/eval split, which is where the logic
lives.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from flybrain.experiment import (
    COLOUR_BANDS,
    ColourReadout,
    ExperimentConfig,
    TemperatureColourLoop,
)


class _StubLoop(TemperatureColourLoop):
    """The real class with the connectome parts skipped, so the maths can be tested."""

    def __init__(self, config=None):
        self.config = config or ExperimentConfig()
        self.band_centres = np.array([k for _, k in COLOUR_BANDS], dtype=np.float64)
        self.band_names = [n for n, _ in COLOUR_BANDS]
        self._last_fit_temperatures = None


# ------------------------------------------------------------------ encoding


def test_rate_increases_with_temperature():
    loop = _StubLoop()
    c = loop.config
    cold = loop.rate_for_temperature(c.temp_min_c)
    warm = loop.rate_for_temperature(c.temp_max_c)
    # The cold end must stay above zero: a zero rate injects no current at all, leaving
    # the whole readout population silent and the decoder with no signal to read.
    assert cold == c.min_rate_hz
    assert cold > 0.0
    assert warm == c.max_rate_hz
    # Monotone across the range.
    temps = np.linspace(c.temp_min_c, c.temp_max_c, 25)
    rates = [loop.rate_for_temperature(t) for t in temps]
    assert all(b >= a for a, b in pairwise(rates))


def test_rate_is_clamped_outside_the_range():
    loop = _StubLoop()
    assert loop.rate_for_temperature(-100.0) == loop.config.min_rate_hz
    assert loop.rate_for_temperature(200.0) == loop.config.max_rate_hz


def test_ideal_kelvin_is_a_straight_ramp_over_the_band_range():
    loop = _StubLoop()
    c = loop.config
    lo, hi = COLOUR_BANDS[0][1], COLOUR_BANDS[-1][1]
    assert loop.ideal_kelvin(c.temp_min_c) == pytest.approx(lo)
    assert loop.ideal_kelvin(c.temp_max_c) == pytest.approx(hi)
    assert loop.ideal_kelvin((c.temp_min_c + c.temp_max_c) / 2) == pytest.approx((lo + hi) / 2)
    # Clamped outside the configured range.
    assert loop.ideal_kelvin(-100.0) == pytest.approx(lo)
    assert loop.ideal_kelvin(200.0) == pytest.approx(hi)


def test_band_centres_ascend_in_kelvin():
    """Bands are ordered coldest-looking first, so kelvin must ascend."""
    kelvin = [k for _, k in COLOUR_BANDS]
    assert kelvin == sorted(kelvin)
    assert all(k > 0 for k in kelvin)


def test_band_for_kelvin_picks_the_nearest_centre():
    loop = _StubLoop()
    centres = loop.band_centres
    assert loop.band_for_kelvin(centres[0]) == loop.band_names[0]
    assert loop.band_for_kelvin(centres[-1]) == loop.band_names[-1]
    assert loop.band_for_kelvin(centres[0] - 10_000) == loop.band_names[0]
    assert loop.band_for_kelvin(centres[-1] + 10_000) == loop.band_names[-1]


def test_evaluation_temperatures_do_not_overlap_training_temperatures():
    """`evaluate` must report held-out error, not in-sample fit quality."""
    loop = _StubLoop()
    train = loop.training_temperatures(21)
    loop._last_fit_temperatures = train
    evals = loop.evaluation_temperatures(9)
    assert evals.size == 9
    # Every evaluation temperature lies strictly between training points.
    assert not np.any(np.isclose(evals[:, None], train[None, :], atol=1e-9))
    assert evals.min() > train.min()
    assert evals.max() < train.max()


# ------------------------------------------------------------------ readout


def test_colour_readout_features_have_a_bias_and_check_length():
    r = ColourReadout(n_features=4)
    f = r.features(np.array([1.0, 2.0, 3.0]))
    assert f.shape == (4,)
    assert f[-1] == 1.0
    with pytest.raises(ValueError):
        r.features(np.array([1.0, 2.0]))


def test_colour_readout_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    true_w = np.array([10.0, -4.0, 2.5, 0.0, 7.0])
    y = X @ true_w + 5000.0
    r = ColourReadout(n_features=6, l2=1e-6)
    r.fit(X, y)
    pred = r.predict(X)
    assert np.mean(np.abs(pred - y)) < 1.0
    # A single row must give the same answer as a stack of one.
    assert r.predict(X[0])[0] == pytest.approx(pred[0])


def test_colour_readout_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=30) * 100 + 5000
    r = ColourReadout(n_features=4, l2=0.5)
    r.fit(X, y)
    path = r.save(tmp_path / "readout.npz")
    back = ColourReadout(n_features=1).load(path)
    assert back.n_features == r.n_features
    assert back.l2 == r.l2
    np.testing.assert_allclose(back.W, r.W)
    np.testing.assert_allclose(back.predict(X), r.predict(X))
