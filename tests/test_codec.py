"""Tests for :mod:`flybrain.codec` and :mod:`flybrain.learn`.

These tests use only plain arrays and the shared dataclasses: no simulator,
no Home Assistant, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.codec import ChannelSpec, SpikeDecoder, SpikeEncoder
from flybrain.learn import ReadoutLearner
from flybrain.types import Episode, Signal, SignalKind, SpikeTrain

ENTITY = "sensor.living_room_temperature"
NEURONS = (10, 11, 12)
ACTION_MAP = {
    "light.kitchen.turn_on": [100, 101],
    "light.kitchen.turn_off": [102],
    "climate.hall.set_temperature": [103, 104, 105],
}


def make_channel(
    entity_id: str = ENTITY,
    indices: tuple[int, ...] = NEURONS,
    vmin: float = 0.0,
    vmax: float = 100.0,
    gain: float = 1.0,
) -> ChannelSpec:
    return ChannelSpec(
        entity_id=entity_id,
        kind=SignalKind.TEMPERATURE,
        neuron_indices=np.asarray(indices, dtype=np.int32),
        vmin=vmin,
        vmax=vmax,
        gain=gain,
    )


def make_signal(entity_id: str, value: float) -> Signal:
    return Signal(
        entity_id=entity_id,
        kind=SignalKind.TEMPERATURE,
        value=value,
        state=str(value),
        timestamp=0.0,
        unit="C",
        attributes={},
    )


# --------------------------------------------------------------------------- #
# ChannelSpec
# --------------------------------------------------------------------------- #
def test_channel_spec_from_config_parses_plain_dict() -> None:
    spec = ChannelSpec.from_config(
        {
            "entity_id": "sensor.outdoor_temperature",
            "kind": "temperature",
            "neuron_indices": [5, 6, 7],
            "vmin": -10.0,
            "vmax": 40.0,
            "gain": 2.0,
        }
    )
    assert spec.entity_id == "sensor.outdoor_temperature"
    assert spec.kind is SignalKind.TEMPERATURE
    assert spec.neuron_indices.dtype == np.int32
    np.testing.assert_array_equal(spec.neuron_indices, [5, 6, 7])
    assert spec.vmin == -10.0
    assert spec.vmax == 40.0
    assert spec.gain == 2.0


def test_channel_spec_equality_is_value_based() -> None:
    assert make_channel() == make_channel()
    assert make_channel() != make_channel(entity_id="sensor.other")


# --------------------------------------------------------------------------- #
# SpikeEncoder
# --------------------------------------------------------------------------- #
def test_encode_is_deterministic_for_fixed_seed() -> None:
    channel = make_channel()
    signals = [make_signal(ENTITY, 60.0)]
    first = SpikeEncoder([channel], dt_ms=10.0, seed=42).encode(signals, 500.0)[ENTITY]
    second = SpikeEncoder([channel], dt_ms=10.0, seed=42).encode(signals, 500.0)[ENTITY]
    np.testing.assert_array_equal(first.spike_times_ms, second.spike_times_ms)
    np.testing.assert_array_equal(first.rate_hz, second.rate_hz)

    # A second call on the same encoder must repeat itself too.
    encoder = SpikeEncoder([channel], dt_ms=10.0, seed=42)
    again = encoder.encode(signals, 500.0)[ENTITY]
    np.testing.assert_array_equal(first.spike_times_ms, again.spike_times_ms)


def test_encode_missing_signal_produces_zero_spikes() -> None:
    encoder = SpikeEncoder([make_channel()], dt_ms=10.0, seed=1)
    train = encoder.encode([make_signal("sensor.something_else", 50.0)], 500.0)[ENTITY]
    assert train.spike_times_ms.shape == (0,)
    assert train.rate_hz.shape == (len(NEURONS),)
    assert float(train.rate_hz.sum()) == 0.0
    assert train.neuron_indices.dtype == np.int32
    assert train.spike_times_ms.dtype == np.float32


def test_higher_sensor_value_increases_firing_rate() -> None:
    channel = make_channel(vmax=100.0)
    counts: list[int] = []
    mean_rates: list[float] = []
    for value in (0.0, 25.0, 50.0, 75.0, 100.0):
        encoder = SpikeEncoder([channel], dt_ms=10.0, seed=7)
        train = encoder.encode([make_signal(ENTITY, value)], 1000.0)[ENTITY]
        counts.append(int(train.spike_times_ms.size))
        mean_rates.append(float(train.rate_hz.mean()))

    assert counts == sorted(counts), counts
    assert mean_rates == sorted(mean_rates), mean_rates
    assert counts[0] == 0
    assert counts[-1] > counts[1] > 0
    # 100% of range -> 200 Hz for 1 s -> roughly 200 spikes.
    assert 100 <= counts[-1] <= 300


def test_encode_clips_values_outside_configured_range() -> None:
    channel = make_channel(vmin=0.0, vmax=100.0)
    encoder = SpikeEncoder([channel], dt_ms=10.0, seed=5)
    below = encoder.encode([make_signal(ENTITY, -500.0)], 500.0)[ENTITY]
    above = encoder.encode([make_signal(ENTITY, 500.0)], 500.0)[ENTITY]
    assert encoder.rate_for(channel, -500.0)[1] == 0.0
    assert encoder.rate_for(channel, 500.0)[1] == encoder.max_rate_hz
    assert float(below.rate_hz.sum()) == 0.0
    assert float(above.rate_hz.sum()) > 0.0


def test_spike_times_stay_inside_window_and_use_channel_neurons() -> None:
    channel = make_channel()
    encoder = SpikeEncoder([channel], dt_ms=10.0, seed=11)
    train = encoder.encode([make_signal(ENTITY, 100.0)], 500.0)[ENTITY]
    assert train.spike_times_ms.size > 0
    assert float(train.spike_times_ms.min()) >= 0.0
    assert float(train.spike_times_ms.max()) <= 500.0
    assert np.all(np.diff(train.spike_times_ms) >= 0.0)
    assert set(train.neuron_indices.tolist()) == set(NEURONS)


def test_to_current_shape_dtype_and_monotonicity() -> None:
    channel = make_channel()
    encoder = SpikeEncoder([channel], dt_ms=10.0, seed=3)
    low = encoder.encode([make_signal(ENTITY, 10.0)], 500.0)[ENTITY]
    high = encoder.encode([make_signal(ENTITY, 90.0)], 500.0)[ENTITY]

    idx_low, cur_low = encoder.to_current(low, 500.0, 1.5)
    _, cur_high = encoder.to_current(high, 500.0, 1.5)

    assert idx_low.shape == cur_low.shape == (len(NEURONS),)
    assert idx_low.dtype == np.int32
    assert cur_low.dtype == np.float32
    assert set(idx_low.tolist()) <= set(channel.neuron_indices.tolist())
    assert np.all(cur_high >= cur_low)
    assert float(cur_high.sum()) > float(cur_low.sum())


def test_to_current_is_linear_in_rate() -> None:
    encoder = SpikeEncoder([make_channel()], dt_ms=10.0, seed=0)
    train = SpikeTrain(
        neuron_indices=np.asarray([1, 2], dtype=np.int32),
        spike_times_ms=np.zeros(0, dtype=np.float32),
        rate_hz=np.asarray([0.0, 100.0], dtype=np.float32),
    )
    # duration 1000 ms at dt 10 ms -> 100 steps; 100 Hz * 1 s = 100 spikes.
    _, current = encoder.to_current(train, 1000.0, 1.0)
    np.testing.assert_allclose(current, [0.0, 1.0], rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# SpikeDecoder
# --------------------------------------------------------------------------- #
def test_decoder_feature_layout_is_stable() -> None:
    decoder = SpikeDecoder(ACTION_MAP, window_ms=1000.0)
    assert decoder.action_keys == sorted(ACTION_MAP)
    assert decoder.n_actions == 3
    assert decoder.n_features == 4

    counts = {100: 3, 101: 5, 102: 0, 103: 0, 104: 0, 105: 0}
    feats = decoder.features(counts, 1000.0)
    assert feats.shape == (4,)
    assert feats[-1] == 1.0
    # Mean count of group [100, 101] is (3 + 5) / 2 = 4 -> 4 Hz over 1 s.
    on_index = decoder.action_keys.index("light.kitchen.turn_on")
    assert feats[on_index] == 4.0
    assert feats[decoder.action_keys.index("light.kitchen.turn_off")] == 0.0

    # Feature order is unchanged by the counts passed in.
    other = decoder.features({103: 6}, 1000.0)
    assert float(other[decoder.action_keys.index("climate.hall.set_temperature")]) == 2.0


def test_decoder_feature_scale_divides_rate_features_only() -> None:
    decoder = SpikeDecoder(ACTION_MAP, feature_scale=2.0)
    feats = decoder.features({100: 4, 101: 4}, 1000.0)
    on_index = decoder.action_keys.index("light.kitchen.turn_on")
    assert feats[on_index] == 2.0
    assert feats[-1] == 1.0


def test_untrained_decoder_emits_no_actions() -> None:
    decoder = SpikeDecoder(ACTION_MAP)
    command = decoder.decode({100: 5, 102: 7}, 500.0)
    assert command.actions == []
    assert set(command.logits) == set(ACTION_MAP)
    assert all(score == 0.0 for score in command.logits.values())
    assert command.spike_counts == {100: 5, 102: 7}
    assert command.window_ms == 500.0


def test_decoder_with_weights_emits_only_the_favoured_action() -> None:
    decoder = SpikeDecoder(ACTION_MAP, threshold=0.5)
    weights = np.zeros((decoder.n_actions, decoder.n_features), dtype=np.float64)
    favored = "light.kitchen.turn_on"
    weights[decoder.action_keys.index(favored), -1] = 5.0
    decoder.set_weights(weights)

    command = decoder.decode({}, 500.0)
    assert len(command.actions) == 1
    action = command.actions[0]
    assert action.key == favored
    assert action.entity_id == "light.kitchen"
    assert action.service == "turn_on"
    assert action.data == {}
    assert 0.5 < action.confidence < 1.0
    assert command.logits[favored] == 5.0


def test_decoder_respects_threshold() -> None:
    decoder = SpikeDecoder(ACTION_MAP, threshold=10.0)
    weights = np.zeros((decoder.n_actions, decoder.n_features), dtype=np.float64)
    weights[:, -1] = 5.0
    decoder.set_weights(weights)
    assert decoder.decode({}, 500.0).actions == []


def test_decoder_features_use_window_when_duration_missing() -> None:
    decoder = SpikeDecoder(ACTION_MAP, window_ms=1000.0)
    feats = decoder.features({100: 2, 101: 2}, 0.0)
    assert feats[decoder.action_keys.index("light.kitchen.turn_on")] == 2.0


def test_decoder_save_load_roundtrip(tmp_path: Path) -> None:
    decoder = SpikeDecoder(ACTION_MAP, window_ms=250.0, threshold=0.25, feature_scale=2.0)
    weights = np.arange(12, dtype=np.float64).reshape(3, 4) / 10.0
    decoder.set_weights(weights)
    path = decoder.save(tmp_path / "decoder.npz")

    restored = SpikeDecoder({"light.kitchen.turn_on": [1]})  # deliberately different
    restored.load(path)

    np.testing.assert_array_equal(restored.W, decoder.W)
    assert restored.action_keys == decoder.action_keys
    np.testing.assert_array_equal(
        restored.action_map["light.kitchen.turn_off"],
        decoder.action_map["light.kitchen.turn_off"],
    )
    assert restored.window_ms == 250.0
    assert restored.threshold == 0.25
    assert restored.feature_scale == 2.0

    via_classmethod = SpikeDecoder.from_npz(path)
    np.testing.assert_array_equal(via_classmethod.W, decoder.W)
    assert via_classmethod.action_keys == decoder.action_keys


def test_decoder_features_align_with_learner_weights() -> None:
    """End-to-end: decoder.features -> learner.fit -> set_weights -> decode."""
    action_map = {"light.kitchen.turn_on": [100, 101], "light.kitchen.turn_off": [102]}
    decoder = SpikeDecoder(action_map, window_ms=1000.0)
    # Sorted keys: turn_off first, turn_on second, then the bias.
    X = np.vstack(
        [
            decoder.features({102: 100}, 1000.0),  # turn_off active
            decoder.features({100: 50, 101: 50}, 1000.0),  # turn_on active
        ]
    )
    Y = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    learner = ReadoutLearner(decoder.n_features, decoder.n_actions, l2=1e-9)
    learner.extend(X, Y)
    decoder.set_weights(learner.fit(method="ridge"))
    assert learner.accuracy(X, Y) == 1.0

    on = decoder.decode({100: 50, 101: 50}, 1000.0)
    off = decoder.decode({102: 100}, 1000.0)
    assert [action.key for action in on.actions] == ["light.kitchen.turn_on"]
    assert [action.key for action in off.actions] == ["light.kitchen.turn_off"]


# --------------------------------------------------------------------------- #
# ReadoutLearner
# --------------------------------------------------------------------------- #
def separable_dataset(
    n: int = 400, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly separable 0/1 dataset with a comfortable margin."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 3))
    raw = np.hstack([raw, np.ones((n, 1))])
    true_w = np.asarray(
        [[2.0, -1.0, 0.5, 0.25], [-2.0, 1.0, -0.5, -0.25]],
        dtype=np.float64,
    )
    margins = raw @ true_w.T
    keep = np.all(np.abs(margins) > 0.5, axis=1)
    X = raw[keep]
    Y = (X @ true_w.T > 0.0).astype(np.float64)
    return X, Y, true_w


def test_ridge_fit_separates_synthetic_data() -> None:
    X, Y, _ = separable_dataset()
    learner = ReadoutLearner(n_features=X.shape[1], n_actions=Y.shape[1], l2=1e-3)
    learner.extend(X, Y)
    W = learner.fit(method="ridge")

    assert W.shape == (2, 4)
    assert np.all(np.isfinite(W))
    assert learner.accuracy(X, Y) > 0.95
    per_action = learner.per_action_accuracy(X, Y)
    assert set(per_action) == {0, 1}
    assert all(value > 0.95 for value in per_action.values())


def test_add_and_add_episode_match_extend() -> None:
    X, Y, _ = separable_dataset(n=50)
    batched = ReadoutLearner(4, 2, l2=1e-3)
    batched.extend(X, Y)

    stepwise = ReadoutLearner(4, 2, l2=1e-3)
    for i in range(X.shape[0]):
        stepwise.add_episode(Episode(features=X[i], target=Y[i], context={}))

    assert len(stepwise) == X.shape[0]
    np.testing.assert_array_equal(batched.fit(), stepwise.fit())


def test_ridge_l2_penalty_shrinks_weight_norm() -> None:
    X, _, true_w = separable_dataset(seed=1)
    rng = np.random.default_rng(2)
    Y = X @ true_w.T + 0.2 * rng.normal(size=(X.shape[0], 2))

    regularized = ReadoutLearner(4, 2, l2=1.0)
    regularized.extend(X, Y)
    W_reg = regularized.fit(method="ridge")

    unregularized = ReadoutLearner(4, 2, l2=0.0)
    unregularized.extend(X, Y)
    W_unreg = unregularized.fit(method="ridge")

    assert np.linalg.norm(W_reg) < np.linalg.norm(W_unreg)


def test_delta_rule_reduces_error_and_stays_finite() -> None:
    X, Y, _ = separable_dataset(n=200)
    learner = ReadoutLearner(4, 2, l2=0.0)
    learner.extend(X, Y)
    before = float(np.mean((Y - X @ learner.W.T) ** 2))

    W = learner.fit(method="delta", lr=0.05, epochs=200)
    after = float(np.mean((Y - X @ W.T) ** 2))

    assert np.all(np.isfinite(W))
    assert after < before


def test_fit_without_samples_returns_finite_zeros() -> None:
    learner = ReadoutLearner(3, 2, l2=0.0)
    W = learner.fit(method="ridge")
    np.testing.assert_array_equal(W, np.zeros((2, 3)))
    assert learner.accuracy(np.zeros((0, 3)), np.zeros((0, 2))) == 0.0
    assert learner.per_action_accuracy(np.zeros((0, 3)), np.zeros((0, 2))) == {0: 0.0, 1: 0.0}


def test_single_class_targets_produce_no_nans() -> None:
    X, _, _ = separable_dataset(n=100)
    for value in (0.0, 1.0):
        Y = np.full((X.shape[0], 2), value, dtype=np.float64)
        learner = ReadoutLearner(4, 2, l2=1.0)
        learner.extend(X, Y)
        W = learner.fit(method="ridge")
        assert np.all(np.isfinite(W))
        assert np.isfinite(learner.accuracy(X, Y))
        assert all(np.isfinite(v) for v in learner.per_action_accuracy(X, Y).values())

        delta = ReadoutLearner(4, 2, l2=0.0)
        delta.extend(X, Y)
        assert np.all(np.isfinite(delta.fit(method="delta", lr=0.01, epochs=20)))


def test_learner_save_load_roundtrip(tmp_path: Path) -> None:
    X, Y, _ = separable_dataset(n=64)
    learner = ReadoutLearner(4, 2, l2=0.5)
    learner.extend(X, Y)
    learner.fit(method="ridge")
    path = learner.save(tmp_path / "readout.npz")

    restored = ReadoutLearner(4, 2)
    restored.load(path)

    np.testing.assert_array_equal(restored.W, learner.W)
    assert restored.l2 == 0.5
    assert restored.n_features == 4
    assert restored.n_actions == 2
    assert len(restored) == len(learner)
    np.testing.assert_array_equal(restored.predict(X), learner.predict(X))

    via_classmethod = ReadoutLearner.from_npz(path)
    np.testing.assert_array_equal(via_classmethod.W, learner.W)
