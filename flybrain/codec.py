"""Spike encoding and decoding for the connectome-driven HA controller.

This module is the boundary between Home Assistant and the frozen FlyWire LIF
network:

* :class:`SpikeEncoder` turns :class:`~flybrain.types.Signal` readings into
  Poisson spike trains on selected neuron populations (rate coding), and can
  convert a train into a steady subthreshold current for injection.
* :class:`SpikeDecoder` turns output-population spike counts into a
  :class:`~flybrain.types.BrainCommand` through a small linear readout whose
  weights are learned by :mod:`flybrain.learn`.

Everything here works on plain NumPy arrays and the shared dataclasses from
:mod:`flybrain.types`; no simulator and no network access is required.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.types import Action, BrainCommand, Signal, SignalKind, SpikeTrain

__all__ = ["ChannelSpec", "SpikeDecoder", "SpikeEncoder"]


def _npz_path(path: str | Path) -> Path:
    """Return ``path`` with an explicit ``.npz`` suffix (``np.savez`` adds it)."""
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_name(p.name + ".npz")
    return p


def _split_action_key(key: str) -> tuple[str, str]:
    """Split ``"<entity_id>.<service>"`` on the last dot."""
    entity_id, sep, service = key.rpartition(".")
    if not sep:
        return key, ""
    return entity_id, service


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


@dataclass(eq=False)
class ChannelSpec:
    """How one Home Assistant entity maps onto a population of neurons.

    Attributes:
        entity_id: Home Assistant entity id, e.g. ``"sensor.living_room_temperature"``.
        kind: Semantic signal class, used for documentation/selection by callers.
        neuron_indices: ``int32`` connectome indices that receive this channel's spikes.
        vmin: Sensor value that corresponds to zero drive.
        vmax: Sensor value that corresponds to maximum drive.
        gain: Multiplicative scale applied after the tuning curve.
    """

    entity_id: str
    kind: SignalKind
    neuron_indices: np.ndarray
    vmin: float
    vmax: float
    gain: float = 1.0
    #: Flip the tuning curve. Needed for a sensor that reads high when the sense it stands in
    #: for is low (an outdoor probe standing in for "how warm the house feels").
    invert: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SignalKind):
            self.kind = SignalKind(self.kind)
        self.entity_id = str(self.entity_id)
        self.neuron_indices = np.asarray(self.neuron_indices, dtype=np.int32).reshape(-1)
        self.vmin = float(self.vmin)
        self.vmax = float(self.vmax)
        self.gain = float(self.gain)
        self.invert = bool(self.invert)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChannelSpec):
            return NotImplemented
        return (
            self.entity_id == other.entity_id
            and self.kind == other.kind
            and np.array_equal(self.neuron_indices, other.neuron_indices)
            and self.vmin == other.vmin
            and self.vmax == other.vmax
            and self.gain == other.gain
            and self.invert == other.invert
        )

    @classmethod
    def from_config(cls, d: Mapping[str, Any]) -> ChannelSpec:
        """Build a spec from a plain mapping, e.g. one YAML channel entry.

        Accepted keys mirror the field names; ``entity_id``/``entity`` and
        ``neuron_indices``/``neurons``/``indices`` are interchangeable and
        ``kind`` may be a :class:`SignalKind` or its string value.
        """
        entity_id = d.get("entity_id", d.get("entity"))
        if entity_id is None:
            raise ValueError("channel config needs an 'entity_id' (or 'entity') key")
        raw_kind = d.get("kind", SignalKind.OTHER)
        kind = raw_kind if isinstance(raw_kind, SignalKind) else SignalKind(str(raw_kind))
        raw_indices = d.get(
            "neuron_indices",
            d.get("neurons", d.get("indices", ())),
        )
        return cls(
            entity_id=str(entity_id),
            kind=kind,
            neuron_indices=np.asarray(raw_indices, dtype=np.int32).reshape(-1),
            vmin=float(d.get("vmin", 0.0)),
            vmax=float(d.get("vmax", 1.0)),
            gain=float(d.get("gain", 1.0)),
            invert=bool(d.get("invert", False)),
        )


class SpikeEncoder:
    """Rate-codes HA signals into Poisson spike trains on chosen neurons.

    For every :class:`ChannelSpec` the matching :class:`Signal` value is
    normalized to ``u = clip((value - vmin) / (vmax - vmin), 0, 1)`` and mapped
    to a firing rate ``rate = max_rate_hz * tuning(u) * gain`` in Hz.  Spikes are
    drawn as a homogeneous Poisson process and snapped to the ``dt_ms`` grid.

    Sampling is deterministic for a fixed ``seed``: the random stream for a
    channel is derived from ``(seed, entity_id)``, so repeated calls with the
    same inputs return identical trains and a monotonically increasing sensor
    value yields a monotonically non-decreasing spike count.
    """

    def __init__(
        self,
        channels: list[ChannelSpec],
        dt_ms: float = 100.0,
        seed: int | None = None,
        max_rate_hz: float = 200.0,
        tuning: str = "linear",
    ) -> None:
        """Configure the encoder.

        Args:
            channels: Population mapping per Home Assistant entity.
            dt_ms: Integration timestep in milliseconds; spike times are snapped
                to bin centres of this width.
            seed: Seed for ``np.random.default_rng``; ``None`` gives fresh
                randomness on every call, a fixed value is reproducible.
            max_rate_hz: Peak firing rate at ``u == 1`` before ``gain``.
            tuning: ``"linear"`` (``u``) or ``"gaussian"``, a Gaussian bump
                centred on ``u == 1`` renormalized to 0 at ``u == 0``.  Both are
                monotone non-decreasing on ``[0, 1]``.

        Raises:
            ValueError: For a non-positive ``dt_ms``, negative ``max_rate_hz``,
                or unknown ``tuning``.
        """
        if dt_ms <= 0.0:
            raise ValueError("dt_ms must be positive")
        if max_rate_hz < 0.0:
            raise ValueError("max_rate_hz must be non-negative")
        if tuning not in ("linear", "gaussian"):
            raise ValueError(f"unknown tuning curve: {tuning!r}")
        self.channels: list[ChannelSpec] = list(channels)
        self.dt_ms = float(dt_ms)
        self.seed = None if seed is None else int(seed)
        self.max_rate_hz = float(max_rate_hz)
        self.tuning = tuning

    @property
    def channel_map(self) -> dict[str, ChannelSpec]:
        """Channels keyed by entity id (last one wins for duplicates)."""
        return {spec.entity_id: spec for spec in self.channels}

    def _rng_for(self, entity_id: str) -> np.random.Generator:
        """Deterministic per-entity generator when a seed is configured."""
        if self.seed is None:
            return np.random.default_rng()
        tag = zlib.crc32(entity_id.encode("utf-8"))
        return np.random.default_rng([self.seed, int(tag)])

    def _tuning_value(self, u: float) -> float:
        """Map normalized drive ``u in [0, 1]`` to a factor in ``[0, 1]``."""
        if self.tuning == "linear":
            return u
        # Gaussian bump centred on u=1, renormalized so tuning(0) == 0.
        sigma = 0.5
        peak = math.exp(-0.5 * ((u - 1.0) / sigma) ** 2)
        floor = math.exp(-0.5 * (1.0 / sigma) ** 2)
        return (peak - floor) / (1.0 - floor)

    def rate_for(self, spec: ChannelSpec, value: float) -> tuple[float, float]:
        """Return ``(u, rate_hz)`` for a raw sensor ``value`` on ``spec``."""
        span = spec.vmax - spec.vmin
        if span <= 0.0:
            u = 1.0 if value >= spec.vmax else 0.0
        else:
            u = min(1.0, max(0.0, (float(value) - spec.vmin) / span))
        rate = self.max_rate_hz * self._tuning_value(u) * spec.gain
        return u, max(0.0, float(rate))

    def _sample_spikes(
        self, spec: ChannelSpec, rate_hz: float, duration_ms: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw Poisson spike times and per-neuron counts for one channel.

        Uses exponential inter-arrival times (the exact homogeneous Poisson
        construction): with a fixed unit-exponential stream ``E_i`` and rate
        ``r``, spike ``i`` occurs at ``sum_{j<=i} E_j / r``.  The count is
        therefore non-decreasing in ``r`` for a fixed stream, which is what
        makes the encoder monotone in sensor value.
        """
        n_neurons = int(spec.neuron_indices.size)
        n_bins = max(1, round(duration_ms / self.dt_ms))
        duration_s = max(float(duration_ms), 0.0) / 1000.0
        if n_neurons == 0 or duration_s <= 0.0 or rate_hz <= 0.0:
            return np.zeros(0, dtype=np.float32), np.zeros(n_neurons, dtype=np.int64)

        rng = self._rng_for(spec.entity_id)
        expected = rate_hz * duration_s
        pool = math.ceil(expected + 6.0 * math.sqrt(expected) + 16.0)
        gaps = rng.exponential(1.0, size=pool)
        cumulative = np.cumsum(gaps)
        n_spikes = min(int(np.searchsorted(cumulative, expected, side="right")), pool)

        times_ms = cumulative[:n_spikes] / rate_hz * 1000.0
        bin_idx = np.floor(times_ms / self.dt_ms).astype(np.int64)
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
        spike_times = ((bin_idx.astype(np.float64) + 0.5) * self.dt_ms).astype(np.float32)

        assignment = np.arange(n_spikes) % n_neurons
        counts = np.bincount(assignment, minlength=n_neurons).astype(np.int64)
        return spike_times, counts

    def encode(self, signals: list[Signal], duration_ms: float) -> dict[str, SpikeTrain]:
        """Encode ``signals`` into one :class:`SpikeTrain` per configured entity.

        Each spike is assigned round-robin across the channel's neurons, so the
        population is driven evenly.  A configured entity with no matching
        signal produces an empty train with all-zero rates instead of raising.
        Returns a dict keyed by ``entity_id``; if several channels share an
        entity id, the last one wins.
        """
        duration_ms = float(duration_ms)
        duration_s = max(duration_ms, 0.0) / 1000.0
        by_entity = {signal.entity_id: signal for signal in signals}

        trains: dict[str, SpikeTrain] = {}
        for spec in self.channels:
            signal = by_entity.get(spec.entity_id)
            if signal is None:
                counts = np.zeros(int(spec.neuron_indices.size), dtype=np.int64)
                spike_times = np.zeros(0, dtype=np.float32)
            else:
                _, rate_hz = self.rate_for(spec, signal.value)
                spike_times, counts = self._sample_spikes(spec, rate_hz, duration_ms)
            if duration_s > 0.0:
                rates = (counts / duration_s).astype(np.float32)
            else:
                rates = np.zeros(int(spec.neuron_indices.size), dtype=np.float32)
            trains[spec.entity_id] = SpikeTrain(
                neuron_indices=spec.neuron_indices.copy(),
                spike_times_ms=spike_times,
                rate_hz=rates,
            )
        return trains

    def to_current(
        self,
        train: SpikeTrain,
        duration_ms: float,
        current_per_spike_mv: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a train into a steady current for each driven neuron.

        A neuron firing at ``rate_hz`` emits ``rate_hz * duration_ms / 1000``
        spikes per window on average.  Spreading one ``current_per_spike_mv``
        unit of current per spike evenly over the ``duration_ms / dt_ms`` steps
        of the window gives the equivalent constant drive::

            n_steps   = duration_ms / dt_ms
            expected  = rate_hz * duration_ms / 1000
            current   = expected / n_steps * current_per_spike_mv
                      = rate_hz * dt_ms / 1000 * current_per_spike_mv

        The result is monotone in ``rate_hz`` and therefore safe to feed to
        ``ConnectomeSim.set_drive``.  Returns ``(neuron_indices, current_mv)``
        with ``int32`` and ``float32`` arrays aligned with ``train``.
        """
        indices = np.asarray(train.neuron_indices, dtype=np.int32).reshape(-1)
        rates = np.asarray(train.rate_hz, dtype=np.float64).reshape(-1)
        if indices.shape != rates.shape:
            raise ValueError(
                f"train arrays misaligned: {indices.shape} indices vs {rates.shape} rates"
            )
        n_steps = max(float(duration_ms) / self.dt_ms, 1.0)
        expected_spikes = rates * (max(float(duration_ms), 0.0) / 1000.0)
        current_mv = expected_spikes / n_steps * float(current_per_spike_mv)
        return indices, current_mv.astype(np.float32)


class SpikeDecoder:
    """Linear readout from output-population spikes to Home Assistant actions.

    The feature vector is stable and defined as, for the action keys in sorted
    order ``k_0 < k_1 < ... < k_{A-1}``::

        features = [rate(k_0), rate(k_1), ..., rate(k_{A-1}), 1.0]

    where ``rate(k_j)`` is the mean firing rate in Hz of the neurons mapped to
    ``k_j`` over the window, optionally divided by ``feature_scale``.  The last
    entry is an unregularized bias term.  ``W`` has shape ``(A, A + 1)`` and
    starts at zeros so an untrained decoder never emits an action.
    """

    def __init__(
        self,
        action_map: dict[str, list[int]],
        window_ms: float = 500.0,
        threshold: float = 0.5,
        feature_scale: float | None = None,
    ) -> None:
        """Configure the decoder.

        Args:
            action_map: ``"<entity_id>.<service>"`` -> output neuron indices
                whose spikes vote for that action.
            window_ms: Nominal accumulation window; used when ``decode`` gets a
                non-positive duration.
            threshold: Logit above which an action is emitted.
            feature_scale: Optional divisor applied to rate features (e.g. a
                nominal peak rate) to keep logits well conditioned.
        """
        self.action_map: dict[str, np.ndarray] = {
            str(key): np.asarray(indices, dtype=np.int32).reshape(-1)
            for key, indices in action_map.items()
        }
        self._keys: tuple[str, ...] = tuple(sorted(self.action_map))
        self.window_ms = float(window_ms)
        self.threshold = float(threshold)
        self.feature_scale = None if feature_scale is None else float(feature_scale)
        self.W: np.ndarray = np.zeros(self._shape(), dtype=np.float64)

    def _shape(self) -> tuple[int, int]:
        n_actions = len(self._keys)
        return n_actions, n_actions + 1

    @property
    def action_keys(self) -> list[str]:
        """Stable, sorted action-key order used for features and ``W`` rows."""
        return list(self._keys)

    @property
    def n_actions(self) -> int:
        """Number of action keys / rows of ``W``."""
        return len(self._keys)

    @property
    def n_features(self) -> int:
        """Feature-vector length (``n_actions`` rate features plus a bias)."""
        return len(self._keys) + 1

    def _effective_window(self, duration_ms: float) -> float:
        duration_ms = float(duration_ms)
        return duration_ms if duration_ms > 0.0 else self.window_ms

    def features(self, spike_counts: dict[int, int], duration_ms: float) -> np.ndarray:
        """Build the deterministic ``(n_actions + 1,)`` feature vector.

        Rate features follow the sorted :attr:`action_keys` order and the bias
        term is always the last element, so learned weights stay aligned.
        """
        window_ms = self._effective_window(duration_ms)
        duration_s = max(window_ms, 0.0) / 1000.0
        feats = np.zeros(self.n_features, dtype=np.float64)
        if duration_s > 0.0:
            for j, key in enumerate(self._keys):
                group = self.action_map[key]
                if group.size == 0:
                    continue
                total = 0
                for idx in group:
                    total += int(spike_counts.get(int(idx), 0))
                feats[j] = (total / int(group.size)) / duration_s
        if self.feature_scale is not None and self.feature_scale > 0.0:
            feats[:-1] /= self.feature_scale
        feats[-1] = 1.0
        return feats

    def decode(self, spike_counts: dict[int, int], duration_ms: float) -> BrainCommand:
        """Score every action key and emit those above ``threshold``.

        Emits no actions until weights are set (``W`` is zeros by default) and
        never raises for unknown neuron indices in ``spike_counts``.
        """
        feats = self.features(spike_counts, duration_ms)
        scores = self.W @ feats
        window_ms = self._effective_window(duration_ms)

        logits: dict[str, float] = {}
        actions: list[Action] = []
        for j, key in enumerate(self._keys):
            score = float(scores[j])
            logits[key] = score
            if score > self.threshold:
                entity_id, service = _split_action_key(key)
                actions.append(
                    Action(
                        entity_id=entity_id,
                        service=service,
                        confidence=_sigmoid(score),
                        data={},
                    )
                )
        return BrainCommand(
            actions=actions,
            spike_counts={int(k): int(v) for k, v in spike_counts.items()},
            logits=logits,
            window_ms=window_ms,
        )

    def set_weights(self, W: np.ndarray) -> None:
        """Install a weight matrix, validating ``(n_actions, n_features)``."""
        arr = np.asarray(W, dtype=np.float64)
        if arr.shape != self._shape():
            raise ValueError(
                f"weights must have shape {self._shape()}, got {arr.shape}"
            )
        self.W = np.array(arr, dtype=np.float64, copy=True)

    def save(self, path: str | Path) -> Path:
        """Persist weights, action-key order, neuron groups and settings to ``.npz``."""
        p = _npz_path(path)
        keys = np.asarray(self._keys, dtype=np.str_)
        if self._keys:
            group_concat = np.concatenate([self.action_map[k] for k in self._keys])
        else:
            group_concat = np.zeros(0, dtype=np.int32)
        group_lengths = np.asarray(
            [int(self.action_map[k].size) for k in self._keys], dtype=np.int64
        )
        scale = np.nan if self.feature_scale is None else self.feature_scale
        np.savez(
            p,
            W=self.W,
            action_keys=keys,
            group_concat=group_concat.astype(np.int32),
            group_lengths=group_lengths,
            window_ms=self.window_ms,
            threshold=self.threshold,
            feature_scale=scale,
        )
        return p

    def load(self, path: str | Path) -> SpikeDecoder:
        """Load a decoder previously written by :meth:`save`; returns ``self``."""
        p = _npz_path(path)
        with np.load(p, allow_pickle=False) as data:
            keys = [str(k) for k in data["action_keys"].tolist()]
            concat = np.asarray(data["group_concat"], dtype=np.int32)
            lengths = np.asarray(data["group_lengths"], dtype=np.int64).tolist()
            action_map: dict[str, np.ndarray] = {}
            offset = 0
            for key, length in zip(keys, lengths):
                action_map[key] = concat[offset : offset + length].copy()
                offset += int(length)
            self._keys = tuple(keys)
            self.action_map = action_map
            self.W = np.array(data["W"], dtype=np.float64)
            self.window_ms = float(data["window_ms"])
            self.threshold = float(data["threshold"])
            scale = float(data["feature_scale"])
            self.feature_scale = None if math.isnan(scale) else scale
        return self

    @classmethod
    def from_npz(cls, path: str | Path) -> SpikeDecoder:
        """Convenience constructor: load a decoder without building an ``action_map``."""
        return cls({}).load(path)
