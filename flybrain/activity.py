"""Drive a connectome engine and expose *activity* to the live view.

``MemoryBrain`` is a small adapter that steps a simulator incrementally and answers the
only three questions the dashboard asks:

* how much did each neuron fire during the last window (a dense intensity buffer),
* what just spiked (a sparse list, for the raster), and
* what is the aggregate usage (active count, total spikes, mean rate).

It wraps :class:`flybrain.sim.ConnectomeSim` today. Any engine exposing ``n_neurons``,
``step(n_steps)``, ``spike_counts()`` and ``annotation_table`` can be substituted, which
is how the fast engine will slot in later without touching the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActivitySettings:
    """Knobs the dashboard can change live."""

    #: Multiplier applied to per-window spike counts before display quantization.
    gain: float = 8.0
    #: Spike count considered "full brightness" for the colormap.
    saturation: float = 4.0
    #: Gamma applied when mapping to intensity; <1 lifts dim activity into view.
    gamma: float = 0.5
    #: Neural time advanced per published frame, in milliseconds.
    window_ms: float = 50.0
    #: Frames published per second (wall clock).
    fps: float = 20.0
    #: How many neurons to broadcast in the sparse recent-spike channel.
    sparse_limit: int = 600
    #: Background drive applied to a random subset each window, if any.
    background_drive_mv: float = 0.0
    #: Fraction of neurons receiving background drive each window.
    background_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "gain": self.gain,
            "saturation": self.saturation,
            "gamma": self.gamma,
            "window_ms": self.window_ms,
            "fps": self.fps,
            "sparse_limit": self.sparse_limit,
            "background_drive_mv": self.background_drive_mv,
            "background_fraction": self.background_fraction,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ActivitySettings:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Frame:
    """One published slice of activity."""

    seq: int
    sim_ms: float
    intensity: np.ndarray  # uint8, length n_neurons
    counts: np.ndarray  # int32, length n_neurons, raw window counts
    sparse_indices: np.ndarray  # int32, most active neurons this window
    sparse_counts: np.ndarray  # int32
    total_spikes: int
    active_neurons: int
    metrics: dict = field(default_factory=dict)


class MemoryBrain:
    """Incremental, single-timestep-at-a-time driver for a connectome engine."""

    def __init__(self, sim, settings: ActivitySettings | None = None) -> None:
        if sim._torch is None:
            raise RuntimeError("simulator must be loaded before wrapping it")
        self.sim = sim
        self.settings = settings or ActivitySettings()
        self.n_neurons = int(sim.n_neurons)
        self._seq = 0
        self._rng = np.random.default_rng(7)
        self._total_spikes_ever = 0
        self._ema_rate_hz = 0.0
        #: Optional persistent drive, set by the encoder/experiment layer.
        self._drive_indices: np.ndarray | None = None
        self._drive_current = 0.0
        #: Spike counts and window length from the most recent window, for the usage panel.
        self._last_counts: np.ndarray | None = None
        self._last_window_ms = 0.0

    # ------------------------------------------------------------------ drive

    def set_input_drive(self, indices, current_mv: float) -> None:
        """Apply a persistent drive to ``indices`` on every window.

        The drive is held by the *simulator* (``ConnectomeSim._persistent_drive``), which
        re-applies it on every step. Keeping only a local copy of the indices is not
        enough: it must be pushed through to the simulator, and cleared there as well.
        """
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            self.clear_input_drive()
            return
        self._drive_indices = idx
        self._drive_current = float(current_mv)
        self.sim.set_drive(idx, self._drive_current)

    def clear_input_drive(self) -> None:
        """Remove the persistent drive from both this driver and the simulator."""
        self._drive_indices = None
        self._drive_current = 0.0
        self.sim.clear_drive()

    # ---------------------------------------------------------------- stepping

    def advance(self) -> Frame:
        """Advance one display window and return the resulting :class:`Frame`."""
        s = self.settings
        n_steps = max(1, round(s.window_ms / self.sim.params.dt_ms))

        # The simulator re-applies its own persistent drive on every step, so there is no
        # need to re-set it here; doing so was harmless but obscured who owned the state.
        if s.background_drive_mv > 0 and s.background_fraction > 0:
            k = int(self.n_neurons * s.background_fraction)
            if k > 0:
                pick = self._rng.choice(self.n_neurons, size=k, replace=False)
                self.sim.inject(pick, s.background_drive_mv)

        self.sim.step(n_steps)
        counts = self.sim.spike_counts(reset=True).astype(np.int32)
        # Keep the window's counts so the usage panel can read them later. Reading the
        # simulator's spike tensor directly raced with this reset and returned nothing
        # almost every time.
        self._last_counts = counts
        self._last_window_ms = n_steps * self.sim.params.dt_ms

        intensity = self.quantize(counts, s)
        total = int(counts.sum())
        active = int((counts > 0).sum())
        self._total_spikes_ever += total

        window_s = (n_steps * self.sim.params.dt_ms) / 1000.0
        inst_rate = (total / max(window_s, 1e-9)) / max(self.n_neurons, 1)
        self._ema_rate_hz = 0.9 * self._ema_rate_hz + 0.1 * inst_rate

        order = np.argsort(counts)[::-1][: s.sparse_limit]
        order = order[counts[order] > 0]
        self._seq += 1

        return Frame(
            seq=self._seq,
            sim_ms=self.sim.sim_time_ms,
            intensity=intensity,
            counts=counts,
            sparse_indices=order.astype(np.int32),
            sparse_counts=counts[order].astype(np.int32),
            total_spikes=total,
            active_neurons=active,
            metrics={
                "total_spikes_ever": self._total_spikes_ever,
                "mean_rate_hz": round(self._ema_rate_hz, 4),
                "window_ms": n_steps * self.sim.params.dt_ms,
                "fps": s.fps,
            },
        )

    @staticmethod
    def quantize(counts: np.ndarray, s: ActivitySettings) -> np.ndarray:
        """Map raw spike counts to a uint8 intensity buffer for the shader.

        ``intensity = clip(counts * gain / saturation, 0, 1) ** gamma * 255``
        """
        scaled = counts.astype(np.float32) * float(s.gain)
        norm = np.clip(scaled / max(float(s.saturation), 1e-6), 0.0, 1.0)
        norm = np.power(norm, float(s.gamma))
        return (norm * 255.0).astype(np.uint8)

    # ---------------------------------------------------------------- readout

    def region_usage(self, top: int = 12) -> list[dict]:
        """Per-cell-class activity for the most recent window, for the 'usage' panel.

        Reads the cached counts from the last :meth:`advance` rather than the simulator's
        spike tensor, which has already been reset by the time a request arrives.
        """
        table = getattr(self.sim, "annotation_table", None)
        if table is None or self._last_counts is None:
            return []
        arr = self._last_counts
        window_s = max(self._last_window_ms, 1e-6) / 1000.0
        cls = table["cell_class"].fillna("unclassified").to_numpy()
        out: list[dict] = []
        for name in np.unique(cls):
            mask = cls == name
            total = int(arr[mask].sum())
            if total:
                out.append(
                    {
                        "name": str(name),
                        "neurons": int(mask.sum()),
                        "spikes": total,
                        "rate_hz": round(total / max(int(mask.sum()), 1) / window_s, 2),
                    }
                )
        out.sort(key=lambda d: d["spikes"], reverse=True)
        return out[:top]
