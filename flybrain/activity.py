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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.families import ALL_FAMILIES, family_ids
from flybrain.types import coerce_patch

#: Rejection bounds for :class:`ActivitySettings`, checked by
#: :meth:`ActivitySettings.validate_patch`. ``(low, high)``, inclusive; ``None`` means unbounded
#: on that side.
#:
#: These are *rejection* bounds, not clamps, and the difference is deliberate. The control
#: loop's sensitivity limits are clamped, because a value outside them still means something
#: ("use as much of the range as is usable"). Here a negative gain or a frame rate of zero means
#: nothing at all — it is a mistake — and silently clamping it to a legal value would hide the
#: mistake rather than report it. A frame rate of zero is also the kind of setting whose failure
#: mode is a dashboard that has quietly frozen.
FIELD_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "gain": (0.0, 1000.0),
    "saturation": (0.01, 1e6),
    "gamma": (0.01, 10.0),
    "window_ms": (0.1, 10_000.0),
    "fps": (1.0, 240.0),
    "sparse_limit": (0.0, 1_000_000.0),
    "background_drive_mv": (0.0, 1000.0),
    "background_fraction": (0.0, 1.0),
}


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

    @classmethod
    def validate_patch(cls, patch: Mapping[str, Any]) -> dict[str, Any]:
        """Coerce and range-check a patch, returning it **without applying it**.

        The caller commits the whole result or none of it. The code this replaces cast each
        field and assigned it inside the *same* loop, so ``{"fps": 30, "gain": "abc"}`` applied
        the new frame rate and then failed: a partial update, reported to the operator as a
        rejected one. There was also no range check at all, on either the REST or the WebSocket
        path, so ``fps: 0`` or a negative gain were accepted and then silently papered over
        downstream (``run_loop`` wraps fps in ``max(1.0, ...)``).

        Raises:
            KeyError: the patch names an unknown setting.
            ValueError: a value is the wrong type, or outside :data:`FIELD_BOUNDS`.
        """
        coerced = coerce_patch(cls(), patch, allowed=set(cls.__dataclass_fields__))
        for key, value in coerced.items():
            lo, hi = FIELD_BOUNDS.get(key, (None, None))
            if lo is not None and value < lo:
                raise ValueError(f"{key} must be at least {lo}, got {value}")
            if hi is not None and value > hi:
                raise ValueError(f"{key} must be at most {hi}, got {value}")
        return coerced

    def apply_patch(self, patch: Mapping[str, Any]) -> None:
        """Validate *patch* and commit it, or raise without changing anything.

        The single entry point both the REST route and the WebSocket handler use, so the two
        cannot drift apart again — they previously had separate, differently-broken loops.
        """
        for key, value in self.validate_patch(patch).items():
            setattr(self, key, value)


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
        #: Per-neuron integer labels for the two groupings the dashboard colours by, built lazily
        #: by :meth:`_ensure_indices`. The alternative — a boolean mask per class per request —
        #: was O(classes x neurons) on every poll.
        self._indices_for: object = "unbuilt"
        self._class_index: np.ndarray | None = None
        self._class_names: list[str] = []
        self._class_sizes = np.zeros(0, dtype=np.int64)
        self._family_ids: np.ndarray | None = None
        self._family_index: np.ndarray | None = None

    # ------------------------------------------------------------------ indices

    def _build_index(self, table, column: str) -> tuple[np.ndarray | None, list[str]]:
        """Turn an annotation column into a dense ``int8`` label array.

        Returned as int8 rather than object strings so that per-class totals are a single
        ``np.bincount`` instead of one full-array comparison per class. The names list is the
        ``id -> name`` mapping that bincount's output indexes into.
        """
        if table is None or column not in table.columns:
            return None, []
        values = table[column].fillna("unlabelled").astype(str).to_numpy()
        names, codes = np.unique(values, return_inverse=True)
        if len(names) > 127:
            raise ValueError(
                f"{column} has {len(names)} distinct values, which overflows the int8 index"
            )
        return codes.astype(np.int8), [str(n) for n in names]

    def _ensure_indices(self) -> None:
        """Build the label indices from the annotation table, once per table.

        Lazy rather than eager in ``__init__`` on purpose: the table can legitimately be attached
        *after* the driver is constructed — ``ConnectomeSim.load()`` sets it, and tests attach a
        synthetic one — so an eager build would freeze whichever table happened to exist first and
        then report no regions at all. Comparing identity makes a replacement table take effect
        without needing a rebuild API.
        """
        table = getattr(self.sim, "annotation_table", None)
        if table is self._indices_for:
            return
        self._indices_for = table
        self._class_index, self._class_names = self._build_index(table, "cell_class")
        self._class_sizes = (
            np.bincount(self._class_index, minlength=len(self._class_names))
            if self._class_index is not None
            else np.zeros(0, dtype=np.int64)
        )
        # The per-neuron family mapping is built here rather than in families.py so that both
        # groupings share one join onto connectome order. `_family_index` is the same array under
        # the name the region code reads; keeping one object avoids computing the join twice.
        self._family_ids = (
            family_ids(table["super_class"].to_numpy())
            if table is not None and "super_class" in table.columns
            else None
        )
        self._family_index = self._family_ids

    @property
    def n_classes(self) -> int:
        """How many distinct ``cell_class`` values this build knows about."""
        self._ensure_indices()
        return len(self._class_names)

    @property
    def seq(self) -> int:
        """The sequence number of the most recent :meth:`advance`.

        Public because it is the identity that ties a decision on the chart to a recorded brain
        state: the Jev inspector is asked about a ``seq``, and every history row carries the one
        it was decided on.
        """
        return int(self._seq)

    def class_names(self) -> list[str]:
        """``cell_class`` names, in the id order :meth:`counts_by_class` returns totals in."""
        self._ensure_indices()
        return list(self._class_names)

    def class_index(self) -> np.ndarray | None:
        """Per-neuron ``cell_class`` id, or ``None`` when no annotation table is loaded."""
        self._ensure_indices()
        return self._class_index

    def family_index(self) -> np.ndarray | None:
        """Per-neuron ``super_class`` id as published, or ``None`` without annotations."""
        self._ensure_indices()
        return self._family_index

    def counts_by_class(self, counts: np.ndarray) -> np.ndarray:
        """Spike totals per ``cell_class`` id, by bincount rather than per-class masking."""
        self._ensure_indices()
        index = self._class_index
        if index is None:
            return np.zeros(0, dtype=np.int64)
        return np.bincount(index, weights=counts.astype(np.float64), minlength=len(self._class_names))

    # ------------------------------------------------------------------ drive

    def set_input_drive(self, indices, current_mv: float) -> None:
        """Apply a persistent drive to ``indices`` on every window.

        The drive is held by the *simulator* (``ConnectomeSim.set_drive``), which
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

    def family_usage(self, top: int = 0) -> list[dict]:
        """Per-family activity for the most recent window, for the cell-family legend.

        Same contract as :meth:`region_usage` but grouped by the broad ``super_class`` families
        the dashboard colours by, so the legend and the cloud cannot disagree about which group a
        neuron is in.
        """
        self._ensure_indices()
        if self._family_ids is None or self._last_counts is None:
            return []
        window_s = max(self._last_window_ms, 1e-6) / 1000.0
        totals = np.bincount(
            self._family_ids,
            weights=self._last_counts.astype(np.float64),
            minlength=len(ALL_FAMILIES) + 1,
        )
        sizes = np.bincount(self._family_ids, minlength=len(ALL_FAMILIES) + 1)
        out: list[dict] = []
        for fam in ALL_FAMILIES:
            # int() of a whole float64 is exact here and yields a Python int, which is what
            # the JSON encoder needs: a numpy scalar would raise at serialisation time.
            spikes = int(totals[fam.id])
            neurons = int(sizes[fam.id])
            out.append(
                {
                    "id": fam.id,
                    "key": fam.key,
                    "spikes": spikes,
                    "neurons": neurons,
                    "rate_hz": round(spikes / max(neurons, 1) / window_s, 3),
                }
            )
        out.sort(key=lambda d: d["spikes"], reverse=True)
        return out[:top] if top else out

    def region_usage(self, top: int = 12) -> list[dict]:
        """Per-cell-class activity for the most recent window, for the 'usage' panel.

        Reads the cached counts from the last :meth:`advance` rather than the simulator's
        spike tensor, which has already been reset by the time a request arrives.
        """
        self._ensure_indices()
        if self._class_index is None or self._last_counts is None:
            return []
        window_s = max(self._last_window_ms, 1e-6) / 1000.0
        totals = self.counts_by_class(self._last_counts)
        out: list[dict] = []
        for i, name in enumerate(self._class_names):
            total = int(totals[i])
            if not total:
                continue
            neurons = int(self._class_sizes[i])
            out.append(
                {
                    "name": name,
                    "neurons": neurons,
                    "spikes": total,
                    "rate_hz": round(total / max(neurons, 1) / window_s, 2),
                }
            )
        out.sort(key=lambda d: d["spikes"], reverse=True)
        return out[:top]
