"""The first end-to-end loop: temperature drives the colour of a light.

One sensor, one light, and a learned readout. No Home Assistant needed to test it.

    temperature ──► rate-coded spikes ──► fly thermosensory + ascending neurons
                                                            │
                                                            ▼
                                                frozen FlyWire v783 network
                                                            │
                                                            ▼
                                   firing rates of a readout population ──► ridge fit
                                                            │
                                                            ▼
                                              colour temperature (Kelvin)

Why colour rather than on/off: the mapping is continuous, so a mistake is visible as the
wrong shade instead of a silently missing event, and it transfers straight to Home
Assistant's ``light.turn_on`` with ``color_temp_kelvin``.

The brain is frozen and only the readout is learned, so the interesting question is what
the readout sees. Two choices dominate the accuracy, and both were measured rather than
guessed (``tools/readout_probe.py``, held-out temperature split):

1. **Features.** Feed the readout the firing rate of *every* neuron in the readout pool.
   Collapsing the pool to one scalar per colour band - the original design - makes the
   features almost perfectly collinear (r = 0.998-1.000), because three random slices of
   one population all report the same thing. Per-neuron rates take the achievable error
   from ~120 K to ~21 K.

2. **Target.** Regress the Kelvin value directly. The original design fitted a soft
   Gaussian over three colour bands and applied a softmax; that label scheme is itself
   153 K away from the ideal mapping, so no feature set could have done better.

Run::

    .venv/bin/python -m flybrain.experiment
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flybrain.activity import ActivitySettings, MemoryBrain
from flybrain.learn import ReadoutLearner
from flybrain.mapping import RoleResolver
from flybrain.sim import ConnectomeSim
from flybrain.types import Action, Signal, SignalKind

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data/experiments"

#: Colour-temperature bands, ascending in Kelvin. These set the endpoints of the reported
#: range; the readout output itself is continuous and is not restricted to them.
#:
#: Naming follows *lighting* convention, not temperature convention, because that is what
#: the label is read as next to a colour swatch: a low Kelvin light looks warm/orange and a
#: high Kelvin light looks cool/blue. An earlier revision named 4000 K "cool" (because a
#: cold room maps to it), which put the word "cool white" directly beside an orange swatch.
COLOUR_BANDS: list[tuple[str, float]] = [
    ("warm", 2700.0),
    ("neutral", 5000.0),
    ("cool", 6500.0),
]


def rate_for_temperature(celsius: float, config: ExperimentConfig) -> float:
    """Firing rate (Hz) the sensory population runs at for a temperature.

    Warmer means faster, matching how the fly's thermosensory neurons actually behave. The
    rate runs from ``min_rate_hz`` to ``max_rate_hz`` rather than from zero, so the brain is
    driven at every temperature and the readout always carries signal.

    This is module-level, and both the offline experiment and the live loop call it, because
    the learned readout is only valid if the live encoding is *bit-for-bit* the one it was
    fitted on. Two copies of this formula would be a silent correctness bug.
    """
    u = (float(celsius) - config.temp_min_c) / max(config.temp_max_c - config.temp_min_c, 1e-9)
    u = float(np.clip(u, 0.0, 1.0))
    return float(config.min_rate_hz + u * (config.max_rate_hz - config.min_rate_hz))


def drive_current_for_rate(rate_hz: float, dt_ms: float, config: ExperimentConfig) -> float:
    """Convert a sensory firing rate into the equivalent steady injected current."""
    return float(rate_hz) * dt_ms / 1000.0 * config.current_per_spike_mv


@dataclass
class ExperimentConfig:
    """Everything the loop needs, so a run is reproducible."""

    temperature_entity: str = "sensor.living_room_temperature"
    light_entity: str = "light.kitchen"
    temp_min_c: float = 10.0
    temp_max_c: float = 35.0
    #: Neurons the temperature drives. The fly's real thermosensory population is only 29
    #: cells, so it is pooled with ascending neurons to carry a usable rate code.
    input_roles: list[str] = field(
        default_factory=lambda: ["thermosensory", "hygrosensory"]
    )
    input_pool_size: int = 256
    #: Population the readout listens to. NOTE: the fly's descending and motor neurons
    #: are almost unreachable from a sensory drive in this connectome (a strong drive
    #: produced ~23 extra spikes across all 1,299 descending neurons), so they are a poor
    #: readout here. The antennal-lobe populations are what actually respond within a
    #: 300 ms window, so they are the default.
    readout_roles: list[str] = field(
        default_factory=lambda: ["antennial_projection", "antennial_local"]
    )
    readout_pool_size: int = 512
    #: Brain time simulated per training sample, and per control tick.
    window_ms: float = 300.0
    #: Neuron drive per spike-equivalent. This is the main calibration knob.
    #: Drive per spike-equivalent. Below ~5 mV nothing crosses threshold at all; this is
    #: the main calibration knob between "silent" and "saturated".
    current_per_spike_mv: float = 20.0
    #: Firing rate of the sensory code at the bottom and top of the temperature range.
    #: The floor must NOT be zero: with no baseline the coldest temperature injects no
    #: current at all, the whole readout population is silent, and the decoder has
    #: literally nothing to read. Real thermoreceptors have spontaneous activity too.
    min_rate_hz: float = 20.0
    max_rate_hz: float = 120.0
    #: Ridge penalty for the readout. Measured on a held-out split, accuracy is flat
    #: between 0.1 and 100, so this is not a sensitive knob.
    ridge_l2: float = 10.0
    seed: int = 0


class ColourReadout:
    """Linear map from readout-population firing rates to a colour temperature.

    Features are the firing rate (Hz) of every neuron in the readout pool over one
    window, plus an unregularized bias; ``W`` is a single row fitted by ridge regression
    straight onto the target Kelvin value.
    """

    def __init__(self, n_features: int, l2: float = 10.0) -> None:
        self.n_features = int(n_features)
        self.l2 = float(l2)
        self.W: np.ndarray = np.zeros(self.n_features, dtype=np.float64)

    def features(self, rates_hz: np.ndarray) -> np.ndarray:
        """Append the bias term to a per-neuron rate vector."""
        r = np.asarray(rates_hz, dtype=np.float64).reshape(-1)
        if r.size != self.n_features - 1:
            raise ValueError(
                f"expected {self.n_features - 1} rates, got {r.size}"
            )
        return np.concatenate([r, [1.0]])

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "ridge") -> np.ndarray:
        """Fit the readout. ``X`` is ``(n_samples, n_features-1)`` rate rows."""
        Xb = np.hstack([np.asarray(X, dtype=np.float64),
                        np.ones((len(X), 1))])
        learner = ReadoutLearner(n_features=self.n_features, n_actions=1, l2=self.l2)
        learner.extend(Xb, np.asarray(y, dtype=np.float64).reshape(-1, 1))
        self.W = learner.fit(method=method).reshape(-1)
        return self.W

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict Kelvin for one rate row or a stack of them."""
        Xb = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xb = np.hstack([Xb, np.ones((Xb.shape[0], 1))])
        return Xb @ self.W

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez(path, W=self.W, n_features=self.n_features, l2=self.l2)
        return path

    def load(self, path: str | Path) -> ColourReadout:
        with np.load(Path(path), allow_pickle=False) as d:
            self.W = np.array(d["W"], dtype=np.float64)
            self.n_features = int(d["n_features"])
            self.l2 = float(d["l2"])
        return self


class TemperatureColourLoop:
    """Drives the connectome from a temperature and reads out a colour."""

    def __init__(self, config: ExperimentConfig | None = None, sim=None) -> None:
        self.config = config or ExperimentConfig()
        self.sim = sim or ConnectomeSim().load()
        self.roles = RoleResolver.from_sim(self.sim)
        self.settings = ActivitySettings(window_ms=self.config.window_ms, fps=20.0)
        self.brain = MemoryBrain(self.sim, self.settings)

        self.input_indices = self.roles.pool(
            self.config.input_roles, self.config.input_pool_size, seed=self.config.seed
        )
        self.readout_indices = self.roles.pool(
            self.config.readout_roles, self.config.readout_pool_size, seed=self.config.seed + 1
        )

        self.band_centres = np.array([k for _, k in COLOUR_BANDS], dtype=np.float64)
        self.band_names = [name for name, _ in COLOUR_BANDS]
        self.readout = ColourReadout(
            n_features=int(self.readout_indices.size) + 1, l2=self.config.ridge_l2
        )
        self._last_fit_temperatures: np.ndarray | None = None

    # ------------------------------------------------------------- encoding

    def rate_for_temperature(self, celsius: float) -> float:
        """Firing rate (Hz) the sensory population runs at for a temperature."""
        return rate_for_temperature(celsius, self.config)

    def stimulus(self, celsius: float) -> Signal:
        return Signal(
            entity_id=self.config.temperature_entity,
            kind=SignalKind.TEMPERATURE,
            value=float(celsius),
            state=str(float(celsius)),
            unit="C",
            timestamp=time.time(),
        )

    def ideal_kelvin(self, celsius: float) -> float:
        """The colour the sensor reading ought to map to: a straight ramp over the range."""
        c = self.config
        lo, hi = COLOUR_BANDS[0][1], COLOUR_BANDS[-1][1]
        u = (float(celsius) - c.temp_min_c) / max(c.temp_max_c - c.temp_min_c, 1e-9)
        return float(lo + np.clip(u, 0.0, 1.0) * (hi - lo))

    def band_for_kelvin(self, kelvin: float) -> str:
        """Nearest band label, for display only - the output itself is continuous."""
        return self.band_names[int(np.argmin(np.abs(self.band_centres - kelvin)))]

    # ------------------------------------------------------------- sampling

    def sweep(self, temperatures) -> np.ndarray:
        """Drive the brain *continuously* through ``temperatures``; one window each.

        This is the regime the live control loop actually runs in, and it is deliberately
        not the same as running each temperature from a fresh reset:

        * the drive for each window is set before that window runs, so nothing is
          predicted from a temperature that has not happened yet;
        * the network carries its ongoing activity from the previous window, exactly as it
          does in a live loop where the sensor drifts.

        A from-rest window and an already-running window are different enough that a linear
        probe separates them perfectly (``tools/loop_regime_probe.py``), so training on one
        and running on the other silently costs accuracy.
        """
        steps = max(1, round(self.config.window_ms / self.sim.params.dt_ms))
        window_s = self.config.window_ms / 1000.0

        self.sim.reset()
        self.brain.clear_input_drive()
        rows = []
        for celsius in temperatures:
            rate = rate_for_temperature(float(celsius), self.config)
            current = drive_current_for_rate(rate, self.sim.params.dt_ms, self.config)
            self.brain.set_input_drive(self.input_indices, current)
            self.sim.step(steps)
            counts = self.sim.spike_counts(reset=True)
            rows.append(counts[self.readout_indices].astype(np.float64) / window_s)
        return np.asarray(rows, dtype=np.float64)

    def sample(self, celsius: float) -> tuple[dict[int, int], np.ndarray]:
        """One window at ``celsius`` from a resting brain; counts and rates.

        Single-shot convenience for tooling. Note this is the *from-rest* regime, not the
        live one - prefer :meth:`sweep` for anything the live loop will consume.
        """
        self.sim.reset()
        self.brain.clear_input_drive()
        rate = rate_for_temperature(float(celsius), self.config)
        current = drive_current_for_rate(rate, self.sim.params.dt_ms, self.config)
        self.brain.set_input_drive(self.input_indices, current)
        self.sim.step(max(1, round(self.config.window_ms / self.sim.params.dt_ms)))
        counts_all = self.sim.spike_counts(reset=True)

        counts = {
            int(i): int(counts_all[i])
            for i in self.readout_indices
            if counts_all[i] > 0
        }
        window_s = self.config.window_ms / 1000.0
        rates = counts_all[self.readout_indices].astype(np.float64) / window_s
        return counts, rates

    def rates(self, celsius: float) -> np.ndarray:
        return self.sample(celsius)[1]

    # ------------------------------------------------------------- training

    def training_temperatures(self, n: int = 21) -> np.ndarray:
        c = self.config
        return np.linspace(c.temp_min_c, c.temp_max_c, n)

    def evaluation_temperatures(self, n: int = 9) -> np.ndarray:
        """Temperatures that lie strictly *between* the training points.

        Held-out evaluation matters here: ``evaluate`` used to reuse the training grid, so
        its error was in-sample fit quality dressed up as accuracy. Taking midpoints of
        consecutive training pairs guarantees the two sets never touch - a symmetric
        offset happens to collide with the central training point.
        """
        c = self.config
        fitted = self._last_fit_temperatures
        if fitted is None or fitted.size < 2:
            return np.linspace(c.temp_min_c, c.temp_max_c, n)
        midpoints = (fitted[:-1] + fitted[1:]) / 2.0
        picks = np.unique(np.round(np.linspace(0, midpoints.size - 1, n)).astype(int))
        return midpoints[picks]

    def fit(self, n_samples: int = 21, method: str = "ridge") -> dict:
        """Sweep the brain across the temperature range and fit the readout."""
        temperatures = self.training_temperatures(n_samples)
        started = time.time()
        X = self.sweep(temperatures)
        Y = np.asarray([self.ideal_kelvin(t) for t in temperatures], dtype=np.float64)
        self.readout.fit(X, Y, method=method)
        self._last_fit_temperatures = temperatures
        pred = self.readout.predict(X)
        return {
            "n_samples": len(temperatures),
            "seconds": round(time.time() - started, 1),
            "train_mean_abs_error_k": float(np.mean(np.abs(pred - Y))),
            "features": int(self.readout.n_features),
        }

    # ------------------------------------------------------------ inference

    def predict_kelvin(self, celsius: float) -> float:
        """Single-shot prediction: warm the brain up, then read one window.

        Runs two windows at the same temperature and decodes the second, so the brain is
        in the running state the readout was fitted on rather than a cold one.
        """
        rates = self.sweep([celsius, celsius])[-1]
        lo, hi = COLOUR_BANDS[0][1], COLOUR_BANDS[-1][1]
        return float(np.clip(self.readout.predict(rates)[0], lo, hi))

    def to_action(self, celsius: float, confidence: float = 1.0) -> Action:
        """The Home Assistant service call this temperature produces."""
        kelvin = self.predict_kelvin(celsius)
        return Action(
            entity_id=self.config.light_entity,
            service="turn_on",
            confidence=confidence,
            data={"color_temp_kelvin": round(kelvin)},
        )

    # ------------------------------------------------------------ evaluation

    def evaluate(self, n: int = 9, held_out: bool = True) -> dict:
        """Compare predicted colour temperature against the ideal linear mapping.

        Runs as a single continuous sweep, so what is measured here is the same regime the
        live loop runs in, not a friendlier one.
        """
        temps = (
            self.evaluation_temperatures(n) if held_out
            else self.training_temperatures(n)
        )
        X = self.sweep(temps)
        preds = np.clip(self.readout.predict(X), self.band_centres[0], self.band_centres[-1])
        rows = []
        for t, got in zip(temps, preds):
            want = self.ideal_kelvin(float(t))
            # Store *integers* and derive the per-row error from those same integers. The
            # Home Assistant call rounds the Kelvin value anyway, and mixing
            # `round(got - want)` in the table with `round(got) - round(want)` in the
            # summary makes the printed table impossible to reconcile with the printed
            # mean (they differ by up to 1 K per row).
            k, w = round(float(got)), round(want)
            rows.append(
                {
                    "celsius": round(float(t), 2),
                    "kelvin": k,
                    "ideal_kelvin": w,
                    "error_k": k - w,
                    "band": self.band_for_kelvin(float(k)),
                }
            )
        got = np.array([r["kelvin"] for r in rows], dtype=np.float64)
        want = np.array([r["ideal_kelvin"] for r in rows], dtype=np.float64)
        # Monotonicity: does it ever go the wrong way as it warms?
        diffs = np.diff(got)
        return {
            "rows": rows,
            "held_out": bool(held_out),
            "mean_abs_error_k": float(np.mean(np.abs(got - want))),
            "max_abs_error_k": float(np.max(np.abs(got - want))),
            "monotone_steps": int((diffs >= 0).sum()),
            "total_steps": int(diffs.size),
            "correlation": float(np.corrcoef(got, want)[0, 1]) if got.size > 2 else float("nan"),
        }

    # ------------------------------------------------------------ lifecycle

    def save(self, directory: str | Path | None = None) -> Path:
        directory = Path(directory or ARTIFACT_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        self.readout.save(directory / "colour_readout.npz")
        meta = {
            "readout": "linear ridge regression, per-neuron rates -> Kelvin",
            "input_roles": self.config.input_roles,
            "input_pool": int(self.input_indices.size),
            "readout_roles": self.config.readout_roles,
            "readout_pool": int(self.readout_indices.size),
            "readout_features": int(self.readout.n_features),
            "window_ms": self.config.window_ms,
            #: The live loop needs `seed` to rebuild exactly these neuron pools. Without
            #: it, the loop would pick a different random subset and the learned weights
            #: would be meaningless.
            "seed": self.config.seed,
            "regime": "continuous sweep, one window per temperature", 
            "temp_range_c": [self.config.temp_min_c, self.config.temp_max_c],
            "sensor_rate_hz": [self.config.min_rate_hz, self.config.max_rate_hz],
            "band_centres_k": {n: k for n, k in COLOUR_BANDS},
            "current_per_spike_mv": self.config.current_per_spike_mv,
            "ridge_l2": self.config.ridge_l2,
            "params": {
                "recurrent_scale": self.sim.params.recurrent_scale,
                "w_scale_mv": self.sim.params.w_scale_mv,
            },
        }
        (directory / "colour_meta.json").write_text(json.dumps(meta, indent=2))
        return directory


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    print("building the loop (loads the 138,639-neuron connectome, ~5 s) ...")
    loop = TemperatureColourLoop()
    print(
        f"  sensor drives {loop.input_indices.size} neurons "
        f"from {loop.config.input_roles} at "
        f"{loop.config.min_rate_hz:.0f}-{loop.config.max_rate_hz:.0f} Hz"
    )
    print(
        f"  readout listens to {loop.readout_indices.size} neurons "
        f"from {loop.config.readout_roles}"
    )
    print(f"  readout features per sample: {loop.readout.n_features}")

    print(f"\ntraining over {loop.config.temp_min_c:.0f}-{loop.config.temp_max_c:.0f} C ...")
    stats = loop.fit()
    print(f"  {stats['n_samples']} brain windows in {stats['seconds']}s")
    print(f"  training mean |error|: {stats['train_mean_abs_error_k']:.1f} K")

    print("\ntemperature -> light colour (held-out temperatures)")
    print(f"  {'temp':>7}  {'colour':>7}  {'ideal':>7}  {'err':>7}   band")
    report = loop.evaluate()
    for row in report["rows"]:
        print(
            f"  {row['celsius']:6.2f}C  {row['kelvin']:5d}K  {row['ideal_kelvin']:5d}K  "
            f"{row['error_k']:+5d}K   {row['band']}"
        )
    print(
        f"\n  held-out mean |error| {report['mean_abs_error_k']:.0f} K  "
        f"(max {report['max_abs_error_k']:.0f} K)   "
        f"monotone steps {report['monotone_steps']}/{report['total_steps']}   "
        f"correlation {report['correlation']:.4f}"
    )

    path = loop.save()
    print(f"\nsaved readout + metadata to {path}")

    action = loop.to_action(28.0)
    print(
        f"\nexample: 28C -> HA call  {action.entity_id} {action.service} "
        f"{action.data}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
