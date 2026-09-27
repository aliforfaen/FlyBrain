"""Record what the brain saw, so that new readouts can be taught afterwards.

The live loop runs and forgets. That is fine for the one thing it was trained for and fatal for
everything else: a second readout needs *examples* of the state you want it to recognise, and a
state that was never written down cannot be replayed. This module is the prerequisite for every
idea in ``docs/roadmap.md``.

What is written, per completed window:

* **features** — the readout population's firing rates, one ``float32`` row per window. This is
  exactly the input a linear readout is fitted on, so a recording is directly trainable.
* **sensors** — every Home Assistant entity value at that moment, so a target can be built from
  what the house was actually doing.
* **labels** — timestamped, append-only, written by hand when something worth learning happens.

Storage is deliberately crude and crash-safe: raw little-endian ``float32`` appended to one binary
file, plus two JSONL files. There is no database, no compression and no schema migration, because
the whole point is that a power cut costs the last window and nothing else. Rows are flushed on
every write for the same reason.

    data/recordings/<name>/
        meta.json      feature_dim, window_ms, roles, start time
        features.f32   n_windows * feature_dim float32, row per window
        windows.jsonl  one {"t":..,"sensors":{..}} per recorded window
        labels.jsonl   one {"t":..,"label":..,"source":..} per label event
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

logger = logging.getLogger(__name__)

#: Default location for recordings; gitignored (see .gitignore).
DEFAULT_ROOT = Path("data/recordings")

FEATURES_FILE = "features.f32"
WINDOWS_FILE = "windows.jsonl"
LABELS_FILE = "labels.jsonl"
META_FILE = "meta.json"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        fh.flush()


def _repair_partial_row(path: Path, row_bytes: int) -> int:
    """Truncate trailing bytes that do not form a whole row; returns how many were dropped.

    A hard kill during ``write()`` can leave a fragment. Leaving it there is worse than it
    sounds: the next ``record()`` appends *after* the fragment, so from that point on every row
    is shifted by a constant offset relative to ``windows.jsonl``. The row *count* still looks
    plausible, so nothing downstream notices — one moment's sensor reading is silently paired
    with another moment's spike vector. Repairing on open is what stops the file ever getting a
    partial row in the middle, which no reader could disentangle after the fact.
    """
    if not path.exists():
        return 0
    size = path.stat().st_size
    usable = (size // row_bytes) * row_bytes
    if usable == size:
        return 0
    with path.open("r+b") as fh:
        fh.truncate(usable)
    return size - usable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line is the expected consequence of a hard kill mid-write;
                # losing it is better than refusing to open the recording at all.
                logger.warning("ignoring malformed trailing line in %s", path)
    return out


@dataclass(frozen=True)
class Label:
    """One hand-applied annotation of a moment in time."""

    t: float
    label: str
    source: str = "manual"


class Recorder:
    """Append-only writer for one recording session.

    Cheap enough to call once per control window (roughly every 2.3 s at present throughput).
    """

    def __init__(
        self,
        root: Path,
        feature_dim: int,
        window_ms: float,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.feature_dim = int(feature_dim)
        self.window_ms = float(window_ms)
        self.root.mkdir(parents=True, exist_ok=True)

        self._features_path = self.root / FEATURES_FILE
        self._windows_path = self.root / WINDOWS_FILE
        self._labels_path = self.root / LABELS_FILE
        self._meta_path = self.root / META_FILE

        # Repair a partial trailing row *before* opening for append. This is the root-cause fix
        # for the misalignment: once a fragment is there and the next row is appended after it,
        # no reader can work out where the rows really start.
        row_bytes = 4 * self.feature_dim
        dropped = _repair_partial_row(self._features_path, row_bytes)
        if dropped:
            logger.warning(
                "truncated %d stray byte(s) of a partial feature row in %s",
                dropped,
                self._features_path,
            )

        self._fh = self._features_path.open("ab")
        self._n_windows = self._features_path.stat().st_size // row_bytes

        if not self._meta_path.exists():
            payload = {
                "feature_dim": self.feature_dim,
                "window_ms": self.window_ms,
                "started": time.time(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            if meta:
                payload.update(meta)
            self._meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            existing = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if int(existing.get("feature_dim", -1)) != self.feature_dim:
                raise ValueError(
                    f"recording at {self.root} has feature_dim "
                    f"{existing.get('feature_dim')}, not {self.feature_dim}"
                )

    # ------------------------------------------------------------------ write

    @property
    def n_windows(self) -> int:
        """Windows recorded so far. Derived from file size, so it survives a restart."""
        return self._n_windows

    def record(
        self,
        features: np.ndarray,
        sensors: dict[str, float] | None = None,
        t: float | None = None,
    ) -> None:
        """Append one window's feature vector and the sensor values that produced it."""
        row = np.asarray(features, dtype=np.float32).reshape(-1)
        if row.size != self.feature_dim:
            raise ValueError(f"expected {self.feature_dim} features, got {row.size}")

        self._fh.write(row.tobytes())
        self._fh.flush()
        self._n_windows += 1
        _append_jsonl(
            self._windows_path,
            {"t": time.time() if t is None else float(t), "sensors": sensors or {}},
        )

    def label(self, text: str, t: float | None = None, source: str = "manual") -> Label:
        """Attach a label to *now*.

        Labels are separate events rather than per-window fields on purpose: a label describes a
        moment, and the windows around it are what a readout learns from. Writing it as an event
        means a label never has to be applied before the data it refers to exists.
        """
        if not text or not text.strip():
            raise ValueError("label text must not be empty")
        event = Label(t=time.time() if t is None else float(t), label=text.strip(), source=source)
        _append_jsonl(
            self._labels_path,
            {"t": event.t, "label": event.label, "source": event.source},
        )
        logger.info("labelled %s as %r", time.strftime("%H:%M:%S", time.localtime(event.t)), event.label)
        return event

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Recording:
    """Read-only view of a finished (or still-growing) recording."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        meta_path = self.root / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"no recording at {self.root}")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_dim = int(self.meta["feature_dim"])
        self.window_ms = float(self.meta.get("window_ms", 0.0))

    def _feature_rows(self) -> int:
        """Whole rows present in ``features.f32``, ignoring any trailing partial row."""
        path = self.root / FEATURES_FILE
        if not path.exists():
            return 0
        return path.stat().st_size // (4 * self.feature_dim)

    def paired_count(self) -> int:
        """How many rows are certainly paired across both files.

        ``Recorder.record`` appends the feature row and *then* the matching ``windows.jsonl``
        line, so a crash between the two writes leaves the files one apart — and nothing in
        either file records which row is the orphan. Rather than guess, this takes the shorter
        of the two. At worst one window is discarded, and it is discarded *deterministically*
        instead of pairing one moment's sensor reading with another moment's spike vector.
        """
        return min(self._feature_rows(), len(_read_jsonl(self.root / WINDOWS_FILE)))

    @property
    def features(self) -> np.ndarray:
        """``(n_windows, feature_dim)`` float32 firing rates, one row per window.

        Truncated to :meth:`paired_count` so it always lines up with :attr:`windows`. The rows
        kept are the *first* n, because both files are append-only and any orphan is therefore
        at the end.
        """
        raw = np.fromfile(self.root / FEATURES_FILE, dtype=np.float32)
        usable = (raw.size // self.feature_dim) * self.feature_dim
        if usable != raw.size:
            # A kill during a write can leave a partial row; drop it rather than fail.
            logger.warning("dropping %d trailing floats from a partial window", raw.size - usable)
        rows = usable // self.feature_dim
        keep = min(rows, self.paired_count())
        if keep < rows:
            logger.warning(
                "ignoring %d feature row(s) with no matching window (interrupted write?)",
                rows - keep,
            )
        return raw[: keep * self.feature_dim].reshape(-1, self.feature_dim)

    @property
    def windows(self) -> list[dict[str, Any]]:
        """One row per window, truncated to :meth:`paired_count` to match :attr:`features`."""
        rows = _read_jsonl(self.root / WINDOWS_FILE)
        keep = min(len(rows), self.paired_count())
        if keep < len(rows):
            logger.warning(
                "ignoring %d window row(s) with no matching feature row (interrupted write?)",
                len(rows) - keep,
            )
        return rows[:keep]

    @property
    def labels(self) -> list[Label]:
        return [Label(**row) for row in _read_jsonl(self.root / LABELS_FILE)]

    def sensor_matrix(self, entities: list[str]) -> tuple[np.ndarray, list[str]]:
        """Sensor values as a matrix, plus the entity names that were actually present.

        Missing readings become ``NaN`` rather than ``0.0``: an absent sensor is not a sensor
        reporting zero, and silently conflating them would teach a readout something false.
        """
        rows = self.windows
        present = [e for e in entities if any(e in r.get("sensors", {}) for r in rows)]
        out = np.full((len(rows), len(present)), np.nan, dtype=np.float64)
        for i, row in enumerate(rows):
            sensors = row.get("sensors", {})
            for j, entity in enumerate(present):
                value = sensors.get(entity)
                if value is not None:
                    out[i, j] = value
        return out, present

    def labelled(self, horizon_s: float = 120.0) -> tuple[np.ndarray, np.ndarray, list[Label]]:
        """Build ``(X, y, used)`` for supervised fitting, by nearest preceding label.

        A window takes the label of the most recent label event, provided that event is no more
        than ``horizon_s`` earlier. Windows further from any label are dropped: they describe a
        moment nobody annotated, and guessing at them is how a readout learns noise.
        """
        windows = self.windows
        labels = sorted(self.labels, key=lambda lb: lb.t)
        if not labels or not windows:
            return np.empty((0, self.feature_dim)), np.empty((0,), dtype=object), []

        ts = np.array([w.get("t", 0.0) for w in windows], dtype=np.float64)
        lab_ts = np.array([lb.t for lb in labels], dtype=np.float64)
        # For each window, the index of the last label at or before it.
        idx = np.searchsorted(lab_ts, ts, side="right") - 1

        keep = idx >= 0
        if not keep.any():
            return np.empty((0, self.feature_dim)), np.empty((0,), dtype=object), []
        age = ts[keep] - lab_ts[idx[keep]]
        fresh = age <= float(horizon_s)
        rows = np.flatnonzero(keep)[fresh]

        used_idx = idx[rows]
        features = self.features[rows]
        used = [labels[i] for i in used_idx]
        y = np.array([lb.label for lb in used], dtype=object)
        return features, y, used


def open_recording(path: str | Path) -> Recording:
    """Open a recording by directory path."""
    return Recording(Path(path))


def latest(root: str | Path = DEFAULT_ROOT) -> Path | None:
    """Most recently modified recording under ``root``, or ``None``."""
    base = Path(root)
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if (p / META_FILE).exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / META_FILE).stat().st_mtime)


def _main(argv: list[str] | None = None) -> int:
    """Small CLI so labelling needs no Python.

    ``python -m flybrain.recorder --label busy`` is the entire "label button": it can be bound to
    a phone shortcut, an HA ``shell_command``, or a key on the desktop.
    """
    parser = argparse.ArgumentParser(description="Inspect a recording, or label the latest one.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="recordings directory")
    parser.add_argument("--name", help="recording to act on (default: most recent)")
    parser.add_argument("--label", help="append this label to the recording, timestamped now")
    parser.add_argument("--source", default="cli", help="where the label came from")
    parser.add_argument("--list", action="store_true", help="list recordings and exit")
    parser.add_argument("--summary", action="store_true", help="show a recording's contents")
    args = parser.parse_args(argv)

    root = Path(args.root)

    if args.list:
        if not root.exists():
            print(f"no recordings in {root}")
            return 0
        for path in sorted(root.iterdir()):
            if (path / META_FILE).exists():
                rec = Recording(path)
                print(
                    f"{path.name:24} {rec.features.shape[0]:6d} windows  "
                    f"{len(rec.labels):4d} labels  dim={rec.feature_dim}"
                )
        return 0

    target = root / args.name if args.name else latest(root)
    if target is None:
        print(f"no recordings found in {root}")
        return 1

    if args.label:
        # Label-only writes must not disturb the feature file, so open it in the same way the
        # loop would; the constructor is idempotent with respect to an existing meta.json.
        rec = Recording(target)
        with Recorder(target, rec.feature_dim, rec.window_ms) as writer:
            writer.label(args.label, source=args.source)
        print(f"labelled {target.name}: {args.label}")
        return 0

    if args.summary:
        rec = Recording(target)
        print(f"{target}")
        print(f"  windows   {rec.features.shape[0]}")
        print(f"  dim       {rec.feature_dim}")
        print(f"  window_ms {rec.window_ms}")
        print(f"  labels    {len(rec.labels)}")
        for lb in rec.labels:
            print(f"    {time.strftime('%H:%M:%S', time.localtime(lb.t))}  {lb.label}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(_main())


def _iter_recordings(root: str | Path = DEFAULT_ROOT) -> Iterator[Recording]:
    """Yield every recording under ``root``, oldest first."""
    base = Path(root)
    if not base.exists():
        return
    for path in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime):
        if (path / META_FILE).exists():
            yield Recording(path)
