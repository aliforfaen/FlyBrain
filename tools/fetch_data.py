"""Fetch the connectome and annotation tables this project runs on.

The repository itself stays small: the four source tables below are downloaded from their
public upstream locations on first run. All four are **anonymous** — no account, no token, no
Codex sign-in, nothing to configure.

    .venv/bin/python tools/fetch_data.py            # download whatever is missing
    .venv/bin/python tools/fetch_data.py --check    # report status, download nothing
    .venv/bin/python tools/fetch_data.py --force    # re-download everything

Each download is verified against the upstream byte count and only moved into place once the
count matches. An interrupted transfer therefore cannot leave a truncated parquet behind that
fails much later inside the simulator with a confusing error.

The data is **FlyWire v783 (FAFB)** and is licensed **CC BY-NC 4.0 — non-commercial**. See
`docs/licensing.md` before shipping anything built on it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1 << 20  # 1 MiB
TIMEOUT_S = 120


@dataclass(frozen=True)
class Source:
    """One upstream file, with the byte count used to verify it."""

    dest: str
    url: str
    size: int
    what: str
    licence: str


SOURCES: tuple[Source, ...] = (
    Source(
        dest="vendor/fly-brain/data/2025_Connectivity_783.parquet",
        url=(
            "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/"
            "data/2025_Connectivity_783.parquet"
        ),
        size=100_804_642,
        what="FlyWire v783 connectivity — 15,091,983 (pre, post) synapse pairs",
        licence="CC BY-NC 4.0",
    ),
    Source(
        dest="vendor/fly-brain/data/2025_Completeness_783.csv",
        url=(
            "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/"
            "data/2025_Completeness_783.csv"
        ),
        size=3_465_987,
        what="FlyWire v783 neuron list — 138,639 root ids; row index == connectome index",
        licence="CC BY-NC 4.0",
    ),
    Source(
        dest="data/annotations/flywire_annotations_supl1.tsv",
        url=(
            "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
            "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
        ),
        size=31_718_505,
        what="Cell typing — 139,248 rows; covers 138,625 of 138,639 neurons",
        licence="no licence file; attribute the papers",
    ),
    Source(
        dest="data/codex/coordinates.csv.gz",
        url="https://storage.googleapis.com/flywire-data/codex/data/fafb/783/coordinates.csv.gz",
        size=5_314_546,
        what="Neuron soma positions (Codex, FAFB v783)",
        licence="CC BY-NC 4.0",
    ),
)

POSITIONS = ROOT / "data/codex/positions_normalized.npy"


def human(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def download(source: Source, force: bool = False) -> bool:
    """Download one file if needed. Returns True if the file is present and complete."""
    dest = ROOT / source.dest
    if dest.exists() and not force:
        if dest.stat().st_size == source.size:
            print(f"  ok    {source.dest}  ({human(source.size)})")
            return True
        print(f"  stale {source.dest}  (have {human(dest.stat().st_size)}, want {human(source.size)})")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  get   {source.dest}  ({human(source.size)})")
    try:
        with requests.get(source.url, stream=True, timeout=TIMEOUT_S) as response:
            response.raise_for_status()
            written = 0
            with part.open("wb") as handle:
                for chunk in response.iter_content(CHUNK):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    # Only worth a progress line when a human is watching; keep logs clean
                    # when this runs from a script.
                    if sys.stdout.isatty():
                        pct = 100 * written / source.size
                        print(f"\r        {pct:5.1f}%  {human(written)}", end="", flush=True)
        if sys.stdout.isatty():
            print()
    except (requests.RequestException, OSError) as exc:
        part.unlink(missing_ok=True)
        print(f"  FAIL  {source.dest}: {exc}")
        return False

    actual = part.stat().st_size
    if actual != source.size:
        part.unlink(missing_ok=True)
        print(f"  FAIL  {source.dest}: got {human(actual)}, expected {human(source.size)}")
        return False
    part.replace(dest)
    print(f"  done  {source.dest}")
    return True


def check() -> int:
    """Report what is present without touching the network."""
    missing = 0
    for source in SOURCES:
        dest = ROOT / source.dest
        if not dest.exists():
            print(f"  MISSING  {source.dest}")
            missing += 1
        elif dest.stat().st_size != source.size:
            print(
                f"  SIZE     {source.dest}  "
                f"({human(dest.stat().st_size)}, want {human(source.size)})"
            )
            missing += 1
        else:
            print(f"  ok       {source.dest}  ({human(source.size)})")

    if POSITIONS.exists():
        print("  ok       data/codex/positions_normalized.npy")
    else:
        print("  MISSING  data/codex/positions_normalized.npy  (derived; needs the coordinates)")
        missing += 1

    print()
    print("everything present" if not missing else f"{missing} file(s) missing or incomplete")
    return 0 if not missing else 1


def build_positions() -> bool:
    """Derive the normalised position buffer the 3D view needs."""
    print("  run   tools/build_positions.py  (derive positions_normalized.npy)")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_positions.py")],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the FlyWire v783 connectome and annotation tables.",
    )
    parser.add_argument("--check", action="store_true", help="report status, download nothing")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--no-positions",
        action="store_true",
        help="skip deriving positions_normalized.npy for the 3D view",
    )
    args = parser.parse_args(argv)

    if args.check:
        return check()

    print("FlyWire v783 (FAFB) — CC BY-NC 4.0, non-commercial. See docs/licensing.md.")
    print()

    failed: list[str] = []
    for source in SOURCES:
        if not download(source, force=args.force):
            failed.append(source.dest)

    if failed:
        print()
        print(f"{len(failed)} download(s) failed. Nothing was left half-written.")
        print("Re-run to retry; completed files are skipped.")
        return 1

    if not args.no_positions and (args.force or not POSITIONS.exists()):
        print()
        if not build_positions():
            print("  FAIL  could not derive positions; the 3D view will fall back to a random layout")
            return 1
    elif POSITIONS.exists():
        print("  ok    data/codex/positions_normalized.npy")

    print()
    print("Data ready. Next:  .venv/bin/python -m flybrain.server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
