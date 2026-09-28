"""Build the normalised neuron position buffer used by the 3D live view.

Downloads the public Codex coordinate table if it is missing, joins it onto the
connectome's neuron order, converts FlyWire voxel units to microns, centres the cloud
and rescales it to roughly [-1, 1] so the renderer can use a simple camera.

    .venv/bin/python tools/build_positions.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODEX_DIR = ROOT / "data/codex"
COORDS_GZ = CODEX_DIR / "coordinates.csv.gz"
COMPLETENESS = ROOT / "vendor/fly-brain/data/2025_Completeness_783.csv"
OUT = CODEX_DIR / "positions_normalized.npy"
#: Provenance for the buffer, committed alongside the readout and documented in docs/data.md.
#: It was written by hand once and the script stopped producing it, which is how a documented
#: artifact goes stale; write it here so a regeneration cannot leave the two disagreeing.
META = CODEX_DIR / "positions_meta.json"

URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/coordinates.csv.gz"
VOXEL_NM = np.array([4.0, 4.0, 40.0], dtype=np.float32)  # FlyWire FAFB voxel size


def ensure_coordinates() -> Path:
    if COORDS_GZ.exists():
        return COORDS_GZ
    CODEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, COORDS_GZ)
    return COORDS_GZ


def main() -> int:
    path = ensure_coordinates()
    raw = pd.read_csv(path)
    # Multiple supervoxels share a root id; keep the first position per neuron.
    raw["root_id"] = raw["root_id"].astype(np.int64)
    raw = raw.drop_duplicates(subset="root_id")
    # `position` looks like "[352484 175164 229040]" and splits into 4 tokens naively.
    parts = raw["position"].astype(str).str.extract(r"\[\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\]")
    raw = raw.assign(x=parts[0], y=parts[1], z=parts[2]).dropna(subset=["x", "y", "z"])
    table = raw.drop_duplicates(subset="root_id").set_index("root_id")[["x", "y", "z"]]

    ids = pd.read_csv(COMPLETENESS, index_col=0).index.to_numpy(np.int64)
    xyz = table.reindex(ids).to_numpy(np.float32)
    if np.isnan(xyz).any():
        raise SystemExit(f"{(np.isnan(xyz).any(axis=1)).sum()} neurons have no coordinate")

    microns = (xyz * VOXEL_NM) / 1000.0
    microns -= microns.mean(axis=0)
    normalised = microns / float(np.abs(microns).max())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, normalised.astype(np.float32))
    # Deliberately byte-identical to the committed artifact, down to the integer nanometres and
    # the absent trailing newline: if a regeneration produced a spurious diff, the file would stop
    # being trustworthy as provenance for the buffer.
    META.write_text(
        json.dumps(
            {
                "n": int(normalised.shape[0]),
                "scale_um": float(np.abs(microns).max()),
                "voxel_nm": [int(v) for v in VOXEL_NM.tolist()],
                "source": "codex fafb 783 coordinates.csv.gz",
            },
            indent=2,
        )
    )
    extent = (microns.max(axis=0) - microns.min(axis=0)).round(1)
    print(f"wrote {OUT}  shape={normalised.shape}  extent={extent.tolist()} um")
    print(f"wrote {META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
