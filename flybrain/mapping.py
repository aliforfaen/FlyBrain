"""Resolve logical sensorimotor roles onto connectome neuron indices.

The FlyWire v783 connectome comes with a public annotation table
(``data/annotations/flywire_annotations_supl1.tsv``, from the
``flyconnectome/flywire_annotations`` release) covering 138,625 of the 138,639
neurons. This module joins that table onto connectome row indices and resolves
roles such as ``thermosensory`` or ``descending`` to concrete neuron indices.

Roles are deliberately biological: a Home Assistant temperature sensor is encoded
onto the fly's actual thermosensory neurons, not an arbitrary slice of the network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Role:
    """A named population of connectome neurons.

    Attributes
    ----------
    name:
        Logical role, e.g. ``thermosensory``.
    indices:
        Connectome row indices belonging to the role.
    description:
        Human-readable note about the underlying biology.
    """

    name: str
    indices: np.ndarray
    description: str = ""

    def __len__(self) -> int:
        return int(self.indices.size)


#: Roles used by the sensorimotor loop, as (name, predicate-description).
#: Each entry maps a role to a filter over the annotation table.
ROLE_SPECS: dict[str, dict] = {
    "thermosensory": {
        "column": "cell_class",
        "values": ["thermosensory"],
        "description": "Hot/cold sensing (TRN_VP1m/VP2/VP3a/VP3b), 3rd-order thermosensory neurons",
    },
    "hygrosensory": {
        "column": "cell_class",
        "values": ["hygrosensory"],
        "description": "Humidity sensing (HRN_VP1d/VP1l/VP4/VP5)",
    },
    "mechanosensory": {
        "column": "cell_class",
        "values": ["mechanosensory"],
        "description": "Touch/vibration sensing, useful for motion and contact",
    },
    "olfactory": {
        "column": "cell_class",
        "values": ["olfactory"],
        "description": "Odour sensing (ORN/ALPN), a large sensory population",
    },
    "gustatory": {
        "column": "cell_class",
        "values": ["gustatory"],
        "description": "Taste sensing",
    },
    "visual": {
        "column": "cell_class",
        "values": ["visual"],
        "description": "Photoreceptors R1-6/R7/R8, useful for illuminance",
    },
    "antennial_projection": {
        "column": "cell_class",
        "values": ["ALPN"],
        "description": "Antennal-lobe projection neurons: second-order olfactory output",
    },
    "antennial_local": {
        "column": "cell_class",
        "values": ["ALLN"],
        "description": "Antennal-lobe local neurons",
    },
    "kenyon_cells": {
        "column": "cell_class",
        "values": ["Kenyon_Cell"],
        "description": "Mushroom-body intrinsic neurons: associative learning and memory",
    },
    "descending": {
        "column": "super_class",
        "values": ["descending"],
        "description": "Descending neurons carrying brain output to the ventral nerve cord",
    },
    "motor": {
        "column": "super_class",
        "values": ["motor"],
        "description": "Brain and neck motor neurons: the fly's motor output",
    },
    "mushroom_body_output": {
        "column": "cell_class",
        "values": ["MBON"],
        "description": "Mushroom-body output neurons",
    },
    # NOTE: listed by explicit ``cell_type``, deliberately *not* by a name prefix. The
    # neighbouring types ``lLN1_bc``, ``lLN2P_a``, ``lLN2X12``, ... read like "large lateral
    # neurons" and a loose ``^lLN`` match pulls in 158 of them — but their ``cell_class`` is
    # ``ALLN``, antennal-lobe **local** neurons, i.e. olfactory interneurons, and none of them
    # is a clock cell. That prefix both over-counts by 3x and misses the real clock. The
    # circadian set is small and its members are named individually.
    "clock": {
        "column": "cell_type",
        "values": [
            "l-LNv",
            "s-LNv_a",
            "s-LNv_b",
            "LNd_a",
            "LNd_b",
            "LNd_c",
            "DN1a",
            "DN1pA",
            "DN1pB",
            "DN1-l",
        ],
        "description": "Circadian clock neurons (l-LNv/s-LNv/LNd/DN1), 48 cells: slow context",
    },
}


def default_sensor_roles() -> list[str]:
    """Roles suitable for encoding Home Assistant *sensor* values."""
    return ["thermosensory", "hygrosensory", "mechanosensory", "visual", "olfactory"]


def default_output_roles() -> list[str]:
    """Roles suitable for decoding *actions* from (brain output)."""
    return ["descending", "motor"]


class RoleResolver:
    """Resolves role names to connectome indices using the annotation table."""

    def __init__(self, annotation_table, n_neurons: int | None = None) -> None:
        self.table = annotation_table
        self.n_neurons = int(n_neurons) if n_neurons else len(annotation_table)
        self._cache: dict[str, Role] = {}

    @classmethod
    def from_sim(cls, sim) -> RoleResolver:
        """Build from a loaded :class:`flybrain.sim.ConnectomeSim`."""
        if sim.annotation_table is None:
            raise RuntimeError(
                "simulator has no annotation table; load it with annotations enabled"
            )
        return cls(sim.annotation_table, sim.n_neurons)

    # ------------------------------------------------------------- resolution

    def resolve(self, role: str) -> Role:
        """Return the :class:`Role` for ``role``, resolving and caching it."""
        if role in self._cache:
            return self._cache[role]
        if role not in ROLE_SPECS:
            raise KeyError(
                f"unknown role {role!r}; known roles: {sorted(ROLE_SPECS)}"
            )
        spec = ROLE_SPECS[role]
        mask = self.table[spec["column"]].isin(spec["values"])
        if "extra_mask" in spec:
            mask &= spec["extra_mask"](self.table)
        indices = np.flatnonzero(mask.to_numpy()).astype(np.int32)
        resolved = Role(name=role, indices=indices, description=spec["description"])
        self._cache[role] = resolved
        logger.info("role %-20s -> %6d neurons", role, len(resolved))
        return resolved

    def resolve_many(self, roles: list[str]) -> dict[str, Role]:
        return {r: self.resolve(r) for r in roles}

    def resolve_subtypes(self, role: str) -> dict[str, np.ndarray]:
        """Break a role down by ``cell_type``, e.g. TRN_VP1m vs TRN_VP2.

        Returns a mapping of cell type to connectome indices.
        """
        resolved = self.resolve(role)
        if resolved.indices.size == 0:
            return {}
        sub = self.table.iloc[resolved.indices]
        frame = sub.assign(_idx=resolved.indices)
        out: dict[str, np.ndarray] = {}
        for cell_type, group in frame.groupby(frame["cell_type"].fillna("unknown")):
            out[str(cell_type)] = group["_idx"].to_numpy().astype(np.int32)
        return out

    def pool(
        self,
        roles: list[str],
        size: int,
        seed: int = 0,
        balanced: bool = True,
    ) -> np.ndarray:
        """Build a neuron pool of exactly ``size`` indices drawn from ``roles``.

        Small biological populations are cycled (with repetition) so that any sensor can
        drive a pool large enough to carry a rate code; ``balanced=True`` spreads the pool
        evenly across roles before cycling.
        """
        if size <= 0:
            return np.zeros(0, dtype=np.int32)
        rng = np.random.default_rng(seed)
        pools = [self.resolve(r).indices for r in roles]
        pools = [p for p in pools if p.size]
        if not pools:
            raise ValueError(f"none of the roles {roles} resolved to any neurons")

        if balanced and len(pools) > 1:
            per = max(1, size // len(pools))
            chunks = []
            for p in pools:
                if p.size >= per:
                    chunks.append(rng.choice(p, size=per, replace=False))
                else:
                    reps = int(np.ceil(per / p.size))
                    chunks.append(np.tile(p, reps)[:per])
            base = np.concatenate(chunks)
        else:
            base = np.concatenate(pools)

        if base.size >= size:
            chosen = rng.choice(base, size=size, replace=False)
        else:
            reps = int(np.ceil(size / base.size))
            chosen = np.tile(base, reps)[:size]
        return chosen.astype(np.int32)

    def summary(self) -> dict:
        """Counts for every known role, for inspection."""
        out = {}
        for role in ROLE_SPECS:
            out[role] = len(self.resolve(role))
        return out

    # ------------------------------------------------------------ persistence

    def export_roles(self, roles: list[str], path: str | Path) -> Path:
        """Write resolved roles to JSON so downstream tools need not re-resolve."""
        payload = {
            r: {
                "indices": self.resolve(r).indices.tolist(),
                "description": self.resolve(r).description,
            }
            for r in roles
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    @staticmethod
    def load_roles(path: str | Path) -> dict[str, Role]:
        raw = json.loads(Path(path).read_text())
        return {
            name: Role(name=name, indices=np.asarray(v["indices"], dtype=np.int32),
                       description=v.get("description", ""))
            for name, v in raw.items()
        }
