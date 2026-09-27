"""Plain-English names for broad cell families, and which end of the brain is which.

The connectome's own vocabulary is precise and unreadable. ``super_class`` gives ten values —
``optic``, ``central``, ``visual_projection`` — and ``cell_class`` gives fifty more like ``ME>LO``
and ``ALPN``. Those are the right labels for the data and the wrong ones for a person looking at
the screen, which is why this module exists: it maps the publisher's values onto names and
descriptions that say what the cells *do*, for a reader who is not a neurobiologist.

Two disciplines are inherited from :mod:`flybrain.pet` and are worth stating, because both are
easy to erode:

1. **The vocabulary is closed and lives in code, not in the data.** Every family the connectome
   can report has an entry here with a name, a description and a colour, and there is a test that
   the set matches the published values. A value that arrives without an entry is rendered as
   unlabelled rather than guessed at.
2. **Nothing here is invented biology.** The descriptions restate what ``mapping.py`` and the
   annotation table already assert. Where the anatomy is a *fact about this dataset* rather than
   something derivable — which end is the front — it is stated as a constant and *verified by a
   test against the raw data*, not recomputed by a heuristic at runtime that could drift.

Families are broader than cell classes on purpose. Colouring fifty classes at once produces
confetti, and 23% of neurons have no ``cell_class`` at all (``super_class`` is missing for only 14
of 138,639).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Family id 0 is reserved for neurons the annotation table does not classify, so the legend can
#: account for every point in the cloud. Real families are numbered from 1.
UNLABELLED_ID = 0


@dataclass(frozen=True)
class Family:
    """One broad group of cells, named for a person rather than for the dataset."""

    id: int
    key: str
    """The publisher's ``super_class`` value, verbatim."""
    label: str
    """Plain English, for the legend."""
    blurb: str
    """One sentence on what these cells do."""
    colour: str
    """Legend swatch and the hue the shader tints this family with."""


#: The ten published ``super_class`` values, ordered by size (largest first) so the legend reads
#: top-down by how much of the brain each one is. ``id`` is baked into the binary id buffer, so
#: **the numbers must not be reordered** without also treating it as a wire-format change.
FAMILIES: tuple[Family, ...] = (
    Family(
        1, "optic", "The eyes' own processing",
        "The optic lobe: the first place visual signals are sorted, before they reach the brain.",
        "#4b7fd0",
    ),
    Family(
        2, "central", "Central brain — where it comes together",
        "The integration machinery: learning, memory, and the circuits that decide what to do.",
        "#a06fd0",
    ),
    Family(
        3, "sensory", "Senses coming in",
        "Sensory neurons reporting what the body and the antennae are detecting.",
        "#35b0a4",
    ),
    Family(
        4, "visual_projection", "From the eyes into the brain",
        "Cells that carry what the eyes saw out of the optic lobe and into the central brain.",
        "#56c8e8",
    ),
    Family(
        5, "ascending", "Messages up from the body",
        "Neurons bringing information from the body up into the brain.",
        "#e0a33c",
    ),
    Family(
        6, "descending", "Commands down to the body",
        "Neurons carrying the brain's decisions down towards the body.",
        "#e5764a",
    ),
    Family(
        7, "sensory_ascending", "Senses coming up from the body",
        "Sensory cells whose signals travel up from the body rather than entering the head.",
        "#c9d15e",
    ),
    Family(
        8, "visual_centrifugal", "The brain talking back to the eyes",
        "Feedback from the brain into the optic lobe — the eyes are not a one-way camera.",
        "#6fbf6a",
    ),
    Family(
        9, "motor", "Movement commands",
        "Neurons that drive muscles, including the ones that steer the head and neck.",
        "#e05a86",
    ),
    Family(
        10, "endocrine", "Hormone signalling",
        "Cells that release hormones rather than talking to other neurons directly.",
        "#c06fd6",
    ),
)

#: Cells the table does not classify. Only 14 neurons in v783, and the optic lobe's are excluded
#: from this table, so a large remainder here would mean the join went wrong rather than that the
#: fly is unlabelled — the legend shows the count so that is visible.
UNLABELLED = Family(
    UNLABELLED_ID, "unlabelled", "Not yet classified",
    "Neurons the annotation table does not place in any family.",
    "#5a6472",
)

#: Every family the legend can show, unlabelled last.
ALL_FAMILIES: tuple[Family, ...] = (*FAMILIES, UNLABELLED)

#: ``key`` -> family, for turning publisher values into ids.
FAMILY_BY_KEY: dict[str, Family] = {f.key: f for f in FAMILIES}


@dataclass(frozen=True)
class Sense:
    """One sensory pathway, as a thing a person can choose to follow."""

    id: int
    key: str
    """Also the :data:`flybrain.mapping.ROLE_SPECS` key that resolves to neuron indices."""
    label: str
    blurb: str
    trained: bool = False
    """True for the one pathway the colour readout was actually fitted on."""


#: The sensory roles in :data:`flybrain.mapping.ROLE_SPECS`, in a fixed order. Ids are baked into
#: the binary id buffer alongside the families, so this order is also a wire format.
#:
#: ``gustatory`` is included even though ``default_sensor_roles()`` omits it and this house has no
#: taste sensor: the neurons exist, they can still be spotlit, and the panel says the pathway is
#: not wired to anything rather than pretending it is absent.
SENSE_GROUPS: tuple[Sense, ...] = (
    Sense(
        1, "thermosensory", "Warmth and cold",
        "The pathway the light colour is read from. Temperature drives these, this drives the light.",
        trained=True,
    ),
    Sense(2, "hygrosensory", "Humidity", "Antennal cells that report how damp the air is."),
    Sense(3, "mechanosensory", "Touch and vibration", "Cells that feel contact, wind and movement."),
    Sense(4, "olfactory", "Smell", "Odour detectors in the antennae and the lobe behind them."),
    Sense(5, "gustatory", "Taste", "Taste receptors. This house has no sensor wired to them."),
    Sense(6, "visual", "Light and sight", "The photoreceptors: how much light is falling on the fly."),
)

SENSE_BY_KEY: dict[str, Sense] = {s.key: s for s in SENSE_GROUPS}


# --------------------------------------------------------------------------- ids


def family_ids(super_class: np.ndarray) -> np.ndarray:
    """Map a per-neuron array of ``super_class`` values to uint8 family ids.

    Takes an array rather than a DataFrame so it can be tested without pandas or the connectome.
    Anything unrecognised — including NaN, which is what the 14 unclassified cells carry — becomes
    :data:`UNLABELLED_ID` rather than raising: a rendering path must not fail because the
    publisher added a value, it must show the gap.
    """
    values = np.asarray(super_class, dtype=object)
    out = np.full(values.shape[0], UNLABELLED_ID, dtype=np.uint8)
    for family in FAMILIES:
        # `==` against an object array containing NaN yields False, which is what we want.
        out[values == family.key] = family.id
    return out


def sense_ids(role_indices: dict[str, np.ndarray], n_neurons: int) -> np.ndarray:
    """Map resolved role indices to uint8 sense ids, 0 meaning "not part of a sense pathway".

    A neuron belongs to at most one sensory pathway here — the roles are disjoint by construction
    in ``mapping.py`` — but if two ever overlapped the later id would win, which is why the caller
    passes a mapping rather than a list.
    """
    out = np.zeros(int(n_neurons), dtype=np.uint8)
    for sense in SENSE_GROUPS:
        indices = role_indices.get(sense.key)
        if indices is None or len(indices) == 0:
            continue
        out[np.asarray(indices, dtype=np.int64)] = sense.id
    return out


def group_id_buffer(family: np.ndarray, sense: np.ndarray) -> bytes:
    """Interleave the two id arrays into the 2-bytes-per-neuron wire buffer.

    Family first, then sense, so the client can read either without knowing the stride.
    """
    fam = np.ascontiguousarray(family, dtype=np.uint8)
    sen = np.ascontiguousarray(sense, dtype=np.uint8)
    if fam.shape != sen.shape:
        raise ValueError(f"family and sense id arrays differ in shape: {fam.shape} vs {sen.shape}")
    interleaved = np.empty((fam.shape[0], 2), dtype=np.uint8)
    interleaved[:, 0] = fam
    interleaved[:, 1] = sen
    return interleaved.tobytes()


def family_counts(family: np.ndarray) -> dict[int, int]:
    """Neuron count per family id, including ids with no members in this build."""
    fam = np.asarray(family, dtype=np.uint8)
    counts = np.bincount(fam, minlength=len(ALL_FAMILIES) + 1)
    return {family_.id: int(counts[family_.id]) for family_ in ALL_FAMILIES}


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Marker:
    """A labelled place to put an orientation marker, in the same units as the positions."""

    key: str
    label: str
    anchor: tuple[float, float, float]


def orientation_from(positions: np.ndarray, *, inset_frac: float = 0.09) -> dict[str, Marker]:
    """Where to draw the front/back/left/right markers.

    **Which way is front is a fact about this dataset, not a heuristic**, and it is not derivable
    from the positions alone. It is established by three landmarks in the published annotations,
    all of which agree that the front of the head is *negative z*:

    ============================================  =========  ==========================
    landmark                                      mean z     why it settles the question
    ============================================  =========  ==========================
    photoreceptors ``R1-6`` (the retina)           −0.175    the retina is the front of the eye
    ``TRN`` antennal thermoreceptors               −0.830    the antennae sit in front of the brain
    ``ascending`` neurons (from the body)          +0.38     the neck is at the back
    ============================================  =========  ==========================

    ``left`` is negative x, confirmed against the annotation table's ``side`` column: neurons
    marked ``left`` average x = −0.123 and ``right`` averages +0.124, covering all 138,639 cells.
    No claim is made about up and down — the cloud is only ~0.24 deep in y, and the renderer uses
    the same axis for depth so a label there would be more confusing than useful.

    The markers sit **on** the ends of the brain, inset from the extreme by a fraction of each
    axis's own span. Two reasons, and the second is the one that forced it:

    * an anatomical figure marks the ends of the structure rather than floating labels beside it;
    * **the cloud is wider than the gap between the dashboard's two columns.** A label at the
      literal extreme is projected under a panel and cannot be read. Insetting pulls it into the
      visible strip, and the labels are kept short for the same reason — the full anatomy is
      spelled out in the guided walkthrough and in these docstrings, where there is room for it.
    """
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if pos.shape[0] == 0:
        raise ValueError("cannot orient an empty point cloud")
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    mid = (lo + hi) / 2.0
    cx, cy, cz = (float(v) for v in mid)
    ins = span * float(inset_frac)

    return {
        "front": Marker("front", "front · eyes", (cx, cy, float(lo[2]) + float(ins[2]))),
        "back": Marker("back", "back · body", (cx, cy, float(hi[2]) - float(ins[2]))),
        "left": Marker("left", "left", (float(lo[0]) + float(ins[0]), cy, cz)),
        "right": Marker("right", "right", (float(hi[0]) - float(ins[0]), cy, cz)),
    }


def front_back_evidence(cell_type: np.ndarray, positions: np.ndarray) -> dict[str, float | None]:
    """Mean z of the landmarks that establish which end is the front.

    Shipped in the payload and asserted in the tests, so the claim in
    :func:`orientation_from` is reproducible from the data rather than taken on trust. Returns
    ``None`` for a landmark this build cannot find, which the caller should show as "not checked"
    rather than as zero.
    """
    types = np.asarray(cell_type, dtype=object)
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    out: dict[str, float | None] = {}
    for name, pattern in (("retina_r1_6", "R1-6"), ("antennal_trn", "TRN")):
        mask = np.array([pattern in str(v) for v in types], dtype=bool)
        out[name] = float(pos[mask, 2].mean()) if mask.any() else None
    return out


# --------------------------------------------------------------------------- payload


def families_payload(family: np.ndarray) -> list[dict]:
    """The family list for ``/api/groups``: name, description, colour and neuron count."""
    counts = family_counts(family)
    return [
        {
            "id": f.id,
            "key": f.key,
            "label": f.label,
            "blurb": f.blurb,
            "colour": f.colour,
            "neurons": counts[f.id],
        }
        for f in ALL_FAMILIES
    ]


def senses_payload(
    sense: np.ndarray, wired: dict[str, bool] | None = None
) -> list[dict]:
    """The sense list for ``/api/groups``.

    ``wired`` says whether a live Home Assistant sensor feeds this pathway. It is reported rather
    than filtered on: a pathway with no sensor is still a real set of neurons that can be shown,
    and hiding it would make the menu disagree with the brain on screen.
    """
    idle = wired or {}
    sense = np.asarray(sense, dtype=np.uint8)
    counts = np.bincount(sense, minlength=len(SENSE_GROUPS) + 1)
    return [
        {
            "id": s.id,
            "key": s.key,
            "label": s.label,
            "blurb": s.blurb,
            "trained": s.trained,
            "wired": bool(idle.get(s.key, False)),
            "neurons": int(counts[s.id]),
        }
        for s in SENSE_GROUPS
    ]


def groups_payload(
    *,
    n_neurons: int,
    family: np.ndarray,
    sense: np.ndarray,
    positions: np.ndarray,
    cell_type: np.ndarray | None = None,
    wired: dict[str, bool] | None = None,
) -> dict:
    """Everything the client needs to colour by family and label the axes, in one response.

    Deliberately one payload rather than three: the legend, the id buffer and the orientation
    markers are only meaningful together, and fetching them separately would let a client render
    a legend whose ids do not match the buffer it is colouring.
    """
    markers = orientation_from(positions)
    payload: dict = {
        "n": int(n_neurons),
        "families": families_payload(family),
        "senses": senses_payload(sense, wired),
        "orientation": {
            key: {"key": m.key, "label": m.label, "anchor": list(m.anchor)}
            for key, m in markers.items()
        },
    }
    if cell_type is not None:
        payload["evidence"] = front_back_evidence(cell_type, positions)
    return payload
