"""Which Home Assistant entity drives which part of the fly brain.

``loop.py`` wires exactly one sensor — a thermometer — onto one role, ``thermosensory``, and
turns the result into one light colour. That is a demonstration, not an architecture. This module
is the architecture: a declarative map from Home Assistant entities onto the fly's *real* sensory
populations, plus discovery that reads a live instance and proposes the wiring for you.

The point is to use the right pathway, not merely a working one. The connectome is not uniform:
it has 10,855 visual neurons, 2,656 mechanosensory, 2,279 olfactory, and 29 thermosensory. Wiring
a camera into a random slice of the brain throws away the part that evolved for seeing. So each
entity is matched to the population that would actually process it:

===================  ==========================  =========================================
Home Assistant       Fly pathway                 Why
===================  ==========================  =========================================
camera motion /      ``visual``                  a moving body is what the visual motion
person / pet                                     pathway detects (T4/T5 in the lobula plate)
illuminance          ``visual``                  photoreceptors R1-6/R7/R8
contact / door /     ``mechanosensory``          touch and vibration
window
temperature          ``thermosensory``           TRN_VP1m/VP2/VP3
humidity             ``hygrosensory``            HRN_VP1d/VP1l/VP4/VP5
VOC / CO2 / PM2.5    ``olfactory``               the fly's largest sensory investment
acoustic events      ``mechanosensory``          the fly hears through mechanosensory
(bark, meow, glass                               structures, not eyes
break, sound)
===================  ==========================  =========================================

The ordering rule is load-bearing. A camera may expose ``..._bark_detection`` as a binary sensor
whose ``device_class`` is **motion**, so classifying by device class alone wires an acoustic event
into the visual system. Sound is therefore matched before motion. Note the converse, found by
running discovery against a real Tapo C120: its bark/meow/glass-break entities are ``select.*``
*sensitivity settings*, not event sensors, so that camera currently produces **no** acoustic
events at all. The rule is ready for an acoustic sensor; this hardware does not have one.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flybrain.codec import ChannelSpec
from flybrain.mapping import RoleResolver, default_sensor_roles
from flybrain.types import Signal, SignalKind

logger = logging.getLogger(__name__)

#: Domains that represent something the brain can *sense*. Actuators (``light``, ``switch``,
#: ``media_player``, ``siren``) are deliberately excluded: they are things the loop would drive,
#: and feeding a lamp's on/off state back in as a sense would be circular.
SENSOR_DOMAINS = ("sensor.", "binary_sensor.")

#: Default value span per signal kind, i.e. "what counts as zero" and "what counts as full".
#: Binary kinds are 0..1 because that is what they are; a span of 0..1 on a temperature would
#: make every room look identical.
KIND_RANGES: dict[SignalKind, tuple[float, float]] = {
    SignalKind.TEMPERATURE: (10.0, 35.0),
    SignalKind.HUMIDITY: (20.0, 80.0),
    SignalKind.ILLUMINANCE: (0.0, 500.0),
    SignalKind.AUDIO: (0.0, 1.0),
    SignalKind.MOTION: (0.0, 1.0),
    SignalKind.CONTACT: (0.0, 1.0),
    SignalKind.POWER: (0.0, 500.0),
    SignalKind.OTHER: (0.0, 1.0),
}

#: ``(keywords, fly role, rationale)``. First match wins, so this tuple is ordered, not a set.
ROLE_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("bark", "meow", "glass", "sound", "audio", "noise"),
        "mechanosensory",
        "acoustic event: the fly hears through mechanosensory structures",
    ),
    (
        ("temperature", "temp"),
        "thermosensory",
        "thermosensory neurons (TRN_VP1m/VP2/VP3)",
    ),
    (
        ("humidity", "humid"),
        "hygrosensory",
        "hygrosensory neurons (HRN_VP1d/VP1l/VP4/VP5)",
    ),
    (
        ("voc", "co2", "co_2", "pm25", "pm2_5", "air_quality", "tvoc"),
        "olfactory",
        "odour sensing (ORN/ALPN): the fly's largest sensory population",
    ),
    (
        ("illuminance", "lux", "light_level", "brightness"),
        "visual",
        "photoreceptors R1-6/R7/R8",
    ),
    (
        ("motion", "movement", "person", "occupancy", "cell_motion", "pet"),
        "visual",
        "a moving body is what the visual motion pathway detects",
    ),
    (
        ("contact", "window", "opening", "garage", "vibration", "door"),
        "mechanosensory",
        "touch and vibration: mechanosensory",
    ),
)

#: Entity states that mean "this sensor is not reporting". A pathway on one of these is real
#: wiring with no data behind it yet, which is worth showing rather than silently dropping.
DEAD_STATES = {"unavailable", "unknown", "none", ""}


@dataclass(frozen=True)
class Pathway:
    """One wired connection: a Home Assistant entity driving a fly sensory population."""

    entity_id: str
    role: str
    kind: SignalKind
    vmin: float
    vmax: float
    gain: float = 1.0
    invert: bool = False
    note: str = ""

    def to_channel(self, indices: np.ndarray) -> ChannelSpec:
        """Build the encoder channel for this pathway over a given neuron pool."""
        return ChannelSpec(
            entity_id=self.entity_id,
            kind=self.kind,
            neuron_indices=np.asarray(indices, dtype=np.int32),
            vmin=self.vmin,
            vmax=self.vmax,
            gain=self.gain,
            invert=self.invert,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "role": self.role,
            "kind": self.kind.value,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "gain": self.gain,
            "invert": self.invert,
            "note": self.note,
        }


def _phrase(entity_id: str) -> str:
    """Entity id as a space-separated phrase, for whole-word matching.

    Substring matching is wrong here and produces confident nonsense: ``"temp"`` matches
    ``sensor.backup_last_attem**p**ted_automatic_backup``, and ``"temperature"`` matches
    ``sensor.rack_gpu**temperature`` (a GPU, not a room). Splitting on separators means a
    keyword has to be a whole word, so both false positives disappear.
    """
    return " " + entity_id.lower().replace(".", " ").replace("_", " ") + " "


def role_for(entity_id: str, kind: SignalKind | None = None) -> tuple[str, str] | None:
    """Best fly role for an entity, as ``(role, rationale)``, or ``None`` if unmatched.

    ``kind`` is accepted so callers can pass the already-classified signal, but the decision is
    made on the *name*: the name is what distinguishes a bark detector from a motion detector,
    and both arrive as ``SignalKind.MOTION``.
    """
    phrase = _phrase(entity_id)
    for keywords, role, note in ROLE_RULES:
        for k in keywords:
            if " " + k.replace("_", " ") + " " in phrase:
                return role, note
    return None


def pathway_for(signal: Signal) -> Pathway | None:
    """Build a :class:`Pathway` for one signal, or ``None`` if it has no sensible role."""
    match = role_for(signal.entity_id, signal.kind)
    if match is None:
        return None
    role, note = match
    vmin, vmax = KIND_RANGES.get(signal.kind, KIND_RANGES[SignalKind.OTHER])
    return Pathway(
        entity_id=signal.entity_id,
        role=role,
        kind=signal.kind,
        vmin=vmin,
        vmax=vmax,
        note=note,
    )


@dataclass
class Wiring:
    """The proposed (or configured) wiring for one home."""

    #: Pathways whose entity is currently reporting a usable value.
    pathways: list[Pathway]
    #: Pathways that match a role but whose entity reports nothing right now, with the state.
    #: Kept visible on purpose: a camera whose detectors are dark is a wiring problem to fix,
    #: not a wiring decision to hide.
    dormant: list[tuple[Pathway, str]]
    #: Sensor entities with no plausible pathway, with the reason.
    ignored: list[tuple[str, str]]

    # --------------------------------------------------------------- inspection

    def roles(self) -> list[str]:
        """Roles that have at least one live pathway, in a stable order."""
        return sorted({p.role for p in self.pathways})

    def all_roles(self) -> list[str]:
        """Roles that have any pathway at all, live or dormant."""
        return sorted({p.role for p in self.pathways} | {p.role for p, _ in self.dormant})

    def unreachable_roles(self) -> list[str]:
        """Known sensory roles this home has no sensor for.

        Worth reporting explicitly, because "no olfactory pathway" and "olfactory pathway that
        is silently broken" look identical from the brain's side.
        """
        return [r for r in default_sensor_roles() if r not in self.all_roles()]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for p in self.pathways:
            out[p.role] += 1
        return dict(sorted(out.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "pathways": [p.as_dict() for p in self.pathways],
            "dormant": [{"pathway": p.as_dict(), "state": s} for p, s in self.dormant],
            "ignored": [{"entity_id": e, "reason": r} for e, r in self.ignored],
            "roles": self.roles(),
            "unreachable_roles": self.unreachable_roles(),
        }

    # -------------------------------------------------------------- to channels

    def to_channels(
        self, resolver: RoleResolver, neurons_per_pathway: int = 64
    ) -> list[ChannelSpec]:
        """Turn the wiring into encoder channels with **disjoint** neuron pools per pathway.

        Disjointness matters: if two motion sensors drove the same neurons, the brain would
        receive their sum and the readout could never tell which one fired. Each role's
        population is therefore split evenly between the pathways on that role, then subsampled
        to a common width so every channel carries comparable weight.
        """
        by_role: dict[str, list[Pathway]] = defaultdict(list)
        for p in self.pathways:
            by_role[p.role].append(p)

        channels: list[ChannelSpec] = []
        for role, paths in sorted(by_role.items()):
            try:
                indices = resolver.resolve(role).indices
            except KeyError:
                # A wiring file naming a role this connectome does not have should cost that
                # channel, not the whole loop. Silence would be worse: log it.
                logger.warning("unknown role %r; skipping %d pathway(s)", role, len(paths))
                continue
            if indices.size == 0:
                logger.warning("role %r resolved to no neurons; skipping %d pathways", role, len(paths))
                continue
            paths = sorted(paths, key=lambda p: p.entity_id)
            per = max(1, indices.size // len(paths))
            for i, path in enumerate(paths):
                chunk = indices[i * per : (i + 1) * per]
                if chunk.size == 0:                    # more pathways than neurons
                    chunk = indices
                if chunk.size > neurons_per_pathway:
                    # Deterministic, so a restart re-creates the identical wiring.
                    rng = np.random.default_rng(_seed_for(path.entity_id))
                    chunk = np.sort(rng.choice(chunk, size=neurons_per_pathway, replace=False))
                channels.append(path.to_channel(chunk))
        return channels


def _seed_for(entity_id: str) -> int:
    """Stable per-entity seed, so neuron selection survives a restart."""
    digest = hashlib.sha256(entity_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def discover(signals: Iterable[Signal], *, include_ignored: bool = True) -> Wiring:
    """Propose a :class:`Wiring` from a live set of Home Assistant signals."""
    pathways: list[Pathway] = []
    dormant: list[tuple[Pathway, str]] = []
    ignored: list[tuple[str, str]] = []

    for sig in signals:
        if not sig.entity_id.startswith(SENSOR_DOMAINS):
            continue
        # ``enum`` sensors are categorical states, not measurements: a phone's ringer set to
        # "silent", a media session "Playing". They carry no magnitude for a rate code to encode,
        # and are better used as a *readout target* than as a sensory drive.
        if sig.attributes.get("device_class") == "enum":
            if include_ignored:
                ignored.append((sig.entity_id, "categorical state, not a measurement"))
            continue
        pathway = pathway_for(sig)
        if pathway is None:
            if include_ignored:
                ignored.append((sig.entity_id, "no matching fly sensory pathway"))
            continue
        state = (sig.state or "").strip().lower()
        if state in DEAD_STATES:
            dormant.append((pathway, sig.state or "empty"))
        else:
            pathways.append(pathway)

    pathways.sort(key=lambda p: (p.role, p.entity_id))
    dormant.sort(key=lambda ps: (ps[0].role, ps[0].entity_id))
    return Wiring(pathways=pathways, dormant=dormant, ignored=ignored)


# ------------------------------------------------------------------ presentation


def format_wiring(wiring: Wiring) -> str:
    """Human-readable summary, for the CLI and for pasting into docs."""
    lines: list[str] = []
    counts = wiring.counts()
    lines.append(f"live pathways: {len(wiring.pathways)}  across {len(counts)} role(s)")
    for role, n in counts.items():
        lines.append(f"  {role:20} {n}")
    if wiring.pathways:
        lines.append("")
        lines.append(f"  {'entity':54} {'role':20} {'kind':10} range")
        for p in wiring.pathways:
            lines.append(
                f"  {p.entity_id:54} {p.role:20} {p.kind.value:10} {p.vmin:g}..{p.vmax:g}"
            )
    if wiring.dormant:
        lines.append("")
        lines.append(f"dormant ({len(wiring.dormant)}) - wired to a role, but reporting nothing:")
        for p, state in wiring.dormant:
            lines.append(f"  {p.entity_id:54} {p.role:20} state={state}")
    missing = wiring.unreachable_roles()
    if missing:
        lines.append("")
        lines.append("no sensor at all for: " + ", ".join(missing))
    return "\n".join(lines)


async def _read_signals() -> list[Signal]:
    from flybrain.ha import make_home_assistant

    ha = make_home_assistant()
    try:
        return await ha.get_signals()
    finally:
        await ha.aclose()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Propose a fly-brain wiring for a Home Assistant instance."
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--channels",
        type=int,
        metavar="N",
        help="also build encoder channels (needs the connectome) with N neurons each",
    )
    args = parser.parse_args(argv)

    signals = asyncio.run(_read_signals())
    wiring = discover(signals)

    if args.json:
        print(json.dumps(wiring.as_dict(), indent=2))
    else:
        print(format_wiring(wiring))

    if args.channels:
        from flybrain.sim import ConnectomeSim

        sim = ConnectomeSim().load()
        resolver = RoleResolver.from_sim(sim)
        channels = wiring.to_channels(resolver, neurons_per_pathway=args.channels)
        print()
        print(f"encoder channels: {len(channels)}")
        for ch in channels:
            print(
                f"  {ch.entity_id:54} {ch.kind.value:10} "
                f"{ch.neuron_indices.size:5d} neurons"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(_main())


def load_wiring(path: str | Path) -> Wiring:
    """Load a wiring previously written by ``--json`` (or hand-written to match).

    Lets a chosen wiring be pinned to disk and reviewed, rather than re-discovered on every
    start from whatever the house happens to be reporting.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pathways = [
        Pathway(
            entity_id=p["entity_id"],
            role=p["role"],
            kind=SignalKind(p["kind"]),
            vmin=float(p["vmin"]),
            vmax=float(p["vmax"]),
            gain=float(p.get("gain", 1.0)),
            invert=bool(p.get("invert", False)),
            note=p.get("note", ""),
        )
        for p in payload.get("pathways", [])
    ]
    dormant = [
        (
            Pathway(
                entity_id=d["pathway"]["entity_id"],
                role=d["pathway"]["role"],
                kind=SignalKind(d["pathway"]["kind"]),
                vmin=float(d["pathway"]["vmin"]),
                vmax=float(d["pathway"]["vmax"]),
                gain=float(d["pathway"].get("gain", 1.0)),
                invert=bool(d["pathway"].get("invert", False)),
                note=d["pathway"].get("note", ""),
            ),
            d.get("state", ""),
        )
        for d in payload.get("dormant", [])
    ]
    ignored = [(i["entity_id"], i["reason"]) for i in payload.get("ignored", [])]
    return Wiring(pathways=pathways, dormant=dormant, ignored=ignored)
