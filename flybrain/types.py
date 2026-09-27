"""Shared data types for the flybrain-ha connectome controller.

Every module imports its types from here. See CONTRACT.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

#: Entity states that mean "this sensor is not reporting".
#:
#: Defined **once**, here, because three modules had each grown their own identical copy
#: (``ha.UNAVAILABLE_STATES``, ``wiring.DEAD_STATES``, ``loop._DEAD_STATES``). The copies being
#: identical is precisely what made them dangerous: the primary temperature path in
#: :meth:`flybrain.loop.LiveLoop.drive_channels` checked that the entity was *present* but
#: forgot this set, while the extra-channel path a few lines below remembered it. An
#: ``unavailable`` thermometer was therefore encoded as its ``0.0`` fallback and could drive a
#: real light. A duplicated constant is an invitation to exactly that bug; a single import is
#: not, which is why the old names are removed rather than aliased.
DEAD_STATES = frozenset({"unavailable", "unknown", "none", ""})


class SignalKind(str, Enum):
    """Semantic class of a Home Assistant signal, used to pick encoder tuning."""

    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    ILLUMINANCE = "illuminance"
    MOTION = "motion"
    CONTACT = "contact"
    POWER = "power"
    #: Sound-derived events: a camera's bark/meow/glass-break/sound detectors, a microphone
    #: level. The fly hears through mechanosensory structures, so this is a distinct class.
    AUDIO = "audio"
    OTHER = "other"


@dataclass
class Signal:
    """One Home Assistant entity state, normalized for the brain."""

    entity_id: str
    kind: SignalKind
    value: float
    state: str
    timestamp: float
    unit: str = ""
    attributes: dict = field(default_factory=dict)


def signal_is_dead(signal: Signal | None) -> bool:
    """True when a signal is missing, or carries no usable reading.

    Two checks, because they catch different things and neither alone is enough:

    * the ``state`` string catches the literal values Home Assistant reports for a sensor
      that is offline, unknown or never seen;
    * the ``unavailable`` attribute is set by ``HAClient._parse_state`` whenever the state is
      dead **or** the value failed to parse, so it also catches a state that is present but
      not a number. Checking only the string would let that case through as the ``0.0``
      fallback.

    This lives beside :class:`Signal` rather than in the loop because it is a statement about
    the type, not about the loop's policy — and because both the loop and the pacing trigger
    need it, so putting it in either one would make the other import it in a circle. It exists
    at all because the temperature path used to check only that the entity was *present*, while
    the extra-channel path two lines below checked the state: an ``unavailable`` thermometer was
    therefore driven as a genuine 0 °C reading. See :data:`DEAD_STATES`.
    """
    if signal is None:
        return True
    if signal.attributes.get("unavailable"):
        return True
    return signal.state.strip().lower() in DEAD_STATES


@dataclass
class Action:
    """One Home Assistant service call proposed by the brain."""

    entity_id: str
    service: str
    confidence: float = 1.0
    data: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.entity_id}.{self.service}"


@dataclass
class SpikeTrain:
    """Spikes generated for one sensor channel over one integration window."""

    neuron_indices: np.ndarray
    spike_times_ms: np.ndarray
    rate_hz: np.ndarray


@dataclass
class BrainCommand:
    """A decoder decision for a single control tick."""

    actions: list[Action]
    spike_counts: dict[int, int]
    logits: dict[str, float]
    window_ms: float


@dataclass
class Episode:
    """One recorded training sample: features seen and actions that should follow."""

    features: np.ndarray
    target: np.ndarray
    context: dict = field(default_factory=dict)


def _as_bool(value: object) -> bool:
    """Coerce to ``bool`` the way the rest of the project reads boolean settings.

    ``bool("false")`` is ``True``, which is the kind of surprise that turns a "yes, keep it
    read-only" into a satisfied service call. Strings are therefore matched against the same
    set of falsy words the environment loader uses.
    """
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def coerce_patch(instance: object, patch: Mapping[str, object], *, allowed: set[str]) -> dict:
    """Coerce a settings patch to each field's type, **without applying it**.

    Returns the coerced values so the caller can commit them in one go. Coercing first and
    assigning second is the entire point: applying field-by-field means a bad value part-way
    through a patch leaves the earlier fields already changed, and the caller reports an error
    for an update that half happened. That matters most where a setting is safety-relevant — a
    ``dry_run`` that half-applied is worse than one that refused outright.

    Raises:
        KeyError: the patch names a field that is not settable.
        ValueError: a value cannot be coerced to its field's type.
    """
    unknown = set(patch) - set(allowed)
    if unknown:
        raise KeyError(", ".join(sorted(unknown)))

    out: dict = {}
    for key, value in patch.items():
        current = getattr(instance, key)
        try:
            if isinstance(current, bool):
                out[key] = _as_bool(value)
            elif isinstance(current, str):
                out[key] = str(value)
            else:
                out[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bad value for {key}: {value!r}") from exc
    return out
