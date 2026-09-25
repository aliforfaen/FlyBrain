"""Shared data types for the flybrain-ha connectome controller.

Every module imports its types from here. See CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


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
