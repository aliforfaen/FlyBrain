"""Jev: the optional judgment layer, and the discipline that keeps it affordable.

Jev is TypeSafe's "System One" model. It does not write prose: you hand it a state and a set of
typed questions, and it returns a typed answer per question. That interface is the reason it is
here at all — not accuracy, which measured as a **tie** with a general chat model
(``docs/jev.md``). What it adds is a decision-shaped interface and a ``confidence`` per answer,
which is what makes it safe to gate anything on.

Three things about this module are deliberate and worth stating before the code:

**It is off unless configured.** With no ``JEV_API_KEY`` (or ``TYPESAFE_API_KEY``) the
client refuses to make a
request and :func:`JevClient.available` says *why*, because "off" and "broken" have to look
different on screen. There is no fallback that invents an answer.

**The three call disciplines are code, not convention** (``docs/jev.md``): trim the state
(:func:`build_state`), batch every question into one request (:meth:`JevClient.ask` — one HTTP
call, always), and route on confidence (:func:`route`). Batching is not an optimisation: five
questions in one request cost and take the same as one, while three requests pay for the state
three times.

**Hand-rolled ``httpx`` rather than the official SDK.** The SDK is real and good — its retry
handling and its OpenAPI-derived models are what this file was checked against rather than
guessed at — but it depends on ``httpx2`` + ``httpcore2`` + ``truststore``, a second HTTP stack
in a project that already depends on ``httpx``, for a feature that is off by default. The
retry logic below is ~30 lines and the wire shapes are pinned by tests, including one that runs
against the live endpoint. If that trade ever stops being right, ``flybrain/jev.py`` is the only
file that changes.

Wire shapes below are taken from the vendor's generated schema (``typesafe_sdk/_schemas/models``)
and from live unauthenticated probes, and two of them are NOT what a reasonable guess would
produce — see :func:`choice` and the note on ``probabilities`` keys in :func:`parse_answer`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import socket
import ssl
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self
from urllib.parse import urlparse

import httpx

from flybrain.env import load_dotenv

logger = logging.getLogger(__name__)

#: The endpoint the configured key actually works against. Note this is **not** the host in the
#: vendor's public quickstart: `api.typesafe.ai` served the account this was first written for and
#: answered every request with HTTP 401, while this one works. Two differences matter and both are
#: handled below rather than assumed away:
#:
#: * the error envelope is ``{"error": "..."}``, not ``{"detail": {...}}``;
#: * there is **no** cheap ``/v1/models`` probe, so :meth:`JevClient.available` has to spend a
#:   real (tiny) question instead of a free GET.
DEFAULT_BASE_URL = "https://jevtypesafeai.com/api/v1/decide"
DECIDE_PATH = "/api/v1/decide"

#: Accepted spellings of the credential, in priority order. ``JEV_API_KEY`` is this provider's
#: name; ``TYPESAFE_API_KEY`` is kept so an existing ``.env`` is not silently ignored, and both
#: come from the same vendor.
API_KEY_VARS = ("JEV_API_KEY", "TYPESAFE_API_KEY")

#: The **versioned** id, never the ``jev-latest`` alias. The alias moves when a release ships,
#: which would silently invalidate every confidence threshold calibrated against it — and routing
#: on confidence is the whole design. The vendor's own SDK defaults to the alias; this is a
#: deliberate disagreement with it, for the reason in docs/jev.md.
DEFAULT_MODEL = "jev-1.13.0"

#: $42 per billion input tokens. Output tokens are free, so the bill is entirely a function of
#: state length — which is why trimming the state is the first discipline and not an
#: optimisation. The API reports token counts but no cost field; the multiplication is ours.
USD_PER_INPUT_TOKEN = 42.0 / 1e9

#: How long an availability probe result is trusted. The probe is a real request, so it must not
#: run on every dashboard refresh.
PROBE_TTL_S = 300.0

QuestionKind = Literal["choice", "score", "noul"]


class JevError(RuntimeError):
    """Anything that stops a judgment being made. Never carries the API key."""


class JevAuthError(JevError):
    """The request was refused for credential reasons.

    ``reason`` is one of the states the live endpoint actually distinguishes, plus one this
    client adds:

    * ``"disabled"`` — switched off here. Nothing was sent, and a key may well be configured.
    * ``"no_key"`` — HTTP 403, ``"Must supply an API key!"``. Nothing was sent.
    * ``"unauthorized"`` — HTTP 401, ``"Cannot authenticate with the server"``. A key is present
      and the server will not accept it: revoked, mistyped, or for a different environment.

    Keeping these apart matters because the fix is different: one is a switch, one is a missing
    setting, and the other is a credential that looks entirely well-formed.
    """

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


class JevModelMismatch(JevError):
    """The response came from a different model than the pinned one.

    This is a refusal rather than a warning. Confidence thresholds are calibrated against one
    version, so an answer from an unknown version is not a slightly worse answer — it is an
    uninterpretable one. Turning the feature off is the honest response.
    """


# --------------------------------------------------------------------------- risk and routing


@dataclass(frozen=True)
class Risk:
    """How much damage a wrong answer does, which is what sets the threshold.

    "A confidence threshold is not one number." The floor depends entirely on what the judgment
    is allowed to affect, and this project has a natural ladder: a verdict painted on a chart
    costs a glance, a label written into ``labels.jsonl`` silently corrupts the only training
    data there is, and gating a Home Assistant action can change the house.
    """

    name: str
    #: At or above this, act without asking.
    act_at: float
    #: At or above this (but below ``act_at``), propose and let a human confirm.
    confirm_at: float


#: The three tiers. The defaults come from docs/jev.md, measured on this project's own states
#: rather than inherited: a state that *determines* the answer scores 0.94-0.99, and one that
#: genuinely under-determines it collapses to ~0.34. ``label``'s 0.95 therefore lets the strongest
#: answers write while a determined-but-not-maximal 0.94 only proposes, and 0.34 never acts. The
#: separation is wide, but a **byte-identical** request varies by up to 0.17 across calls, which is
#: why the floors sit with margin instead of on a measured value. **``paint`` is the tier placement A
#: actually uses** (J2, the decision inspector), and its first real answer landed in the middle band:
#: `throttled` at 0.24 confidence, which `needs_human` carries to the card as "not enough to call it"
#: rather than a label. ``label`` and ``ha_action`` belong to placements B and C, which are **not
#: built**, so those two numbers are calibrated against captured fixtures and have never gated a real
#: write. Every tier is overridable from the environment (``JEV_LABEL_ACT`` and friends) rather than
#: fixed here.
PAINT = Risk("paint", act_at=0.70, confirm_at=0.50)
LABEL = Risk("label", act_at=0.95, confirm_at=0.80)
HA_ACTION = Risk("ha_action", act_at=0.98, confirm_at=0.90)

DEFAULT_RISKS: Mapping[str, Risk] = {
    PAINT.name: PAINT,
    LABEL.name: LABEL,
    HA_ACTION.name: HA_ACTION,
}


@dataclass(frozen=True)
class Routing:
    """What to do with an answer, and why."""

    action: Literal["act", "confirm", "needs_human"]
    reason: str
    confidence: float | None


# ------------------------------------------------------------------------------------ answers


@dataclass(frozen=True)
class Answer:
    """One typed answer, normalized across the three question kinds.

    The fields are flattened rather than kept as a union of three classes. There are exactly
    three shapes and they are read in one place, so a union would add ceremony without adding
    safety — but the *asymmetry* is preserved and documented: ``confidence`` is ``None`` for a
    ``noul``, and :func:`route` refuses to gate on it.
    """

    question_id: str
    kind: QuestionKind
    #: ``None`` for ``noul``. That is not a missing value; the API does not return one.
    confidence: float | None
    #: ``choice``: the highest-probability label.
    choice: str | None = None
    #: ``score``: probability-weighted mean of the rubric levels, so it can be fractional.
    score: float | None = None
    #: ``noul``: probability of yes/true, 0..1.
    noul: float | None = None
    #: ``choice``: keyed by **label**. ``score``: keyed by **level** as a string ("0", "1", ...).
    probabilities: Mapping[str, float] = field(default_factory=dict)
    #: ``score`` only: the requested rubric, level string -> description.
    legend: Mapping[str, Any] | None = None

    @property
    def value(self) -> Any:
        """The answer itself, whichever kind it is."""
        if self.kind == "choice":
            return self.choice
        if self.kind == "score":
            return self.score
        return self.noul

    @property
    def gateable(self) -> bool:
        """Whether this answer can be routed on confidence. False for every ``noul``."""
        return self.confidence is not None


def parse_answer(question_id: str, raw: Mapping[str, Any]) -> Answer | None:
    """Parse one entry of the response's ``answers`` object.

    **The ``probabilities`` keys are not the same for the two kinds that have them**, which is the
    easiest thing here to get wrong:

    * ``choice`` returns ``probabilities`` keyed by the **choice label** (``{"calm": 0.8, ...}``),
      because the labels are the ones you supplied in ``criteria``.
    * ``score`` returns ``probabilities`` and ``legend`` keyed by the **score level as a string**
      (``{"0": 0.1, "2": 0.8}``), because the levels are positional and were never named.

    Returns ``None`` for an unrecognized ``type``, so that a future API version adding an answer
    kind does not fail the whole response. The caller logs it.
    """
    kind = raw.get("type")
    if kind == "choice":
        return Answer(
            question_id=question_id,
            kind="choice",
            choice=str(raw["choice"]),
            confidence=float(raw["confidence"]),
            probabilities={str(k): float(v) for k, v in dict(raw.get("probabilities", {})).items()},
        )
    if kind == "score":
        return Answer(
            question_id=question_id,
            kind="score",
            score=float(raw["score"]),
            confidence=float(raw["confidence"]),
            probabilities={str(k): float(v) for k, v in dict(raw.get("probabilities", {})).items()},
            legend={str(k): v for k, v in dict(raw.get("legend", {})).items()},
        )
    if kind == "noul":
        return Answer(
            question_id=question_id,
            kind="noul",
            noul=float(raw["noul"]),
            confidence=None,
        )
    return None


def reconcile_score(answer: Answer, tolerance: float = 0.02) -> float | None:
    """Check a ``score`` against its own ``probabilities``; return the absolute gap.

    ``score`` is the probability-weighted mean of the rubric **levels**, so recomputing it from
    the returned probabilities is a free consistency check. It will not be exact: the
    probabilities are rounded to two decimals while ``score`` is computed from full precision, so
    a small mismatch is expected and is **not** a bug. ``None`` when there is nothing to check,
    which is every kind except ``score``.
    """
    if answer.kind != "score" or answer.score is None or not answer.probabilities:
        return None
    total = 0.0
    weight = 0.0
    for level, probability in answer.probabilities.items():
        try:
            index = int(level)
        except (TypeError, ValueError):
            return None  # a non-numeric key means this is not a level-keyed map
        total += index * probability
        weight += probability
    if weight <= 0.0:
        return None
    return abs(answer.score - total / weight)


def route(answer: Answer, risk: Risk) -> Routing:
    """Decide whether an answer is safe to act on, given what acting would do.

    A ``noul`` answer is always ``needs_human``, and that is structural rather than cautious: the
    API returns no ``confidence`` for a ``noul`` at all, so there is nothing to gate on. The fix
    is to ask a two-option ``choice`` instead, which is what ``docs/jev.md`` says — a ``noul`` is
    a composed fact, never a gate.
    """
    if answer.confidence is None:
        return Routing(
            "needs_human",
            f"{answer.kind} carries no confidence; ask a two-option choice if this must gate",
            None,
        )
    if answer.confidence >= risk.act_at:
        return Routing(
            "act", f"confidence {answer.confidence:.3f} >= act_at {risk.act_at}", answer.confidence
        )
    if answer.confidence >= risk.confirm_at:
        return Routing(
            "confirm",
            f"confidence {answer.confidence:.3f} in [{risk.confirm_at}, {risk.act_at})",
            answer.confidence,
        )
    return Routing(
        "needs_human",
        f"confidence {answer.confidence:.3f} below confirm_at {risk.confirm_at}",
        answer.confidence,
    )


# ---------------------------------------------------------------------------------- questions


def noul(instructions: str, *, criteria: Mapping[str, Any] | None = None) -> dict:
    """A yes/no question. Returns a bare probability and **no confidence**.

    Fine as a composed fact. Not fine as a gate — see :func:`route`.
    """
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = dict(criteria)
    return question


def choice(instructions: str, criteria: Mapping[str, Any]) -> dict:
    """A question that selects between named alternatives.

    ``criteria`` is a mapping of **label -> description**, and it is required. This is the field
    most likely to be written wrongly: it is not ``options`` and it is not a list. The labels are
    what come back in ``choice``, and what key ``probabilities``.
    """
    if not criteria:
        raise ValueError("a choice question needs at least one labelled criterion")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: str, criteria: Sequence[Any]) -> dict:
    """A graded question. ``criteria`` is an ordered sequence of level descriptions, from zero.

    The levels are positional and unnamed, which is why the response keys them ``"0"``, ``"1"``
    and so on, and why ``legend`` exists to map them back.
    """
    if not criteria:
        raise ValueError("a score question needs at least one level")
    return {"type": "score", "instructions": instructions, "criteria": list(criteria)}


# ------------------------------------------------------------------------ placements


#: The closed vocabulary for the decision inspector — placement A in ``docs/jev.md``.
#:
#: Every entry is a failure mode this project already documents *elsewhere*, which is the whole
#: design: Jev is used as an **auditor**, not a narrator. It does not describe a decision in
#: prose, it sorts one into a category somebody can act on, and those categories exist because
#: each of them has already cost this project real time. A label the rest of the docs cannot
#: explain would be a label nobody can verify.
FAILURE_MODES: Mapping[str, str] = {
    "healthy": "Tracks the ideal mapping within tolerance. Nothing is wrong with this decision.",
    "saturated_sensory": (
        "The sensory population was at its documented ceiling, so more input cannot change the "
        "answer. See engine.md: the antennal lobe saturates at a receptor rate of 25 Hz or more."
    ),
    "regime_mismatch": (
        "The window began from rest rather than the running regime the readout was fitted on, so "
        "the decode is being asked a question it was not trained for. See AGENTS.md #2."
    ),
    "too_few_spikes": (
        "The readout population barely responded, so the decoded colour rests on very little "
        "evidence and could be near-arbitrary."
    ),
    "sensor_stale": (
        "The temperature reading driving this window was old rather than fresh, so the colour "
        "describes the room as it was some time ago."
    ),
    "throttled": (
        "The colour moved but the deadband suppressed the service call, so the light was "
        "deliberately left where it was."
    ),
    "unknown": "None of the above describes this decision.",
}

#: The fields a failure-mode judgement is allowed to see.
#:
#: "Trim the state" is the first of the three call disciplines in ``docs/jev.md``: every field
#: sent costs tokens and, worse, invites reasoning from something irrelevant. Each entry here is
#: named by at least one row of :data:`FAILURE_MODES`, and nothing else is sent.
DECISION_STATE_FIELDS: tuple[str, ...] = (
    "temperature_c",
    "ideal_kelvin",
    "kelvin",
    "band",
    "active_neurons",
    "total_spikes",
    "window_ms",
    "reading_age_s",
    "top_regions",
    "settings",
)


def decision_state(row: Mapping[str, Any]) -> dict:
    """Trim a recorded decision to the fields the failure-mode vocabulary can use.

    Missing fields are omitted rather than sent as ``null``, so the model cannot reason about a
    value the project never measured as though it were a zero.
    """
    return {key: row[key] for key in DECISION_STATE_FIELDS if row.get(key) is not None}


def session_spend(client: JevClient) -> float:
    """Total reported cost of everything this client has asked, including probes.

    Defined here rather than summed at each call site because the probe is easy to forget, and a
    figure labelled "spent" that quietly excludes the dashboard's own requests is worse than no
    figure at all. Costs the vendor did not report are counted as zero rather than guessed at.
    """
    return round(sum(float(c.get("cost_usd") or 0.0) for c in client.calls), 8)


def failure_mode_question() -> dict:
    """The one ``choice`` question placement A asks."""
    return choice(
        "This is one recorded decision from a fruit-fly connectome driving a room light. "
        "Given the state, decide which single label best explains how the colour came out. "
        "Choose 'healthy' only if the chosen colour tracks the ideal within a small tolerance; "
        "prefer a specific failure mode whenever the state shows one.",
        FAILURE_MODES,
    )


async def classify_decision(
    client: JevClient,
    row: Mapping[str, Any],
    *,
    risk: Risk = PAINT,
    question_id: str = "failure_mode",
) -> dict:
    """Ask Jev which failure mode a recorded decision looks like.

    Returns a plain dict rather than a :class:`JevResponse` because the caller is an HTTP
    endpoint that has to serialise it, and because the *routing* is part of the answer: a verdict
    is only usable if you also know how much to trust it. A ``noul`` or a low confidence comes
    back as ``needs_human`` with the reason attached, which is a real outcome and not an error —
    this project's own measurement produced 0.34 for a state containing a contradiction.
    """
    response = await client.ask(decision_state(row), {question_id: failure_mode_question()})
    answer = response.answers.get(question_id)
    if answer is None:
        raise JevError(f"Jev answered without a {question_id!r} answer")
    routing = route(answer, risk)
    probabilities = getattr(answer, "probabilities", None) or {}
    return {
        "label": getattr(answer, "choice", None),
        "confidence": answer.confidence,
        "action": routing.action,
        "routing_reason": routing.reason,
        "probabilities": {str(k): float(v) for k, v in probabilities.items()},
        "model": response.model,
        "cost_usd": response.cost_usd,
        "cost_reported": response.cost_reported,
        "credits_remaining_usd": response.credits_remaining_usd,
        "latency_ms": round(response.latency_ms, 1),
        "floor_ms": response.network_floor_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }


# ------------------------------------------------------------------------------------- config


def _number(raw: str | None, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric Jev setting %r", raw)
        return default


def _risk_from_env(env: Mapping[str, str], default: Risk) -> Risk:
    """Override one tier's thresholds from ``JEV_<TIER>_ACT`` / ``JEV_<TIER>_CONFIRM``.

    A configuration error must not silently *loosen* a safety threshold, which is the failure
    that matters here: a typo in ``JEV_LABEL_ACT`` should not turn "propose a label" into "write
    one". Anything unusable — non-numeric, out of range, or inverted so that `confirm_at` exceeds
    `act_at` — falls back to the default and says so.
    """
    prefix = f"JEV_{default.name.upper()}"
    act = _number(env.get(f"{prefix}_ACT"), default.act_at)
    confirm = _number(env.get(f"{prefix}_CONFIRM"), default.confirm_at)
    if not (0.0 <= confirm <= act <= 1.0):
        logger.warning(
            "ignoring %s thresholds act_at=%r confirm_at=%r: need 0 <= confirm <= act <= 1",
            default.name, act, confirm,
        )
        return default
    return Risk(default.name, act_at=act, confirm_at=confirm)


@dataclass
class JevConfig:
    """Everything Jev needs, all of it optional.

    ``enabled`` is the only thing the rest of the project has to check, and it is deliberately
    just "is there a key" — the endpoint being unreachable is a *runtime* condition reported by
    :meth:`JevClient.available`, not a configuration error.
    """

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 30.0
    #: Confidence thresholds per risk tier. Overridable, because they are calibrated against data
    #: and this project's data is the only data that counts.
    risks: Mapping[str, Risk] = field(default_factory=lambda: dict(DEFAULT_RISKS))
    #: Measured TCP+TLS time to the API, in milliseconds. Recorded beside every latency number,
    #: because a latency claim without the floor says more about geography than about Jev.
    #: ``None`` means "not measured yet" and is reported as such rather than replaced by 0.
    network_floor_ms: float | None = None
    #: Refuse an answer whose reported ``model`` is not the pinned id. See ``JevModelMismatch``.
    strict_model: bool = True
    #: The on/off switch, independent of whether a key is present. Default **False**.
    #:
    #: Defaulting to off is a deliberate cost decision rather than diffidence. Merely having the
    #: dashboard open used to spend one probe question every ``PROBE_TTL_S`` (about $0.03/day) to
    #: feed a badge, and a dashboard is a thing people leave running for weeks. Nothing should
    #: spend money as a side effect of being *watched*; the switch makes the spend something a
    #: person asked for. ``JEV_ENABLED=1`` turns it on.
    enabled_flag: bool = False

    def __post_init__(self) -> None:
        """Normalize the key once, so `enabled`, `redacted` and the header all agree.

        A key of ``"   "`` is not a key. Without this, a whitespace-only value from a templating
        mistake would report the feature as *enabled* and then fail at the far end with a 401 —
        an "unauthorized" when the truth is "nothing was configured", which sends the operator
        looking at the wrong thing.
        """
        if self.api_key is not None:
            stripped = self.api_key.strip()
            self.api_key = stripped or None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> JevConfig:
        """Read the credentials from the environment, loading ``.env`` if not given one.

        Same precedence as the rest of the project: real environment variables win over ``.env``,
        so a systemd unit or container env block works without a file.
        """
        if env is None:
            load_dotenv()
        e = os.environ if env is None else env
        # Both spellings, first one wins: an existing `.env` carrying `TYPESAFE_API_KEY` keeps
        # working, and the provider's own name takes precedence when both are present.
        raw_key = next((e.get(name) for name in API_KEY_VARS if e.get(name)), None)
        floor = e.get("JEV_NETWORK_FLOOR_MS")
        return cls(
            api_key=(raw_key or "").strip() or None,
            model=(e.get("JEV_MODEL") or DEFAULT_MODEL).strip(),
            base_url=(e.get("JEV_BASE_URL") or DEFAULT_BASE_URL).strip(),
            timeout_s=_number(e.get("JEV_TIMEOUT_S"), 30.0),
            network_floor_ms=None if not floor else _number(floor, 0.0),
            strict_model=(e.get("JEV_STRICT_MODEL") or "1").strip().lower()
            not in {"0", "false", "no", "off"},
            enabled_flag=(e.get("JEV_ENABLED") or "0").strip().lower()
            in {"1", "true", "yes", "on"},
            risks={name: _risk_from_env(e, risk) for name, risk in DEFAULT_RISKS.items()},
        )

    def risk(self, name: str) -> Risk:
        """The threshold tier for a placement, by name."""
        try:
            return self.risks[name]
        except KeyError as exc:
            raise JevError(f"unknown risk tier {name!r}; have {sorted(self.risks)}") from exc

    @property
    def enabled(self) -> bool:
        """Whether Jev may be called at all: a key *and* the switch.

        Both are required, and they are kept separate on purpose. "Off" is a choice, "no key" is
        an omission, and they send a person to different places — which is the whole reason
        :class:`JevStatus` distinguishes them rather than reporting one "unavailable".
        """
        return self.enabled_flag and bool(self.api_key)

    @property
    def key_present(self) -> bool:
        """Whether a credential exists, regardless of the switch."""
        return bool(self.api_key)

    @property
    def endpoint(self) -> str:
        """The ``/decide`` URL, tolerating either the full path or the bare host.

        A reader copies the full URL out of the docs, while a base-URL habit writes only the host;
        both are accepted so neither is a mistake. There is no separate root to expose: unlike the
        other host, this one has no sibling endpoints worth naming.
        """
        url = self.base_url.strip().rstrip("/")
        if not url:
            return DEFAULT_BASE_URL
        if url.endswith(DECIDE_PATH):
            return url
        return f"{url}{DECIDE_PATH}"

    def redacted(self) -> dict:
        """Config, safe to log or return over HTTP. Never the key itself."""
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "network_floor_ms": self.network_floor_ms,
            "strict_model": self.strict_model,
            "enabled": self.enabled,
            "key_present": self.key_present,
            "key_hint": None if not self.api_key else f"...{self.api_key[-4:]}",
        }


@dataclass(frozen=True)
class JevStatus:
    """Why Jev is or is not usable, in a form the UI can say out loud."""

    available: bool
    reason: Literal["ok", "no_key", "unauthorized", "unreachable", "disabled", "unprobed"]
    detail: str = ""
    models: Sequence[str] = ()
    floor_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "detail": self.detail,
            "models": list(self.models),
            "floor_ms": self.floor_ms,
        }


@dataclass(frozen=True)
class JevResponse:
    """One completed request: the answers, and the accounting that goes with them."""

    answers: Mapping[str, Answer]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    #: The cost of this call. The vendor reports it on this host; when it does not, this is
    #: ``input_tokens x $0.042/1e6`` and ``cost_reported`` is False.
    cost_usd: float
    network_floor_ms: float | None
    #: Credit left on the account, when the API reports it. Shown in the dashboard because a
    #: silent zero is how a feature stops working without anyone noticing.
    credits_remaining_usd: float | None = None
    cost_reported: bool = False

    def routed(self, risk: Risk) -> dict[str, Routing]:
        return {name: route(answer, risk) for name, answer in self.answers.items()}


# ------------------------------------------------------------------------------------- client


class JevClient:
    """One HTTP call per judgment. Deliberately not one call per question.

    ``transport`` exists so tests can run the entire client against recorded responses with no
    network, and so a test can assert the call *count* — the batching rule is the kind of thing
    that silently regresses into one call per question, which costs the state every time.
    """

    def __init__(
        self,
        config: JevConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock=time.monotonic,
        sleep=asyncio.sleep,
        max_attempts: int = 3,
    ) -> None:
        self.config = config or JevConfig.from_env()
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._max_attempts = max(1, int(max_attempts))
        self._client: httpx.AsyncClient | None = None
        self._status: JevStatus | None = None
        self._status_at: float | None = None
        #: Every call, for the cost log and for a test to assert "exactly one".
        self.calls: list[dict] = []

    # ------------------------------------------------------------------ plumbing

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=self.config.timeout_s,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _headers(self) -> dict[str, str]:
        if not self.config.api_key:
            raise JevAuthError("no_key", f"no {' or '.join(API_KEY_VARS)} in the environment")
        return {"Authorization": f"Bearer {self.config.api_key}"}

    def _retry_after_s(self, response: httpx.Response, attempt: int) -> float:
        """How long to wait before the next attempt.

        Both spellings are honoured because the vendor sends both: ``retry-after-ms`` when it
        wants sub-second precision, ``retry-after`` otherwise. Falling back is exponential with
        jitter, so a herd of clients does not synchronise.
        """
        ms = response.headers.get("retry-after-ms")
        if ms:
            try:
                return max(0.0, float(ms) / 1000.0)
            except ValueError:
                pass
        after = response.headers.get("retry-after")
        if after:
            try:
                return max(0.0, float(after))
            except ValueError:
                pass
        return min(8.0, 0.5 * (2**attempt)) * (0.5 + random.random())

    async def available(self, *, refresh: bool = False, probe: bool = True) -> JevStatus:
        """Report whether Jev can be used right now, and if not, why.

        This asks one minimal question rather than probing a cheap endpoint, because **this host
        has no cheap endpoint to probe** — ``/api/v1/models`` answers ``unknown_endpoint``. So the
        check costs a fraction of a cent rather than nothing, which is why the result is cached for
        :data:`PROBE_TTL_S` and why ``refresh`` is explicit rather than implied.

        ``probe=False`` is what the *status* path passes. A dashboard that is merely being looked
        at must not spend money, so this returns the last known answer and otherwise says
        ``unprobed`` — a state the UI shows as "not checked yet", not as a failure.

        The failure reasons are kept apart on purpose — see :class:`JevAuthError`:
        ``disabled`` means a person switched it off, ``no_key`` means nothing was configured,
        ``unprobed`` means nobody has asked yet, and ``unauthorized`` means the far end refused a
        credential that does exist. Four different actions follow from those four words.
        """
        if not self.config.api_key:
            # Checked *before* the switch, deliberately. With no key at all, "nothing is
            # configured" is the fact that sends someone to the right place; "switched off" would
            # be true but useless, since there was never anything to switch on.
            return JevStatus(
                False, "no_key", f"no {' or '.join(API_KEY_VARS)} in the environment",
                floor_ms=self.config.network_floor_ms,
            )
        if not self.config.enabled_flag:
            return JevStatus(
                False,
                "disabled",
                "Jev is switched off (JEV_ENABLED=0). A key is configured and unused; "
                "nothing is sent and nothing is spent.",
                floor_ms=self.config.network_floor_ms,
            )
        now = self._clock()
        if (
            not refresh
            and self._status is not None
            and self._status_at is not None
            and (now - self._status_at) < PROBE_TTL_S
        ):
            return self._status
        if not probe:
            if self._status is not None:
                # Stale, but it is the last thing actually observed. Reporting it beats
                # reporting "unknown" when we do in fact know something.
                return self._status
            return JevStatus(
                False,
                "unprobed",
                "not checked yet — the dashboard has not asked, and asking costs a request.",
                floor_ms=self.config.network_floor_ms,
            )

        status = await self._probe()
        self._status, self._status_at = status, now
        return status

    async def _probe(self) -> JevStatus:
        """One trivial question. Cheap, but not free — say so rather than implying it is."""
        payload = {
            "state": {"probe": "reachability check"},
            "questions": {"ok": noul("Is this request being answered?")},
        }
        try:
            response = await self.client.post(
                self.config.endpoint, headers=self._headers(), json=payload
            )
        except httpx.HTTPError as exc:
            return JevStatus(
                False, "unreachable", f"{type(exc).__name__}: {exc}",
                floor_ms=self.config.network_floor_ms,
            )
        if response.status_code == 200:
            self._record_probe(response)
            return JevStatus(True, "ok", "", floor_ms=self.config.network_floor_ms)
        return self._status_for_error(response)

    def _record_probe(self, response: httpx.Response) -> None:
        """Log the probe's real cost.

        The probe asks a real question, so it really does cost money, and leaving it out of
        :attr:`calls` made ``spent_usd`` under-report the session by exactly the amount the
        dashboard spends on itself — the one number a person would use to decide whether to leave
        it running. The entry carries ``probe: True`` so an availability check can still be told
        apart from a judgment.
        """
        entry: dict[str, Any] = {
            "model": self.config.model,
            "questions": 1,
            "probe": True,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": None,
            "floor_ms": self.config.network_floor_ms,
            "cost_usd": 0.0,
            "cost_reported": False,
            "credits_remaining_usd": None,
        }
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            usage = body.get("usage")
            if isinstance(usage, dict):
                entry["input_tokens"] = int(usage.get("input_tokens") or 0)
                entry["output_tokens"] = int(usage.get("output_tokens") or 0)
                if usage.get("cost_usd") is not None:
                    entry["cost_usd"] = float(usage["cost_usd"])
                    entry["cost_reported"] = True
                if usage.get("credits_remaining_usd") is not None:
                    entry["credits_remaining_usd"] = float(usage["credits_remaining_usd"])
            if body.get("model"):
                entry["model"] = str(body["model"])
        self.calls.append(entry)

    def _status_for_error(self, response: httpx.Response) -> JevStatus:
        """Map a refusal to a reason the operator can act on.

        **The status code alone is not enough on this host.** A *missing* key and an *invalid* key
        both come back as 401; only the message distinguishes them:

        * ``"Missing API key. Send 'Authorization: Bearer jv_live_...'."``
        * ``"Invalid or revoked API key."``

        Those send an operator to different places (set a variable vs mint a new key), so the
        message is read rather than the code being trusted.
        """
        detail = self._error_detail(response)
        lowered = detail.lower()
        if "missing api key" in lowered or "must supply an api key" in lowered:
            return JevStatus(False, "no_key", detail, floor_ms=self.config.network_floor_ms)
        if response.status_code in (401, 403) or "invalid or revoked" in lowered:
            return JevStatus(False, "unauthorized", detail, floor_ms=self.config.network_floor_ms)
        return JevStatus(
            False,
            "unreachable",
            f"HTTP {response.status_code}: {detail}",
            floor_ms=self.config.network_floor_ms,
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        """Pull the message out of an error body, truncated.

        Both envelopes the vendor's hosts use are handled, because they differ and guessing
        produced a real bug: this host answers ``{"error": "Invalid or revoked API key."}``, while
        ``api.typesafe.ai`` answers ``{"detail": {"error_type": ..., "message": ...}}``.
        """
        try:
            body = response.json()
        except ValueError:
            return response.text[:200]
        if not isinstance(body, dict):
            return str(body)[:200]
        if isinstance(body.get("error"), str):
            return body["error"][:200]
        detail = body.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("message") or detail.get("error_type") or "")[:200]
        return str(detail or "")[:200]

    # -------------------------------------------------------------------- asking

    async def ask(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        model: str | None = None,
    ) -> JevResponse:
        """Ask every question in **one** request and return the answers keyed by question id.

        ``questions`` is a mapping of your id -> question object built by :func:`choice`,
        :func:`score` or :func:`noul`. The ids are yours alone: the model never sees them, so
        each question's ``instructions`` has to be complete on its own.
        """
        if not questions:
            raise JevError("at least one question is required")
        if not self.config.api_key:
            raise JevAuthError("no_key", f"no {' or '.join(API_KEY_VARS)} in the environment")
        if not self.config.enabled_flag:
            # Deliberately not reported as "no_key": a key may be sitting right there in `.env`.
            # Saying "no key in the environment" when one exists sends a person to edit the wrong
            # thing, which is the exact failure this whole distinction exists to prevent.
            raise JevAuthError(
                "disabled", "Jev is switched off (JEV_ENABLED=0); no request was sent"
            )

        payload = {
            "state": state,
            "model": model or self.config.model,
            "questions": dict(questions),
        }
        started = self._clock()
        response = await self._post(self.config.endpoint, payload)
        latency_ms = (self._clock() - started) * 1000.0

        try:
            body = response.json()
        except ValueError as exc:
            raise JevError(f"Jev returned a non-JSON body: {response.text[:200]}") from exc
        if not isinstance(body, dict):
            raise JevError(f"Jev returned {type(body).__name__}, expected an object")

        answered_by = str(body.get("model", ""))
        if self.config.strict_model and answered_by != self.config.model:
            raise JevModelMismatch(
                f"asked {self.config.model!r} but {answered_by!r} answered; confidence "
                "thresholds are calibrated against one version, so this answer is not usable"
            )

        raw_answers = body.get("answers")
        if not isinstance(raw_answers, dict):
            raise JevError("response has no 'answers' object")

        answers: dict[str, Answer] = {}
        for name, raw in raw_answers.items():
            if not isinstance(raw, dict):
                raise JevError(f"answer {name!r} is not an object")
            parsed = parse_answer(str(name), raw)
            if parsed is None:
                logger.warning("ignoring answer %r with unrecognized type %r", name, raw.get("type"))
                continue
            answers[str(name)] = parsed

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        # This host **does** report the cost, which the SDK schema does not describe and the first
        # draft of docs/jev.md therefore claimed it did not. Prefer the vendor's own number: it is
        # the figure that will be billed, and recomputing it ourselves would drift from it silently
        # the day the price or the rounding changes. The multiplication stays as the fallback for a
        # host that omits the field.
        reported_cost = usage.get("cost_usd")
        cost = (
            float(reported_cost)
            if isinstance(reported_cost, (int, float))
            else input_tokens * USD_PER_INPUT_TOKEN
        )
        credits = usage.get("credits_remaining_usd")

        result = JevResponse(
            answers=answers,
            model=answered_by,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            network_floor_ms=self.config.network_floor_ms,
            credits_remaining_usd=float(credits) if isinstance(credits, (int, float)) else None,
            # Whether the figure above is the vendor's or ours. The dashboard says which, because
            # "cost" and "our estimate of cost" are different claims.
            cost_reported=isinstance(reported_cost, (int, float)),
        )
        self._log_call(result, payload)
        return result

    def _log_call(self, result: JevResponse, payload: Mapping[str, Any]) -> None:
        """One line per call, with the floor beside the latency.

        The floor is not decoration. Without it, "Jev took 300 ms" is a statement about this
        machine's distance from the API, and the two are indistinguishable in a log that only
        records one number.
        """
        floor = result.network_floor_ms
        question_count = len(payload.get("questions") or {})
        self.calls.append(
            {
                "model": result.model,
                "questions": question_count,
                "input_tokens": result.input_tokens,
                "latency_ms": round(result.latency_ms, 1),
                "floor_ms": floor,
                "cost_usd": result.cost_usd,
                "cost_reported": result.cost_reported,
                "credits_remaining_usd": result.credits_remaining_usd,
            }
        )
        credits = (
            "" if result.credits_remaining_usd is None
            else f", ${result.credits_remaining_usd:.4f} credit left"
        )
        logger.info(
            "jev: %d question(s), %d input tokens, $%.6f%s%s, %.0f ms%s, answered by %s",
            question_count,
            result.input_tokens,
            result.cost_usd,
            "" if result.cost_reported else " (computed, not reported)",
            credits,
            result.latency_ms,
            "" if floor is None else f" (network floor {floor:.0f} ms)",
            result.model,
        )

    async def _post(self, url: str, payload: Mapping[str, Any]) -> httpx.Response:
        """POST with bounded retries. Retries 429 and 5xx only; never a 4xx that means "wrong"."""
        last: httpx.Response | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self.client.post(url, headers=self._headers(), json=dict(payload))
            except httpx.HTTPError as exc:
                if attempt + 1 >= self._max_attempts:
                    raise JevError(f"could not reach Jev: {type(exc).__name__}: {exc}") from exc
                await self._sleep(min(8.0, 0.5 * (2**attempt)))
                continue

            if response.status_code in (401, 403):
                # Retrying a credential failure only spends time; the key will not become valid.
                raise JevAuthError(
                    "unauthorized" if response.status_code == 401 else "no_key",
                    self._error_detail(response),
                )
            if response.status_code == 200:
                return response
            if response.status_code == 429 or response.status_code >= 500:
                last = response
                if attempt + 1 >= self._max_attempts:
                    break
                delay = self._retry_after_s(response, attempt)
                logger.warning(
                    "jev: HTTP %d, retrying in %.2fs (attempt %d/%d)",
                    response.status_code, delay, attempt + 1, self._max_attempts,
                )
                await self._sleep(delay)
                continue
            raise JevError(f"Jev rejected the request: HTTP {response.status_code} {self._error_detail(response)}")

        detail = "" if last is None else f": {self._error_detail(last)}"
        raise JevError(f"Jev failed after {self._max_attempts} attempts{detail}")


# ------------------------------------------------------------------ state construction (trim)


def _round(value: Any, digits: int = 1) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def estimate_tokens(state: Any) -> int:
    """A rough token count for a state object: about four characters per token.

    A *proxy*, and labelled as one. The real count comes back as ``usage.input_tokens`` and is
    what the cost log uses; this exists so a test can assert the state stays inside a budget
    without spending a request, and so a regression that starts shipping the whole entity dump
    fails in CI instead of on the bill.
    """
    return max(1, len(json.dumps(state, separators=(",", ":"), default=str)) // 4)


def build_state(
    *,
    frame: Mapping[str, Any],
    decision: Mapping[str, Any],
    regions: Sequence[Mapping[str, Any]] = (),
    previous: Mapping[str, Any] | None = None,
    region_limit: int = 6,
) -> dict:
    """Build the trimmed ``state`` object — the first of the three disciplines.

    State is the *only* thing billed and the main thing that moves latency, so trimming is the
    cost model rather than an optimisation on top of it. A first window of the recorded session
    carries **289 Home Assistant entities**, most of them ``update.*``/``button.*`` noise that
    bears on no judgment in ``docs/jev.md``.

    Four rules, all applied here:

    * **Named fields, not an entity dump.** The relationships between the parts stay legible.
    * **Only what the question needs.** ``frame``/``decision`` are the loop's own summaries; no
      raw entity list and no 512-element rate vector goes in.
    * **Only what changed**, when ``previous`` is supplied. "What is new" is the honest
      representation, and the reservoir already holds the history.
    * **Rounded numbers.** ``21.4`` is one token; a full float is five.

    ``regions`` is truncated to the busiest ``region_limit`` rows and to non-zero ones, because
    ``region_usage()`` collapses 138,639 neurons to about a dozen rows and the empty ones are
    still tokens.
    """
    state: dict[str, Any] = {
        "window": {
            "sim_ms": _round(frame.get("sim_ms"), 0),
            "active_neurons": frame.get("active_neurons"),
            "spikes": frame.get("total_spikes"),
            "mean_rate_hz": _round(frame.get("mean_rate_hz"), 2),
        },
        "decision": {
            "sensor_c": _round(decision.get("temperature_c"), 2),
            "chosen_k": _round(decision.get("kelvin"), 0),
            "ideal_k": _round(decision.get("ideal_kelvin"), 0),
            "delta_k": _round(decision.get("error_k"), 0),
        },
        "sensing": {
            "entity": decision.get("temperature_entity"),
            "age_s": _round(decision.get("reading_age_s"), 1),
            "stale": bool(decision.get("reading_stale")),
        },
        "settings": dict(decision.get("settings") or {}),
    }

    busiest = sorted(
        (r for r in regions if float(r.get("spikes", 0) or 0) > 0),
        key=lambda r: float(r.get("spikes", 0) or 0),
        reverse=True,
    )[:region_limit]
    if busiest:
        state["regions"] = [
            {
                "class": r.get("class") or r.get("cell_class") or r.get("name"),
                "spikes": r.get("spikes"),
                "rate_hz": _round(r.get("rate_hz"), 1),
            }
            for r in busiest
        ]

    # The settings block is a small, closed set: send only what moved, like everything else.
    if previous is not None:
        return _changed_only(state, previous)
    return state


def _changed_only(state: dict, previous: Mapping[str, Any]) -> dict:
    """Keep the shaped keys from ``state``, but drop values identical to ``previous``.

    Deliberately shallow and structural: a sub-object survives with only the values that carry
    news, and one whose every value is unchanged is dropped rather than sent empty.

    **An empty result is meaningful, not a bug.** If nothing in the window changed there is no
    judgment to make, and the honest thing is not to make one — an empty state would be a request
    that costs tokens to tell the model nothing. Callers should treat ``{}`` as "skip this" rather
    than sending it. The ``regions`` list is compared wholesale, because a row-by-row diff of a
    dozen classes would cost more in shape than it saves in values.
    """
    out: dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(value, dict):
            if previous.get(key) != value:
                out[key] = value
            continue
        changed = {k: v for k, v in value.items() if previous.get(key, {}).get(k) != v}
        if changed:
            out[key] = changed
    return out


# ------------------------------------------------------------------------- network floor (J1)


def measure_network_floor_ms(host: str = "", *, port: int = 443, timeout: float = 8.0) -> float:
    """Time a bare TCP connect plus a TLS handshake, in milliseconds.

    This is the number the latency discipline exists for. Every Jev latency figure is
    ``floor + server time``, and a figure published without the floor describes the *network path*
    rather than the model — 199 ms of a ~460-560 ms measurement in the original source, and
    154 ms measured from the machine this was developed on (the first request it ever made to the
    host configured at the time; a re-measure on a warm path is ~49 ms).

    Measured with a raw socket rather than ``httpx`` on purpose: an HTTP request would include the
    server's own time, which is the thing that should be reported separately. It blocks, so call
    it through ``asyncio.to_thread`` from async code.

    Defaults to the host actually configured, so the number beside a latency figure describes the
    path that latency came from rather than the path of whichever host was written here first.
    """
    if not host:
        host = urlparse(JevConfig.from_env().endpoint).hostname or "jevtypesafeai.com"
    started = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout) as raw:
        context = ssl.create_default_context()
        with context.wrap_socket(raw, server_hostname=host):
            pass
    return (time.perf_counter() - started) * 1000.0
