"""Jev: the optional judgment layer, and the discipline that keeps it affordable.

Jev is TypeSafe's "System One" model. It does not write prose: you hand it a state and a set of
typed questions, and it returns a typed answer per question. That interface is the reason it is
here at all — not accuracy, which measured as a **tie** with a general chat model
(``docs/jev.md``). What it adds is a decision-shaped interface and a ``confidence`` per answer,
which is what makes it safe to gate anything on.

Three things about this module are deliberate and worth stating before the code:

**It is off unless configured.** With no ``TYPESAFE_API_KEY`` the client refuses to make a
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

import httpx

from flybrain.env import load_dotenv

logger = logging.getLogger(__name__)

#: Root of the first-party API. ``/v1/systemone`` is the only endpoint that answers questions;
#: ``/v1/models`` also exists and is used here as the cheap availability probe.
DEFAULT_BASE_URL = "https://api.typesafe.ai"
SYSTEM_ONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"

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

    ``reason`` is one of the two states the live endpoint actually distinguishes:

    * ``"no_key"`` — HTTP 403, ``"Must supply an API key!"``. Nothing was sent.
    * ``"unauthorized"`` — HTTP 401, ``"Cannot authenticate with the server"``. A key is present
      and the server will not accept it: revoked, mistyped, or for a different environment.

    Keeping these apart matters because the fix is different: one is a missing setting, the other
    is a credential that looks entirely well-formed.
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


#: The three tiers. The defaults are calibrated against the numbers in docs/jev.md rather than
#: invented: the measured confidence gap is narrow — 0.979 on clear cases, 0.841 on deliberately
#: ambiguous ones — so ``label`` is set to let the *measured clear* case write a label and the
#: *measured ambiguous* case only propose one. The margin is thin enough that these must be
#: re-calibrated on this project's own data before anything is trusted, which is why every tier
#: is overridable from the environment (``JEV_LABEL_ACT`` and friends) rather than fixed here.
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
        raw_key = e.get("TYPESAFE_API_KEY")
        floor = e.get("JEV_NETWORK_FLOOR_MS")
        return cls(
            api_key=(raw_key or "").strip() or None,
            model=(e.get("JEV_MODEL") or DEFAULT_MODEL).strip(),
            base_url=(e.get("JEV_BASE_URL") or DEFAULT_BASE_URL).strip(),
            timeout_s=_number(e.get("JEV_TIMEOUT_S"), 30.0),
            network_floor_ms=None if not floor else _number(floor, 0.0),
            strict_model=(e.get("JEV_STRICT_MODEL") or "1").strip().lower()
            not in {"0", "false", "no", "off"},
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
        return bool(self.api_key)

    @property
    def root(self) -> str:
        """The API root, with any endpoint path stripped off.

        ``JEV_BASE_URL`` is documented as the full ``.../v1/systemone`` URL, because that is the
        value a reader copies out of the docs. The vendor's SDK treats its base URL as a root, so
        both spellings arrive in the wild; normalizing here means neither is a mistake.
        """
        url = self.base_url.rstrip("/")
        for path in (SYSTEM_ONE_PATH, MODELS_PATH):
            url = url.removesuffix(path)
        return url.rstrip("/") or DEFAULT_BASE_URL

    @property
    def endpoint(self) -> str:
        return f"{self.root}{SYSTEM_ONE_PATH}"

    @property
    def models_url(self) -> str:
        return f"{self.root}{MODELS_PATH}"

    def redacted(self) -> dict:
        """Config, safe to log or return over HTTP. Never the key itself."""
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "network_floor_ms": self.network_floor_ms,
            "strict_model": self.strict_model,
            "key_present": self.enabled,
            "key_hint": None if not self.api_key else f"...{self.api_key[-4:]}",
        }


@dataclass(frozen=True)
class JevStatus:
    """Why Jev is or is not usable, in a form the UI can say out loud."""

    available: bool
    reason: Literal["ok", "no_key", "unauthorized", "unreachable", "disabled"]
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
    #: Computed, because the API reports token counts but no cost.
    cost_usd: float
    network_floor_ms: float | None

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
            raise JevAuthError("no_key", "TYPESAFE_API_KEY is not set")
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

    async def available(self, *, refresh: bool = False) -> JevStatus:
        """Report whether Jev can be used right now, and if not, why.

        Probes ``GET /v1/models`` rather than asking a real question: it exercises the same
        credential for no length of state, so a dashboard refresh cannot cost money. The result
        is cached for :data:`PROBE_TTL_S`, which is why ``refresh`` exists.

        The two credential failures are kept apart on purpose — see :class:`JevAuthError`.
        """
        if not self.config.enabled:
            return JevStatus(False, "no_key", "TYPESAFE_API_KEY is not set", floor_ms=self.config.network_floor_ms)
        now = self._clock()
        if (
            not refresh
            and self._status is not None
            and self._status_at is not None
            and (now - self._status_at) < PROBE_TTL_S
        ):
            return self._status

        status = await self._probe()
        self._status, self._status_at = status, now
        return status

    async def _probe(self) -> JevStatus:
        try:
            response = await self.client.get(self.config.models_url, headers=self._headers())
        except httpx.HTTPError as exc:
            return JevStatus(
                False, "unreachable", f"{type(exc).__name__}: {exc}",
                floor_ms=self.config.network_floor_ms,
            )
        if response.status_code == 200:
            models = ()
            try:
                body = response.json()
                listing = body.get("models") if isinstance(body, dict) else None
                if isinstance(listing, list):
                    models = tuple(
                        str(item.get("id", item)) if isinstance(item, dict) else str(item)
                        for item in listing
                    )
            except ValueError:
                pass  # reachable and authorized; the listing is a bonus, not the point
            return JevStatus(
                True, "ok", "", models=models, floor_ms=self.config.network_floor_ms
            )
        return self._status_for_error(response)

    def _status_for_error(self, response: httpx.Response) -> JevStatus:
        detail = self._error_detail(response)
        if response.status_code == 401:
            return JevStatus(False, "unauthorized", detail, floor_ms=self.config.network_floor_ms)
        if response.status_code == 403:
            return JevStatus(False, "no_key", detail, floor_ms=self.config.network_floor_ms)
        return JevStatus(
            False,
            "unreachable",
            f"HTTP {response.status_code}: {detail}",
            floor_ms=self.config.network_floor_ms,
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        """Pull the message out of ``{"detail": {...}}``, truncated.

        The vendor's own SDK truncates error bodies, and the shape here is confirmed against the
        live endpoint rather than guessed: a 403 arrives as
        ``{"detail": {"error_type": "authentication_error", "message": "..."}}``.
        """
        try:
            body = response.json()
        except ValueError:
            return response.text[:200]
        detail = body.get("detail") if isinstance(body, dict) else None
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
        if not self.config.enabled:
            raise JevAuthError("no_key", "TYPESAFE_API_KEY is not set")

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
        cost = input_tokens * USD_PER_INPUT_TOKEN

        result = JevResponse(
            answers=answers,
            model=answered_by,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            network_floor_ms=self.config.network_floor_ms,
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
            }
        )
        logger.info(
            "jev: %d question(s), %d input tokens, $%.8f, %.0f ms%s, answered by %s",
            question_count,
            result.input_tokens,
            result.cost_usd,
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


def measure_network_floor_ms(host: str = "api.typesafe.ai", *, port: int = 443, timeout: float = 8.0) -> float:
    """Time a bare TCP connect plus a TLS handshake, in milliseconds.

    This is the number the latency discipline exists for. Every Jev latency figure is
    ``floor + server time``, and a figure published without the floor describes the *network path*
    rather than the model — 199 ms of a ~460-560 ms measurement in the original source, and
    154 ms measured from the machine this was developed on.

    Measured with a raw socket rather than ``httpx`` on purpose: an HTTP request would include the
    server's own time, which is the thing that should be reported separately. It blocks, so call
    it through ``asyncio.to_thread`` from async code.
    """
    started = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout) as raw:
        context = ssl.create_default_context()
        with context.wrap_socket(raw, server_hostname=host):
            pass
    return (time.perf_counter() - started) * 1000.0
