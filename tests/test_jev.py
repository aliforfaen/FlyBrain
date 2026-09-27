"""The Jev client: wire shapes, the three call disciplines, and the credential surface.

No network, no key, and no ``typesafe_sdk`` is needed to run any of this. The transport is
injected, so the whole client — including retries and error mapping — is exercised against the
fixtures in ``tests/fixtures/jev/``, which is what makes the batching rule testable at all.

The properties worth defending:

1. **One request per judgment.** Five questions in one call cost the same as one; five calls pay
   for the state five times. This is asserted by counting calls, because a refactor that quietly
   turns batching into a loop is invisible on the bill until it is large.
2. **A `choice` question uses `criteria`, and `choice` answers key `probabilities` by label.**
   Both were wrong in the first draft of ``docs/jev.md``, which is why they are pinned here.
3. **A `noul` cannot be gated on.** It carries no confidence, so routing it is a refusal rather
   than a low score.
4. **A rejected credential is not a retry.** 401 and 403 stop immediately; 429 and 5xx retry.
5. **The key never appears in anything loggable.**
"""

from __future__ import annotations

import asyncio
import itertools
import json
import pathlib

import httpx
import pytest

from flybrain.jev import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    HA_ACTION,
    LABEL,
    PAINT,
    Answer,
    JevAuthError,
    JevClient,
    JevConfig,
    JevError,
    JevModelMismatch,
    Risk,
    build_state,
    choice,
    estimate_tokens,
    noul,
    parse_answer,
    reconcile_score,
    route,
    score,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "jev"


def fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def live_config(**kwargs) -> JevConfig:
    """A configuration that is actually switched on.

    Since ``JEV_ENABLED`` defaults to off, a key alone no longer means Jev will be called, so
    every test that is *about* a working layer has to say so. Tests about the switch itself build
    ``JevConfig(...)`` directly.
    """
    kwargs.setdefault("api_key", "jv_live_test")
    kwargs.setdefault("enabled_flag", True)
    # The real deployment has this measured and set, and a latency figure without it describes
    # geography rather than the model — so the default test config carries it too.
    kwargs.setdefault("network_floor_ms", 49.0)
    return JevConfig(**kwargs)


def client_for(response_text: str, *, status: int = 200, config: JevConfig | None = None, calls=None):
    """A client whose transport always answers with one canned response."""
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return httpx.Response(status, text=response_text, headers={"content-type": "application/json"})

    return JevClient(
        config or live_config(),
        transport=httpx.MockTransport(handler),
    )


def client_seq(responses, *, calls=None, config=None):
    """A client that answers with a different response per attempt."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        status, text, headers = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(status, text=text, headers=headers or {"content-type": "application/json"})

    return JevClient(config or live_config(), transport=httpx.MockTransport(handler))


def run(coro):
    """``pytest-asyncio`` is not a project dependency; the repo drives coroutines by hand."""
    return asyncio.run(coro)


# ------------------------------------------------------------------------------ configuration


class TestConfig:
    def test_the_model_defaults_to_the_pinned_version_not_the_alias(self) -> None:
        """The alias moves when a release ships and would silently invalidate every threshold."""
        assert JevConfig().model == DEFAULT_MODEL == "jev-1.13.0"
        assert "latest" not in JevConfig().model

    def test_no_key_means_disabled(self) -> None:
        assert JevConfig(api_key=None).enabled is False
        assert JevConfig(api_key="   ").enabled is False

    def test_the_key_comes_from_a_welcoming_environment(self) -> None:
        cfg = JevConfig.from_env({"TYPESAFE_API_KEY": "  jv_live_x  "})
        assert cfg.api_key == "jv_live_x", "surrounding whitespace must not become part of a key"

    def test_the_base_url_is_normalized_whichever_way_it_is_written(self) -> None:
        """A bare host and the full endpoint are the same endpoint.

        Both spellings arrive in the wild — a reader copies the full path out of the docs, a
        base-URL habit writes only the host — so neither is treated as a mistake.
        """
        host = "https://jevtypesafeai.com"
        assert JevConfig(base_url=host).endpoint == DEFAULT_BASE_URL
        assert JevConfig(base_url=DEFAULT_BASE_URL).endpoint == DEFAULT_BASE_URL

    def test_a_trailing_slash_is_tolerated(self) -> None:
        assert JevConfig(base_url=f"{DEFAULT_BASE_URL}/").endpoint == DEFAULT_BASE_URL

    def test_an_empty_url_falls_back_to_the_default(self) -> None:
        assert JevConfig(base_url="   ").endpoint == DEFAULT_BASE_URL

    def test_the_redacted_config_carries_no_key(self) -> None:
        secret = "jv_live_supersecretvalue"
        redacted = JevConfig(api_key=secret).redacted()
        assert secret not in json.dumps(redacted)
        assert redacted["key_present"] is True
        # A hint is allowed and useful: it distinguishes two keys without revealing either.
        assert redacted["key_hint"] == "..." + secret[-4:]

    def test_redacted_says_when_there_is_no_key(self) -> None:
        assert JevConfig(api_key=None).redacted()["key_hint"] is None

    def test_a_bad_timeout_falls_back_rather_than_raising(self) -> None:
        assert JevConfig.from_env({"JEV_TIMEOUT_S": "soon"}).timeout_s == 30.0

    def test_the_floor_is_unknown_until_measured(self) -> None:
        """`None` rather than 0, so an unmeasured floor cannot be reported as a measurement."""
        assert JevConfig.from_env({}).network_floor_ms is None
        assert JevConfig.from_env({"JEV_NETWORK_FLOOR_MS": "154"}).network_floor_ms == 154.0


class TestThresholds:
    """The thresholds are settings, not literals, because they need calibrating on our own data."""

    def test_the_defaults_match_the_documented_measurements(self) -> None:
        """The floor must separate the extremes *measured here*, not the borrowed ones.

        A state that determines the answer scored 0.94-0.99; a state that does not settle the
        question scored ~0.34. The floor has to sit above the latter so it can never write, and at
        or below the former so a determined answer can at least propose a label.
        """
        cfg = JevConfig()
        assert cfg.risk("label").confirm_at <= 0.94, "a determined answer must at least propose"
        assert cfg.risk("label").act_at > 0.34, "an under-determined answer must never write"
        assert cfg.risk("label").act_at <= 0.99, "the most determined answer must be able to write"

    def test_every_tier_is_stricter_than_the_one_below_it(self) -> None:
        cfg = JevConfig()
        ladder = [cfg.risk("paint"), cfg.risk("label"), cfg.risk("ha_action")]
        for lower, higher in itertools.pairwise(ladder):
            assert higher.act_at >= lower.act_at
            assert higher.confirm_at >= lower.confirm_at

    def test_a_threshold_can_be_overridden(self) -> None:
        cfg = JevConfig.from_env({"JEV_LABEL_ACT": "0.99", "JEV_LABEL_CONFIRM": "0.9"})
        assert cfg.risk("label").act_at == 0.99
        assert cfg.risk("label").confirm_at == 0.9

    def test_an_inverted_threshold_falls_back_instead_of_loosening(self) -> None:
        """A typo must not turn 'propose a label' into 'write one'."""
        cfg = JevConfig.from_env({"JEV_LABEL_ACT": "0.5", "JEV_LABEL_CONFIRM": "0.9"})
        assert cfg.risk("label") == LABEL

    def test_a_nonsense_threshold_falls_back(self) -> None:
        assert JevConfig.from_env({"JEV_LABEL_ACT": "soon"}).risk("label") == LABEL

    def test_an_out_of_range_threshold_falls_back(self) -> None:
        assert JevConfig.from_env({"JEV_HA_ACTION_ACT": "1.4"}).risk("ha_action") == HA_ACTION

    def test_every_tier_reads_the_name_derived_from_its_own_name(self) -> None:
        """The knob is ``JEV_<TIER>_ACT``, and this is what keeps the docs honest about it.

        ``ha_action`` was documented -- and shipped in `.env.example` -- as **`JEV_HA_ACT`**, which
        the code never read. The test that was supposed to cover it used the same wrong name, so it
        asserted the default and passed whether or not the override worked. Deriving the name here
        means renaming a tier cannot orphan a documented setting in silence.

        The values differ from every default on purpose: an override that happens to equal the
        default proves nothing.
        """
        for name in ("paint", "label", "ha_action"):
            env = {f"JEV_{name.upper()}_ACT": "0.93", f"JEV_{name.upper()}_CONFIRM": "0.50"}
            risk = JevConfig.from_env(env).risk(name)
            assert (risk.act_at, risk.confirm_at) == (0.93, 0.50), name

    def test_an_unknown_tier_is_an_error_not_a_silent_default(self) -> None:
        """Silently defaulting would let a typo in a placement name pick its own threshold."""
        with pytest.raises(JevError):
            JevConfig().risk("lable")

    def test_overriding_one_tier_leaves_the_others_alone(self) -> None:
        cfg = JevConfig.from_env({"JEV_PAINT_ACT": "0.6"})
        assert cfg.risk("paint").act_at == 0.6
        assert cfg.risk("label") == LABEL


# -------------------------------------------------------------------------------- questions


class TestQuestionBuilders:
    def test_a_choice_uses_criteria_not_options(self) -> None:
        """The field is `criteria`, a label -> description mapping. `options` is not a field."""
        question = choice("Which failure mode?", {"healthy": "nothing wrong", "stale": "no reading"})
        assert question["type"] == "choice"
        assert question["criteria"] == {"healthy": "nothing wrong", "stale": "no reading"}
        assert "options" not in question

    def test_a_choice_without_labels_is_rejected_here_not_by_the_api(self) -> None:
        with pytest.raises(ValueError):
            choice("Which?", {})

    def test_a_score_uses_an_ordered_sequence_of_levels(self) -> None:
        question = score("How busy?", ["empty", "quiet", "normal", "busy"])
        assert question["criteria"] == ["empty", "quiet", "normal", "busy"]

    def test_a_score_without_levels_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            score("How busy?", [])

    def test_a_noul_needs_nothing_but_a_question(self) -> None:
        assert noul("Is the house quiet?") == {
            "type": "noul",
            "instructions": "Is the house quiet?",
        }

    def test_the_ids_are_not_sent_to_the_model(self) -> None:
        """Question ids are for our code; the instructions have to stand alone."""
        question = choice("Which failure mode?", {"healthy": "fine"})
        assert "id" not in question and "name" not in question


# ---------------------------------------------------------------------------------- parsing


class TestParseAnswer:
    def test_a_choice_answer_keys_probabilities_by_label(self) -> None:
        """Not by index. The doc originally claimed index strings for both kinds; only score does."""
        answer = parse_answer("failure_mode", fixture("response_choice.json")["answers"]["failure_mode"])
        assert answer.kind == "choice"
        assert answer.choice == "healthy"
        assert answer.confidence == pytest.approx(0.56)
        # Keyed by LABEL, which is the half of this that the first draft of the docs got wrong.
        assert set(answer.probabilities) == {"healthy", "stale_sensor", "drifting", "unknown"}

    def test_a_score_answer_carries_score_legend_and_confidence(self) -> None:
        answer = parse_answer("how_busy", fixture("response_score.json")["answers"]["how_busy"])
        assert answer.kind == "score"
        assert answer.score == pytest.approx(1.35)
        assert answer.confidence == pytest.approx(0.44)
        assert answer.legend["3"] == "busy"

    def test_a_score_answer_keys_probabilities_by_level(self) -> None:
        answer = parse_answer("how_busy", fixture("response_score.json")["answers"]["how_busy"])
        assert set(answer.probabilities) == {"0", "1", "2", "3", "4"}

    def test_a_noul_answer_has_no_confidence(self) -> None:
        """Absence, not zero. Zero would read as 'certainly uncertain' and route confidently."""
        answer = parse_answer("is_responding", fixture("response_noul.json")["answers"]["is_responding"])
        assert answer.kind == "noul"
        assert answer.noul == pytest.approx(0.67)
        assert answer.confidence is None
        assert answer.gateable is False

    def test_the_value_property_unwraps_every_kind(self) -> None:
        assert parse_answer("q", {"type": "noul", "noul": 0.5}).value == 0.5
        assert parse_answer("q", {"type": "score", "score": 1.5, "confidence": 0.9,
                                  "legend": {}, "probabilities": {}}).value == 1.5
        assert parse_answer("q", {"type": "choice", "choice": "x", "confidence": 0.9,
                                  "probabilities": {}}).value == "x"

    def test_an_unrecognised_type_is_skipped_rather_than_fatal(self) -> None:
        """Forward compatibility: a new answer kind must not break the whole response."""
        assert parse_answer("q", {"type": "ranking", "order": []}) is None


class TestScoreReconciliation:
    def test_a_consistent_score_reconciles(self) -> None:
        answer = parse_answer("how_busy", fixture("response_score.json")["answers"]["how_busy"])
        # A real response: 0.16*0 + 0.33*1 + 0.5*2 + 0.01*3 = 1.35, and the API reported 1.35.
        assert reconcile_score(answer) < 0.02

    def test_the_two_decimal_rounding_is_within_tolerance(self) -> None:
        """The mismatch is expected: probabilities are rounded, the score is not.

        The true distribution was ``{0.001, 0.051, 0.221, 0.727}``, whose weighted mean is exactly
        ``2.674``. The response rounds the probabilities to two decimals and computes the score at
        full precision, so recomputing it from what was *returned* gives ``2.68``. A gap of ~0.006
        is the documented behaviour; a gap of 0.5 would mean the score and its own distribution
        disagree, which is the thing this check exists to catch.
        """
        answer = parse_answer("q", {
            "type": "score", "score": 2.674, "confidence": 0.9,
            "legend": {"0": "empty", "1": "quiet", "2": "normal", "3": "busy"},
            "probabilities": {"0": 0.0, "1": 0.05, "2": 0.22, "3": 0.73},
        })
        assert reconcile_score(answer) < 0.02

    def test_an_inconsistent_score_is_caught(self) -> None:
        answer = parse_answer("q", {
            "type": "score", "score": 3.0, "confidence": 0.9, "legend": {"0": "a", "1": "b"},
            "probabilities": {"0": 0.5, "1": 0.5},
        })
        assert reconcile_score(answer) == pytest.approx(2.5)

    def test_non_score_answers_have_nothing_to_reconcile(self) -> None:
        assert reconcile_score(parse_answer("q", {"type": "noul", "noul": 0.9})) is None
        assert reconcile_score(parse_answer("q", {"type": "choice", "choice": "a",
                                                  "confidence": 0.9, "probabilities": {}})) is None

    def test_a_non_numeric_legend_key_is_not_a_level_map(self) -> None:
        answer = parse_answer("q", {"type": "score", "score": 1.0, "confidence": 0.9,
                                    "legend": {"low": "a"}, "probabilities": {"low": 1.0}})
        assert reconcile_score(answer) is None


# ---------------------------------------------------------------------------------- routing


def answer_with_confidence(confidence: float | None) -> Answer:
    return Answer(question_id="q", kind="choice", choice="x", confidence=confidence)


class TestRouting:
    def test_high_confidence_acts(self) -> None:
        assert route(answer_with_confidence(0.99), LABEL).action == "act"

    def test_medium_confidence_proposes(self) -> None:
        assert route(answer_with_confidence(0.85), LABEL).action == "confirm"

    def test_low_confidence_goes_to_a_human(self) -> None:
        assert route(answer_with_confidence(0.55), LABEL).action == "needs_human"

    def test_the_bands_are_closed_at_the_bottom(self) -> None:
        """Exactly `confirm_at` proposes; exactly `act_at` acts. No off-by-one at the boundary."""
        risk = Risk("t", act_at=0.9, confirm_at=0.8)
        assert route(answer_with_confidence(0.8), risk).action == "confirm"
        assert route(answer_with_confidence(0.9), risk).action == "act"
        assert route(answer_with_confidence(0.7999), risk).action == "needs_human"

    def test_the_same_confidence_routes_differently_by_risk(self) -> None:
        """The ladder is monotone: what is good enough to paint may not be good enough to gate.

        0.85 paints a verdict, proposes a label, and cannot even confirm a house action. Each step
        costs more to get wrong, so each step demands more certainty.
        """
        answer = answer_with_confidence(0.85)
        assert route(answer, PAINT).action == "act"
        assert route(answer, LABEL).action == "confirm"
        assert route(answer, HA_ACTION).action == "needs_human"

    def test_a_very_confident_answer_can_still_gate_an_action(self) -> None:
        answer = answer_with_confidence(0.96)
        assert route(answer, LABEL).action == "act"
        assert route(answer, HA_ACTION).action == "confirm"

    def test_an_answer_at_the_label_floor_writes(self) -> None:
        """A routing boundary probe, not a measured confidence: exactly at ``act_at``."""
        cfg = JevConfig()
        assert route(answer_with_confidence(cfg.risk("label").act_at), LABEL).action == "act"

    def test_an_answer_just_under_the_label_floor_only_proposes(self) -> None:
        """One step below ``act_at`` must fall back to confirm, not write.

        If this ever starts acting, the floor has been lowered to where a borderline answer writes
        a label — and a wrong label corrupts the training set silently.
        """
        cfg = JevConfig()
        answer = answer_with_confidence(cfg.risk("label").act_at - 0.01)
        assert route(answer, LABEL).action == "confirm"

    def test_gating_a_house_action_needs_more_than_writing_a_label(self) -> None:
        assert route(answer_with_confidence(0.96), LABEL).action == "act"
        assert route(answer_with_confidence(0.96), HA_ACTION).action == "confirm"

    def test_a_noul_cannot_be_gated_even_at_certainty(self) -> None:
        """There is no confidence field to gate on, so a `noul` is a fact, never a gate."""
        answer = parse_answer("q", {"type": "noul", "noul": 1.0})
        routing = route(answer, PAINT)
        assert routing.action == "needs_human"
        assert "two-option choice" in routing.reason

    def test_every_routing_explains_itself(self) -> None:
        for confidence in (0.99, 0.85, 0.5):
            assert route(answer_with_confidence(confidence), LABEL).reason


# ---------------------------------------------------------------------------- the client


class TestAsking:
    def test_the_answers_come_back_keyed_by_question_id(self) -> None:
        client = client_for(fixture_text("response_choice.json"))
        response = run(client.ask({"window": {}}, {"failure_mode": choice("Which?", {"a": "b"})}))
        assert response.answers["failure_mode"].choice == "healthy"

    def test_every_question_goes_in_one_request(self) -> None:
        """Batching: five questions cost and take the same as one, so there must be one call."""
        calls: list[httpx.Request] = []
        client = client_for(fixture_text("response_choice.json"), calls=calls)
        questions = {"a": noul("Is it quiet?"), "b": noul("Is it dark?"), "c": noul("Is it cold?")}
        run(client.ask({"x": 1}, questions))
        assert len(calls) == 1
        body = json.loads(calls[0].content)
        assert set(body["questions"]) == {"a", "b", "c"}

    def test_the_model_is_pinned_in_the_request(self) -> None:
        calls: list[httpx.Request] = []
        run(client_for(fixture_text("response_noul.json"), calls=calls).ask({}, {"q": noul("?")}))
        assert json.loads(calls[0].content)["model"] == "jev-1.13.0"

    def test_the_authorization_header_is_a_bearer_token(self) -> None:
        calls: list[httpx.Request] = []
        run(client_for(fixture_text("response_noul.json"), calls=calls).ask({}, {"q": noul("?")}))
        assert calls[0].headers["authorization"] == "Bearer jv_live_test"

    def test_the_reported_cost_is_preferred_over_our_own_multiplication(self) -> None:
        """The vendor's number is the one that will be billed; ours would drift from it."""
        response = run(client_for(fixture_text("response_choice.json")).ask({}, {"q": noul("?")}))
        assert response.input_tokens == 458
        assert response.cost_reported is True
        assert response.cost_usd == pytest.approx(0.000193)
        assert response.credits_remaining_usd is not None

    def test_a_host_that_reports_no_cost_falls_back_to_computing_it(self) -> None:
        body = fixture("response_noul.json")
        del body["usage"]["cost_usd"]
        body["usage"].pop("credits_remaining_usd", None)
        response = run(client_for(json.dumps(body)).ask({}, {"q": noul("?")}))
        assert response.cost_reported is False
        assert response.cost_usd == pytest.approx(305 * 42.0 / 1e9)

    def test_a_judgment_costs_a_fraction_of_a_cent(self) -> None:
        """The order of magnitude is what makes generous labelling affordable."""
        response = run(client_for(fixture_text("response_choice.json")).ask({}, {"q": noul("?")}))
        assert response.cost_usd < 0.001

    def test_the_call_log_records_tokens_latency_and_the_floor(self) -> None:
        """A latency number without the floor beside it describes geography, not the model."""
        client = JevClient(
            live_config(network_floor_ms=154.0),
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, text=fixture_text("response_noul.json"))
            ),
        )
        run(client.ask({}, {"q": noul("?")}))
        entry = client.calls[0]
        assert entry["floor_ms"] == 154.0
        assert entry["input_tokens"] == 305
        assert entry["questions"] == 1

    def test_an_empty_question_set_is_refused(self) -> None:
        with pytest.raises(JevError):
            run(client_for(fixture_text("response_noul.json")).ask({}, {}))

    def test_asking_without_a_key_refuses_before_any_request(self) -> None:
        calls: list[httpx.Request] = []
        client = JevClient(
            JevConfig(api_key=None),
            transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, text="{}")),
        )
        with pytest.raises(JevAuthError) as excinfo:
            run(client.ask({}, {"q": noul("?")}))
        assert excinfo.value.reason == "no_key"
        assert calls == []

    def test_the_response_model_is_asserted_against_the_pin(self) -> None:
        """A pinned version is only pinned if a mismatch is noticed.

        The vendor's own docs say the reported model 'may differ from the alias supplied in the
        request', so this field is meaningful rather than decorative.
        """
        body = fixture("response_noul.json")
        body["model"] = "jev-1.14.0"
        client = client_for(json.dumps(body))
        with pytest.raises(JevModelMismatch):
            run(client.ask({}, {"q": noul("?")}))

    def test_a_model_mismatch_can_be_downgraded_to_a_warning(self) -> None:
        body = fixture("response_noul.json")
        body["model"] = "jev-1.14.0"
        client = client_for(json.dumps(body), config=live_config(api_key="k", strict_model=False))
        assert run(client.ask({}, {"q": noul("?")})).model == "jev-1.14.0"

    def test_an_unknown_answer_kind_does_not_fail_the_others(self) -> None:
        body = fixture("response_noul.json")
        body["answers"]["future"] = {"type": "ranking", "order": ["a"]}
        response = run(client_for(json.dumps(body)).ask({}, {"q": noul("?")}))
        assert set(response.answers) == {"is_responding"}

    def test_a_non_json_body_is_an_error_not_a_crash(self) -> None:
        with pytest.raises(JevError):
            run(client_for("<html>nope</html>").ask({}, {"q": noul("?")}))

    def test_a_response_without_answers_is_an_error(self) -> None:
        with pytest.raises(JevError):
            run(client_for('{"model": "jev-1.13.0", "usage": {}}').ask({}, {"q": noul("?")}))


class TestErrorHandling:
    def test_a_missing_key_is_403_and_says_no_key(self) -> None:
        client = client_seq([(403, fixture_text("error_403.json"), None)])
        with pytest.raises(JevAuthError) as excinfo:
            run(client.ask({}, {"q": noul("?")}))
        assert excinfo.value.reason == "no_key"

    def test_a_rejected_key_is_401_and_says_unauthorized(self) -> None:
        """Distinct from 'no key', because the fix is different — the key looks fine."""
        client = client_seq([(401, fixture_text("error_401.json"), None)])
        with pytest.raises(JevAuthError) as excinfo:
            run(client.ask({}, {"q": noul("?")}))
        assert excinfo.value.reason == "unauthorized"
        assert "Cannot authenticate" in str(excinfo.value)

    def test_a_credential_failure_is_never_retried(self) -> None:
        """Retrying a 401 only spends time; the key does not become valid."""
        calls: list[httpx.Request] = []
        client = client_seq([(401, fixture_text("error_401.json"), None)], calls=calls)
        with pytest.raises(JevAuthError):
            run(client.ask({}, {"q": noul("?")}))
        assert len(calls) == 1

    def test_a_429_is_retried_and_the_retry_after_is_honoured(self) -> None:
        slept: list[float] = []
        calls: list[httpx.Request] = []
        client = client_seq(
            [
                (429, '{"detail": "slow down"}', {"retry-after": "2.5"}),
                (200, fixture_text("response_noul.json"), None),
            ],
            calls=calls,
        )

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        client._sleep = fake_sleep
        response = run(client.ask({}, {"q": noul("?")}))
        assert response.answers["is_responding"].noul == pytest.approx(0.67)
        assert len(calls) == 2
        assert slept == [2.5], "the server's own backoff instruction must win over ours"

    def test_retry_after_ms_is_preferred_when_present(self) -> None:
        """The vendor sends both spellings; millisecond precision matters below one second."""
        slept: list[float] = []
        client = client_seq(
            [
                (429, "{}", {"retry-after": "5", "retry-after-ms": "250"}),
                (200, fixture_text("response_noul.json"), None),
            ]
        )

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        client._sleep = fake_sleep
        run(client.ask({}, {"q": noul("?")}))
        assert slept == [0.25]

    def test_a_server_error_is_retried_then_given_up_on(self) -> None:
        calls: list[httpx.Request] = []
        client = client_seq([(503, "{}", None)], calls=calls)

        async def no_sleep(seconds: float) -> None:
            return None

        client._sleep = no_sleep
        with pytest.raises(JevError):
            run(client.ask({}, {"q": noul("?")}))
        assert len(calls) == 3, "three bounded attempts, not an unbounded loop"

    def test_a_4xx_that_is_not_a_rate_limit_is_not_retried(self) -> None:
        """A 400 means the request is wrong; sending it again will not make it right."""
        calls: list[httpx.Request] = []
        client = client_seq([(400, '{"detail": "bad state"}', None)], calls=calls)
        with pytest.raises(JevError):
            run(client.ask({}, {"q": noul("?")}))
        assert len(calls) == 1

    def test_a_caller_can_drive_routing_straight_off_the_response(self) -> None:
        """A real capture at confidence 0.56 is not allowed to touch the training set.

        The risk ladder doing its job on captured data rather than on a synthetic number: the same
        answer may paint a verdict (>= 0.50) and may not write a label (< 0.80), because a wrong
        label corrupts the only training data this project has.
        """
        response = run(client_for(fixture_text("response_choice.json")).ask({}, {"q": noul("?")}))
        assert response.routed(LABEL)["failure_mode"].action == "needs_human"
        assert response.routed(PAINT)["failure_mode"].action == "confirm"


class TestAvailability:
    def test_no_key_says_so_without_asking_the_network(self) -> None:
        calls: list[httpx.Request] = []
        client = JevClient(
            JevConfig(api_key=None),
            transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, text="{}")),
        )
        status = run(client.available())
        assert status.reason == "no_key" and status.available is False
        assert calls == []

    def test_a_good_key_reports_ok(self) -> None:
        client = client_for(fixture_text("response_noul.json"))
        status = run(client.available())
        assert status.available is True and status.reason == "ok"

    def test_a_reported_credit_balance_is_surfaced(self) -> None:
        """A silent zero is how a feature stops working without anyone noticing."""
        response = run(client_for(fixture_text("response_choice.json")).ask({}, {"q": noul("?")}))
        assert response.credits_remaining_usd == pytest.approx(4.997265)

    def test_a_rejected_key_says_unauthorized(self) -> None:
        client = client_for(fixture_text("error_401.json"), status=401)
        assert run(client.available()).reason == "unauthorized"

    def test_a_missing_key_header_says_no_key(self) -> None:
        client = client_for(fixture_text("error_403.json"), status=403)
        assert run(client.available()).reason == "no_key"

    def test_an_unreachable_endpoint_says_unreachable(self) -> None:
        def boom(request):
            raise httpx.ConnectError("no route to host")

        client = JevClient(live_config(), transport=httpx.MockTransport(boom))
        status = run(client.available())
        assert status.reason == "unreachable"
        assert "ConnectError" in status.detail

    def test_the_probe_asks_one_trivial_question(self) -> None:
        """This host has no free endpoint to probe, so a check costs a fraction of a cent.

        `/api/v1/models` answers `unknown_endpoint` here, which is why the probe is a real (tiny)
        question rather than a GET — and why the answer is cached and `refresh` is explicit.
        """
        calls: list[httpx.Request] = []
        client = client_for(fixture_text("response_noul.json"), calls=calls)
        run(client.available())
        assert calls[0].method == "POST"
        assert calls[0].url.path.endswith("/decide")
        assert len(json.loads(calls[0].content)["questions"]) == 1

    def test_the_probe_is_cached(self) -> None:
        calls: list[httpx.Request] = []
        client = client_for(fixture_text("probe_200.json"), calls=calls)
        run(client.available())
        run(client.available())
        assert len(calls) == 1

    def test_the_cache_can_be_bypassed(self) -> None:
        calls: list[httpx.Request] = []
        client = client_for(fixture_text("models_200.json"), calls=calls)
        run(client.available())
        run(client.available(refresh=True))
        assert len(calls) == 2

    def test_the_status_carries_the_floor_when_it_is_known(self) -> None:
        client = client_for(
            fixture_text("models_200.json"),
            config=live_config(api_key="k", network_floor_ms=154.0),
        )
        assert run(client.available()).to_dict()["floor_ms"] == 154.0

    def test_the_status_serializes_to_json(self) -> None:
        status = run(client_for(fixture_text("models_200.json")).available())
        assert json.loads(json.dumps(status.to_dict()))["reason"] == "ok"


# --------------------------------------------------------------------------------- state


def frame(**kw):
    return {"sim_ms": 300.0, "total_spikes": 38412, "active_neurons": 11204, **kw}


def decision(**kw):
    base = {
        "temperature_c": 21.4,
        "kelvin": 4120.0,
        "ideal_kelvin": 4090.0,
        "error_k": -30.0,
        "temperature_entity": "sensor.hallway_temperature",
        "reading_age_s": 1.234,
        "reading_stale": False,
        "settings": {"deadband_k": 25, "interval_s": 15, "smooth_ms": 5000},
    }
    base.update(kw)
    return base


REGIONS = [
    {"class": "ALPN", "spikes": 812, "rate_hz": 41.234},
    {"class": "Kenyon", "spikes": 0, "rate_hz": 0.0},
    {"class": "Clock", "spikes": 130, "rate_hz": 3.9},
]


class TestBuildState:
    def test_it_sends_named_fields_not_an_entity_dump(self) -> None:
        state = build_state(frame=frame(), decision=decision())
        assert set(state) == {"window", "decision", "sensing", "settings"}
        assert state["window"]["spikes"] == 38412
        assert state["decision"]["sensor_c"] == 21.4

    def test_numbers_are_rounded(self) -> None:
        """`21.4` is one token; a full float is five."""
        state = build_state(frame=frame(sim_ms=300.4567), decision=decision(reading_age_s=1.23456))
        assert state["window"]["sim_ms"] == 300.0
        assert state["sensing"]["age_s"] == 1.2

    def test_empty_regions_are_dropped_and_the_rest_are_sorted(self) -> None:
        state = build_state(frame=frame(), decision=decision(), regions=REGIONS)
        assert [r["class"] for r in state["regions"]] == ["ALPN", "Clock"]

    def test_the_region_list_is_capped(self) -> None:
        many = [{"class": f"c{i}", "spikes": 100 - i, "rate_hz": 1.0} for i in range(30)]
        state = build_state(frame=frame(), decision=decision(), regions=many, region_limit=4)
        assert len(state["regions"]) == 4

    def test_no_regions_means_no_regions_key(self) -> None:
        assert "regions" not in build_state(frame=frame(), decision=decision())

    def test_only_changes_are_sent_when_a_previous_state_is_given(self) -> None:
        """An empty result is meaningful: nothing changed, so there is no judgment to make.

        Sending `{}` would spend tokens to tell the model nothing, so the caller is expected to
        skip the request. This is asserted rather than assumed because "send only what changed"
        quietly becoming "send everything twice" is what the cost model would not notice.
        """
        first = build_state(frame=frame(), decision=decision())
        same = build_state(frame=frame(), decision=decision(), previous=first)
        assert same == {}

    def test_an_unchanged_region_list_is_not_resent(self) -> None:
        first = build_state(frame=frame(), decision=decision(), regions=REGIONS)
        again = build_state(frame=frame(), decision=decision(), regions=REGIONS, previous=first)
        assert "regions" not in again

    def test_a_changed_value_survives_the_diff(self) -> None:
        first = build_state(frame=frame(), decision=decision())
        second = build_state(frame=frame(), decision=decision(temperature_c=22.9), previous=first)
        assert second["decision"]["sensor_c"] == 22.9
        assert "chosen_k" not in second["decision"], "unchanged values are not resent"

    def test_the_state_stays_inside_a_token_budget(self) -> None:
        """A regression that ships the whole entity dump should fail here, not on the bill.

        The budget is a character proxy because no successful response could be captured to read
        a real `input_tokens` from — the key was rejected throughout. Re-base it on the measured
        figure as soon as a valid key exists; `tests/fixtures/jev/README.md` says so.
        """
        state = build_state(frame=frame(), decision=decision(), regions=REGIONS)
        assert estimate_tokens(state) < 400

    def test_settings_are_sent_whole_the_first_time(self) -> None:
        state = build_state(frame=frame(), decision=decision())
        assert state["settings"]["interval_s"] == 15

    def test_a_missing_field_does_not_raise(self) -> None:
        """A partial snapshot must not take the judgment path down."""
        state = build_state(frame={}, decision={})
        assert state["window"]["spikes"] is None
        assert state["sensing"]["entity"] is None


# ------------------------------------------------------------------------- the HTTP surface


class _StubClient:
    """Stands in for the client so the endpoint is testable without touching the network."""

    def __init__(self, status) -> None:
        self._status = status
        self.config = live_config(api_key="jv_live_donotleakme")
        self.calls: list = []

    async def available(self, *, refresh: bool = False, probe: bool = True):
        return self._status

    async def aclose(self) -> None:
        return None


class TestStatusEndpoint:
    """The credential surface, as the dashboard would see it.

    No placement has a UI yet, so this endpoint is the only thing that exposes the layer — which
    makes "it never leaks the key" worth asserting rather than assuming.
    """

    def _payload(self, status, monkeypatch):
        from flybrain import server

        monkeypatch.setattr(server.service, "_jev", _StubClient(status))
        return run(server.get_jev_status())

    def test_a_missing_key_is_reported_as_such(self, monkeypatch) -> None:
        from flybrain.jev import JevStatus

        payload = self._payload(JevStatus(False, "no_key", "TYPESAFE_API_KEY is not set"), monkeypatch)
        assert payload["available"] is False
        assert payload["reason"] == "no_key"

    def test_a_rejected_key_is_reported_as_unauthorized_not_as_missing(self, monkeypatch) -> None:
        from flybrain.jev import JevStatus

        payload = self._payload(JevStatus(False, "unauthorized", "Cannot authenticate"), monkeypatch)
        assert payload["reason"] == "unauthorized"

    def test_a_working_endpoint_reports_the_models_it_could_use(self, monkeypatch) -> None:
        from flybrain.jev import JevStatus

        payload = self._payload(JevStatus(True, "ok", "", models=("jev-1.13.0",)), monkeypatch)
        assert payload["available"] is True
        assert payload["models"] == ["jev-1.13.0"]

    def test_the_key_never_reaches_the_response(self, monkeypatch) -> None:
        from flybrain.jev import JevStatus

        payload = self._payload(JevStatus(True, "ok"), monkeypatch)
        assert "jv_live_donotleakme" not in json.dumps(payload)
        assert payload["config"]["key_hint"] == "...akme"
        assert payload["config"]["model"] == "jev-1.13.0"

    def test_the_payload_is_json_serializable(self, monkeypatch) -> None:
        from flybrain.jev import JevStatus

        payload = self._payload(JevStatus(False, "unreachable", "ConnectError"), monkeypatch)
        assert json.loads(json.dumps(payload))["reason"] == "unreachable"


class TestJevlessMode:
    """The switch, and the promise that a watched dashboard cannot spend money.

    This matters more than it looks. The probe is a real request, and a dashboard is something
    people leave open for weeks — so "spends ~$0.03/day because it is on screen" is a silent cost
    with no benefit until a placement exists. These tests pin both halves: that off means off, and
    that the *status* path never pays for a badge.
    """

    def test_enabled_is_off_by_default_even_with_a_key(self) -> None:
        """A key is a credential, not a decision to spend."""
        config = JevConfig(api_key="jv_live_test")
        assert config.key_present is True
        assert config.enabled is False

    def test_the_switch_and_the_key_are_reported_separately(self) -> None:
        """`key_present` must not be dragged down by the switch, or the UI would hide the key."""
        redacted = JevConfig(api_key="jv_live_secret").redacted()
        assert redacted["key_present"] is True
        assert redacted["enabled"] is False
        assert redacted["key_hint"] == "...cret"

    def test_the_environment_switch_turns_it_on(self) -> None:
        assert JevConfig.from_env({"JEV_API_KEY": "k"}).enabled is False
        assert JevConfig.from_env({"JEV_API_KEY": "k", "JEV_ENABLED": "1"}).enabled is True

    def test_the_environment_switch_accepts_the_usual_spellings(self) -> None:
        for truthy in ("1", "true", "yes", "on", "TRUE", " On "):
            cfg = JevConfig.from_env({"JEV_API_KEY": "k", "JEV_ENABLED": truthy})
            assert cfg.enabled is True, truthy
        for falsy in ("0", "false", "no", "off", ""):
            cfg = JevConfig.from_env({"JEV_API_KEY": "k", "JEV_ENABLED": falsy})
            assert cfg.enabled is False, falsy

    def test_switched_off_with_a_key_says_disabled_not_no_key(self) -> None:
        """The distinction that sends a person to the right fix."""
        calls: list[httpx.Request] = []
        client = JevClient(
            JevConfig(api_key="jv_live_test"),
            transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, text="{}")),
        )
        status = run(client.available())
        assert status.reason == "disabled"
        assert status.available is False
        assert calls == [], "a switched-off layer must not touch the network at all"

    def test_no_key_still_wins_over_the_switch(self) -> None:
        """With nothing configured, "you never set this up" beats "you turned it off"."""
        client = JevClient(
            JevConfig(api_key=None),
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text="{}")),
        )
        assert run(client.available()).reason == "no_key"

    def test_asking_while_switched_off_raises_disabled_not_no_key(self) -> None:
        """A key that is present but unused must not be reported as a missing key."""
        calls: list[httpx.Request] = []
        client = JevClient(
            JevConfig(api_key="jv_live_test"),
            transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, text="{}")),
        )
        with pytest.raises(JevAuthError) as excinfo:
            run(client.ask({}, {"q": noul("?")}))
        assert excinfo.value.reason == "disabled"
        assert calls == []

    def test_the_status_path_does_not_probe(self) -> None:
        """`probe=False` is what /api/status passes, and it must not spend."""
        calls: list[httpx.Request] = []
        client = client_for(fixture_text("models_200.json"), calls=calls)
        status = run(client.available(probe=False))
        assert status.reason == "unprobed"
        assert calls == []

    def test_the_status_path_returns_the_last_known_answer_after_a_probe(self) -> None:
        """Stale-but-real beats "unknown" when something genuinely was observed."""
        calls: list[httpx.Request] = []
        client = client_for(fixture_text("models_200.json"), calls=calls)
        run(client.available())
        assert run(client.available(probe=False)).reason == "ok"
        assert len(calls) == 1, "reading the cached answer must not re-ask"

    def test_the_reason_vocabulary_is_closed(self) -> None:
        """Six reasons, each a different action for the person reading it."""
        import typing

        from flybrain.jev import JevStatus

        # `get_type_hints`, not `__dataclass_fields__[...].type`: the module uses
        # `from __future__ import annotations`, so the field type is the *string*
        # "Literal[...]" and `get_args` on it returns nothing — which silently makes the
        # companion test below pass on an empty set.
        allowed = set(typing.get_args(typing.get_type_hints(JevStatus)["reason"]))
        assert allowed == {
            "ok", "no_key", "disabled", "unprobed", "unauthorized", "unreachable",
        }

    def test_every_reason_has_a_plain_english_label_in_the_client(self) -> None:
        """A reason with no label renders as a raw token in the UI."""
        import typing
        from pathlib import Path

        from flybrain.jev import JevStatus

        reasons = typing.get_args(typing.get_type_hints(JevStatus)["reason"])
        assert reasons, "the annotation did not resolve; this test would pass vacuously"
        app = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text(encoding="utf-8")
        missing = [r for r in reasons if f"{r}:" not in app]
        assert not missing, f"Jev reasons with no client label: {missing}"


class TestDecisionInspector:
    """Placement A: the closed vocabulary, the trimmed state, and the verdict shape.

    This class exists because the first version of :func:`classify_decision` read
    ``routing.detail``, which does not exist — the field is ``reason``. Nothing caught it, so a
    real call was paid for, answered, and then turned into a 500 that was not cached. A test over
    the assembled dict is exactly what was missing.
    """

    def _row(self) -> dict:
        return {
            "t": 1790530920.67,
            "temperature_c": 26.484,
            "kelvin": 4790,
            "ideal_kelvin": 5336,
            "band": "neutral",
            "seq": 222,
            "active_neurons": 8506,
            "total_spikes": 157939,
            "window_ms": 300.0,
            "reading_age_s": 0.4,
            "top_regions": [{"name": "ALLN", "spikes": 120}],
            "settings": {"interval_s": 8.0, "deadband_k": 25.0},
        }

    def test_the_vocabulary_is_the_seven_documented_modes(self) -> None:
        from flybrain.jev import FAILURE_MODES

        assert set(FAILURE_MODES) == {
            "healthy", "saturated_sensory", "regime_mismatch", "too_few_spikes",
            "sensor_stale", "throttled", "unknown",
        }

    def test_every_mode_explains_itself(self) -> None:
        """A label nobody can verify is a label nobody should trust."""
        from flybrain.jev import FAILURE_MODES

        for label, meaning in FAILURE_MODES.items():
            assert meaning and len(meaning) > 20, label

    def test_the_question_offers_every_mode_as_a_criterion(self) -> None:
        from flybrain.jev import FAILURE_MODES, failure_mode_question

        q = failure_mode_question()
        assert q["type"] == "choice"
        assert set(q["criteria"]) == set(FAILURE_MODES)

    def test_the_state_is_trimmed_to_what_a_mode_can_use(self) -> None:
        from flybrain.jev import DECISION_STATE_FIELDS, decision_state

        state = decision_state({**self._row(), "an_irrelevant_field": 1})
        assert "an_irrelevant_field" not in state
        assert set(state) <= set(DECISION_STATE_FIELDS)
        assert state["temperature_c"] == 26.484

    def test_unmeasured_fields_are_omitted_rather_than_sent_as_null(self) -> None:
        """A null invites the model to read "not measured" as "zero"."""
        from flybrain.jev import decision_state

        row = self._row()
        row["reading_age_s"] = None
        assert "reading_age_s" not in decision_state(row)

    def test_a_real_captured_response_becomes_a_verdict(self) -> None:
        """The shape the HTTP endpoint returns, against a verbatim capture."""
        from flybrain.jev import FAILURE_MODES, classify_decision

        client = client_for(fixture_text("response_failure_mode.json"))
        verdict = run(classify_decision(client, self._row()))

        assert verdict["label"] in set(FAILURE_MODES)
        assert 0.0 <= verdict["confidence"] <= 1.0
        assert verdict["action"] in {"act", "confirm", "needs_human"}
        # The bug that motivated this class: these keys must exist, and `routing_reason` is the
        # one that was wrong.
        assert "routing_reason" in verdict
        assert verdict["routing_reason"]
        assert set(verdict["probabilities"]) == set(FAILURE_MODES)
        assert verdict["cost_usd"] > 0
        assert verdict["floor_ms"] == 49.0

    def test_a_low_confidence_verdict_only_proposes(self) -> None:
        """The capture scored 0.50, which must land on `confirm` rather than being acted on."""
        from flybrain.jev import classify_decision

        client = client_for(fixture_text("response_failure_mode.json"))
        verdict = run(classify_decision(client, self._row()))
        assert verdict["confidence"] == pytest.approx(0.5)
        assert verdict["action"] == "confirm"

    def test_a_noul_answer_comes_back_as_needs_human_not_as_an_error(self) -> None:
        """A `noul` carries no confidence, so there is nothing to gate on — a real outcome."""
        from flybrain.jev import classify_decision

        client = client_for(fixture_text("response_noul.json"), config=None)
        with pytest.raises(JevError):
            # `response_noul.json` answers under a different id, so the missing answer is an
            # error rather than a silent `unknown` verdict.
            run(classify_decision(client, self._row(), question_id="failure_mode"))

    def test_the_request_carries_the_trimmed_state_and_one_question(self) -> None:
        calls: list[httpx.Request] = []
        from flybrain.jev import DECISION_STATE_FIELDS, classify_decision

        client = client_for(fixture_text("response_failure_mode.json"), calls=calls)
        run(classify_decision(client, self._row()))
        body = json.loads(calls[0].content)
        assert len(body["questions"]) == 1
        assert "failure_mode" in body["questions"]
        assert set(body["state"]) <= set(DECISION_STATE_FIELDS)


class TestProbeAccounting:
    """The probe is a real request that really costs money.

    Leaving it out of the call log made `spent_usd` under-report the session by exactly what the
    dashboard spends on itself — the number a person uses to decide whether to leave it running.
    """

    def test_a_probe_is_recorded_with_its_cost(self) -> None:
        calls: list[httpx.Request] = []
        client = client_for(fixture_text("probe_200.json"), calls=calls)
        status = run(client.available())
        assert status.reason == "ok"
        assert len(client.calls) == 1
        entry = client.calls[0]
        assert entry["probe"] is True
        assert entry["questions"] == 1
        assert entry["cost_usd"] > 0, "a probe costs money and the log has to say so"
        assert entry["cost_reported"] is True

    def test_the_probe_cost_counts_towards_the_session_total(self) -> None:
        from flybrain.jev import session_spend

        client = client_for(fixture_text("probe_200.json"))
        run(client.available())
        run(client.available(refresh=True))
        assert len(client.calls) == 2
        assert session_spend(client) > 0

    def test_a_failed_probe_records_nothing(self) -> None:
        """Nothing was charged for a connection that never opened."""
        def boom(request):
            raise httpx.ConnectError("no route to host")

        client = JevClient(live_config(), transport=httpx.MockTransport(boom))
        assert run(client.available()).reason == "unreachable"
        assert client.calls == []

    def test_a_cached_answer_does_not_add_a_second_charge(self) -> None:
        calls: list[httpx.Request] = []
        client = client_for(fixture_text("models_200.json"), calls=calls)
        run(client.available())
        run(client.available())
        assert len(calls) == 1
        assert len(client.calls) == 1
