"""Live checks against the real Jev endpoint. Skipped unless asked for.

Run them with::

    FLYBRAIN_LIVE_TESTS=1 .venv/bin/python -m pytest tests/test_jev_live.py -q -s

They are excluded by default because they need the network, they cost money (a fraction of a
cent, but not zero), and a CI runner has neither a key nor a reason to reach this API.

**These tests do not assert that Jev works.** They assert that our *diagnosis* of it is truthful,
which is a different and more useful claim: they pass whether the credential is good, missing, or
rejected, as long as :meth:`JevClient.available` says the right one. A test that failed whenever
the key was bad would be a test people learn to ignore, and the whole point of the credential
surface is that "off" and "broken" look different.

One of them needs no credential at all and is worth running on any new machine: the network floor.
Every latency number in ``docs/jev.md`` is meaningless without it.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from flybrain.jev import (
    JevClient,
    JevConfig,
    measure_network_floor_ms,
    noul,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FLYBRAIN_LIVE_TESTS", "0") in {"0", "false", "no", ""},
    reason="set FLYBRAIN_LIVE_TESTS=1 to run the live Jev checks",
)

KNOWN_REASONS = {"ok", "no_key", "unauthorized", "unreachable"}


def test_the_network_floor_is_measurable_without_a_credential() -> None:
    """A latency claim without this number describes the network path, not the model.

    No key is needed to measure it, which is why this runs even on a machine with no TypeSafe
    account — and why it should be re-run whenever the answer "is Jev fast enough?" comes up.
    """
    floor = measure_network_floor_ms()
    print(f"\nnetwork floor to api.typesafe.ai: {floor:.1f} ms")
    assert 0.0 < floor < 5000.0, "a floor outside this range means the measurement itself is wrong"


def test_available_reports_a_truthful_reason() -> None:
    config = JevConfig.from_env()
    client = JevClient(config)

    async def run():
        # One event loop for both calls: an httpx AsyncClient is bound to the loop it opened its
        # connections on, so closing it from a second `asyncio.run` raises "Event loop is closed".
        try:
            return await client.available(refresh=True)
        finally:
            await client.aclose()

    status = asyncio.run(run())

    print(f"\nJev status: available={status.available} reason={status.reason} detail={status.detail!r}")
    assert status.reason in KNOWN_REASONS
    # The three failure reasons are mutually exclusive, and only "ok" may claim availability.
    assert status.available == (status.reason == "ok")
    if status.reason == "ok":
        assert status.models, "an authorised probe should list the models on offer"
        assert config.model in status.models, (
            f"pinned model {config.model!r} is not in {list(status.models)}; the pin is wrong or "
            "the version has been retired"
        )
    else:
        print(f"  -> Jev is not usable from here ({status.reason}). The client is still correct.")


def test_a_real_judgment_round_trips_if_a_credential_works() -> None:
    """The only test that exercises a real answer. Skipped, loudly, without a working key."""
    config = JevConfig.from_env()
    client = JevClient(config)

    async def run():
        status = await client.available(refresh=True)
        if not status.available:
            return None, status
        try:
            response = await client.ask(
                {"window": {"sim_ms": 300}, "sensing": {"entity": "sensor.test", "stale": False}},
                {
                    "quiet": noul("Is the house quiet right now, given nothing has changed?"),
                    "moving": noul("Is there any sign of movement in this state?"),
                },
            )
        finally:
            await client.aclose()
        return response, status

    response, status = asyncio.run(run())
    if response is None:
        pytest.skip(f"no usable credential ({status.reason}: {status.detail})")

    print(
        f"\nlive call: {response.input_tokens} input tokens, "
        f"${response.cost_usd:.8f}, {response.latency_ms:.0f} ms "
        f"(floor {response.network_floor_ms}), answered by {response.model}"
    )
    assert response.model == config.model, "the pinned model is not the one that answered"
    assert set(response.answers) == {"quiet", "moving"}
    # Two questions in one request: the batching rule, measured rather than assumed.
    assert len(client.calls) == 1
    for answer in response.answers.values():
        assert answer.kind == "noul"
        assert 0.0 <= answer.noul <= 1.0
        # A noul genuinely has no confidence field, so it can never gate anything.
        assert answer.confidence is None
