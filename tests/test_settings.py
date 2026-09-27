"""The settings-update contract: validate everything, then commit everything.

This file exists because of one specific failure. Both settings entry points used to coerce and
assign inside the *same* loop, so a patch whose later value was bad still applied its earlier
values — and the caller was then told the update had been rejected. Neither path checked ranges,
so ``fps: 0`` was accepted and silently papered over downstream by ``max(1.0, fps)`` in
``run_loop``. And on the WebSocket path there was no ``try`` at all: a bad value raised out of
the receive loop into the blanket handler, which logged "websocket failed" and closed the
connection, so a single bad number took the dashboard's whole activity stream with it.

The contract pinned down here:

1. A patch is validated in full before anything is assigned.
2. A rejected patch changes nothing at all.
3. Ranges are enforced where a value is nonsense rather than merely unusual.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from flybrain.activity import FIELD_BOUNDS, ActivitySettings
from flybrain.loop import LoopConfig

# --------------------------------------------------------------- activity settings


def test_validate_patch_returns_without_applying() -> None:
    """Validation is a pure step: it must not touch the object it was asked about."""
    settings = ActivitySettings()
    before = settings.to_dict()
    coerced = ActivitySettings.validate_patch({"fps": 30, "gain": 4})
    assert coerced == {"fps": 30.0, "gain": 4.0}
    assert settings.to_dict() == before


def test_apply_patch_commits_accepted_values() -> None:
    settings = ActivitySettings()
    settings.apply_patch({"fps": 30, "gain": 4})
    assert settings.fps == pytest.approx(30.0)
    assert settings.gain == pytest.approx(4.0)


def test_a_rejected_patch_changes_nothing_at_all() -> None:
    """The regression. The good value used to land before the bad one raised."""
    settings = ActivitySettings()
    before = settings.to_dict()
    with pytest.raises(ValueError):
        settings.apply_patch({"fps": 30, "gain": "not-a-number"})
    assert settings.to_dict() == before


def test_unknown_settings_are_rejected_before_anything_is_applied() -> None:
    settings = ActivitySettings()
    before = settings.to_dict()
    with pytest.raises(KeyError):
        settings.apply_patch({"fps": 30, "nonsense": 1})
    assert settings.to_dict() == before


@pytest.mark.parametrize(
    "patch",
    [
        {"fps": 0},
        {"fps": -1},
        {"fps": 5000},
        {"gain": -1},
        {"gamma": 0},
        {"background_fraction": 1.5},
        {"window_ms": 0},
    ],
)
def test_out_of_range_values_are_rejected(patch: dict) -> None:
    settings = ActivitySettings()
    before = settings.to_dict()
    with pytest.raises(ValueError):
        settings.apply_patch(patch)
    assert settings.to_dict() == before


def test_every_activity_field_has_bounds() -> None:
    """A new field must not silently arrive without a range decision.

    ``validate_patch`` treats a missing entry as unbounded, which is right for a field that
    genuinely is and wrong for one nobody thought about. This makes "nobody thought about it"
    visible at the point the field is added.
    """
    assert set(FIELD_BOUNDS) == set(ActivitySettings.__dataclass_fields__)


def test_numeric_strings_are_coerced_not_rejected() -> None:
    """The dashboard sends JSON, but a form field or a shell shortcut may send text."""
    settings = ActivitySettings()
    settings.apply_patch({"fps": "30"})
    assert settings.fps == pytest.approx(30.0)


# --------------------------------------------------------------- loop settings


def test_a_string_false_does_not_become_true() -> None:
    """``bool("false")`` is ``True``.

    That is how "keep the real light read-only" quietly turns into a live service call, so
    string booleans are matched against the same falsy words the environment loader uses.
    """
    loop = LoopConfig()
    loop.apply({"dry_run": "false"})
    assert loop.dry_run is False
    loop.apply({"dry_run": "true"})
    assert loop.dry_run is True
    loop.apply({"dry_run": "0"})
    assert loop.dry_run is False


def test_loop_config_apply_is_all_or_nothing() -> None:
    """``dry_run`` lives in this dataclass, so a half-applied patch is a safety problem.

    A caller that is told "bad value, nothing changed" must be able to believe it.
    """
    loop = LoopConfig()
    before = loop.to_dict()
    with pytest.raises(ValueError):
        loop.apply({"dry_run": False, "deadband_k": "not-a-number"})
    assert loop.to_dict() == before
    assert loop.dry_run is True


# --------------------------------------------------------------- the socket path


def test_a_bad_settings_message_is_reported_and_does_not_raise(monkeypatch) -> None:
    """One malformed message must not take the activity stream down with it.

    The socket is how the dashboard sees anything at all. This used to raise out of the receive
    loop into a blanket handler that closed the connection, so a single out-of-range number
    froze the whole view.
    """
    from flybrain import server

    sent: list[str] = []

    class _WS:
        async def send_text(self, text: str) -> None:
            sent.append(text)

    monkeypatch.setattr(server.service, "settings", ActivitySettings())

    async def run() -> None:
        await server._apply_dashboard_settings(
            _WS(), {"type": "settings", "settings": {"fps": 0}}
        )
        # ...and the socket still works afterwards, which is the half that matters.
        await server._apply_dashboard_settings(
            _WS(), {"type": "settings", "settings": {"fps": 30}}
        )

    asyncio.run(run())

    assert json.loads(sent[0])["type"] == "error"
    assert "fps" in json.loads(sent[0])["error"]
    assert server.service.settings.fps == pytest.approx(30.0)
