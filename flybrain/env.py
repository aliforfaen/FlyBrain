"""Minimal `.env` loading, so house-specific configuration is never hardcoded.

Everything about a real installation — the Home Assistant URL, its token, which entity is the
thermometer, which light to drive — is configuration, not code. It lives in `.env`, which is
gitignored so that a real address and a real token cannot be committed by accident.

Requiring the operator to `source .env` by hand in every shell is a foot-gun: forget once and
the loop quietly runs against the **mock** home, which looks like the brain doing nothing.

`load_dotenv()` reads the repository's `.env` and fills `os.environ` for any key that is not
already set. Real environment variables always win, so `FLYBRAIN_INTERVAL_S=5
.venv/bin/python -m flybrain.server`, a systemd unit, or a container env block all keep
working exactly as before.

The parser is deliberately tiny rather than depending on `python-dotenv`: the format used
here is `KEY=VALUE` one per line, and the whole file is 5 KB.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / ".env"


def parse(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines.

    Blank lines and ``#`` comments are skipped, a leading ``export `` is tolerated so a file
    that is also valid to `source` works, and one layer of matching quotes is stripped.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(
    path: str | Path | None = None,
    env: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Load `.env` into ``env`` (``os.environ`` by default).

    Returns the values that were actually applied. Existing variables are left alone unless
    ``override`` is set, and a missing file is not an error — the project has to run against
    the mock home with no configuration at all.
    """
    target = ROOT / DEFAULT_PATH if path is None else Path(path)
    store = os.environ if env is None else env
    if not target.is_file():
        return {}
    try:
        values = parse(target.read_text())
    except OSError:
        return {}

    applied: dict[str, str] = {}
    for key, value in values.items():
        if override or key not in store:
            store[key] = value
            applied[key] = value
    return applied
