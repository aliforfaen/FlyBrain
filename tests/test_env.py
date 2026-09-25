"""Tests for `.env` loading.

These never touch the developer's real `.env`: every case passes an explicit path and an
explicit mapping, which is also how the loader is used in production (only the entry points
call it with defaults).
"""

from __future__ import annotations

from pathlib import Path

from flybrain.env import load_dotenv, parse


class TestParse:
    def test_simple_pairs(self) -> None:
        assert parse("A=1\nB=two\n") == {"A": "1", "B": "two"}

    def test_comments_and_blank_lines_are_skipped(self) -> None:
        text = "# a comment\n\nA=1\n   \n# another\nB=2\n"
        assert parse(text) == {"A": "1", "B": "2"}

    def test_matching_quotes_are_stripped(self) -> None:
        assert parse("A=\"hello world\"\nB='x y'\n") == {"A": "hello world", "B": "x y"}

    def test_unbalanced_quote_is_left_alone(self) -> None:
        assert parse("A=\"oops\n") == {"A": '"oops'}

    def test_export_prefix_is_tolerated(self) -> None:
        # The file stays valid to `source`, which is what many homelab setups do.
        assert parse("export A=1\n") == {"A": "1"}

    def test_value_may_contain_equals(self) -> None:
        assert parse("TOKEN=abc=def=\n") == {"TOKEN": "abc=def="}

    def test_lines_without_an_equals_are_ignored(self) -> None:
        assert parse("JUST_A_WORD\nA=1\n") == {"A": "1"}

    def test_empty_value_is_kept(self) -> None:
        assert parse("A=\n") == {"A": ""}


class TestLoadDotenv:
    def test_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        assert load_dotenv(tmp_path / "nope.env", {}) == {}

    def test_fills_absent_keys_and_reports_them(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("A=1\nB=2\n")
        env: dict[str, str] = {"A": "already-set"}
        applied = load_dotenv(path, env)
        assert env == {"A": "already-set", "B": "2"}
        assert applied == {"B": "2"}

    def test_override_replaces_existing(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("A=from-file\n")
        env = {"A": "from-shell"}
        load_dotenv(path, env, override=True)
        assert env["A"] == "from-file"

    def test_real_environment_wins_by_default(self, tmp_path: Path) -> None:
        """`KEY=value cmd` and systemd units must keep overriding the file."""
        path = tmp_path / ".env"
        path.write_text("HA_MODE=rest\n")
        env = {"HA_MODE": "mock"}
        load_dotenv(path, env)
        assert env["HA_MODE"] == "mock"

    def test_empty_file_applies_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        path.write_text("")
        assert load_dotenv(path, {}) == {}
