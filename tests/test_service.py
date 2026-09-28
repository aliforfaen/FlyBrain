"""Tests for the systemd user service under ``deploy/``.

A unit file is configuration with a syntax, and the failure modes are quiet: a wrong
``WorkingDirectory`` sends recordings to ``$HOME`` while the dashboard still works, and a
``ProtectHome=`` added later makes recording fail at the first completed window rather than at
startup. Neither shows up in a normal test run, so the decisions that keep the service working are
pinned here.

These tests do not start, stop or install anything. They read the shipped files, and they use
``systemd-analyze`` and ``bash`` only when those exist, so the suite stays offline and dependency
free like the rest of it.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "deploy" / "flybrain.service"
INSTALLER = ROOT / "deploy" / "install-service.sh"
SERVICE_DOC = ROOT / "docs" / "service.md"


def unit_text() -> str:
    return UNIT.read_text(encoding="utf-8")


def directives() -> list[str]:
    """Unit lines that systemd actually reads, with comments and blank lines removed.

    Needed because the file's comments discuss directives it deliberately omits (``ProtectHome=``
    and friends), so a naive substring check cannot tell a decision from an accident.
    """
    return [
        line.strip()
        for line in unit_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ";"))
    ]


def rendered_unit() -> str:
    """The unit after the installer's path substitution, pointed at this checkout."""
    return unit_text().replace("%h/FlyBrain", str(ROOT))


def rendered_unit_for_verify() -> str:
    """The rendered unit with `ExecStart` aimed at an interpreter that exists.

    `systemd-analyze verify` treats a non-existent `ExecStart` as an error, so on a fresh clone or
    in CI — anywhere without `.venv` — a *syntax* check would fail for an environmental reason and
    say nothing about the unit. The shipped path is asserted in its own test, where its absence is
    a skip rather than a failure.
    """
    text = rendered_unit()
    shipped = str(ROOT / ".venv" / "bin" / "python")
    if not Path(shipped).is_file():
        text = text.replace(shipped, sys.executable)
    return text


class TestUnitFile:
    def test_it_runs_the_same_entry_point_the_readme_documents(self) -> None:
        assert "ExecStart=%h/FlyBrain/.venv/bin/python -m flybrain.server" in unit_text()

    def test_it_sets_a_working_directory(self) -> None:
        """`recorder.DEFAULT_ROOT` is the *relative* path `data/recordings`.

        Without `WorkingDirectory`, a service quietly records into `$HOME` instead of the
        repository. `.env` loading does not depend on the cwd, so nothing else notices.
        """
        assert "WorkingDirectory=%h/FlyBrain" in unit_text()

    def test_it_is_enabled_for_the_user_manager(self) -> None:
        assert "WantedBy=default.target" in directives()

    def test_it_restarts_after_a_crash(self) -> None:
        assert "Restart=on-failure" in directives()

    def test_it_does_not_make_home_read_only(self) -> None:
        """The readout, positions and recordings all live inside the checkout under `$HOME`."""
        joined = "\n".join(directives())
        assert "ProtectHome=" not in joined
        assert "ProtectSystem=strict" not in joined

    def test_the_shipped_path_is_the_documented_clone_location(self) -> None:
        """The installer rewrites this; a hand-copy expects the README's `~/FlyBrain`."""
        assert unit_text().count("%h/FlyBrain") == 3  # Documentation, WorkingDirectory, ExecStart

    def test_no_environment_file_so_there_is_only_one_loader(self) -> None:
        """`.env` is read by `flybrain/env.py`; a second parser would be a second set of rules."""
        assert not any(line.startswith("EnvironmentFile=") for line in directives())

    def test_the_documentation_link_names_a_file_that_exists(self) -> None:
        assert SERVICE_DOC.is_file()
        assert "Documentation=file://%h/FlyBrain/docs/service.md" in unit_text()


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze not installed")
class TestUnitParses:
    def test_the_rendered_unit_passes_verify(self, tmp_path: Path) -> None:
        """`systemd-analyze verify` is the only real parser available here; use it.

        The `--user` form is used because this is a user unit: a system-scope verify would not
        check the same search paths.
        """
        path = tmp_path / "flybrain.service"
        path.write_text(rendered_unit_for_verify(), encoding="utf-8")
        result = subprocess.run(
            ["systemd-analyze", "--user", "verify", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        # verify exits 0 with no output when the unit is clean; a warning is not a failure but
        # should be visible if it ever starts complaining.
        assert result.returncode == 0, result.stderr or result.stdout

    def test_the_rendered_execstart_python_exists_in_this_checkout(self) -> None:
        if not (ROOT / ".venv" / "bin" / "python").exists():
            pytest.skip("no .venv in this checkout")
        exec_line = next(
            line for line in rendered_unit().splitlines() if line.startswith("ExecStart=")
        )
        interpreter = exec_line.split("=", 1)[1].split(" -m ", 1)[0]
        assert Path(interpreter).is_file()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
class TestInstaller:
    def test_it_is_executable(self) -> None:
        assert INSTALLER.stat().st_mode & 0o111

    def test_it_parses(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(INSTALLER)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_it_refuses_to_install_without_a_venv(self, tmp_path: Path) -> None:
        """`ExecStart` points at `.venv/bin/python`; installing without one is a crash loop.

        Run from a copy with a `.service` template but no `.venv` beside it. A stub `systemctl`
        that always succeeds stands in for the user manager, so the test reaches the venv guard
        deterministically instead of depending on whether the machine running the suite has a
        session bus.
        """
        fake = tmp_path / "repo" / "deploy"
        fake.mkdir(parents=True)
        shutil.copy(INSTALLER, fake / "install-service.sh")
        shutil.copy(UNIT, fake / "flybrain.service")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "systemctl"
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

        env = dict(os.environ)
        env.update(HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / "config"))
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

        result = subprocess.run(
            ["bash", str(fake / "install-service.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode != 0
        assert "uv sync" in result.stderr


class TestDocumentationIsWiredUp:
    def test_the_service_guide_is_indexed(self) -> None:
        assert "docs/service.md" in (ROOT / "README.md").read_text(encoding="utf-8")
        assert "docs/service.md" in (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        assert "service.md" in (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    def test_the_guide_documents_the_flag_a_service_needs(self) -> None:
        """`FLYBRAIN_ALWAYS_ON` is the difference between a service that decides and one that idles."""
        assert "FLYBRAIN_ALWAYS_ON" in SERVICE_DOC.read_text(encoding="utf-8")

    def test_the_guide_is_substantial(self) -> None:
        assert SERVICE_DOC.stat().st_size > 2000
