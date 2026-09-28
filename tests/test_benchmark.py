"""Tests for the engine benchmark harness and the numbers it is supposed to keep honest.

`tools/benchmark.py` cannot run here — it needs the connectome and a GPU, and it takes tens of
seconds — so these tests check the parts that rot silently instead: that the CLI still parses
without a GPU, that the scenarios it offers are the ones the documentation's table names, and that
the one constant the dashboard derives its power estimate from still agrees with the figure the
documentation quotes. That last check is not ceremony: `STEP_COST_S` sat at `2.34` for a while
after the step time became `0.53 ms`, and the only symptom was a slightly wrong wattage on a panel.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tools" / "benchmark.py"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BENCHMARK), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


class TestCli:
    def test_the_script_exists(self) -> None:
        assert BENCHMARK.is_file()

    def test_help_parses_without_a_gpu_or_the_connectome(self) -> None:
        result = run("--help")
        assert result.returncode == 0, result.stderr
        assert "docs/engine.md" in result.stdout

    @pytest.mark.parametrize("scenario", ["quiet", "column", "busy"])
    def test_every_documented_scenario_is_selectable(self, scenario: str) -> None:
        assert scenario in run("--help").stdout

    @pytest.mark.parametrize("engine", ["dense", "active"])
    def test_both_engines_are_selectable(self, engine: str) -> None:
        assert engine in run("--help").stdout

    def test_zero_steps_is_rejected_rather_than_dividing_by_zero(self) -> None:
        result = run("--steps", "0")
        assert result.returncode == 2
        assert "positive" in result.stderr


class TestTheNumbersStayHonest:
    def test_the_engine_chapter_points_at_the_harness(self) -> None:
        engine_doc = (ROOT / "docs" / "engine.md").read_text(encoding="utf-8")
        assert "tools/benchmark.py" in engine_doc

    def test_agents_md_lists_the_harness(self) -> None:
        assert "tools/benchmark.py" in (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    def test_the_pacing_step_cost_matches_the_documented_figure(self) -> None:
        from flybrain.pacing import STEP_COST_S

        engine_doc = (ROOT / "docs" / "engine.md").read_text(encoding="utf-8")
        # docs/engine.md states the duty cycle as `1.60 s / interval_s`; the dashboard reads
        # STEP_COST_S. If one changes without the other, this fails rather than mis-reporting watts.
        assert f"{STEP_COST_S:.2f} s / interval_s" in engine_doc
