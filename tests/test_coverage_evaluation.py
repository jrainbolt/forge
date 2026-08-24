from pathlib import Path

from forge.evaluation import run_coverage_v1


def test_coverage_v1_runs_six_deterministic_tasks(tmp_path: Path) -> None:
    result = run_coverage_v1(tmp_path / "coverage")
    assert result.tasks_passed == result.tasks_total == 6
    assert result.required_goals == result.covered_goals
    assert result.premature_finals >= 4
    assert result.source_reads >= 10
