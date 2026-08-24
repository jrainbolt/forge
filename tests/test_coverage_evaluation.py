from pathlib import Path

from forge.evaluation import run_coverage_v1, run_production_decomposition_v1


def test_coverage_v1_runs_six_deterministic_tasks(tmp_path: Path) -> None:
    result = run_coverage_v1(tmp_path / "coverage")
    assert result.tasks_passed == result.tasks_total == 6
    assert result.required_goals == result.covered_goals
    assert result.premature_finals >= 4
    assert result.source_reads >= 10


def test_production_decomposition_v1(tmp_path: Path) -> None:
    result = run_production_decomposition_v1(tmp_path / "decomposition")
    assert (result.tasks_passed, result.tasks_total) == (4, 4)
    assert result.production_plans_created == 4
    assert result.multi_goal_plans == 4
    assert result.goals_failed == 1
    assert result.premature_finals == 1
    assert result.wrong_goal_reads == 1
    assert result.coverage_complete_tasks == 3
