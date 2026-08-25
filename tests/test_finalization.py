from pathlib import Path

from forge.evaluation.finalization import run_finalization_v1


def test_finalization_v1_runs_six_full_orchestration_tasks(tmp_path: Path) -> None:
    result = run_finalization_v1(tmp_path / "finalization")
    assert result.tasks_passed == result.tasks_total == 6
    assert result.finalization_entries == 5
    assert result.post_coverage_tool_attempts == 1
    assert result.post_coverage_tools_executed == 0
    assert result.protocol_corrections == 1
    assert result.required_goals_represented >= 6
