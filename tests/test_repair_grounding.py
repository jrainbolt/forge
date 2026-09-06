from pathlib import Path

from forge.evaluation import REPAIR_GROUNDING_V1, run_repair_grounding_v1


def test_repair_grounding_v1_runs_eight_production_cases(tmp_path: Path) -> None:
    result = run_repair_grounding_v1(tmp_path / REPAIR_GROUNDING_V1)

    assert result.tasks_passed == result.tasks_total == 8
    tasks = {task.task_id: task for task in result.tasks}
    for task_id in {"R01", "R02", "R03", "R04", "R05", "R08"}:
        task = tasks[task_id]
        assert task.status == "completed_repaired_verified"
        assert task.diagnosis == task.source_refreshes == task.ready_entries == 1
        assert task.repair_proposals == task.repair_mutations == 1
        assert task.reverification_result == "passed"
    assert tasks["R06"].status == "repair_verification_failed"
    assert tasks["R06"].repair_mutations == 1
    assert tasks["R06"].reverification_result == "failed"
    assert tasks["R07"].status == "mutated_verification_failed"
    assert tasks["R07"].diagnosis == tasks["R07"].ready_entries == 0
