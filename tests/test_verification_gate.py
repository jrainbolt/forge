from pathlib import Path

from forge.evaluation import VERIFICATION_GATE_V1, run_verification_gate_v1
from forge.orchestration import CodingTaskState, CodingTaskStatus


def test_verification_gate_v1_runs_all_eight_production_cases(tmp_path: Path) -> None:
    result = run_verification_gate_v1(tmp_path / VERIFICATION_GATE_V1)

    assert result.tasks_passed == result.tasks_total == 8
    tasks = {task.task_id: task for task in result.tasks}
    assert tasks["V01"].status == CodingTaskStatus.COMPLETED_VERIFIED.value
    assert tasks["V01"].ready_entries == tasks["V01"].verification_tools == 1
    assert tasks["V01"].model_calls == 3
    assert tasks["V02"].result == "passed"
    assert tasks["V03"].status == CodingTaskStatus.COMPLETED_UNVERIFIED.value
    assert tasks["V03"].verification_tools == 0
    assert tasks["V04"].ready_entries == tasks["V04"].verification_tools == 0
    assert tasks["V05"].status == CodingTaskStatus.MUTATED_VERIFICATION_FAILED.value
    assert tasks["V06"].status == CodingTaskStatus.COMPLETED_REPAIRED_VERIFIED.value
    assert tasks["V06"].ready_entries == tasks["V06"].verification_tools == 2
    assert tasks["V07"].result == "skipped"
    assert tasks["V08"].status == CodingTaskStatus.COMPLETED_VERIFIED.value


def test_verified_status_requires_current_successful_verification() -> None:
    state = CodingTaskState(0)
    state.mutation_succeeded(
        "repository.apply_patch",
        {"path": "value.py", "old_sha256": "a", "new_sha256": "b"},
        1,
    )

    result = state.finish("A model cannot assert verification.")

    assert result.status is CodingTaskStatus.COMPLETED_UNVERIFIED
    assert result.verification_gate_metrics.verification_tools == 0
