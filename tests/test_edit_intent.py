from __future__ import annotations

from pathlib import Path

from forge.evaluation import EDIT_INTENT_V1, run_edit_intent_v1


def test_edit_intent_v1_runs_eight_production_cases(tmp_path: Path) -> None:
    result = run_edit_intent_v1(tmp_path / EDIT_INTENT_V1)

    assert result.tasks_passed == result.tasks_total == 8
    tasks = {task.task_id: task for task in result.tasks}
    assert tasks["I01"].attempts == 1
    assert tasks["I01"].corrections == 0
    assert tasks["I02"].no_ops == tasks["I02"].corrections == 1
    assert tasks["I02"].correction_successes == 1
    assert tasks["I03"].no_ops == 2
    assert tasks["I03"].previews == tasks["I03"].mutations == 0
    assert tasks["I04"].deltas == 1
    assert tasks["I07"].mutations == 2
