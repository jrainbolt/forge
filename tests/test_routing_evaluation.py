from pathlib import Path

from forge.evaluation import (
    ROUTING_V1,
    ROUTING_V1_TASKS,
    load_routing_suite,
    run_routing_v1,
)


def test_routing_v1_is_fixed_six_task_suite() -> None:
    assert load_routing_suite(ROUTING_V1) is ROUTING_V1_TASKS
    assert [task.task_id for task in ROUTING_V1_TASKS] == [
        "R01",
        "R02",
        "R03",
        "R04",
        "R05",
        "R06",
    ]


def test_routing_v1_runs_production_orchestration(tmp_path: Path) -> None:
    result = run_routing_v1(tmp_path / "routing")
    assert result.tasks_passed == result.tasks_total == 6
    assert result.broad_discoveries_attempted > result.broad_discoveries_executed
    assert result.suppressed_broad_discoveries >= 3
    assert result.candidate_inspections >= 6
    assert result.candidate_failures >= 2
    assert result.candidate_set_repetitions >= 1
    assert result.source_files_acquired >= 6
    assert all(task.retrieval_state_final == "source_acquired" for task in result.tasks)
