from forge.evaluation import SEMANTIC_V1, SEMANTIC_V1_TASKS, load_suite


def test_semantic_v1_is_separate_deterministic_suite() -> None:
    tasks = load_suite(SEMANTIC_V1)
    assert tasks == SEMANTIC_V1_TASKS
    assert tuple(task.task_id for task in tasks) == ("S01", "S02", "S03", "S04", "S05")
    assert tuple(task.required_files[0] for task in tasks) == (
        "src/forge/interaction.py",
        "src/forge/context_planner.py",
        "src/forge/orchestration/coding_task.py",
        "src/forge/repository_index.py",
        "src/forge/tools/project.py",
    )
