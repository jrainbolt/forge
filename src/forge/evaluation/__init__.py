"""Deterministic coding evaluation harness."""

from forge.evaluation.reporting import (
    render_terminal_report,
    run_to_dict,
    write_json_report,
)
from forge.evaluation.runner import EvaluationRunner
from forge.evaluation.scoring import score_task
from forge.evaluation.tasks import (
    CODING_V1,
    CODING_V1_TASKS,
    fixture_workspace,
    load_suite,
)
from forge.evaluation.types import (
    SCHEMA_VERSION,
    SUITE_VERSION,
    DimensionScore,
    EvaluationRun,
    EvaluationTask,
    FailureReason,
    RequiredFact,
    TaskCategory,
    TaskResult,
    TaskScores,
    ToolRecord,
)

__all__ = [
    "CODING_V1",
    "CODING_V1_TASKS",
    "SCHEMA_VERSION",
    "SUITE_VERSION",
    "DimensionScore",
    "EvaluationRun",
    "EvaluationRunner",
    "EvaluationTask",
    "FailureReason",
    "RequiredFact",
    "TaskCategory",
    "TaskResult",
    "TaskScores",
    "ToolRecord",
    "fixture_workspace",
    "load_suite",
    "render_terminal_report",
    "run_to_dict",
    "score_task",
    "write_json_report",
]
