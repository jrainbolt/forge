from __future__ import annotations

import json

from forge.evaluation import (
    CONTEXT_BUDGET_V1,
    CONTEXT_BUDGET_V1_TASKS,
    EvaluationRunner,
    fixture_workspace,
    load_suite,
    run_to_dict,
)
from forge.models import MockModel


def call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def test_context_budget_v1_is_separate_and_stable() -> None:
    assert load_suite(CONTEXT_BUDGET_V1) is CONTEXT_BUDGET_V1_TASKS
    assert [task.task_id for task in CONTEXT_BUDGET_V1_TASKS] == [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
    ]


def test_context_budget_report_captures_planner_metrics() -> None:
    task = CONTEXT_BUDGET_V1_TASKS[0]
    model = MockModel(
        (
            call(
                "symbol",
                "repository.find_symbol",
                {"symbol": "RetryPolicy.should_retry"},
            ),
            call(
                "range",
                "repository.read_range",
                {
                    "path": "src/tinyqueue/retry.py",
                    "start_line": 1,
                    "end_line": 20,
                },
            ),
            json.dumps(
                {
                    "type": "final",
                    "answer": "RetryPolicy.should_retry in src/tinyqueue/retry.py "
                    "compares attempts against max_attempts.",
                }
            ),
        )
    )

    run = EvaluationRunner(
        "fixture", model, fixture_workspace().resolve(strict=True)
    ).run(CONTEXT_BUDGET_V1, (task,))
    context = run_to_dict(run)["tasks"][0]["context"]  # type: ignore[index]

    assert context["range_reads"] == 1
    assert context["whole_file_reads"] == 0
    assert context["estimated_context_admitted"] > 0
    assert context["estimated_context_peak"] > 0
