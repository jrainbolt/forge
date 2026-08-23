from __future__ import annotations

import json

from forge.evaluation import (
    CONTEXT_V1,
    CONTEXT_V1_TASKS,
    EvaluationRunner,
    fixture_workspace,
    load_suite,
    run_to_dict,
)
from forge.models import MockModel


def call(call_id: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": call_id, "tool": tool, "arguments": arguments}
    )


def final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def test_context_v1_has_five_stable_isolated_tasks() -> None:
    assert load_suite(CONTEXT_V1) is CONTEXT_V1_TASKS
    assert [task.task_id for task in CONTEXT_V1_TASKS] == [
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
    ]
    assert len({task.prompt for task in CONTEXT_V1_TASKS}) == 5


def test_context_runner_records_structural_targeted_context_metrics() -> None:
    task = CONTEXT_V1_TASKS[1]
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
            final(
                "RetryPolicy.should_retry in src/tinyqueue/retry.py compares attempts "
                "with max_attempts using less than or equal (<=)."
            ),
        )
    )
    run = EvaluationRunner(
        "fixture", model, fixture_workspace().resolve(strict=True)
    ).run(CONTEXT_V1, (task,))
    result = run.task_results[0]
    assert result.success
    assert result.scores.total == result.scores.maximum
    assert result.files_inspected == ("src/tinyqueue/retry.py",)
    assert [tool.name for tool in result.tools] == [
        "repository.find_symbol",
        "repository.read_range",
    ]
    assert result.tools[1].returned_bytes is not None
    assert result.tools[1].returned_lines is not None
    payload = run_to_dict(run)
    tools = payload["tasks"][0]["tools"]  # type: ignore[index]
    assert tools[1]["returned_bytes"] > 0
    assert tools[1]["returned_lines"] > 0


def test_symbol_tool_transcript_remains_ephemeral() -> None:
    task = CONTEXT_V1_TASKS[0]
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
            final(
                "RetryPolicy.should_retry is in src/tinyqueue/retry.py as "
                "RetryPolicy.should_retry."
            ),
        )
    )
    runner = EvaluationRunner(
        "fixture", model, fixture_workspace().resolve(strict=True)
    )
    run = runner.run(CONTEXT_V1, (task,))
    assert run.completed == 1
    assert all(
        "tool_result" not in message.content
        for request in model.requests
        for message in request.messages[:-1]
        if message.role.value == "assistant"
    )
