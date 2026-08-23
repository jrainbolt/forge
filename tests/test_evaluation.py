from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.evaluation import (
    CODING_V1_TASKS,
    EvaluationRunner,
    EvaluationTask,
    RequiredFact,
    TaskCategory,
    ToolRecord,
    fixture_workspace,
    load_suite,
    render_terminal_report,
    run_to_dict,
    score_task,
    write_json_report,
)
from forge.models import MockModel


def call(call_id: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": call_id, "tool": tool, "arguments": arguments}
    )


def final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def sample_task(*, max_tools: int = 3) -> EvaluationTask:
    return EvaluationTask(
        "T01",
        TaskCategory.LOCALIZATION,
        "Where is Widget defined?",
        ("src/widget.py",),
        (RequiredFact(("class widget",)),),
        ("src/widget.py",),
        ("Widget",),
        max_tools,
    )


def test_suite_loading_is_stable_and_unknown_suite_fails() -> None:
    assert load_suite("coding-v1") is CODING_V1_TASKS
    assert [task.task_id for task in CODING_V1_TASKS] == [
        "E01",
        "E02",
        "E03",
        "E04",
        "E05",
        "E06",
        "E07",
        "E08",
    ]
    with pytest.raises(ValueError, match="unknown evaluation suite"):
        load_suite("missing")


def test_each_scoring_dimension_has_passing_and_failing_behavior() -> None:
    task = sample_task(max_tools=2)
    good_tools = (
        ToolRecord("repository.search_files", "success", "discovery"),
        ToolRecord(
            "repository.read_file", "success", "source_content", "src/widget.py"
        ),
    )
    passing = score_task(
        task,
        "Class Widget is defined in src/widget.py.",
        ("src/widget.py",),
        good_tools,
        completed=True,
    )
    assert passing.total == passing.maximum

    failing = score_task(
        task,
        "A different thing exists elsewhere.",
        (),
        (*good_tools, ToolRecord("git.status", "success", "git_working_state")),
        completed=False,
    )
    assert failing.correctness.earned == 0
    assert failing.grounding.earned == 0
    assert failing.localization.earned == 0
    assert failing.efficiency.earned == 0
    assert failing.completion.earned == 0


def test_correct_answer_without_read_loses_grounding_credit() -> None:
    task = sample_task()
    scores = score_task(
        task,
        "Class Widget is defined in src/widget.py.",
        (),
        (ToolRecord("repository.search_files", "success", "discovery"),),
        completed=True,
    )
    assert scores.correctness.earned == scores.correctness.maximum
    assert scores.localization.earned == scores.localization.maximum
    assert scores.grounding.earned == 0


def test_runner_uses_real_orchestration_and_records_grounded_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "widget.py").write_text("class Widget:\n    pass\n")
    model = MockModel(
        (
            call("search", "repository.search_files", {"query": "Widget"}),
            call("read", "repository.read_file", {"path": "src/widget.py"}),
            final("Class Widget is defined in src/widget.py."),
        )
    )
    ticks = iter((1.0, 2.0, 3.0, 4.0))
    run = EvaluationRunner("fixture", model, tmp_path, clock=lambda: next(ticks)).run(
        "test", (sample_task(),)
    )
    result = run.task_results[0]
    assert result.success
    assert result.files_inspected == ("src/widget.py",)
    assert [tool.name for tool in result.tools] == [
        "repository.search_files",
        "repository.read_file",
    ]
    assert result.orchestration_steps == 3
    assert result.scores.total == result.scores.maximum
    assert len(model.requests) == 3
    assert not model.closed


def test_runner_continues_after_failure_and_clears_task_history(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "widget.py").write_text("class Widget:\n    pass\n")
    model = MockModel(
        (
            "not json",
            "still not json",
            call("search", "repository.search_files", {"query": "Widget"}),
            call("read", "repository.read_file", {"path": "src/widget.py"}),
            final("Class Widget is defined in src/widget.py."),
        )
    )
    run = EvaluationRunner("fixture", model, tmp_path).run(
        "test", (sample_task(), sample_task())
    )
    assert not run.task_results[0].success
    assert run.task_results[1].success
    second_initial = model.requests[2].messages
    assert all("not json" not in message.content for message in second_initial)
    assert run.completed == 1
    assert run.failed == 1


def test_json_and_terminal_reports_are_versioned_and_bounded(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "widget.py").write_text("class Widget:\n    pass\n")
    model = MockModel(
        (
            call("search", "repository.search_files", {"query": "Widget"}),
            call("read", "repository.read_file", {"path": "src/widget.py"}),
            final("Class Widget is defined in src/widget.py."),
        )
    )
    run = EvaluationRunner("fixture", model, tmp_path).run("test", (sample_task(),))
    payload = run_to_dict(run)
    assert payload["schema_version"] == 1
    assert payload["suite_version"] == 1
    assert "Class Widget" not in render_terminal_report(run)
    assert "Class Widget" in render_terminal_report(run, verbose=True)
    output = tmp_path / "reports" / "result.json"
    write_json_report(run, output)
    assert json.loads(output.read_text()) == payload
    assert "class Widget:\\n    pass" not in output.read_text()


def test_fixture_ground_truth_is_consistent() -> None:
    workspace = fixture_workspace()
    assert workspace.is_dir()
    for task in CODING_V1_TASKS:
        for path in (*task.required_files, *task.expected_answer_files):
            assert (workspace / path).is_file(), (task.task_id, path)
    retry = (workspace / "src/tinyqueue/retry.py").read_text()
    assert "task.attempts <= self.max_attempts" in retry
    assert "# BUG" not in retry
    assert (
        "test_retry_stops_at_limit" in (workspace / "tests/test_retry.py").read_text()
    )
