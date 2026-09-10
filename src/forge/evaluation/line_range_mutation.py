"""Deterministic production-orchestration evaluation for A35 line-range edits."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.interaction import AutonomyMode
from forge.models import MockModel, MutationRepresentationPolicy
from forge.orchestration import RepositoryChatSession, RepositoryOrchestrationError
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)

LINE_RANGE_MUTATION_V1 = "line-range-mutation-v1"
LINE_RANGE_MUTATION_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class LineRangeMutationTaskResult:
    task_id: str
    completed: bool
    attempts: int
    valid: int
    target_valid: int
    materialized: int
    corrections: int
    previews: int
    mutations: int
    verification: str


@dataclass(frozen=True, slots=True)
class LineRangeMutationEvaluationResult:
    tasks: tuple[LineRangeMutationTaskResult, ...]
    tasks_passed: int
    tasks_total: int


_TASK_IDS = tuple(f"L{number:02d}" for number in range(1, 9))


def run_line_range_mutation_v1(root: Path) -> LineRangeMutationEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(_run(task, root / task.lower()) for task in _TASK_IDS)
    return LineRangeMutationEvaluationResult(
        tasks, sum(task.completed for task in tasks), len(tasks)
    )


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _edit(start: int, end: int, new_text: str) -> str:
    return json.dumps(
        {
            "type": "line_range_edit",
            "path": "main.c",
            "start_line": start,
            "end_line": end,
            "new_text": new_text,
        }
    )


def _final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


class _ExternalEditModel(MockModel):
    def __init__(self, responses: tuple[str, ...], target: Path) -> None:
        super().__init__(responses)
        self._target = target

    def generate(self, request):  # type: ignore[no-untyped-def]
        response = super().generate(request)
        if len(self.requests) == 2:
            self._target.write_text(self._target.read_text() + "/* external */\n")
        return response


def _run(task: str, workspace: Path) -> LineRangeMutationTaskResult:
    workspace.mkdir()
    source = workspace / "main.c"
    source.write_text("prefix();\nreturn 1;\nsuffix();\n")
    before_prefix = b"prefix();\n"
    before_suffix = b"suffix();\n"
    previews: list[str] = []

    def approve(_invocation, preview) -> bool:  # type: ignore[no-untyped-def]
        if hasattr(preview, "diff"):
            previews.append(preview.diff)
        return task != "L07"

    read = _call("read", "repository.read_file", {"path": "main.c"})
    valid = _edit(2, 2, "return 2;")
    responses: tuple[str, ...] = (read, valid, _final("done"))
    expected_mutations = 1
    expected_failure = False
    expected_text = "return 2;"
    mode = AutonomyMode.ASSIST
    if task == "L02":
        source.write_text("prefix();\nint value = 1;\nreturn value;\nsuffix();\n")
        before_suffix = b"suffix();\n"
        responses = (
            read,
            _edit(2, 3, "int value = 2;\nreturn value;"),
            _final("done"),
        )
        expected_text = "int value = 2;\nreturn value;"
    elif task == "L03":
        responses = (read, _edit(0, 2, "bad"), valid, _final("done"))
    elif task == "L04":
        fresh = _call("fresh", "repository.read_file", {"path": "main.c"})
        responses = (read, valid, fresh, valid, _final("done"))
    elif task == "L05":
        read = _call(
            "read",
            "repository.read_range",
            {"path": "main.c", "start_line": 1, "end_line": 1},
        )
        valid = _edit(1, 1, "changed();")
        responses = (read, _edit(3, 3, "changed();"), valid, _final("done"))
        expected_text = "changed();"
    elif task == "L06":
        responses = (read, _edit(2, 2, "return 1;"), _edit(2, 2, "return 1;"))
        expected_mutations = 0
        expected_failure = True
    elif task == "L07":
        responses = (read, valid, _final("rejected"))
        expected_mutations = 0
    elif task == "L08":
        source.write_text("VALUE = 1\n")
        before_prefix = b""
        before_suffix = b""
        read = _call("read", "repository.read_file", {"path": "main.c"})
        responses = (
            read,
            _edit(1, 1, "VALUE = 2"),
            _call("repair-read", "repository.read_file", {"path": "main.c"}),
            _edit(1, 1, "VALUE = 3"),
            _final("repaired"),
        )
        expected_text = "VALUE = 3"
        expected_mutations = 2
        mode = AutonomyMode.REPAIR
    assertion = (
        "from pathlib import Path; "
        f"assert {expected_text!r} in Path('main.c').read_text()"
    )
    commands = ProjectCommands(
        test=ProjectCommand((sys.executable, "-c", assertion), 5)
    )
    model: MockModel = (
        _ExternalEditModel(responses, source) if task == "L04" else MockModel(responses)
    )
    session = RepositoryChatSession(
        LINE_RANGE_MUTATION_V1,
        model,
        workspace,
        mode=mode,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        approval_callback=approve,
        require_relevant_source=False,
        mutation_representation=MutationRepresentationPolicy.LINE_RANGE,
    )
    failed = False
    try:
        response = session.ask("Apply the focused correction and verify it.")
        result = response.coding_task
    except RepositoryOrchestrationError:
        failed = True
        result = session.last_coding_task
    assert result is not None
    metrics = result.structured_mutation_metrics
    data = source.read_bytes()
    untouched = data.startswith(before_prefix) and data.endswith(before_suffix)
    if task == "L04":
        untouched = (
            data.startswith(before_prefix) and b"suffix();\n/* external */\n" in data
        )
    elif task == "L05":
        untouched = data.endswith(before_suffix)
    completed = (
        result.mutation_count == expected_mutations
        and failed is expected_failure
        and untouched
        and metrics.mutation_representation == "line_range"
    )
    if task == "L01":
        completed = completed and result.test.status == "passed"
    elif task == "L02":
        completed = completed and expected_text.encode() in data
    elif task == "L03":
        completed = completed and metrics.line_range_corrections == 1
    elif task == "L04":
        completed = completed and metrics.stale_proposals == 1
    elif task == "L05":
        completed = completed and metrics.line_range_corrections == 1
    elif task == "L06":
        completed = completed and not previews and metrics.no_op_edit_attempts >= 1
    elif task == "L07":
        completed = completed and len(previews) == 1
    elif task == "L08":
        completed = (
            completed
            and result.test.status == "passed"
            and result.repair_succeeded
            and result.mutation_count == 2
        )
    return LineRangeMutationTaskResult(
        task,
        completed,
        metrics.line_range_attempts,
        metrics.line_range_valid,
        metrics.line_range_target_valid,
        metrics.line_range_materialized,
        metrics.line_range_corrections,
        metrics.line_range_preview_created,
        result.mutation_count,
        result.test.status,
    )
