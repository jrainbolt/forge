"""Deterministic production-orchestration evaluation for A28 structured edits."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession, RepositoryOrchestrationError
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)

STRUCTURED_MUTATION_V1 = "structured-mutation-v1"
STRUCTURED_MUTATION_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class StructuredMutationTaskResult:
    task_id: str
    completed: bool
    attempts: int
    valid: int
    corrections: int
    previews: int
    mutations: int


@dataclass(frozen=True, slots=True)
class StructuredMutationEvaluationResult:
    tasks: tuple[StructuredMutationTaskResult, ...]
    tasks_passed: int
    tasks_total: int


def run_structured_mutation_v1(root: Path) -> StructuredMutationEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(_run(task, root / task.lower()) for task in _TASK_IDS)
    return StructuredMutationEvaluationResult(
        tasks, sum(task.completed for task in tasks), len(tasks)
    )


_TASK_IDS = tuple(f"P{number:02d}" for number in range(1, 9))


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _edit(old: str, new: str, path: str = "main.c") -> str:
    return json.dumps(
        {"type": "structured_edit", "path": path, "old_text": old, "new_text": new}
    )


def _final(text: str) -> str:
    return json.dumps({"type": "final", "answer": text})


def _patch(path: Path, old: str, new: str) -> dict[str, object]:
    return {
        "path": path.name,
        "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "edits": [{"old": old, "new": new}],
    }


class _ExternalEditModel(MockModel):
    def __init__(self, responses: tuple[str, ...], target: Path) -> None:
        super().__init__(responses)
        self._target = target

    def generate(self, request):  # type: ignore[no-untyped-def]
        response = super().generate(request)
        if len(self.requests) == 2:
            self._target.write_text(self._target.read_text() + "/* external */\n")
        return response


def _run(task: str, workspace: Path) -> StructuredMutationTaskResult:
    workspace.mkdir()
    source = workspace / "main.c"
    source.write_text("if (value > limit) {\n    return 1;\n}\n")
    preview_diffs: list[str] = []

    def approve(_invocation, preview) -> bool:  # type: ignore[no-untyped-def]
        if hasattr(preview, "diff"):
            preview_diffs.append(preview.diff)
        return task != "P07" or len(preview_diffs) > 1

    read = _call("read", "repository.read_file", {"path": "main.c"})
    good = _edit("value > limit", "value >= limit")
    model: MockModel
    mode = AutonomyMode.ASSIST
    commands = ProjectCommands()
    expected_mutations = 1
    expected_failure = False
    if task == "P02":
        scripted = (
            read,
            _edit("value  > limit", "value >= limit"),
            good,
            _final("done"),
        )
    elif task == "P03":
        source.write_text("int same = 1;\nint same = 1;\n")
        scripted = (
            read,
            _edit("int same = 1;", "int same = 2;"),
            _edit("int same = 1;", "int same = 3;"),
        )
        expected_mutations = 0
        expected_failure = True
    elif task == "P04":
        fresh_read = _call("fresh", "repository.read_file", {"path": "main.c"})
        scripted = (read, good, fresh_read, good, _final("done"))
        model = _ExternalEditModel(scripted, source)
    elif task == "P05":
        source.write_text("trusted();\nline2();\noutside();\n")
        read = _call(
            "read",
            "repository.read_range",
            {"path": "main.c", "start_line": 1, "end_line": 1},
        )
        scripted = (
            read,
            _edit("outside();", "changed();"),
            _edit("trusted();", "changed();"),
            _final("done"),
        )
    elif task == "P06":
        scripted = (read, _edit("x\n" * 201, "y"), good, _final("done"))
    elif task == "P07":
        scripted = (read, good, _final("rejected"))
        expected_mutations = 0
    elif task == "P08":
        source.write_text("int value = 1;\n")
        primary = _patch(source, "value = 1", "value = 2")
        commands = ProjectCommands(
            test=ProjectCommand(
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "assert 'value = 3' in Path('main.c').read_text()",
                ),
                5,
            )
        )
        mode = AutonomyMode.REPAIR
        scripted = (
            read,
            _call("primary", "repository.apply_patch", primary),
            _call("test1", "project.test", {}),
            _call("repair-read", "repository.read_file", {"path": "main.c"}),
            _edit("value = 2", "value = 3"),
            _call("test2", "project.test", {}),
            _final("repaired"),
        )
        expected_mutations = 2
    else:
        scripted = (read, good, _final("done"))
    if task != "P04":
        model = MockModel(scripted)
    session = RepositoryChatSession(
        STRUCTURED_MUTATION_V1,
        model,
        workspace,
        mode=mode,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        approval_callback=approve,
        require_relevant_source=False,
    )
    failed = False
    try:
        response = session.ask("Apply the focused correction and verify if configured.")
        result = response.coding_task
    except RepositoryOrchestrationError:
        failed = True
        result = session.last_coding_task
    assert result is not None
    metrics = result.structured_mutation_metrics
    completed = (
        result.mutation_count == expected_mutations
        and failed is expected_failure
        and (metrics.materialized_previews >= 1 or expected_mutations == 0)
    )
    if task == "P02":
        completed = (
            completed and metrics.corrections == metrics.correction_successes == 1
        )
    if task == "P04":
        completed = completed and metrics.stale_proposals == 1
    if task == "P07":
        completed = completed and len(preview_diffs) == 1
    if task == "P08":
        completed = completed and result.test.status == "passed" and metrics.valid == 1
    return StructuredMutationTaskResult(
        task,
        completed,
        metrics.attempts,
        metrics.valid,
        metrics.corrections,
        metrics.materialized_previews,
        result.mutation_count,
    )
