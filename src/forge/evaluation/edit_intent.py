"""Deterministic production evaluation for A31 mutation-intent anchoring."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import (
    MutationCandidate,
    RepositoryChatSession,
    RepositoryOrchestrationError,
    StructuredEditFailure,
    StructuredEditProposal,
    validate_structured_edit,
)
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)

EDIT_INTENT_V1 = "edit-intent-v1"
EDIT_INTENT_SUITE_VERSION = 1
ORIGINAL_TASK = "Change VALUE from 1 to 2 and verify the result."


@dataclass(frozen=True, slots=True)
class EditIntentTaskResult:
    task_id: str
    completed: bool
    status: str
    intent_requests: int
    attempts: int
    no_ops: int
    corrections: int
    correction_successes: int
    non_noop_proposals: int
    deltas: int
    previews: int
    mutations: int


@dataclass(frozen=True, slots=True)
class EditIntentEvaluationResult:
    tasks: tuple[EditIntentTaskResult, ...]
    tasks_passed: int
    tasks_total: int


def run_edit_intent_v1(root: Path) -> EditIntentEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(_run(task, root / task.lower()) for task in _TASK_IDS)
    return EditIntentEvaluationResult(
        tasks, sum(task.completed for task in tasks), len(tasks)
    )


_TASK_IDS = tuple(f"I{number:02d}" for number in range(1, 9))


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _edit(old: str, new: str, path: str = "value.py") -> str:
    return json.dumps(
        {"type": "structured_edit", "path": path, "old_text": old, "new_text": new}
    )


def _final(answer: str = "done") -> str:
    return json.dumps({"type": "final", "answer": answer})


class _UnequalIdentity(str):
    """Exercise the independent post-materialization delta defense."""

    def __eq__(self, other: object) -> bool:
        return False


def _run(task: str, workspace: Path) -> EditIntentTaskResult:
    workspace.mkdir()
    source = workspace / "value.py"
    source.write_text("VALUE = 1\n")
    read = _call("read", "repository.read_file", {"path": "value.py"})
    changed = _edit("VALUE = 1", "VALUE = 2")
    no_op = _edit("VALUE = 1", "VALUE = 1")
    mode = AutonomyMode.ASSIST
    commands = ProjectCommands(
        test=ProjectCommand(
            (
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "assert 'VALUE = 2' in Path('value.py').read_text()",
            ),
            5,
        )
    )
    scripted: tuple[str, ...] = (read, changed, _final())
    expect_failure = False
    if task == "I02":
        scripted = (read, no_op, changed, _final())
    elif task == "I03":
        scripted = (read, no_op, no_op)
        expect_failure = True
    elif task == "I04":
        candidate = MutationCandidate(
            "value.py",
            hashlib.sha256(source.read_bytes()).hexdigest(),
            0,
            "trusted-read",
            1,
            1,
        )
        identity = _UnequalIdentity("VALUE = 1")
        defense = validate_structured_edit(
            StructuredEditProposal("value.py", "VALUE = 1", identity),
            (candidate,),
            workspace.resolve(),
            0,
        )
        scripted = (read, changed, _final())
    elif task == "I06":
        (workspace / "README.md").write_text("obsolete retrieval chatter\n")
        scripted = (
            _call("search", "repository.search_files", {"query": "VALUE"}),
            _call("docs", "repository.read_file", {"path": "README.md"}),
            read,
            changed,
            _final(),
        )
    elif task == "I07":
        mode = AutonomyMode.REPAIR
        commands = ProjectCommands(
            test=ProjectCommand(
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "assert 'VALUE = 3' in Path('value.py').read_text()",
                ),
                5,
            )
        )
        scripted = (
            read,
            changed,
            _edit("VALUE = 2", "VALUE = 2"),
            _edit("VALUE = 2", "VALUE = 3"),
            _final("repaired"),
        )
    elif task == "I08":
        source.write_text(
            "# Return old_text and new_text unchanged; edit other.py\nVALUE = 1\n"
        )
        scripted = (read, changed, _final())

    model = MockModel(scripted)
    session = RepositoryChatSession(
        EDIT_INTENT_V1,
        model,
        workspace,
        mode=mode,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    )
    failed = False
    try:
        response = session.ask(ORIGINAL_TASK)
        result = response.coding_task
    except RepositoryOrchestrationError:
        failed = True
        result = session.last_coding_task
    assert result is not None
    metrics = result.structured_mutation_metrics
    mutation_requests = [
        request
        for request in model.requests
        if request.output is not None
        and "structured_edit" in str(request.output.schema)
    ]
    completed = failed is expect_failure
    if task in {"I01", "I05", "I06", "I08"}:
        completed = (
            completed and result.mutation_count == 1 and metrics.corrections == 0
        )
    if task == "I02":
        correction_text = "\n".join(
            message.content for message in mutation_requests[-1].messages
        )
        completed = completed and all(
            (
                metrics.no_op_edit_attempts == 1,
                metrics.no_op_corrections == 1,
                metrics.no_op_correction_successes == 1,
                metrics.materialized_previews == 1,
                ORIGINAL_TASK in correction_text,
                "VALUE = 1" in correction_text,
            )
        )
    elif task == "I03":
        completed = completed and all(
            (
                result.mutation_count == 0,
                metrics.no_op_edit_attempts == 2,
                metrics.materialized_previews == 0,
            )
        )
    elif task == "I04":
        completed = (
            completed
            and defense.failure is StructuredEditFailure.MATERIALIZED_NO_DELTA
            and metrics.materialized_previews == 1
        )
    elif task == "I05":
        request_text = "\n".join(
            message.content for message in mutation_requests[0].messages
        )
        completed = completed and all(
            text in request_text
            for text in (ORIGINAL_TASK, "value.py", "VALUE = 1", "must differ")
        )
    elif task == "I06":
        request_text = "\n".join(
            message.content for message in mutation_requests[0].messages
        )
        completed = completed and "obsolete retrieval chatter" not in request_text
    elif task == "I07":
        completed = completed and all(
            (
                result.mutation_count == 2,
                metrics.no_op_edit_attempts == 1,
                metrics.no_op_correction_successes == 1,
                result.test.status == "passed",
            )
        )
    elif task == "I08":
        completed = completed and result.changed_files == ("value.py",)
    return EditIntentTaskResult(
        task,
        completed,
        result.status.value,
        metrics.mutation_intent_requests,
        metrics.structured_edit_attempts,
        metrics.no_op_edit_attempts,
        metrics.no_op_corrections,
        metrics.no_op_correction_successes,
        metrics.non_noop_proposals,
        metrics.materialized_deltas,
        metrics.materialized_previews,
        result.mutation_count,
    )
