"""Model-free production-orchestration evaluation for A27 mutation transition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forge.lexical_index import RepositoryLexicalIndex
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession, RepositoryOrchestrationError
from forge.tools import (
    PermissionDecision,
    RuleBasedPolicy,
    create_assist_repository_policy,
    create_assist_repository_registry,
)

MUTATION_TRANSITION_V1 = "mutation-transition-v1"
MUTATION_TRANSITION_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class MutationTransitionTaskResult:
    task_id: str
    completed: bool
    entries: int
    proposals: int
    post_ready_discovery_attempts: int
    targeted_rereads: int
    successful_mutations: int


@dataclass(frozen=True, slots=True)
class MutationTransitionEvaluationResult:
    tasks: tuple[MutationTransitionTaskResult, ...]
    tasks_passed: int
    tasks_total: int


def run_mutation_transition_v1(root: Path) -> MutationTransitionEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(_run(task, root / task.lower()) for task in _TASK_IDS)
    return MutationTransitionEvaluationResult(
        tasks, sum(task.completed for task in tasks), len(tasks)
    )


_TASK_IDS = ("M01", "M02", "M03", "M04", "M05", "M06")


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def _edit(path: Path, old: str, new: str) -> str:
    return json.dumps(
        {"type": "structured_edit", "path": path.name, "old_text": old, "new_text": new}
    )


def _run(task: str, workspace: Path) -> MutationTransitionTaskResult:
    workspace.mkdir()
    source = workspace / "clock.c"
    source.write_text("int clock_limit(void) { return 10; }\n")
    lexical = RepositoryLexicalIndex(workspace, cache_root=workspace / ".cache")
    registry = create_assist_repository_registry(lexical_index=lexical)
    policy = create_assist_repository_policy()
    callback = None
    edit = _edit(source, "return 10", "return 11")
    if task == "M02":
        scripted = (
            _call("read", "repository.read_range", _range("clock.c", 1, 1)),
            _call("wander", "repository.lexical_search", {"query": "clock"}),
            edit,
            _final("Changed."),
        )
    elif task == "M03":
        scripted = (
            _call("read", "repository.read_range", _range("clock.c", 1, 1)),
            _final("I would change it."),
            edit,
            _final("Changed."),
        )
    elif task == "M04":
        source.write_text(
            "".join(
                "int clock_limit(void) { return 10; }\n"
                if line == 250
                else f"/* filler {line} */\n"
                for line in range(1, 501)
            )
        )
        edit = _edit(source, "return 10", "return 11")
        scripted = (
            _call("read", "repository.read_range", _range("clock.c", 210, 329)),
            _call("reread", "repository.read_range", _range("clock.c", 210, 329)),
            edit,
            _final("Changed."),
        )
    elif task == "M05":
        changed = False

        def mutate(activity) -> None:  # type: ignore[no-untyped-def]
            nonlocal changed
            if activity.invocation_id == "read" and not changed:
                source.write_text("int clock_limit(void) { return 12; }\n")
                changed = True

        callback = mutate
        fresh_edit = _edit(source, "return 12", "return 11")
        scripted = (
            _call("read", "repository.read_range", _range("clock.c", 1, 1)),
            _call("fresh", "repository.read_range", _range("clock.c", 1, 1)),
            fresh_edit,
            _final("Changed."),
        )
    elif task == "M06":
        rules = {
            metadata.name: (
                PermissionDecision.DENY
                if metadata.name.startswith("repository.apply")
                or metadata.name.startswith("repository.write")
                else PermissionDecision.ALLOW
            )
            for metadata in registry.metadata
        }
        policy = RuleBasedPolicy(rules)
        scripted = (
            _call("read", "repository.read_range", _range("clock.c", 1, 1)),
            _final("Write policy prevents the requested change."),
        )
    else:
        scripted = (
            _call("read", "repository.read_range", _range("clock.c", 1, 1)),
            edit,
            _final("Changed."),
        )
    model = MockModel(scripted)
    session = RepositoryChatSession(
        MUTATION_TRANSITION_V1,
        model,
        workspace,
        registry=registry,
        policy=policy,
        lexical_index=lexical,
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
        activity_callback=callback,
    )
    failed = False
    try:
        response = session.execute_task("Correct the clock limit behavior.")
        result = response.coding_task
    except RepositoryOrchestrationError:
        failed = True
        result = session.last_coding_task
    assert result is not None
    metrics = result.transition_metrics
    completed = (
        metrics.entries == (0 if task == "M06" else 2 if task == "M05" else 1)
        and metrics.successful_mutations == (0 if task == "M06" else 1)
        and not failed
    )
    if task == "M02":
        completed = completed and metrics.post_ready_discovery_attempts == 1
    if task == "M03":
        completed = completed and metrics.premature_finals == 1
    if task == "M04":
        completed = completed and metrics.targeted_rereads == 1
    if task == "M05":
        completed = completed and metrics.invalidations == 1
    if task == "M06":
        completed = result.status.value == "failed_before_mutation" and not failed
    return MutationTransitionTaskResult(
        task,
        completed,
        metrics.entries,
        metrics.proposals,
        metrics.post_ready_discovery_attempts,
        metrics.targeted_rereads,
        metrics.successful_mutations,
    )


def _range(path: str, start: int, end: int) -> dict[str, object]:
    return {"path": path, "start_line": start, "end_line": end}
