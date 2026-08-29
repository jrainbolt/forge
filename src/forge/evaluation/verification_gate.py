"""Deterministic full-orchestration evaluation for the A29 verification gate."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import CodingTaskStatus, RepositoryChatSession
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    PermissionDecision,
    RuleBasedPolicy,
    create_assist_repository_registry,
)

VERIFICATION_GATE_V1 = "verification-gate-v1"
VERIFICATION_GATE_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class VerificationGateTaskResult:
    task_id: str
    completed: bool
    status: str
    ready_entries: int
    verification_tools: int
    result: str
    model_calls: int


@dataclass(frozen=True, slots=True)
class VerificationGateEvaluationResult:
    tasks: tuple[VerificationGateTaskResult, ...]
    tasks_passed: int
    tasks_total: int


def run_verification_gate_v1(root: Path) -> VerificationGateEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(_run(task, root / task.lower()) for task in _TASK_IDS)
    return VerificationGateEvaluationResult(
        tasks, sum(task.completed for task in tasks), len(tasks)
    )


_TASK_IDS = tuple(f"V{number:02d}" for number in range(1, 9))


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _edit(old: str, new: str) -> str:
    return json.dumps(
        {
            "type": "structured_edit",
            "path": "value.py",
            "old_text": old,
            "new_text": new,
        }
    )


def _final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def _policy(registry, test: PermissionDecision) -> RuleBasedPolicy:  # type: ignore[no-untyped-def]
    rules = {metadata.name: PermissionDecision.ALLOW for metadata in registry.metadata}
    rules["repository.apply_patch"] = PermissionDecision.ASK
    rules["repository.write_file"] = PermissionDecision.ASK
    rules["project.test"] = test
    rules["project.build"] = test
    return RuleBasedPolicy(rules)


def _run(task: str, workspace: Path) -> VerificationGateTaskResult:
    workspace.mkdir()
    source = workspace / "value.py"
    source.write_text("VALUE = 1\n")
    permission = (
        PermissionDecision.ASK
        if task == "V02"
        else PermissionDecision.DENY
        if task == "V03"
        else PermissionDecision.ALLOW
    )
    commands = ProjectCommands()
    if task != "V04":
        assertion = "VALUE = 3" if task == "V06" else "VALUE = 2"
        if task == "V05":
            assertion = "VALUE = 99"
        commands = ProjectCommands(
            test=ProjectCommand(
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    f"assert {assertion!r} in Path('value.py').read_text()",
                ),
                5,
            )
        )
    registry = create_assist_repository_registry(commands)
    responses = [
        _call("read-primary", "repository.read_file", {"path": "value.py"}),
        _edit("VALUE = 1", "VALUE = 2"),
    ]
    if task == "V06":
        responses = [
            _call("read-primary", "repository.read_file", {"path": "value.py"}),
            _call(
                "primary",
                "repository.apply_patch",
                {
                    "path": "value.py",
                    "expected_sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
                    "edits": [{"old": "VALUE = 1", "new": "VALUE = 2"}],
                },
            ),
            _call("read-repair", "repository.read_file", {"path": "value.py"}),
            _edit("VALUE = 2", "VALUE = 3"),
        ]
    responses.append(_final("Mutation complete."))
    model = MockModel(tuple(responses))
    approvals = 0

    def approve(_invocation, _proposal):  # type: ignore[no-untyped-def]
        nonlocal approvals
        approvals += 1
        return True

    response = RepositoryChatSession(
        VERIFICATION_GATE_V1,
        model,
        workspace,
        mode=AutonomyMode.REPAIR if task == "V06" else AutonomyMode.ASSIST,
        registry=registry,
        policy=_policy(registry, permission),
        approval_callback=approve,
        require_relevant_source=False,
        skip_verification=task == "V07",
        max_tool_executions=3 if task == "V08" else 10,
    ).execute_task("Change VALUE using current source and verify when required.")
    result = response.coding_task
    assert result is not None
    gate = result.verification_gate_metrics
    expected_status = {
        "V01": CodingTaskStatus.COMPLETED_VERIFIED,
        "V02": CodingTaskStatus.COMPLETED_VERIFIED,
        "V03": CodingTaskStatus.COMPLETED_UNVERIFIED,
        "V04": CodingTaskStatus.COMPLETED_UNVERIFIED,
        "V05": CodingTaskStatus.MUTATED_VERIFICATION_FAILED,
        "V06": CodingTaskStatus.COMPLETED_REPAIRED_VERIFIED,
        "V07": CodingTaskStatus.COMPLETED_UNVERIFIED,
        "V08": CodingTaskStatus.COMPLETED_VERIFIED,
    }[task]
    expected_tools = 2 if task == "V06" else 0 if task in {"V03", "V04", "V07"} else 1
    completed = (
        result.status is expected_status
        and gate.verification_tools == expected_tools
        and gate.post_mutation_reads_before_verification == 0
    )
    if task == "V01":
        completed = completed and len(model.requests) == 3
    elif task == "V02":
        completed = completed and gate.approval_requested == gate.approved == 1
    elif task == "V03":
        completed = completed and gate.permission == PermissionDecision.DENY.value
    elif task == "V04":
        completed = completed and gate.required is False
    elif task == "V06":
        completed = completed and result.mutation_count == 2
    elif task == "V07":
        completed = completed and gate.skipped
    elif task == "V08":
        completed = completed and result.test.status == "passed"
    return VerificationGateTaskResult(
        task,
        completed,
        result.status.value,
        gate.ready_entries,
        gate.verification_tools,
        gate.result,
        len(model.requests),
    )
