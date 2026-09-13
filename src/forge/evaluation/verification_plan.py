"""Deterministic A37 full-session verification-plan evaluation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession
from forge.orchestration.coding_task import CodingTaskResult
from forge.project_config import ProjectCommand, ProjectCommands, VerificationPlan
from forge.tools import (
    PermissionDecision,
    RuleBasedPolicy,
    create_assist_repository_registry,
)

VERIFICATION_PLAN_V1 = "verification-plan-v1"
_PLAN = VerificationPlan("build-test", ("project.build", "project.test"))


@dataclass(frozen=True, slots=True)
class VerificationPlanTaskResult:
    task_id: str
    coding: CodingTaskResult
    command_order: tuple[str, ...]
    model_calls: int
    approvals: tuple[str, ...]
    successful_log_exposed: bool


def run_verification_plan_v1(root: Path) -> tuple[VerificationPlanTaskResult, ...]:
    """Run P01-P08 with real A10 subprocesses and no repository-global writes."""
    root.mkdir(parents=True, exist_ok=True)
    return tuple(
        _run(f"P{number:02d}", root / f"p{number:02d}") for number in range(1, 9)
    )


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


def _run(task_id: str, workspace: Path) -> VerificationPlanTaskResult:
    workspace.mkdir()
    (workspace / "value.py").write_text("VALUE = 1\n")
    build_failure = task_id == "P02"
    test_failure = task_id in {"P03", "P08"}
    repair = task_id == "P07"
    build_code = (
        "from pathlib import Path; import sys; "
        "p=Path('steps.log'); "
        "p.write_text(p.read_text() + 'build\\n' if p.exists() else 'build\\n'); "
        "print('A37_BUILD_SUCCESS_LOG'); "
        + ("import time; time.sleep(2); " if task_id == "P09" else "")
        + f"sys.exit({1 if build_failure else 0})"
    )
    test_code = (
        "from pathlib import Path; import sys; "
        "p=Path('steps.log'); "
        "p.write_text(p.read_text() + 'test\\n' if p.exists() else 'test\\n'); "
        + (
            "print('AssertionError: preexisting'); sys.exit(1)"
            if test_failure
            else "sys.exit(0 if 'VALUE = 3' in Path('value.py').read_text() else 1)"
            if repair
            else "sys.exit(0)"
        )
    )
    commands = ProjectCommands(
        ProjectCommand(
            ("forge-a37-nonexistent-executable",)
            if task_id == "P10"
            else (sys.executable, "-c", build_code),
            0.1 if task_id == "P09" else 5,
        ),
        ProjectCommand((sys.executable, "-c", test_code), 5),
        _PLAN,
    )
    registry = create_assist_repository_registry(commands)
    rules = {item.name: PermissionDecision.ALLOW for item in registry.metadata}
    rules["repository.apply_patch"] = PermissionDecision.ASK
    rules["repository.write_file"] = PermissionDecision.ASK
    if task_id == "P04":
        rules["project.build"] = PermissionDecision.DENY
    if task_id == "P05":
        rules["project.test"] = PermissionDecision.ASK
    responses = [
        _call("read", "repository.read_file", {"path": "value.py"}),
        _edit("VALUE = 1", "VALUE = 2"),
    ]
    if repair:
        responses.extend(
            (
                _call("repair-read", "repository.read_file", {"path": "value.py"}),
                _edit("VALUE = 2", "VALUE = 3"),
            )
        )
    responses.append(json.dumps({"type": "final", "answer": "Done."}))
    model = MockModel(tuple(responses))
    approvals: list[str] = []

    def approve(invocation, _preview):  # type: ignore[no-untyped-def]
        approvals.append(invocation.tool_name)
        return not (task_id == "P05" and invocation.tool_name == "project.test")

    response = RepositoryChatSession(
        VERIFICATION_PLAN_V1,
        model,
        workspace,
        mode=AutonomyMode.REPAIR if repair else AutonomyMode.ASSIST,
        registry=registry,
        policy=RuleBasedPolicy(rules),
        approval_callback=approve,
        require_relevant_source=False,
        max_tool_executions=4 if task_id == "P06" else 16,
        verification_plan=_PLAN,
        verification_baseline=task_id == "P08",
    ).execute_task("Change VALUE using current source and verify.")
    assert response.coding_task is not None
    log = workspace / "steps.log"
    return VerificationPlanTaskResult(
        task_id,
        response.coding_task,
        tuple(log.read_text().splitlines()) if log.exists() else (),
        len(model.requests),
        tuple(approvals),
        any(
            "A37_BUILD_SUCCESS_LOG" in message.content
            for request in model.requests
            for message in request.messages
        ),
    )
