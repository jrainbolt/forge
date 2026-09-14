"""Deterministic A38 production configure/build/test orchestration suite."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.evidence_coverage import (
    EvidenceCoverageState,
    EvidenceGoal,
    TaskEvidencePlan,
)
from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession
from forge.orchestration.coding_task import CodingTaskResult
from forge.project_config import ProjectCommand, ProjectCommands, VerificationPlan
from forge.retrieval import SourceKind, classify_source
from forge.tools import (
    PermissionDecision,
    RuleBasedPolicy,
    create_assist_repository_registry,
)

PROJECT_CONFIGURE_V1 = "project-configure-v1"
_PLAN = VerificationPlan(
    "configure-build-test",
    ("project.configure", "project.build", "project.test"),
)


@dataclass(frozen=True, slots=True)
class ProjectConfigureTaskResult:
    task_id: str
    coding: CodingTaskResult
    command_order: tuple[str, ...]
    approvals: tuple[str, ...]
    model_calls: int
    generated_is_metadata: bool
    successful_configure_log_exposed: bool


def run_project_configure_v1(root: Path) -> tuple[ProjectConfigureTaskResult, ...]:
    root.mkdir(parents=True, exist_ok=True)
    return tuple(
        _run(f"C{number:02d}", root / f"c{number:02d}") for number in range(1, 9)
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


def _run(task_id: str, workspace: Path) -> ProjectConfigureTaskResult:
    workspace.mkdir()
    (workspace / "value.py").write_text("VALUE = 1\n")
    configure_code = (
        "from pathlib import Path; import sys; "
        "p=Path('steps.log'); "
        "p.write_text(p.read_text()+'configure\\n' if p.exists() else 'configure\\n'); "
        "Path('build').mkdir(exist_ok=True); "
        "Path('build/config.h').write_text('CONFIGURED=1\\n'); "
        "Path('build/generated.c').write_text('int generated;\\n'); "
        "print('A38_CONFIGURE_SUCCESS_LOG'); "
        + ("import time; time.sleep(2); " if task_id == "C09" else "")
        + f"sys.exit({1 if task_id == 'C02' else 0})"
    )
    build_code = (
        "from pathlib import Path; import sys; "
        "p=Path('steps.log'); "
        "p.write_text(p.read_text()+'build\\n' if p.exists() else 'build\\n'); "
        "assert Path('build/config.h').exists(); "
        "Path('build/output').write_text('built'); "
        + f"sys.exit({1 if task_id == 'C05' else 0})"
    )
    test_code = (
        "from pathlib import Path; import sys; "
        "p=Path('steps.log'); "
        "p.write_text(p.read_text()+'test\\n' if p.exists() else 'test\\n'); "
        "assert Path('build/output').exists(); "
        + (
            "print('AssertionError: preexisting'); sys.exit(1)"
            if task_id == "C08"
            else "sys.exit(0 if 'VALUE = 3' in Path('value.py').read_text() else 1)"
            if task_id == "C07"
            else "sys.exit(0)"
        )
    )
    commands = ProjectCommands(
        build=ProjectCommand((sys.executable, "-c", build_code), 5),
        test=ProjectCommand((sys.executable, "-c", test_code), 5),
        verification_plan=_PLAN,
        configure=ProjectCommand(
            ("forge-a38-nonexistent-executable",)
            if task_id == "C10"
            else (sys.executable, "-c", configure_code),
            0.1 if task_id == "C09" else 5,
        ),
    )
    registry = create_assist_repository_registry(commands)
    rules = {item.name: PermissionDecision.ALLOW for item in registry.metadata}
    rules["repository.apply_patch"] = PermissionDecision.ASK
    rules["repository.write_file"] = PermissionDecision.ASK
    if task_id == "C03":
        rules["project.configure"] = PermissionDecision.DENY
    if task_id == "C04":
        rules["project.configure"] = PermissionDecision.ASK
    approvals: list[str] = []

    def approve(invocation, _preview):  # type: ignore[no-untyped-def]
        approvals.append(invocation.tool_name)
        return not (task_id == "C04" and invocation.tool_name == "project.configure")

    responses = [
        _call("read", "repository.read_file", {"path": "value.py"}),
        _edit("VALUE = 1", "VALUE = 2"),
    ]
    if task_id == "C07":
        responses.extend(
            (
                _call("repair-read", "repository.read_file", {"path": "value.py"}),
                _edit("VALUE = 2", "VALUE = 3"),
            )
        )
    responses.append(json.dumps({"type": "final", "answer": "Done."}))
    model = MockModel(tuple(responses))
    response = RepositoryChatSession(
        PROJECT_CONFIGURE_V1,
        model,
        workspace,
        mode=AutonomyMode.REPAIR
        if task_id in {"C02", "C07", "C08"}
        else AutonomyMode.ASSIST,
        registry=registry,
        policy=RuleBasedPolicy(rules),
        approval_callback=approve,
        require_relevant_source=False,
        max_tool_executions=5 if task_id == "C01" else 18,
        verification_plan=_PLAN,
        verification_baseline=task_id == "C08",
    ).execute_task("Change VALUE using current source and verify.")
    assert response.coding_task is not None
    log = workspace / "steps.log"
    generated = workspace / "build/generated.c"
    source_coverage = EvidenceCoverageState(
        TaskEvidencePlan((EvidenceGoal("impl", "implementation"),))
    )
    generated_rejected = not source_coverage.register_source(
        "impl", "build/generated.c", 0, "generated-observation"
    )
    return ProjectConfigureTaskResult(
        task_id,
        response.coding_task,
        tuple(log.read_text().splitlines()) if log.exists() else (),
        tuple(approvals),
        len(model.requests),
        generated.exists()
        and classify_source("build/generated.c") is SourceKind.GENERATED_METADATA
        and generated_rejected
        and not source_coverage.complete,
        any(
            "A38_CONFIGURE_SUCCESS_LOG" in message.content
            for request in model.requests
            for message in request.messages
        ),
    )
