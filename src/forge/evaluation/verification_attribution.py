"""Eight deterministic A36 production attribution scenarios."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import CodingTaskStatus, RepositoryChatSession
from forge.orchestration.verification_attribution import (
    AttributionResult,
    attribute_failure,
    baseline_evidence,
)
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    PermissionDecision,
    PreparedProjectCommand,
    RuleBasedPolicy,
    create_assist_repository_registry,
)
from forge.tools.project import execute_prepared_project_command
from forge.tools.tool import ToolError

VERIFICATION_ATTRIBUTION_V1 = "verification-attribution-v1"
VERIFICATION_ATTRIBUTION_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class AttributionTaskResult:
    task_id: str
    completed: bool
    status: str
    attribution: str
    initial_attribution: str
    baseline_executed: bool
    repair_ready: bool
    mutation_count: int
    tool_executions: int


@dataclass(frozen=True, slots=True)
class AttributionEvaluationResult:
    tasks: tuple[AttributionTaskResult, ...]
    tasks_passed: int
    tasks_total: int


def run_verification_attribution_v1(root: Path) -> AttributionEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(
        _run(f"A{number:02d}", root / f"a{number:02d}") for number in range(1, 9)
    )
    return AttributionEvaluationResult(
        tasks, sum(task.completed for task in tasks), len(tasks)
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


def _script(task: str) -> str:
    if task == "A01":
        failure = "changed"
        condition = "value != 1"
    elif task in {"A02", "A07"}:
        failure = "preexisting"
        condition = "True"
    elif task == "A03":
        return (
            "from pathlib import Path; import sys; "
            "value=int(Path('value.py').read_text().split('=')[1]); "
            "sys.stderr.write('baseline failed\\n' if value == 1 "
            "else 'changed failed\\n'); "
            "sys.exit(1)"
        )
    elif task == "A04":
        failure = "none"
        condition = "False"
    else:
        failure = "changed"
        condition = "value == 2"
    return (
        "from pathlib import Path; import sys; "
        "value=int(Path('value.py').read_text().split('=')[1]); "
        f"bad={condition}; "
        f"sys.stderr.write('{failure} failed\\n' if bad else ''); "
        "sys.exit(1 if bad else 0)"
    )


def _run(task: str, workspace: Path) -> AttributionTaskResult:
    workspace.mkdir()
    (workspace / "value.py").write_text("VALUE = 1\n")
    if task == "A06":
        return _command_mismatch(workspace)
    argv = (
        (str(workspace / "missing-test-executable"),)
        if task == "A05"
        else (sys.executable, "-c", _script(task))
    )
    registry = create_assist_repository_registry(
        ProjectCommands(test=ProjectCommand(argv, 5))
    )
    rules = {metadata.name: PermissionDecision.ALLOW for metadata in registry.metadata}
    rules["repository.apply_patch"] = PermissionDecision.ASK
    rules["repository.write_file"] = PermissionDecision.ASK
    if task == "A02":
        rules["project.test"] = PermissionDecision.ASK
    policy = RuleBasedPolicy(rules)
    responses = [
        _call("read-primary", "repository.read_file", {"path": "value.py"}),
        _edit("VALUE = 1", "VALUE = 2"),
    ]
    if task == "A08":
        responses.extend(
            (
                _call("read-repair", "repository.read_file", {"path": "value.py"}),
                _edit("VALUE = 2", "VALUE = 3"),
            )
        )
    responses.append(json.dumps({"type": "final", "answer": "Done."}))
    session = RepositoryChatSession(
        VERIFICATION_ATTRIBUTION_V1,
        MockModel(tuple(responses)),
        workspace,
        mode=AutonomyMode.REPAIR if task in {"A07", "A08"} else AutonomyMode.ASSIST,
        registry=registry,
        policy=policy,
        approval_callback=lambda _invocation, _preview: True,
        require_relevant_source=False,
        verification_baseline=True,
    )
    result = session.execute_task("Change VALUE using current source.").coding_task
    assert result is not None
    expected_attribution = {
        "A01": AttributionResult.MUTATION_ASSOCIATED,
        "A02": AttributionResult.PREEXISTING_OR_UNRELATED,
        "A03": AttributionResult.UNATTRIBUTED,
        "A04": AttributionResult.NOT_APPLICABLE,
        "A05": AttributionResult.BASELINE_UNAVAILABLE,
        "A07": AttributionResult.PREEXISTING_OR_UNRELATED,
        "A08": AttributionResult.NOT_APPLICABLE,
    }[task]
    expected_status = {
        "A01": CodingTaskStatus.MUTATED_VERIFICATION_FAILED,
        "A02": CodingTaskStatus.MUTATED_VERIFICATION_BLOCKED_BY_BASELINE_FAILURE,
        "A03": CodingTaskStatus.MUTATED_VERIFICATION_FAILED,
        "A04": CodingTaskStatus.COMPLETED_VERIFIED,
        "A05": CodingTaskStatus.MUTATED_VERIFICATION_FAILED,
        "A07": CodingTaskStatus.MUTATED_VERIFICATION_BLOCKED_BY_BASELINE_FAILURE,
        "A08": CodingTaskStatus.COMPLETED_REPAIRED_VERIFIED,
    }[task]
    baseline = result.verification_baseline
    completed = (
        result.verification_attribution.result is expected_attribution
        and result.status is expected_status
        and baseline is not None
        and (baseline.executed is (task != "A05"))
        and result.mutation_count == (2 if task == "A08" else 1)
        and result.tool_sequence.count("project.test") == (3 if task == "A08" else 2)
    )
    if task == "A07":
        completed = (
            completed and not result.repair_eligible and result.repair_evidence is None
        )
    if task == "A08":
        completed = (
            completed
            and result.repair_grounding_metrics.ready_entries == 1
            and result.attribution_attempts[0].result
            is AttributionResult.MUTATION_ASSOCIATED
        )
    return AttributionTaskResult(
        task,
        completed,
        result.status.value,
        result.verification_attribution.result.value,
        result.attribution_attempts[0].result.value,
        baseline.executed if baseline else False,
        result.repair_grounding_metrics.ready_entries > 0,
        result.mutation_count,
        len(result.tool_sequence),
    )


def _command_mismatch(workspace: Path) -> AttributionTaskResult:
    first = PreparedProjectCommand(
        "test", (sys.executable, "-c", "raise SystemExit(0)"), workspace, 5
    )
    second = PreparedProjectCommand(
        "test", (sys.executable, "-c", "raise SystemExit(1)"), workspace, 5
    )
    first_result = execute_prepared_project_command(first)
    try:
        execute_prepared_project_command(second)
    except ToolError as error:
        second_result = error.output
    else:
        raise AssertionError("second command should fail")
    assert second_result is not None
    attribution = attribute_failure(
        baseline_evidence(first, "success", first_result, 0),
        second,
        "failure",
        second_result,
        0,
    )
    return AttributionTaskResult(
        "A06",
        attribution.result is AttributionResult.COMMAND_MISMATCH,
        "not_verified",
        attribution.result.value,
        attribution.result.value,
        True,
        False,
        0,
        2,
    )
