from __future__ import annotations

import json
import sys
from pathlib import Path
from shutil import copytree

import pytest

from forge.models import MockModel
from forge.orchestration import CodingTaskStatus, RepositoryChatSession
from forge.project_config import (
    ProjectCommand,
    ProjectCommands,
    ProjectConfigurationError,
    parse_project_commands,
)
from forge.structural_completeness import (
    StructuralCompletenessDecision,
    StructuralCompletenessEvaluation,
    StructuralCompletenessMode,
    evaluate_structural_completeness,
)
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)

FIXTURE = Path(__file__).parent / "fixtures/eval_repo"
RETRY_PATH = "src/tinyqueue/retry.py"


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _flow(*tail: str) -> tuple[str, ...]:
    return (
        _call("search", "repository.search_files", {"query": "should_retry"}),
        _call("read", "repository.read_file", {"path": RETRY_PATH}),
        json.dumps(
            {
                "type": "structured_edit",
                "path": RETRY_PATH,
                "old_text": "task.attempts <= self.max_attempts",
                "new_text": "task.attempts < self.max_attempts",
            }
        ),
        *tail,
    )


def _final() -> str:
    return json.dumps({"type": "final", "answer": "done"})


def _session(
    workspace: Path,
    responses: tuple[str, ...],
    mode: StructuralCompletenessMode,
    evaluator,  # type: ignore[no-untyped-def]
    *,
    approve=lambda *_args: True,
    marker: Path | None = None,
) -> RepositoryChatSession:
    commands = ProjectCommands(
        test=(
            ProjectCommand(
                (
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ),
                5,
            )
            if marker is not None
            else None
        )
    )
    return RepositoryChatSession(
        "fixture",
        MockModel(responses),
        workspace,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        approval_callback=approve,
        require_relevant_source=False,
        structural_completeness_mode=mode,
        structural_completeness_evaluator=evaluator,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return Path(copytree(FIXTURE, tmp_path / "workspace"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, StructuralCompletenessMode.SHADOW),
        ("off", StructuralCompletenessMode.OFF),
        ("shadow", StructuralCompletenessMode.SHADOW),
        ("enforce", StructuralCompletenessMode.ENFORCE),
    ],
)
def test_structural_mode_configuration(
    raw: str | None, expected: StructuralCompletenessMode
) -> None:
    project = {} if raw is None else {"structural_completeness_mode": raw}
    assert (
        parse_project_commands({"project": project}).structural_completeness_mode
        is expected
    )
    with pytest.raises(ProjectConfigurationError):
        parse_project_commands({"project": {"structural_completeness_mode": "invalid"}})


def test_enforce_pass_continues_to_verification(workspace: Path) -> None:
    marker = workspace / "verified"
    evaluator = lambda _path: StructuralCompletenessEvaluation(  # noqa: E731
        StructuralCompletenessDecision.PASS
    )
    response = _session(
        workspace,
        _flow(_call("test", "project.test", {}), _final()),
        StructuralCompletenessMode.ENFORCE,
        evaluator,
        marker=marker,
    ).execute_task("fix and verify")
    assert response.coding_task.status is CodingTaskStatus.COMPLETED_VERIFIED
    assert response.coding_task.structural_completeness.decision == "pass"
    assert marker.exists()


def test_enforce_proven_failure_rejects_before_verification(workspace: Path) -> None:
    marker = workspace / "verified"
    approvals: list[str] = []

    def approve(invocation, _preview):  # type: ignore[no-untyped-def]
        approvals.append(invocation.tool_name)
        return True

    evaluator = lambda _path: StructuralCompletenessEvaluation(  # noqa: E731
        StructuralCompletenessDecision.FAIL, ("required-call",)
    )
    response = _session(
        workspace,
        _flow(_final(), _final()),
        StructuralCompletenessMode.ENFORCE,
        evaluator,
        approve=approve,
        marker=marker,
    ).execute_task("fix")
    result = response.coding_task
    assert result.status is CodingTaskStatus.STRUCTURAL_COMPLETENESS_REJECTED
    assert result.mutation_count == 1
    assert result.structural_completeness.failed_check_ids == ("required-call",)
    assert result.test.status == "not_run"
    assert not marker.exists()
    assert approvals == ["repository.apply_patch"]
    assert not result.repair_attempted and not result.repair_eligible


def test_shadow_failure_records_and_continues(workspace: Path) -> None:
    evaluator = lambda _path: StructuralCompletenessEvaluation(  # noqa: E731
        StructuralCompletenessDecision.FAIL, ("required-call",)
    )
    response = _session(
        workspace,
        _flow(_final()),
        StructuralCompletenessMode.SHADOW,
        evaluator,
    ).execute_task("fix")
    assert response.coding_task.status is CodingTaskStatus.COMPLETED_UNVERIFIED
    assert not response.coding_task.structural_completeness.enforced_rejection


@pytest.mark.parametrize(
    "decision",
    [
        StructuralCompletenessDecision.PARTIAL,
        StructuralCompletenessDecision.NOT_APPLICABLE,
    ],
)
def test_enforce_uncertainty_continues(
    workspace: Path, decision: StructuralCompletenessDecision
) -> None:
    evaluator = lambda _path: StructuralCompletenessEvaluation(decision)  # noqa: E731
    response = _session(
        workspace, _flow(_final()), StructuralCompletenessMode.ENFORCE, evaluator
    ).execute_task("fix")
    assert response.coding_task.status is CodingTaskStatus.COMPLETED_UNVERIFIED


def test_checker_exception_fails_open(workspace: Path) -> None:
    def evaluator(_path):  # type: ignore[no-untyped-def]
        raise RuntimeError("checker unavailable")

    response = _session(
        workspace, _flow(_final()), StructuralCompletenessMode.ENFORCE, evaluator
    ).execute_task("fix")
    record = response.coding_task.structural_completeness
    assert response.coding_task.status is CodingTaskStatus.COMPLETED_UNVERIFIED
    assert record.decision == "partial"
    assert record.operational_failures == ("CHECKER_INTERNAL_ERROR",)


def test_compiler_timeout_is_uncertainty_in_enforce(workspace: Path) -> None:
    record = evaluate_structural_completeness(
        StructuralCompletenessMode.ENFORCE,
        workspace,
        lambda _path: StructuralCompletenessEvaluation(
            StructuralCompletenessDecision.PARTIAL,
            operational_failures=("COMPILER_TIMEOUT",),
        ),
    )
    assert record.decision == "partial"
    assert not record.enforced_rejection


def test_structural_record_is_source_free_and_package_excluded(workspace: Path) -> None:
    record = evaluate_structural_completeness(
        StructuralCompletenessMode.ENFORCE,
        workspace,
        lambda _path: StructuralCompletenessEvaluation(
            StructuralCompletenessDecision.FAIL, ("check-identity",)
        ),
    )
    payload = json.dumps(
        {
            "mode": record.mode,
            "decision": record.decision,
            "failed_check_ids": record.failed_check_ids,
            "operational_failures": record.operational_failures,
        }
    )
    assert "source" not in payload and "replacement" not in payload
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "structural_completeness_enforcement_v1" not in configuration


def test_off_does_not_execute_checker(workspace: Path) -> None:
    calls = 0

    def evaluator(_path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return StructuralCompletenessEvaluation(StructuralCompletenessDecision.FAIL)

    response = _session(
        workspace, _flow(_final()), StructuralCompletenessMode.OFF, evaluator
    ).execute_task("fix")
    assert calls == 0
    assert not response.coding_task.structural_completeness.executed


def test_deny_precedes_structural_check(workspace: Path) -> None:
    calls = 0

    def evaluator(_path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return StructuralCompletenessEvaluation(StructuralCompletenessDecision.PASS)

    response = _session(
        workspace,
        _flow(_final()),
        StructuralCompletenessMode.ENFORCE,
        evaluator,
        approve=lambda *_args: False,
    ).execute_task("fix")
    assert response.coding_task.status is CodingTaskStatus.REJECTED
    assert response.coding_task.mutation_count == 0
    assert calls == 0


def test_mode_rollback_is_configuration_only() -> None:
    enforce = parse_project_commands(
        {"project": {"structural_completeness_mode": "enforce"}}
    )
    shadow = parse_project_commands(
        {"project": {"structural_completeness_mode": "shadow"}}
    )
    off = parse_project_commands({"project": {"structural_completeness_mode": "off"}})
    assert enforce.structural_completeness_mode is StructuralCompletenessMode.ENFORCE
    assert shadow.structural_completeness_mode is StructuralCompletenessMode.SHADOW
    assert off.structural_completeness_mode is StructuralCompletenessMode.OFF
