"""A36 trusted baseline comparison and production repair boundaries."""

import json
from pathlib import Path

from forge.evaluation.verification_attribution import (
    VERIFICATION_ATTRIBUTION_V1,
    run_verification_attribution_v1,
)
from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession
from forge.orchestration.verification_attribution import (
    AttributionResult,
    attribute_failure,
    baseline_evidence,
    failure_fingerprint,
)
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    PermissionDecision,
    PreparedProjectCommand,
    RuleBasedPolicy,
    create_assist_repository_registry,
)


def _failure(line: str, *, timeout: bool = False) -> dict[str, object]:
    return {
        "outcome": "timeout" if timeout else "nonzero_exit",
        "exit_code": None if timeout else 1,
        "timed_out": timeout,
        "duration_seconds": 0.1,
        "stdout": "",
        "stderr": line,
        "stdout_truncated": False,
        "stderr_truncated": False,
    }


def test_verification_attribution_v1_runs_eight_real_subprocess_cases(
    tmp_path: Path,
) -> None:
    result = run_verification_attribution_v1(tmp_path / VERIFICATION_ATTRIBUTION_V1)

    assert result.tasks_passed == result.tasks_total == 8
    tasks = {task.task_id: task for task in result.tasks}
    assert tasks["A01"].attribution == AttributionResult.MUTATION_ASSOCIATED.value
    assert tasks["A02"].attribution == AttributionResult.PREEXISTING_OR_UNRELATED.value
    assert tasks["A03"].attribution == AttributionResult.UNATTRIBUTED.value
    assert tasks["A04"].status == "completed_verified"
    assert tasks["A05"].attribution == AttributionResult.BASELINE_UNAVAILABLE.value
    assert tasks["A06"].attribution == AttributionResult.COMMAND_MISMATCH.value
    assert tasks["A07"].repair_ready is False
    assert tasks["A07"].mutation_count == 1
    assert tasks["A08"].repair_ready is True
    assert (
        tasks["A08"].initial_attribution == AttributionResult.MUTATION_ASSOCIATED.value
    )
    assert tasks["A08"].mutation_count == 2


def test_fingerprint_normalizes_only_workspace_and_ansi(tmp_path: Path) -> None:
    left = tmp_path / "one"
    right = tmp_path / "two"
    first = _failure(f"\x1b[31m{left}/test.py: case_alpha FAILED\x1b[0m\n")
    second = _failure(f"{right}/test.py: case_alpha FAILED\n")

    assert failure_fingerprint("test", first, left) == failure_fingerprint(
        "test", second, right
    )
    assert failure_fingerprint("test", first, left) != failure_fingerprint(
        "test", _failure(f"{right}/test.py: case_beta FAILED\n"), right
    )


def test_fingerprint_separates_timeout_launch_error_and_incomplete_logs(
    tmp_path: Path,
) -> None:
    failed = _failure("case_alpha FAILED\n")
    timeout = _failure("case_alpha FAILED\n", timeout=True)
    launch = {**failed, "outcome": "process_start_failed"}
    truncated = {**failed, "stderr_truncated": True}

    assert failure_fingerprint("test", failed, tmp_path) != failure_fingerprint(
        "test", timeout, tmp_path
    )
    assert failure_fingerprint("test", launch, tmp_path) is None
    assert failure_fingerprint("test", truncated, tmp_path) is None


def test_baseline_is_bound_to_workspace_command_and_generation(tmp_path: Path) -> None:
    workspace = tmp_path / "one"
    other = tmp_path / "two"
    workspace.mkdir()
    other.mkdir()
    command = PreparedProjectCommand("test", ("python", "-c", "pass"), workspace, 5)
    baseline = baseline_evidence(
        command, "success", {"outcome": "success", "duration_seconds": 0.1}, 0
    )
    failure = _failure("case_alpha FAILED\n")

    assert attribute_failure(baseline, command, "failure", failure, 0).result is (
        AttributionResult.MUTATION_ASSOCIATED
    )
    assert attribute_failure(baseline, command, "failure", failure, 1).result is (
        AttributionResult.COMMAND_MISMATCH
    )
    different = PreparedProjectCommand("test", command.argv, other, 5)
    assert attribute_failure(baseline, different, "failure", failure, 0).result is (
        AttributionResult.COMMAND_MISMATCH
    )
    different_timeout = PreparedProjectCommand("test", command.argv, workspace, 6)
    assert (
        attribute_failure(baseline, different_timeout, "failure", failure, 0).result
        is AttributionResult.COMMAND_MISMATCH
    )


def test_matching_preexisting_failure_never_claims_verified(tmp_path: Path) -> None:
    command = PreparedProjectCommand("test", ("python", "-c", "pass"), tmp_path, 5)
    failure = _failure("case_alpha FAILED\n")
    baseline = baseline_evidence(command, "failure", failure, 0)
    comparison = attribute_failure(baseline, command, "failure", failure, 0)

    assert comparison.result is AttributionResult.PREEXISTING_OR_UNRELATED
    assert comparison.fingerprint_equal is True
    assert (
        attribute_failure(
            baseline, command, "success", {"outcome": "success"}, 0
        ).result
        is AttributionResult.NOT_APPLICABLE
    )


def test_denied_baseline_never_executes_or_grants_verification(tmp_path: Path) -> None:
    (tmp_path / "value.py").write_text("VALUE = 1\n")
    marker = tmp_path / "should-not-run"
    command = ProjectCommand(
        ("python", "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"),
        5,
    )
    registry = create_assist_repository_registry(ProjectCommands(test=command))
    rules = {item.name: PermissionDecision.ALLOW for item in registry.metadata}
    rules["repository.apply_patch"] = PermissionDecision.ASK
    rules["repository.write_file"] = PermissionDecision.ASK
    rules["project.test"] = PermissionDecision.DENY
    responses = (
        json.dumps(
            {
                "type": "tool_call",
                "id": "read",
                "tool": "repository.read_file",
                "arguments": {"path": "value.py"},
            }
        ),
        json.dumps(
            {
                "type": "structured_edit",
                "path": "value.py",
                "old_text": "VALUE = 1",
                "new_text": "VALUE = 2",
            }
        ),
        json.dumps({"type": "final", "answer": "Done."}),
    )
    result = (
        RepositoryChatSession(
            "attribution-deny",
            MockModel(responses),
            tmp_path,
            mode=AutonomyMode.ASSIST,
            registry=registry,
            policy=RuleBasedPolicy(rules),
            approval_callback=lambda _invocation, _preview: True,
            require_relevant_source=False,
            verification_baseline=True,
        )
        .execute_task("Change VALUE using current source.")
        .coding_task
    )

    assert result is not None
    assert result.verification_baseline is not None
    assert not result.verification_baseline.executed
    assert result.verification_gate_metrics.executed == 0
    assert result.status.value != "completed_verified"
    assert not marker.exists()
