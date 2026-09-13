from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from forge.evaluation.verification_plan import _run, run_verification_plan_v1
from forge.orchestration.coding_task import CodingTaskStatus
from forge.orchestration.verification_attribution import (
    AttributionResult,
    VerificationPlanBaseline,
    attribute_plan_failure,
    baseline_evidence,
)
from forge.project_config import (
    ProjectCommand,
    ProjectCommands,
    ProjectConfigurationError,
    VerificationPlan,
    parse_project_commands,
)
from forge.tools import PreparedProjectCommand


def test_plan_configuration_and_immutability() -> None:
    commands = parse_project_commands(
        {
            "project": {
                "commands": {
                    "build": {"argv": ["cmake", "--build", "build"]},
                    "test": {"argv": ["ctest", "--test-dir", "build"]},
                },
                "verification": {
                    "id": "build-test",
                    "steps": ["project.build", "project.test"],
                },
            }
        }
    )
    assert commands.verification_plan == VerificationPlan(
        "build-test", ("project.build", "project.test")
    )
    with pytest.raises(FrozenInstanceError):
        commands.verification_plan.plan_id = "changed"  # type: ignore[misc,union-attr]


@pytest.mark.parametrize(
    "plan_id,steps",
    [
        ("", ("project.test",)),
        ("bad id", ("project.test",)),
        ("x", ()),
        ("x", ("project.build", "project.build")),
        ("x", ("project.shell",)),
        ("x", ("project.test",) * 5),
    ],
)
def test_invalid_plan_rejected(plan_id: str, steps: tuple[str, ...]) -> None:
    with pytest.raises(ProjectConfigurationError):
        VerificationPlan(plan_id, steps)


def test_unconfigured_plan_step_rejected() -> None:
    with pytest.raises(ProjectConfigurationError, match="not configured"):
        ProjectCommands(
            test=ProjectCommand(("ctest",), 5),
            verification_plan=VerificationPlan("x", ("project.build",)),
        )


def test_valid_explicit_single_step_plan() -> None:
    commands = ProjectCommands(
        test=ProjectCommand(("ctest",), 5),
        verification_plan=VerificationPlan("test-only", ("project.test",)),
    )
    assert commands.verification_plan.steps == ("project.test",)


def test_plan_attribution_requires_same_failed_step_and_prior_pass(
    tmp_path: Path,
) -> None:
    prepared_build = PreparedProjectCommand("build", ("cmake",), tmp_path, 5)
    prepared_test = PreparedProjectCommand("test", ("ctest",), tmp_path, 5)
    build_failure = baseline_evidence(
        prepared_build,
        "failure",
        {"outcome": "nonzero_exit", "exit_code": 1, "stderr": "error: build failed"},
        0,
    )
    baseline = VerificationPlanBaseline(
        "build-test", ("project.build", "project.test"), (build_failure,)
    )
    result = attribute_plan_failure(
        baseline,
        "build-test",
        ("project.build", "project.test"),
        1,
        prepared_test,
        "failure",
        {"outcome": "nonzero_exit", "exit_code": 1, "stderr": "error: tests failed"},
        0,
    )
    assert result.result is AttributionResult.UNATTRIBUTED
    mismatched = attribute_plan_failure(
        baseline,
        "other-plan",
        ("project.build", "project.test"),
        0,
        prepared_build,
        "failure",
        {"outcome": "nonzero_exit", "exit_code": 1, "stderr": "error: build failed"},
        0,
    )
    assert mismatched.result is AttributionResult.COMMAND_MISMATCH


def test_verification_plan_v1_full_orchestration(tmp_path: Path) -> None:
    tasks = {task.task_id: task for task in run_verification_plan_v1(tmp_path / "plan")}
    assert len(tasks) == 8
    p01 = tasks["P01"]
    assert p01.command_order == ("build", "test")
    assert p01.coding.status is CodingTaskStatus.COMPLETED_VERIFIED
    assert p01.coding.verification_plan_runs[-1].outcome == "pass"
    assert p01.coding.verification_plan_runs[-1].completed_steps == 2
    assert p01.model_calls == 3
    assert not p01.successful_log_exposed

    p02 = tasks["P02"]
    assert p02.command_order == ("build",)
    assert p02.coding.status is CodingTaskStatus.MUTATED_VERIFICATION_FAILED
    assert p02.coding.verification_plan_runs[-1].failed_step == "project.build"

    p03 = tasks["P03"]
    assert p03.command_order == ("build", "test")
    assert p03.coding.status is CodingTaskStatus.MUTATED_VERIFICATION_FAILED
    assert p03.coding.verification_plan_runs[-1].failed_step == "project.test"
    assert p03.coding.verification_plan_runs[-1].completed_steps == 1

    p04 = tasks["P04"]
    assert p04.command_order == ()
    assert p04.coding.status is CodingTaskStatus.COMPLETED_UNVERIFIED
    assert p04.coding.verification_plan_runs[-1].outcome == "policy_blocked"

    p05 = tasks["P05"]
    assert p05.command_order == ("build",)
    assert p05.approvals.count("project.test") == 1
    assert p05.coding.status is CodingTaskStatus.COMPLETED_UNVERIFIED
    assert p05.coding.verification_plan_runs[-1].outcome == "approval_rejected"

    p06 = tasks["P06"]
    assert p06.command_order == ("build", "test")
    assert p06.coding.status is CodingTaskStatus.COMPLETED_VERIFIED
    assert p06.coding.verification_gate_metrics.verification_tools == 2

    p07 = tasks["P07"]
    assert p07.command_order == ("build", "test", "build", "test")
    assert p07.coding.status is CodingTaskStatus.COMPLETED_REPAIRED_VERIFIED
    assert tuple(run.outcome for run in p07.coding.verification_plan_runs) == (
        "step_failed",
        "pass",
    )
    assert p07.coding.mutation_count == 2

    p08 = tasks["P08"]
    assert p08.command_order == ("build", "test", "build", "test")
    assert p08.coding.verification_plan_baseline is not None
    assert (
        p08.coding.verification_attribution.result
        is AttributionResult.PREEXISTING_OR_UNRELATED
    )
    assert (
        p08.coding.status
        is CodingTaskStatus.MUTATED_VERIFICATION_BLOCKED_BY_BASELINE_FAILURE
    )


def test_plan_timeout_and_process_start_failure_stop_dependents(tmp_path: Path) -> None:
    timeout = _run("P09", tmp_path / "timeout")
    launch = _run("P10", tmp_path / "launch")
    assert timeout.command_order == ("build",)
    assert timeout.coding.build.outcome == "timeout"
    assert timeout.coding.verification_plan_runs[-1].failed_step == "project.build"
    assert launch.command_order == ()
    assert launch.coding.build.outcome == "process_start_failed"
    assert launch.coding.verification_plan_runs[-1].failed_step == "project.build"
