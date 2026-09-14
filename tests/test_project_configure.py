from __future__ import annotations

import sys
from pathlib import Path

import pytest

from forge.evaluation.project_configure import _run, run_project_configure_v1
from forge.interaction import AutonomyMode, resolve_interaction_policy
from forge.orchestration.coding_task import CodingTaskStatus
from forge.orchestration.verification_attribution import AttributionResult
from forge.project_config import (
    ProjectCommand,
    ProjectCommands,
    ProjectConfigurationError,
    VerificationPlan,
    parse_project_commands,
)
from forge.tools import (
    PermissionDecision,
    ToolCapability,
    ToolEvidence,
    create_repository_registry,
)


def test_configure_parses_only_trusted_argument_arrays() -> None:
    parsed = parse_project_commands(
        {
            "project": {
                "commands": {
                    "configure": {"argv": ["cmake", "-S", ".", "-B", "build"]},
                    "build": {"argv": ["cmake", "--build", "build"]},
                    "test": {"argv": ["ctest", "--test-dir", "build"]},
                },
                "verification": {
                    "id": "configure-build-test",
                    "steps": ["project.configure", "project.build", "project.test"],
                },
            }
        }
    )
    assert parsed.configure == ProjectCommand(("cmake", "-S", ".", "-B", "build"), 120)
    assert parsed.verification_plan.steps == (
        "project.configure",
        "project.build",
        "project.test",
    )


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"argv": []},
        {"argv": "cmake"},
        {"argv": ["cmake"], "timeout_seconds": 0},
        {"argv": ["cmake"], "timeout_seconds": 4000},
        {"argv": ["cmake"], "unexpected": True},
        {"argv": ["cmake"], "cwd": "../"},
    ],
)
def test_invalid_configure_configuration_rejected(raw: object) -> None:
    with pytest.raises(ProjectConfigurationError):
        parse_project_commands({"project": {"commands": {"configure": raw}}})


def test_unconfigured_or_unknown_plan_step_rejected() -> None:
    with pytest.raises(ProjectConfigurationError):
        ProjectCommands(
            verification_plan=VerificationPlan(
                "x", ("project.configure", "project.build")
            )
        )
    with pytest.raises(ProjectConfigurationError):
        VerificationPlan("x", ("project.configure",))
    with pytest.raises(ProjectConfigurationError):
        VerificationPlan("x", ("project.install",))


def test_configure_capability_is_distinct_and_read_mode_cannot_expose_it() -> None:
    commands = ProjectCommands(configure=ProjectCommand((sys.executable, "-V"), 5))
    read = resolve_interaction_policy(AutonomyMode.READ, "trusted-exec")
    assert "project.configure" not in {
        item.name for item in create_repository_registry(read, commands).metadata
    }
    for name, expected in (
        ("safe", PermissionDecision.DENY),
        ("confirm", PermissionDecision.ASK),
        ("trusted-exec", PermissionDecision.ALLOW),
    ):
        policy = resolve_interaction_policy(AutonomyMode.ASSIST, name)
        assert policy.decision_for(ToolCapability.CONFIGURE) is expected
    confirm = resolve_interaction_policy(AutonomyMode.ASSIST, "confirm")
    tool = next(
        item
        for item in create_repository_registry(confirm, commands).metadata
        if item.name == "project.configure"
    )
    assert tool.capability is ToolCapability.CONFIGURE
    assert tool.evidence is ToolEvidence.CONFIGURE_RESULT


def test_project_configure_v1_full_orchestration(tmp_path: Path) -> None:
    tasks = {
        task.task_id: task for task in run_project_configure_v1(tmp_path / "suite")
    }
    assert len(tasks) == 8
    c01 = tasks["C01"]
    assert c01.command_order == ("configure", "build", "test")
    assert c01.coding.status is CodingTaskStatus.COMPLETED_VERIFIED
    assert c01.coding.verification_plan_runs[-1].completed_steps == 3
    assert c01.coding.verification_gate_metrics.verification_tools == 3
    assert c01.model_calls == 3
    assert not c01.successful_configure_log_exposed

    c02 = tasks["C02"]
    assert c02.command_order == ("configure",)
    assert c02.coding.status is CodingTaskStatus.MUTATED_VERIFICATION_FAILED
    assert not c02.coding.repair_eligible
    assert c02.coding.verification_plan_runs[-1].failed_step == "project.configure"

    c03 = tasks["C03"]
    assert c03.command_order == ()
    assert c03.coding.status is CodingTaskStatus.COMPLETED_UNVERIFIED
    assert c03.coding.verification_plan_runs[-1].outcome == "policy_blocked"

    c04 = tasks["C04"]
    assert c04.command_order == ()
    assert c04.approvals.count("project.configure") == 1
    assert c04.coding.verification_plan_runs[-1].outcome == "approval_rejected"

    c05 = tasks["C05"]
    assert c05.command_order == ("configure", "build")
    assert c05.coding.verification_plan_runs[-1].failed_step == "project.build"

    c06 = tasks["C06"]
    assert c06.command_order == ("configure", "build", "test")
    assert c06.generated_is_metadata

    c07 = tasks["C07"]
    assert c07.command_order == (
        "configure",
        "build",
        "test",
        "configure",
        "build",
        "test",
    )
    assert c07.coding.status is CodingTaskStatus.COMPLETED_REPAIRED_VERIFIED
    assert c07.coding.mutation_count == 2

    c08 = tasks["C08"]
    assert c08.command_order == (
        "configure",
        "build",
        "test",
        "configure",
        "build",
        "test",
    )
    assert (
        c08.coding.verification_attribution.result
        is AttributionResult.PREEXISTING_OR_UNRELATED
    )
    assert not c08.coding.repair_eligible


def test_configure_timeout_and_launch_failure_stop_plan(tmp_path: Path) -> None:
    timeout = _run("C09", tmp_path / "timeout")
    launch = _run("C10", tmp_path / "launch")
    assert timeout.command_order == ("configure",)
    assert timeout.coding.configure.outcome == "timeout"
    assert not timeout.coding.repair_eligible
    assert launch.command_order == ()
    assert launch.coding.configure.outcome == "process_start_failed"
    assert not launch.coding.repair_eligible
