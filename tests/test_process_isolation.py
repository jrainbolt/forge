from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from forge.evaluation.process_isolation import run_process_isolation_v1
from forge.interaction import AutonomyMode, resolve_interaction_policy
from forge.orchestration.verification_attribution import command_identity
from forge.process_isolation import (
    ExecutionIsolationMode,
    ExecutionIsolationPolicy,
    MacOSSandboxExec,
    controlled_environment,
)
from forge.project_config import (
    ProjectCommand,
    ProjectCommands,
    ProjectConfigurationError,
    parse_project_commands,
)
from forge.retrieval import SourceKind, classify_source
from forge.tools import (
    AllowAllPolicy,
    ExecutionContext,
    InvocationApproval,
    PermissionDecision,
    RuleBasedPolicy,
    ToolExecutor,
    ToolInvocation,
    ToolRegistry,
    ToolResultStatus,
    create_assist_repository_registry,
    create_repository_registry,
)
from forge.tools.project import PreparedProjectCommand, ProjectCommandTool


def test_process_isolation_v1(tmp_path: Path) -> None:
    results = run_process_isolation_v1(tmp_path / "suite")
    assert tuple(case.case_id for case in results) == tuple(
        f"I{number:02d}" for number in range(1, 9)
    )
    assert all(case.passed for case in results), results


def test_isolation_configuration_is_trusted_and_backward_compatible() -> None:
    assert (
        parse_project_commands({}).execution_isolation.mode
        is ExecutionIsolationMode.NONE
    )
    assert (
        parse_project_commands({"project": {"commands": {}}}).execution_isolation.mode
        is ExecutionIsolationMode.NONE
    )
    for value in ("none", "controlled_env", "strict"):
        parsed = parse_project_commands(
            {"project": {"execution": {"isolation": value}}}
        )
        assert parsed.execution_isolation.mode.value == value
    for raw in ("strict", {"isolation": "other"}, {"isolation": "none", "env": {}}):
        with pytest.raises(ProjectConfigurationError):
            parse_project_commands({"project": {"execution": raw}})


def test_controlled_environment_allowlist_and_none_compatibility(
    tmp_path: Path,
) -> None:
    with patch.dict(
        os.environ, {"FORGE_TEST_SECRET": "super-secret", "PATH": "/usr/bin:/bin"}
    ):
        hardened = controlled_environment(tmp_path)
        assert "FORGE_TEST_SECRET" not in hardened
        assert hardened["PATH"] == "/usr/bin:/bin"
        assert hardened["HOME"].startswith(str(tmp_path))
        assert hardened["TMPDIR"].startswith(str(tmp_path))
        tool = ProjectCommandTool(
            "test", ProjectCommand((sys.executable, "-c", "pass"), 5)
        )
        prepared = tool.prepare(ExecutionContext(tmp_path))
        assert ("FORGE_TEST_SECRET", "super-secret") in prepared.environment


def test_controlled_path_excludes_relative_and_workspace_entries(
    tmp_path: Path,
) -> None:
    workspace_bin = tmp_path / "bin"
    workspace_bin.mkdir()
    environment = controlled_environment(
        tmp_path,
        {"PATH": f".:{workspace_bin}:/usr/bin:/bin", "FORGE_TEST_SECRET": "fake"},
    )
    assert environment["PATH"] == "/usr/bin:/bin"
    assert "FORGE_TEST_SECRET" not in environment


def test_approval_snapshot_rejects_environment_change(tmp_path: Path) -> None:
    tool = ProjectCommandTool(
        "test",
        ProjectCommand(
            (sys.executable, "-c", "from pathlib import Path; Path('ran').touch()"), 5
        ),
        ExecutionIsolationPolicy(ExecutionIsolationMode.CONTROLLED_ENV),
    )
    executor = ToolExecutor(
        ToolRegistry((tool,)), RuleBasedPolicy({"project.test": PermissionDecision.ASK})
    )
    invocation = ToolInvocation("approval", "project.test", {})
    context = ExecutionContext(tmp_path)
    preview = tool.prepare(context)
    with patch.dict(os.environ, {"PATH": "/usr/bin:/bin:/a39-unmatched"}):
        result = executor.execute(
            invocation,
            replace(context, prepared_project_command=preview),
            approval=InvocationApproval.for_invocation(invocation),
        )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["outcome"] == "prepared_command_changed"
    assert not (tmp_path / "ran").exists()


def test_attribution_identity_includes_isolation(tmp_path: Path) -> None:
    base = PreparedProjectCommand("test", ("true",), tmp_path, 5)
    controlled = PreparedProjectCommand(
        "test",
        ("true",),
        tmp_path,
        5,
        ExecutionIsolationPolicy(ExecutionIsolationMode.CONTROLLED_ENV),
    )
    strict = PreparedProjectCommand(
        "test",
        ("true",),
        tmp_path,
        5,
        ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT),
        sandbox_adapter="fake-strict-v1",
    )
    assert (
        len(
            {
                command_identity(base),
                command_identity(controlled),
                command_identity(strict),
            }
        )
        == 3
    )


def test_macos_strict_policy_grants_only_required_toolchain_sysctl(
    tmp_path: Path,
) -> None:
    adapter = MacOSSandboxExec()
    wrapped = adapter.wrap(("/usr/bin/true",), tmp_path)
    profile = wrapped[2]

    assert adapter.identity == "macos-sandbox-exec-toolchain-v2"
    assert "sysctl_read_hw_pagesize_compat" in adapter.capabilities
    assert profile.count("(allow sysctl-read") == 1
    assert '(allow sysctl-read (sysctl-name "hw.pagesize_compat"))' in profile
    assert "(allow sysctl-read)" not in profile
    assert "hw.optional" not in profile
    assert "(allow network" not in profile
    assert profile.count("(allow file-write*") == 1
    assert f"(allow file-write* (subpath {json.dumps(str(tmp_path))}))" in profile


def test_policy_version_invalidates_approval_and_attribution_identity(
    tmp_path: Path,
) -> None:
    class VersionedAdapter:
        capabilities = ()

        def __init__(self, identity: str) -> None:
            self.identity = identity

        def available(self) -> bool:
            return True

        def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
            return argv

    policy = ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT)
    command = ProjectCommand((sys.executable, "-c", "pass"), 5)
    old = ProjectCommandTool(
        "test", command, policy, VersionedAdapter("macos-sandbox-exec-v1")
    ).prepare(ExecutionContext(tmp_path))
    current_tool = ProjectCommandTool(
        "test",
        command,
        policy,
        VersionedAdapter("macos-sandbox-exec-toolchain-v2"),
    )
    current = current_tool.prepare(ExecutionContext(tmp_path))

    assert old != current
    assert command_identity(old) != command_identity(current)

    executor = ToolExecutor(
        ToolRegistry((current_tool,)),
        RuleBasedPolicy({"project.test": PermissionDecision.ASK}),
    )
    invocation = ToolInvocation("stale-policy", "project.test", {})
    result = executor.execute(
        invocation,
        ExecutionContext(tmp_path, prepared_project_command=old),
        approval=InvocationApproval.for_invocation(invocation),
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["outcome"] == "prepared_command_changed"


def test_approval_snapshot_rejects_policy_change(tmp_path: Path) -> None:
    tool = ProjectCommandTool(
        "test",
        ProjectCommand((sys.executable, "-c", "pass"), 5),
        ExecutionIsolationPolicy(ExecutionIsolationMode.CONTROLLED_ENV),
    )
    executor = ToolExecutor(
        ToolRegistry((tool,)),
        RuleBasedPolicy({"project.test": PermissionDecision.ASK}),
    )
    invocation = ToolInvocation("approval", "project.test", {})
    context = ExecutionContext(tmp_path)
    preview = replace(
        tool.prepare(context),
        isolation=ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT),
    )
    result = executor.execute(
        invocation,
        replace(context, prepared_project_command=preview),
        approval=InvocationApproval.for_invocation(invocation),
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["outcome"] == "prepared_command_changed"


def test_hardened_project_command_permission_and_read_exclusion(tmp_path: Path) -> None:
    commands = ProjectCommands(
        test=ProjectCommand((sys.executable, "-c", "pass"), 5),
        execution_isolation=ExecutionIsolationPolicy(
            ExecutionIsolationMode.CONTROLLED_ENV
        ),
    )
    read_policy = resolve_interaction_policy(AutonomyMode.READ, "trusted-exec")
    assert "project.test" not in {
        item.name for item in create_repository_registry(read_policy, commands).metadata
    }
    executor = ToolExecutor(
        create_assist_repository_registry(commands),
        RuleBasedPolicy({"project.test": PermissionDecision.DENY}),
    )
    result = executor.execute(
        ToolInvocation("denied", "project.test", {}), ExecutionContext(tmp_path)
    )
    assert result.status is ToolResultStatus.DENIED
    assert not (tmp_path / ".forge-exec").exists()


def test_hardened_execution_rejects_symlinked_home(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".forge-exec").symlink_to(tmp_path, target_is_directory=True)
    tool = ProjectCommandTool(
        "test",
        ProjectCommand((sys.executable, "-c", "pass"), 5),
        ExecutionIsolationPolicy(ExecutionIsolationMode.CONTROLLED_ENV),
    )
    result = ToolExecutor(ToolRegistry((tool,)), AllowAllPolicy()).execute(
        ToolInvocation("symlink", "project.test", {}), ExecutionContext(workspace)
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["outcome"] == "isolation_failed"


def test_hardened_logs_and_preview_do_not_render_secrets(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with patch.dict(os.environ, {"FORGE_TEST_SECRET": "super-secret"}):
        tool = ProjectCommandTool(
            "test",
            ProjectCommand((sys.executable, "-c", "pass"), 5),
            ExecutionIsolationPolicy(ExecutionIsolationMode.CONTROLLED_ENV),
        )
        preview = tool.prepare(ExecutionContext(tmp_path))
        result = ToolExecutor(ToolRegistry((tool,)), AllowAllPolicy()).execute(
            ToolInvocation("redaction", "project.test", {}), ExecutionContext(tmp_path)
        )
    assert result.status is ToolResultStatus.SUCCESS
    assert "super-secret" not in repr(preview)
    assert "super-secret" not in caplog.text


def test_execution_state_is_generated_metadata() -> None:
    assert (
        classify_source(".forge-exec/home/generated.c") is SourceKind.GENERATED_METADATA
    )
    assert classify_source(".forge-exec/tmp/module.py") is SourceKind.GENERATED_METADATA


def test_strict_wrapper_start_failure_is_isolation_failure(tmp_path: Path) -> None:
    class MissingWrapper:
        identity = "fake-missing-wrapper"

        def available(self) -> bool:
            return True

        def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
            return (str(workspace / "nonexistent-sandbox-wrapper"), *argv)

    tool = ProjectCommandTool(
        "test",
        ProjectCommand((sys.executable, "-c", "pass"), 5),
        ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT),
        MissingWrapper(),
    )
    result = ToolExecutor(ToolRegistry((tool,)), AllowAllPolicy()).execute(
        ToolInvocation("wrapper", "project.test", {}), ExecutionContext(tmp_path)
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["outcome"] == "isolation_failed"
    assert result.output["execution_isolation_failure"] is True
