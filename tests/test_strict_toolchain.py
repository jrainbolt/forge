from __future__ import annotations

from pathlib import Path

import pytest

from forge.evaluation.strict_toolchain import run_strict_toolchain_v1
from forge.orchestration.verification_attribution import command_identity
from forge.process_isolation import (
    ExecutionIsolationMode,
    ExecutionIsolationPolicy,
    StrictFailureClass,
    classify_strict_failure,
)
from forge.project_config import ProjectCommand
from forge.tools import ExecutionContext
from forge.tools.project import ProjectCommandTool


def test_strict_toolchain_v1_deterministic_contracts(tmp_path: Path) -> None:
    result = run_strict_toolchain_v1(tmp_path / "suite")
    assert tuple(case.case_id for case in result.cases) == tuple(
        f"S{number:02d}" for number in range(1, 11)
    )
    assert all(case.passed for case in result.cases), result.cases
    assert all(case.simulated for case in result.cases[:9])
    assert not result.cases[9].simulated
    assert result.simulated_strict_adapter
    assert result.strict_compile_pass
    assert result.strict_link_pass
    assert result.strict_external_write_denied
    assert result.result_scope == "deterministic_fake_adapter_contracts"
    assert result.real_platform_acceptance is None


@pytest.mark.parametrize(
    ("outcome", "stderr", "expected"),
    [
        ("success", "", StrictFailureClass.NONE),
        ("isolation_unavailable", "", StrictFailureClass.RUNTIME_INIT_FAILED),
        (
            "nonzero_exit",
            "Sandbox: ld deny(1) file-write-data",
            StrictFailureClass.WRITE_DENIED,
        ),
        (
            "nonzero_exit",
            "Sandbox: ld deny(1) file-read-data",
            StrictFailureClass.READ_DENIED,
        ),
        (
            "nonzero_exit",
            "Sandbox: ld deny(1) mach-lookup",
            StrictFailureClass.SERVICE_DENIED,
        ),
        (
            "nonzero_exit",
            "Sandbox: ld deny(1) process-exec",
            StrictFailureClass.CHILD_PROCESS_DENIED,
        ),
        (
            "nonzero_exit",
            "Sandbox: ld deny(1) file-map-executable",
            StrictFailureClass.EXEC_DENIED,
        ),
        (
            "nonzero_exit",
            "ld: Assertion failed: header alignment",
            StrictFailureClass.UNKNOWN_FAILURE,
        ),
    ],
)
def test_strict_failure_classification_does_not_infer_denial_from_assertion(
    outcome: str, stderr: str, expected: StrictFailureClass
) -> None:
    assert classify_strict_failure(outcome, stderr) is expected


def test_policy_version_is_bound_into_prepared_approval_identity(
    tmp_path: Path,
) -> None:
    class VaryingAdapter:
        def __init__(self, identity: str) -> None:
            self.identity = identity

        def available(self) -> bool:
            return True

        def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
            return argv

    policy = ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT)
    command = ProjectCommand(("/usr/bin/true",), 5)
    first = ProjectCommandTool("test", command, policy, VaryingAdapter("v1")).prepare(
        ExecutionContext(tmp_path)
    )
    second = ProjectCommandTool("test", command, policy, VaryingAdapter("v2")).prepare(
        ExecutionContext(tmp_path)
    )
    assert first != second
    assert command_identity(first) != command_identity(second)
