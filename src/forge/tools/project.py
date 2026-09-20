"""Controlled execution of trusted, configured project commands."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from forge.process_isolation import (
    DEFAULT_EXECUTION_ISOLATION,
    ExecutionIsolationMode,
    ExecutionIsolationPolicy,
    ExecutionSandbox,
    classify_strict_failure,
    controlled_environment,
    create_execution_directories,
    execution_directories,
    platform_sandbox,
    terminate_process_group,
)
from forge.project_config import ProjectCommand
from forge.tools.tool import Tool, ToolError
from forge.tools.types import (
    ArgumentSchema,
    ExecutionContext,
    StructuredValue,
    ToolCapability,
    ToolEvidence,
    ToolMetadata,
    ToolRisk,
)

MAX_PROCESS_OUTPUT_BYTES = 256 * 1024
LOGGER = logging.getLogger(__name__)
_READ_CHUNK_BYTES = 16 * 1024
_ANSI_ESCAPE = re.compile(rb"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


@dataclass(frozen=True, slots=True)
class PreparedProjectCommand:
    """The exact immutable command shown for approval and then executed."""

    operation: str
    argv: tuple[str, ...]
    workspace: Path
    timeout_seconds: float
    isolation: ExecutionIsolationPolicy = ExecutionIsolationPolicy()
    environment: tuple[tuple[str, str], ...] | None = field(default=None, repr=False)
    execution_home: Path | None = None
    execution_tmp: Path | None = None
    sandbox_adapter: str = "none"
    sandbox: ExecutionSandbox | None = field(default=None, repr=False, compare=False)


class _TailCapture:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._data = bytearray()
        self.truncated = False

    def add(self, chunk: bytes) -> None:
        if len(self._data) + len(chunk) > self._limit:
            self.truncated = True
        self._data.extend(chunk)
        excess = len(self._data) - self._limit
        if excess > 0:
            del self._data[:excess]

    @property
    def data(self) -> bytes:
        return bytes(self._data)


class ProjectCommandTool(Tool):
    def __init__(
        self,
        operation: str,
        command: ProjectCommand | None,
        isolation: ExecutionIsolationPolicy = DEFAULT_EXECUTION_ISOLATION,
        sandbox: ExecutionSandbox | None = None,
    ) -> None:
        if operation not in {"configure", "build", "test"}:
            raise ValueError("project operation must be configure, build, or test")
        self._operation = operation
        self._command = command
        self._isolation = isolation
        self._sandbox = sandbox
        evidence = {
            "configure": ToolEvidence.CONFIGURE_RESULT,
            "build": ToolEvidence.BUILD_RESULT,
            "test": ToolEvidence.TEST_RESULT,
        }[operation]
        self._metadata = ToolMetadata(
            f"project.{operation}",
            f"Run the workspace's predefined user-configured {operation} command "
            "with explicit approval; the model cannot supply the command.",
            ArgumentSchema(),
            ToolRisk.EXECUTE,
            evidence,
            {
                "configure": ToolCapability.CONFIGURE,
                "build": ToolCapability.BUILD,
                "test": ToolCapability.TEST,
            }[operation],
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def configured(self) -> bool:
        return self._command is not None

    def prepare(self, context: ExecutionContext) -> PreparedProjectCommand:
        if self._command is None:
            raise ToolError(
                f"project {self._operation} command is not configured",
                output=_unavailable_result(self._operation, "command_not_configured"),
            )
        home, temporary = (
            execution_directories(context.workspace)
            if self._isolation.mode is not ExecutionIsolationMode.NONE
            else (None, None)
        )
        sandbox = (
            self._sandbox or platform_sandbox()
            if self._isolation.mode is ExecutionIsolationMode.STRICT
            else None
        )
        return PreparedProjectCommand(
            self._operation,
            self._command.argv,
            context.workspace,
            self._command.timeout_seconds,
            self._isolation,
            tuple(
                sorted(project_environment(self._isolation, context.workspace).items())
            ),
            home,
            temporary,
            sandbox.identity if sandbox is not None else "none",
            sandbox,
        )

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        current = self.prepare(context)
        approved = context.prepared_project_command
        if approved is not None:
            if not isinstance(approved, PreparedProjectCommand) or approved != current:
                raise ToolError(
                    "approved project command changed before execution",
                    output=_unavailable_result(
                        self._operation, "prepared_command_changed", current
                    ),
                )
            current = approved
        return execute_prepared_project_command(current)


def execute_prepared_project_command(
    prepared: PreparedProjectCommand,
) -> StructuredValue:
    """Execute one previously prepared snapshot without reparsing configuration."""
    environment = (
        dict(prepared.environment)
        if prepared.environment is not None
        else project_environment(prepared.isolation, prepared.workspace)
    )
    started = time.monotonic()
    sandbox_available = False
    argv = prepared.argv
    if prepared.isolation.mode is ExecutionIsolationMode.STRICT:
        sandbox = prepared.sandbox or platform_sandbox()
        sandbox_available = sandbox.available()
        if not sandbox_available:
            raise ToolError(
                "strict process isolation is unavailable",
                output=_unavailable_result(
                    prepared.operation, "isolation_unavailable", prepared
                ),
            )
        try:
            argv = sandbox.wrap(prepared.argv, prepared.workspace)
        except (OSError, RuntimeError, ValueError) as error:
            raise ToolError(
                "strict process isolation could not be prepared",
                output=_unavailable_result(
                    prepared.operation, "isolation_failed", prepared
                ),
            ) from error
    if prepared.isolation.mode is not ExecutionIsolationMode.NONE:
        try:
            create_execution_directories(prepared.workspace)
        except (OSError, ValueError) as error:
            raise ToolError(
                "workspace-local execution directories are unavailable",
                output=_unavailable_result(
                    prepared.operation, "isolation_failed", prepared
                ),
            ) from error
    try:
        process = subprocess.Popen(
            argv,
            cwd=prepared.workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=environment,
            start_new_session=True,
        )
    except (OSError, ValueError) as error:
        output = _unavailable_result(
            prepared.operation,
            "isolation_failed"
            if prepared.isolation.mode is ExecutionIsolationMode.STRICT
            else "process_start_failed",
            prepared,
        )
        raise ToolError(
            "configured project process could not be started", output=output
        ) from error

    stdout = _TailCapture(MAX_PROCESS_OUTPUT_BYTES)
    stderr = _TailCapture(MAX_PROCESS_OUTPUT_BYTES)
    assert process.stdout is not None and process.stderr is not None
    readers = (
        threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
    )
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        process.wait(timeout=prepared.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(process)
    finally:
        for reader in readers:
            reader.join(timeout=5)
        if process.poll() is None:
            terminate_process_group(process)

    output = _execution_result(
        prepared.operation,
        process.returncode,
        timed_out,
        time.monotonic() - started,
        stdout,
        stderr,
    )
    output.update(_isolation_metrics(prepared, sandbox_available))
    if (
        prepared.isolation.mode is ExecutionIsolationMode.STRICT
        and process.returncode != 0
        and stderr.data.startswith(b"sandbox-exec:")
    ):
        output["outcome"] = "isolation_failed"
        output["execution_isolation_failure"] = True
        output["strict_failure_class"] = classify_strict_failure(
            "isolation_failed", str(output["stderr"])
        ).value
        raise ToolError(
            "strict process isolation could not be established", output=output
        )
    LOGGER.info(
        "Project command completed operation=%s exit_code=%s timed_out=%s "
        "duration=%.3f stdout_truncated=%s stderr_truncated=%s",
        prepared.operation,
        process.returncode,
        timed_out,
        output["duration_seconds"],
        stdout.truncated,
        stderr.truncated,
    )
    if prepared.isolation.mode is ExecutionIsolationMode.STRICT:
        output["strict_failure_class"] = classify_strict_failure(
            str(output["outcome"]), str(output["stderr"])
        ).value
    if timed_out:
        raise ToolError("configured project command timed out", output=output)
    if process.returncode != 0:
        raise ToolError("configured project command exited nonzero", output=output)
    return output


def project_environment(
    isolation: ExecutionIsolationPolicy = DEFAULT_EXECUTION_ISOLATION,
    workspace: Path | None = None,
) -> dict[str, str]:
    """Return the exact noninteractive environment supplied to A10 processes."""
    if isolation.mode is not ExecutionIsolationMode.NONE:
        if workspace is None:
            raise ValueError("hardened project environment requires a workspace")
        return controlled_environment(workspace)
    environment = os.environ.copy()
    environment.update(
        {
            "CI": "1",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _drain(stream: object, capture: _TailCapture) -> None:
    while True:
        chunk = stream.read(_READ_CHUNK_BYTES)  # type: ignore[attr-defined]
        if not chunk:
            return
        capture.add(chunk)


def _execution_result(
    operation: str,
    exit_code: int | None,
    timed_out: bool,
    duration: float,
    stdout: _TailCapture,
    stderr: _TailCapture,
) -> StructuredValue:
    return {
        "operation": operation,
        "outcome": (
            "timeout" if timed_out else "success" if exit_code == 0 else "nonzero_exit"
        ),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "stdout": _decode(stdout.data),
        "stderr": _decode(stderr.data),
        "stdout_truncated": stdout.truncated,
        "stderr_truncated": stderr.truncated,
    }


def _unavailable_result(
    operation: str,
    outcome: str,
    prepared: PreparedProjectCommand | None = None,
) -> StructuredValue:
    result: StructuredValue = {
        "operation": operation,
        "outcome": outcome,
        "exit_code": None,
        "timed_out": False,
        "duration_seconds": 0.0,
        "stdout": "",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    if prepared is not None:
        result.update(
            _isolation_metrics(prepared, False, failure=outcome.startswith("isolation"))
        )
        if prepared.isolation.mode is ExecutionIsolationMode.STRICT:
            result["strict_failure_class"] = classify_strict_failure(outcome, "").value
    return result


def _isolation_metrics(
    prepared: PreparedProjectCommand,
    sandbox_available: bool,
    *,
    failure: bool = False,
) -> StructuredValue:
    hardened = prepared.isolation.mode is not ExecutionIsolationMode.NONE
    return {
        "execution_isolation_mode": prepared.isolation.mode.value,
        "execution_sandbox_adapter": prepared.sandbox_adapter,
        "execution_sandbox_available": sandbox_available,
        "execution_environment_hardened": hardened,
        "execution_home_redirected": hardened,
        "execution_tmp_redirected": hardened,
        "execution_isolation_failure": failure,
        "strict_policy_version": (
            prepared.sandbox_adapter
            if prepared.isolation.mode is ExecutionIsolationMode.STRICT
            else "none"
        ),
        "strict_capabilities": (
            tuple(getattr(prepared.sandbox, "capabilities", ()))
            if prepared.isolation.mode is ExecutionIsolationMode.STRICT
            else ()
        ),
        "strict_failure_class": "none",
    }


def _decode(value: bytes) -> str:
    return _ANSI_ESCAPE.sub(b"", value).decode("utf-8", errors="replace")
