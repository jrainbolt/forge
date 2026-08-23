"""Controlled execution of the two trusted, configured project commands."""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from forge.project_config import ProjectCommand
from forge.tools.tool import Tool, ToolError
from forge.tools.types import (
    ArgumentSchema,
    ExecutionContext,
    StructuredValue,
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
    def __init__(self, operation: str, command: ProjectCommand | None) -> None:
        if operation not in {"build", "test"}:
            raise ValueError("project operation must be build or test")
        self._operation = operation
        self._command = command
        evidence = (
            ToolEvidence.BUILD_RESULT
            if operation == "build"
            else ToolEvidence.TEST_RESULT
        )
        self._metadata = ToolMetadata(
            f"project.{operation}",
            f"Run the workspace's predefined user-configured {operation} command "
            "with explicit approval; the model cannot supply the command.",
            ArgumentSchema(),
            ToolRisk.EXECUTE,
            evidence,
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
        return PreparedProjectCommand(
            self._operation,
            self._command.argv,
            context.workspace,
            self._command.timeout_seconds,
        )

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        return execute_prepared_project_command(self.prepare(context))


def execute_prepared_project_command(
    prepared: PreparedProjectCommand,
) -> StructuredValue:
    """Execute one previously prepared snapshot without reparsing configuration."""
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
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            prepared.argv,
            cwd=prepared.workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=environment,
            start_new_session=True,
        )
    except (OSError, ValueError) as error:
        output = _unavailable_result(prepared.operation, "process_start_failed")
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
        _terminate_process_group(process)
    finally:
        for reader in readers:
            reader.join(timeout=5)
        if process.poll() is None:
            _terminate_process_group(process)

    output = _execution_result(
        prepared.operation,
        process.returncode,
        timed_out,
        time.monotonic() - started,
        stdout,
        stderr,
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
    if timed_out:
        raise ToolError("configured project command timed out", output=output)
    if process.returncode != 0:
        raise ToolError("configured project command exited nonzero", output=output)
    return output


def _drain(stream: object, capture: _TailCapture) -> None:
    while True:
        chunk = stream.read(_READ_CHUNK_BYTES)  # type: ignore[attr-defined]
        if not chunk:
            return
        capture.add(chunk)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=1)
        return
    except subprocess.TimeoutExpired:
        pass
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


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


def _unavailable_result(operation: str, outcome: str) -> StructuredValue:
    return {
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


def _decode(value: bytes) -> str:
    return _ANSI_ESCAPE.sub(b"", value).decode("utf-8", errors="replace")
