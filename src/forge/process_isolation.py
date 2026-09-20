"""Forge-owned isolation policy for trusted project subprocesses."""

from __future__ import annotations

import json
import os
import platform
import signal
import stat
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class ExecutionIsolationMode(StrEnum):
    NONE = "none"
    CONTROLLED_ENV = "controlled_env"
    STRICT = "strict"


class StrictFailureClass(StrEnum):
    NONE = "none"
    EXEC_DENIED = "strict_exec_denied"
    READ_DENIED = "strict_read_denied"
    WRITE_DENIED = "strict_write_denied"
    SERVICE_DENIED = "strict_service_denied"
    CHILD_PROCESS_DENIED = "strict_child_process_denied"
    RUNTIME_INIT_FAILED = "strict_runtime_init_failed"
    UNKNOWN_FAILURE = "strict_unknown_failure"


def classify_strict_failure(outcome: str, stderr: str) -> StrictFailureClass:
    """Classify explicit denial markers only; an assertion is not denial proof."""
    if outcome == "success":
        return StrictFailureClass.NONE
    if outcome in {"isolation_unavailable", "isolation_failed"} or stderr.startswith(
        "sandbox-exec:"
    ):
        return StrictFailureClass.RUNTIME_INIT_FAILED
    if "deny(" in stderr:
        if "file-write" in stderr:
            return StrictFailureClass.WRITE_DENIED
        if "file-read" in stderr:
            return StrictFailureClass.READ_DENIED
        if "mach-lookup" in stderr or "sysctl-read" in stderr:
            return StrictFailureClass.SERVICE_DENIED
        if "process-exec" in stderr or "process-fork" in stderr:
            return StrictFailureClass.CHILD_PROCESS_DENIED
        if "file-map-executable" in stderr:
            return StrictFailureClass.EXEC_DENIED
    return StrictFailureClass.UNKNOWN_FAILURE


@dataclass(frozen=True, slots=True)
class ExecutionIsolationPolicy:
    mode: ExecutionIsolationMode = ExecutionIsolationMode.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ExecutionIsolationMode):
            raise TypeError("execution isolation mode must be ExecutionIsolationMode")


DEFAULT_EXECUTION_ISOLATION = ExecutionIsolationPolicy()


class ExecutionSandbox(Protocol):
    @property
    def identity(self) -> str: ...

    def available(self) -> bool: ...

    def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]: ...


class EphemeralTestSandbox(Protocol):
    """Strict adapter for generated assertions; independent of toolchain writes."""

    @property
    def identity(self) -> str: ...

    def available(self) -> bool: ...

    def wrap(
        self, argv: tuple[str, ...], workspace: Path, temporary: Path
    ) -> tuple[str, ...]: ...


class UnavailableSandbox:
    identity = "unavailable"

    def available(self) -> bool:
        return False

    def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
        raise RuntimeError("strict process sandbox is unavailable")


class MacOSSandboxExec:
    """macOS Seatbelt: all reads, workspace-only writes, no network allowance."""

    identity = "macos-sandbox-exec-toolchain-v2"
    capabilities = (
        "all_filesystem_reads",
        "workspace_filesystem_writes",
        "process_operations",
        "sysctl_read_hw_pagesize_compat",
        "network_not_granted",
    )
    executable = Path("/usr/bin/sandbox-exec")

    def available(self) -> bool:
        if platform.system() != "Darwin" or not self.executable.is_file():
            return False
        try:
            probe = subprocess.run(
                (
                    str(self.executable),
                    "-p",
                    "(version 1)(allow default)",
                    "/usr/bin/true",
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                timeout=3,
                check=False,
                env={"PATH": "/usr/bin:/bin"},
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return probe.returncode == 0

    def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
        resolved = workspace.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("strict sandbox workspace must be a directory")
        quoted = json.dumps(str(resolved), ensure_ascii=True)
        profile = (
            "(version 1)"
            "(deny default)"
            "(allow process*)"
            "(allow file-read*)"
            # Apple ld requires this exact query on the observed macOS toolchain:
            # denying it caused UnsafeHeaderWriter failure; this read alone restored
            # linking while external writes remained denied.
            '(allow sysctl-read (sysctl-name "hw.pagesize_compat"))'
            f"(allow file-write* (subpath {quoted}))"
        )
        return (str(self.executable), "-p", profile, *argv)


def platform_sandbox() -> ExecutionSandbox:
    if platform.system() == "Darwin":
        return MacOSSandboxExec()
    return UnavailableSandbox()


EPHEMERAL_TEMP_POLICY_ID = "forge-private-ephemeral-temp-v1"
EPHEMERAL_ENVIRONMENT_POLICY_ID = "forge-ephemeral-env-v1"


class MacOSEphemeralTestSandbox:
    """Seatbelt profile: project reads, private temp writes, no network/fork."""

    identity = "macos-sandbox-exec-ephemeral-test-v1"

    def available(self) -> bool:
        return MacOSSandboxExec().available()

    def wrap(
        self, argv: tuple[str, ...], workspace: Path, temporary: Path
    ) -> tuple[str, ...]:
        root = workspace.resolve(strict=True)
        scratch = temporary.resolve(strict=True)
        if not root.is_dir() or not scratch.is_dir() or scratch.is_symlink():
            raise ValueError("ephemeral sandbox roots must be real directories")
        interpreter = Path(argv[0]).resolve(strict=True)
        if interpreter != Path(sys.executable).resolve(strict=True):
            raise ValueError("ephemeral sandbox requires the trusted interpreter")
        runtime = Path(sys.base_prefix).resolve(strict=True)
        framework_app = runtime / "Resources/Python.app/Contents/MacOS/Python"
        if not framework_app.is_file():
            raise ValueError("trusted macOS Python framework launcher is unavailable")

        def quoted(path: Path) -> str:
            return json.dumps(str(path), ensure_ascii=True)

        parents = {
            str(parent)
            for selected in (root, scratch, runtime)
            for parent in selected.parents
        }
        parents.update({"/var", "/etc"})
        metadata_reads = "".join(
            f" (literal {json.dumps(path)})" for path in sorted(parents)
        )

        profile = (
            "(version 1)"
            "(deny default)"
            "(allow process-info*)"
            f"(allow process-exec (literal {quoted(interpreter)}))"
            f"(allow process-exec (literal {quoted(framework_app)}))"
            "(allow file-read*"
            f" (subpath {quoted(root)})"
            f" (subpath {quoted(scratch)})"
            f" (subpath {quoted(runtime)})"
            ' (subpath "/System")'
            ' (subpath "/usr")'
            ' (subpath "/Library/Frameworks")'
            ' (subpath "/private/etc")'
            ' (literal "/")'
            ' (literal "/dev/null"))'
            f"(allow file-read-metadata{metadata_reads})"
            f"(allow file-write* (subpath {quoted(scratch)}))"
            '(allow sysctl-read (sysctl-name "hw.pagesize_compat"))'
        )
        return (str(MacOSSandboxExec.executable), "-p", profile, *argv)


def platform_ephemeral_test_sandbox() -> EphemeralTestSandbox:
    if platform.system() == "Darwin":
        return MacOSEphemeralTestSandbox()
    return UnavailableEphemeralTestSandbox()


class UnavailableEphemeralTestSandbox:
    identity = "ephemeral-strict-unavailable"

    def available(self) -> bool:
        return False

    def wrap(
        self, argv: tuple[str, ...], workspace: Path, temporary: Path
    ) -> tuple[str, ...]:
        raise RuntimeError("strict ephemeral sandbox is unavailable")


def ephemeral_test_environment(workspace: Path, temporary: Path) -> dict[str, str]:
    """Fixed no-ambient-secret environment for generated Python assertions."""
    root = workspace.resolve(strict=True)
    scratch = temporary.resolve(strict=True)
    if scratch.is_symlink() or not scratch.is_dir():
        raise ValueError("ephemeral temporary root must be a real directory")
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "TMP": str(scratch),
        "TEMP": str(scratch),
        "PYTHONPATH": str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "LANG": "C",
        "CI": "1",
    }


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """A39 process-group timeout cleanup shared by project and acceptance runs."""
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


def execution_directories(workspace: Path) -> tuple[Path, Path]:
    base = workspace / ".forge-exec"
    return base / "home", base / "tmp"


def create_execution_directories(workspace: Path) -> tuple[Path, Path]:
    """Create controlled HOME/TMP without following pre-existing symlink nodes."""
    root = workspace.resolve(strict=True)
    base = root / ".forge-exec"
    home, temporary = execution_directories(root)
    for path in (base, home, temporary):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            path.mkdir(mode=0o700)
            mode = path.lstat().st_mode
        if (
            not stat.S_ISDIR(mode)
            or path.is_symlink()
            or root not in path.resolve().parents
        ):
            raise ValueError("execution directory is not a workspace-local directory")
    return home, temporary


_ALLOWED_ENVIRONMENT = frozenset(
    {
        "PATH",
        "LANG",
        "TZ",
        "USER",
        "LOGNAME",
        "SYSTEMROOT",
        "WINDIR",
        "SDKROOT",
        "DEVELOPER_DIR",
        "MACOSX_DEPLOYMENT_TARGET",
        "LC_ALL",
        "LC_CTYPE",
        "LC_COLLATE",
        "LC_MESSAGES",
        "LC_MONETARY",
        "LC_NUMERIC",
        "LC_TIME",
        "LC_PAPER",
        "LC_NAME",
        "LC_ADDRESS",
        "LC_TELEPHONE",
        "LC_MEASUREMENT",
        "LC_IDENTIFICATION",
    }
)


def controlled_environment(
    workspace: Path, parent: dict[str, str] | None = None
) -> dict[str, str]:
    source = os.environ if parent is None else parent
    root = workspace.resolve(strict=True)
    path_entries = []
    for entry in source.get("PATH", os.defpath).split(os.pathsep):
        candidate = Path(entry)
        if not candidate.is_absolute():
            continue
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved == root or root in resolved.parents:
            continue
        path_entries.append(entry)
    environment = {
        name: value for name, value in source.items() if name in _ALLOWED_ENVIRONMENT
    }
    home, temporary = execution_directories(workspace)
    environment.update(
        {
            "PATH": os.pathsep.join(path_entries) or os.defpath,
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "CI": "1",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment
