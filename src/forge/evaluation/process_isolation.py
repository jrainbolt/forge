"""Eight deterministic A39 process-isolation scenarios."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from forge.evaluation.project_configure import _run
from forge.process_isolation import ExecutionIsolationMode, ExecutionIsolationPolicy
from forge.project_config import ProjectCommand
from forge.tools import (
    AllowAllPolicy,
    ExecutionContext,
    ToolExecutor,
    ToolInvocation,
    ToolRegistry,
    ToolResultStatus,
)
from forge.tools.project import ProjectCommandTool

PROCESS_ISOLATION_V1 = "process-isolation-v1"
CONTROLLED = ExecutionIsolationPolicy(ExecutionIsolationMode.CONTROLLED_ENV)
STRICT = ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT)


@dataclass(frozen=True, slots=True)
class ProcessIsolationCase:
    case_id: str
    passed: bool
    outcome: str


class _Unavailable:
    identity = "fake-unavailable"

    def available(self) -> bool:
        return False

    def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
        raise AssertionError("unavailable adapter must not wrap")


class _BoundedFake:
    """Deterministically exercises adapter rejection; not an OS sandbox."""

    identity = "fake-bounded-v1"

    def available(self) -> bool:
        return True

    def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
        target = Path(argv[-1]).resolve()
        if workspace.resolve() not in target.parents:
            raise RuntimeError("fake strict adapter denied outside write")
        return argv


def _execute(
    workspace: Path,
    argv: tuple[str, ...],
    policy: ExecutionIsolationPolicy,
    *,
    timeout: float = 5,
    sandbox: object | None = None,
):
    tool = ProjectCommandTool(
        "test",
        ProjectCommand(argv, timeout),
        policy,
        sandbox,  # type: ignore[arg-type]
    )
    executor = ToolExecutor(ToolRegistry((tool,)), AllowAllPolicy())
    return executor.execute(
        ToolInvocation("test", "project.test", {}), ExecutionContext(workspace)
    )


def run_process_isolation_v1(root: Path) -> tuple[ProcessIsolationCase, ...]:
    root.mkdir(parents=True, exist_ok=True)
    cases: list[ProcessIsolationCase] = []
    for number in range(1, 9):
        workspace = root / f"i{number:02d}"
        workspace.mkdir()
        if number == 1:
            command = (
                sys.executable,
                "-c",
                "import os; from pathlib import Path; "
                "Path(os.environ['HOME'], 'output.txt').write_text('home'); "
                "Path(os.environ['TMPDIR'], 'temp.txt').write_text('tmp')",
            )
            result = _execute(workspace, command, CONTROLLED)
            passed = (
                result.status is ToolResultStatus.SUCCESS
                and (workspace / ".forge-exec/home/output.txt").read_text() == "home"
                and (workspace / ".forge-exec/tmp/temp.txt").read_text() == "tmp"
            )
        elif number == 2:
            command = (
                sys.executable,
                "-c",
                "import os,sys; "
                "sys.exit(0 if 'FORGE_TEST_SECRET' not in os.environ else 7)",
            )
            with patch.dict(os.environ, {"FORGE_TEST_SECRET": "super-secret"}):
                result = _execute(workspace, command, CONTROLLED)
            passed = result.status is ToolResultStatus.SUCCESS
        elif number == 3:
            result = _execute(workspace, ("true",), CONTROLLED)
            passed = result.status is ToolResultStatus.SUCCESS
        elif number == 4:
            marker = workspace / "must-not-exist"
            result = _execute(
                workspace,
                (
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('must-not-exist').touch()",
                ),
                STRICT,
                sandbox=_Unavailable(),
            )
            passed = (
                result.status is ToolResultStatus.FAILURE
                and result.output["outcome"] == "isolation_unavailable"
                and not marker.exists()
            )
        elif number in {5, 6}:
            target = (root / "outside.txt") if number == 5 else workspace / "inside.txt"
            result = _execute(
                workspace,
                (
                    sys.executable,
                    "-c",
                    "import sys; from pathlib import Path; "
                    "Path(sys.argv[1]).write_text('write')",
                    str(target),
                ),
                STRICT,
                sandbox=_BoundedFake(),
            )
            passed = (
                result.status
                is (
                    ToolResultStatus.FAILURE
                    if number == 5
                    else ToolResultStatus.SUCCESS
                )
                and target.exists() is (number == 6)
                and result.output["outcome"]
                == ("isolation_failed" if number == 5 else "success")
            )
        elif number == 7:
            marker = workspace / "orphan-marker"
            child = (
                "import sys,time; from pathlib import Path; "
                "time.sleep(0.5); Path(sys.argv[1]).touch()"
            )
            parent = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]]); "
                "time.sleep(5)"
            )
            result = _execute(
                workspace,
                (sys.executable, "-c", parent, str(marker)),
                STRICT,
                timeout=0.1,
                sandbox=_BoundedFake(),
            )
            time.sleep(0.7)
            passed = (
                result.status is ToolResultStatus.FAILURE
                and result.output["outcome"] == "timeout"
                and not marker.exists()
            )
        else:
            task = _run("C01", workspace / "plan", CONTROLLED)
            passed = (
                task.command_order == ("configure", "build", "test")
                and task.model_calls == 3
                and task.coding.status.value == "completed_verified"
            )
            result = None
        cases.append(
            ProcessIsolationCase(
                f"I{number:02d}",
                passed,
                str(result.output.get("outcome", "unknown"))
                if result is not None
                else "pass",
            )
        )
    return tuple(cases)
