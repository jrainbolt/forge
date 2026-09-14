"""Probe the installed macOS strict adapter in disposable local workspaces."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from forge.evaluation.realworld import copy_repository
from forge.process_isolation import (
    ExecutionIsolationMode,
    ExecutionIsolationPolicy,
    MacOSSandboxExec,
)
from forge.project_config import ProjectCommand
from forge.tools import (
    AllowAllPolicy,
    ExecutionContext,
    ToolExecutor,
    ToolInvocation,
    ToolRegistry,
)
from forge.tools.project import ProjectCommandTool


def run_command(workspace: Path, argv: tuple[str, ...]):
    tool = ProjectCommandTool(
        "test",
        ProjectCommand(argv, 5),
        ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT),
    )
    result = ToolExecutor(ToolRegistry((tool,)), AllowAllPolicy()).execute(
        ToolInvocation("strict-probe", "project.test", {}), ExecutionContext(workspace)
    )
    return result


def main() -> int:
    adapter = MacOSSandboxExec()
    if not adapter.available():
        print(json.dumps({"adapter": adapter.identity, "available": False}))
        return 0
    with tempfile.TemporaryDirectory(prefix="forge-a39-strict-") as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        workspace.mkdir()
        outside = root / "outside.txt"
        inside = workspace / "inside.txt"
        harmless = run_command(workspace, (sys.executable, "-c", "pass"))
        code = (
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('ok')"
        )
        inside_result = run_command(
            workspace, (sys.executable, "-c", code, str(inside))
        )
        outside_result = run_command(
            workspace, (sys.executable, "-c", code, str(outside))
        )
        with tempfile.TemporaryDirectory(
            prefix="forge-a40-home-write-probe-", dir=Path.home()
        ) as home_probe:
            home_target = Path(home_probe) / "outside.txt"
            home_result = run_command(
                workspace, (sys.executable, "-c", code, str(home_target))
            )
            home_denied = (
                home_result.status.value == "failure" and not home_target.exists()
            )

        def status(result):  # type: ignore[no-untyped-def]
            return result.status.value, result.output.get("outcome")

        passed = (
            status(harmless) == ("success", "success")
            and status(inside_result) == ("success", "success")
            and inside.read_text() == "ok"
            and outside_result.status.value == "failure"
            and not outside.exists()
            and home_denied
        )
        foundation = None
        if len(sys.argv) > 1:
            foundation_workspace = copy_repository(
                Path(sys.argv[1]), root / "foundation"
            )
            configured = run_command(
                foundation_workspace,
                ("cmake", "-S", ".", "-B", "build", "-DBUILD_TESTING=ON"),
            )
            foundation = {
                "status": configured.status.value,
                "outcome": configured.output.get("outcome"),
                "exit_code": configured.output.get("exit_code"),
                "stderr_tail": str(configured.output.get("stderr", ""))[-1200:],
            }
        print(
            json.dumps(
                {
                    "adapter": adapter.identity,
                    "available": True,
                    "harmless": status(harmless),
                    "inside": status(inside_result),
                    "outside": status(outside_result),
                    "outside_unchanged": not outside.exists(),
                    "home_like_write": status(home_result),
                    "home_like_unchanged": home_denied,
                    "foundation_configure": foundation,
                    "result": "PASS" if passed else "FAIL",
                },
                sort_keys=True,
            )
        )
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
