"""Disposable, bounded STRICT compiler/linker diagnostic ladder."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from forge.evaluation.realworld import copy_repository, hash_workspace
from forge.process_isolation import ExecutionIsolationMode, ExecutionIsolationPolicy
from forge.project_config import ProjectCommand
from forge.tools import (
    AllowAllPolicy,
    ExecutionContext,
    ToolExecutor,
    ToolInvocation,
    ToolRegistry,
)
from forge.tools.project import ProjectCommandTool

STRICT = ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT)
CONTROLLED = ExecutionIsolationPolicy(ExecutionIsolationMode.CONTROLLED_ENV)
NONE = ExecutionIsolationPolicy(ExecutionIsolationMode.NONE)


def execute(
    workspace: Path,
    argv: tuple[str, ...],
    isolation: ExecutionIsolationPolicy = STRICT,
) -> dict[str, object]:
    tool = ProjectCommandTool("test", ProjectCommand(argv, 90), isolation)
    result = ToolExecutor(ToolRegistry((tool,)), AllowAllPolicy()).execute(
        ToolInvocation("probe", "project.test", {}), ExecutionContext(workspace)
    )
    output = result.output or {}
    return {
        "status": result.status.value,
        "outcome": output.get("outcome"),
        "exit_code": output.get("exit_code"),
        "duration_seconds": output.get("duration_seconds"),
        "strict_policy_version": output.get("strict_policy_version"),
        "strict_failure_class": output.get("strict_failure_class"),
        "strict_capabilities": output.get("strict_capabilities"),
        "stderr_tail": str(output.get("stderr", ""))[-1600:],
        "stdout_tail": str(output.get("stdout", ""))[-400:],
    }


def main() -> int:
    compiler = Path("/usr/bin/cc")
    with tempfile.TemporaryDirectory(prefix="forge-a40-probe-") as name:
        root = Path(name)
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "trivial.c").write_text("int main(void) { return 0; }\n")
        ladder = (
            ("T1", (str(compiler), "--version")),
            ("T2", (str(compiler), "-c", "trivial.c", "-o", "trivial.o")),
            ("T3", (str(compiler), "trivial.o", "-o", "trivial")),
            ("T4", (str(workspace / "trivial"),)),
        )
        results: dict[str, object] = {"compiler": str(compiler)}
        for label, argv in ladder:
            if label == "T4" and not (workspace / "trivial").is_file():
                results[label] = {"status": "not_run", "reason": "link_failed"}
                continue
            results[label] = execute(workspace, argv)
        results["plain_compile"] = execute(
            workspace,
            (str(compiler), "-c", "trivial.c", "-o", "trivial-plain.o"),
            NONE,
        )
        results["controlled_compile"] = execute(
            workspace,
            (str(compiler), "-c", "trivial.c", "-o", "trivial-controlled.o"),
            CONTROLLED,
        )
        cmake_workspace = root / "cmake"
        cmake_workspace.mkdir()
        (cmake_workspace / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(A40Trivial C)\n"
            "add_executable(a40-trivial trivial.c)\n"
        )
        (cmake_workspace / "trivial.c").write_text("int main(void) { return 0; }\n")
        results["T5"] = execute(
            cmake_workspace,
            ("/opt/homebrew/bin/cmake", "-S", ".", "-B", "build"),
        )
        if len(sys.argv) > 1:
            canonical = Path(sys.argv[1]).resolve(strict=True)
            before = hash_workspace(canonical)
            foundation = copy_repository(canonical, root / "foundation")
            results["T6"] = execute(
                foundation,
                (
                    "/opt/homebrew/bin/cmake",
                    "-S",
                    ".",
                    "-B",
                    "build",
                    "-DBUILD_TESTING=ON",
                ),
            )
            results["canonical_unchanged"] = hash_workspace(canonical) == before
        print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
