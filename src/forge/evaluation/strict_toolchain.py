"""Deterministic A40 strict-capability and toolchain regression suite.

Fake adapters exercise orchestration contracts; they are not OS isolation proof.
The separate macOS acceptance probe records real Seatbelt behavior.
"""

from __future__ import annotations

import shutil
import sys
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
)
from forge.tools.project import ProjectCommandTool

STRICT_TOOLCHAIN_V1 = "strict-toolchain-v1"
STRICT = ExecutionIsolationPolicy(ExecutionIsolationMode.STRICT)
CONTROLLED = ExecutionIsolationPolicy(ExecutionIsolationMode.CONTROLLED_ENV)


@dataclass(frozen=True, slots=True)
class StrictToolchainCase:
    case_id: str
    passed: bool
    simulated: bool
    outcome: str


@dataclass(frozen=True, slots=True)
class StrictToolchainRun:
    cases: tuple[StrictToolchainCase, ...]
    compiler: str
    simulated_strict_adapter: bool
    strict_compile_pass: bool
    strict_link_pass: bool
    strict_external_write_denied: bool


class _PassthroughFake:
    identity = "fake-toolchain-v1"
    capabilities = ("simulated_toolchain",)

    def available(self) -> bool:
        return True

    def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
        return argv


class _WorkspaceWriteFake(_PassthroughFake):
    identity = "fake-workspace-write-v1"

    def wrap(self, argv: tuple[str, ...], workspace: Path) -> tuple[str, ...]:
        target = Path(argv[-1]).resolve()
        if workspace.resolve() not in target.parents:
            raise RuntimeError("fake strict adapter denied outside write")
        return argv


class _UnavailableFake(_PassthroughFake):
    identity = "fake-unavailable-v1"

    def available(self) -> bool:
        return False


def _execute(
    workspace: Path,
    argv: tuple[str, ...],
    adapter: object,
):
    tool = ProjectCommandTool(
        "test",
        ProjectCommand(argv, 10),
        STRICT,
        adapter,  # type: ignore[arg-type]
    )
    return ToolExecutor(ToolRegistry((tool,)), AllowAllPolicy()).execute(
        ToolInvocation("strict-toolchain", "project.test", {}),
        ExecutionContext(workspace),
    )


def run_strict_toolchain_v1(root: Path) -> StrictToolchainRun:
    root.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("strict-toolchain-v1 requires an installed C compiler")
    workspace = root / "toolchain"
    workspace.mkdir()
    (workspace / "trivial.c").write_text("int main(void) { return 0; }\n")
    adapter = _PassthroughFake()
    cases: list[StrictToolchainCase] = []

    for case_id, argv in (
        ("S01", (compiler, "--version")),
        ("S02", (compiler, "-c", "trivial.c", "-o", "trivial.o")),
        ("S03", (compiler, "trivial.o", "-o", "trivial")),
        ("S04", (str(workspace / "trivial"),)),
    ):
        result = _execute(workspace, argv, adapter)
        cases.append(
            StrictToolchainCase(
                case_id,
                result.status.value == "success",
                True,
                str(result.output.get("outcome")),
            )
        )

    write_code = (
        "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('probe')"
    )
    for case_id, target, allowed in (
        ("S05", workspace / "inside.txt", True),
        ("S06", root / "outside.txt", False),
        ("S07", root / "home-like" / "outside.txt", False),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        result = _execute(
            workspace,
            (sys.executable, "-c", write_code, str(target)),
            _WorkspaceWriteFake(),
        )
        cases.append(
            StrictToolchainCase(
                case_id,
                (result.status.value == "success") is allowed
                and target.exists() is allowed,
                True,
                str(result.output.get("outcome")),
            )
        )

    marker = workspace / "must-not-run"
    unavailable = _execute(
        workspace,
        (
            sys.executable,
            "-c",
            "from pathlib import Path; Path('must-not-run').touch()",
        ),
        _UnavailableFake(),
    )
    cases.append(
        StrictToolchainCase(
            "S08",
            unavailable.output.get("outcome") == "isolation_unavailable"
            and not marker.exists(),
            True,
            str(unavailable.output.get("outcome")),
        )
    )
    with patch("forge.tools.project.platform_sandbox", lambda: adapter):
        strict_plan = _run("C01", root / "strict-plan", STRICT)
    cases.append(
        StrictToolchainCase(
            "S09",
            strict_plan.command_order == ("configure", "build", "test")
            and strict_plan.coding.status.value == "completed_verified"
            and strict_plan.model_calls == 3,
            True,
            strict_plan.coding.status.value,
        )
    )
    controlled_plan = _run("C01", root / "controlled-plan", CONTROLLED)
    cases.append(
        StrictToolchainCase(
            "S10",
            controlled_plan.command_order == ("configure", "build", "test")
            and controlled_plan.coding.status.value == "completed_verified",
            False,
            controlled_plan.coding.status.value,
        )
    )
    return StrictToolchainRun(
        tuple(cases),
        compiler,
        True,
        cases[1].passed,
        cases[2].passed,
        cases[5].passed and cases[6].passed,
    )
