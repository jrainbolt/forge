"""Fixed-shape, read-only Git tools."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from forge.tools.tool import Tool, ToolError
from forge.tools.types import (
    ArgumentSchema,
    ArgumentSpec,
    ArgumentType,
    ExecutionContext,
    StructuredValue,
    ToolMetadata,
    ToolRisk,
)

MAX_GIT_OUTPUT_BYTES = 256 * 1024
GitRunner = Callable[[Path, Sequence[str]], bytes]


class GitStatusTool(Tool):
    _metadata = ToolMetadata(
        "git.status",
        "Inspect structured local Git status for the active workspace.",
        ArgumentSchema(),
        ToolRisk.READ_ONLY,
    )

    def __init__(self, *, runner: GitRunner | None = None) -> None:
        self._runner = runner or _run_git

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self,
        arguments: Mapping[str, str | int | bool],
        context: ExecutionContext,
    ) -> StructuredValue:
        output = self._runner(
            context.workspace,
            ("status", "--porcelain=v1", "--branch", "--untracked-files=all"),
        )
        return _parse_status(output)


class GitDiffTool(Tool):
    _metadata = ToolMetadata(
        "git.diff",
        "Read a bounded working-tree or staged diff in the active workspace.",
        ArgumentSchema(
            (
                ArgumentSpec(
                    "staged",
                    ArgumentType.BOOLEAN,
                    "Read the staged diff instead of the working-tree diff.",
                    False,
                ),
            )
        ),
        ToolRisk.READ_ONLY,
    )

    def __init__(self, *, runner: GitRunner | None = None) -> None:
        self._runner = runner or _run_git

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self,
        arguments: Mapping[str, str | int | bool],
        context: ExecutionContext,
    ) -> StructuredValue:
        staged = bool(arguments.get("staged", False))
        command = ["diff", "--no-ext-diff", "--no-textconv"]
        if staged:
            command.append("--cached")
        output = self._runner(context.workspace, tuple(command))
        try:
            diff = output.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ToolError("Git diff output is not valid UTF-8") from error
        return {"staged": staged, "diff": diff, "truncated": False}


def _run_git(workspace: Path, arguments: Sequence[str]) -> bytes:
    command = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(workspace),
        "--no-pager",
        *arguments,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "PAGER": "cat",
        }
    )
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            shell=False,
            env=environment,
        )
    except FileNotFoundError as error:
        raise ToolError("Git executable is unavailable") from error
    except OSError as error:
        raise ToolError("Git process could not be started") from error
    if result.returncode != 0:
        message = result.stderr[:4096].decode("utf-8", errors="replace").strip()
        raise ToolError(f"Git command failed: {message or 'unknown Git error'}")
    if len(result.stdout) > MAX_GIT_OUTPUT_BYTES:
        raise ToolError(
            f"Git output exceeds the {MAX_GIT_OUTPUT_BYTES}-byte output limit"
        )
    return result.stdout


def _parse_status(output: bytes) -> StructuredValue:
    try:
        lines = output.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ToolError("Git status output is not valid UTF-8") from error
    branch: str | None = None
    entries = []
    for line in lines:
        if line.startswith("## "):
            branch = _parse_branch(line[3:])
            continue
        if len(line) < 3:
            continue
        code = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        entries.append(
            {
                "path": path,
                "kind": _status_kind(code),
                "index_status": None if code[0] in {" ", "?"} else code[0],
                "worktree_status": None if code[1] == " " else code[1],
                "staged": code[0] not in {" ", "?"},
                "worktree": code[1] != " ",
            }
        )
    return {"branch": branch, "clean": not entries, "entries": entries}


def _parse_branch(value: str) -> str | None:
    if value.startswith("No commits yet on "):
        return value.removeprefix("No commits yet on ")
    branch = value.split("...", 1)[0]
    return None if branch.startswith("HEAD ") else branch


def _status_kind(code: str) -> str:
    if code == "??":
        return "untracked"
    if "U" in code or code in {"AA", "DD"}:
        return "conflicted"
    if "R" in code:
        return "renamed"
    if "D" in code:
        return "deleted"
    if "A" in code:
        return "added"
    return "modified"
