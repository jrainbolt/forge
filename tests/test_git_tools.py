from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from forge.tools import (
    AllowAllPolicy,
    ExecutionContext,
    PermissionDecision,
    ToolExecutor,
    ToolInvocation,
    ToolResult,
    ToolResultStatus,
    create_readonly_repository_registry,
)
from forge.tools.git import (
    MAX_GIT_OUTPUT_BYTES,
    GitDiffTool,
    GitStatusTool,
    _run_git,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git unavailable")


def output_mapping(result: ToolResult) -> Mapping[str, object]:
    output = result.output
    assert isinstance(output, Mapping)
    return output


def test_git_status_parses_clean_repository() -> None:
    tool = GitStatusTool(runner=lambda _workspace, _arguments: b"## main\n")
    output = tool.execute({}, ExecutionContext(Path.cwd().resolve()))
    assert isinstance(output, Mapping)
    assert output["branch"] == "main"
    assert output["clean"] is True
    assert output["entries"] == []


def test_git_status_parses_structured_entry_categories() -> None:
    status = (
        b"## feature...origin/feature\n"
        b" M modified.txt\n"
        b"M  staged.txt\n"
        b"?? untracked.txt\n"
        b" D deleted.txt\n"
        b"R  old.txt -> renamed.txt\n"
        b"UU conflict.txt\n"
    )
    tool = GitStatusTool(runner=lambda _workspace, _arguments: status)
    output = tool.execute({}, ExecutionContext(Path.cwd().resolve()))
    assert isinstance(output, Mapping)
    entries = output["entries"]
    assert isinstance(entries, list)
    assert output["branch"] == "feature"
    assert [entry["kind"] for entry in entries] == [
        "modified",
        "modified",
        "untracked",
        "deleted",
        "renamed",
        "conflicted",
    ]
    assert entries[1]["staged"] is True
    assert entries[0]["worktree"] is True
    assert entries[4]["path"] == "renamed.txt"


def test_git_diff_uses_only_fixed_worktree_and_staged_arguments() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(_workspace: Path, arguments: Sequence[str]) -> bytes:
        calls.append(tuple(arguments))
        return b"diff --git a/file b/file\n"

    tool = GitDiffTool(runner=runner)
    context = ExecutionContext(Path.cwd().resolve())
    working = tool.execute({}, context)
    staged = tool.execute({"staged": True}, context)
    assert calls == [
        ("diff", "--no-ext-diff", "--no-textconv"),
        ("diff", "--no-ext-diff", "--no-textconv", "--cached"),
    ]
    assert working["staged"] is False  # type: ignore[index]
    assert staged["staged"] is True  # type: ignore[index]
    assert working["truncated"] is False  # type: ignore[index]


def test_git_process_is_fixed_non_shell_noninteractive_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **options: object):
        captured["command"] = command
        captured.update(options)
        return subprocess.CompletedProcess(command, 0, stdout=b"output", stderr=b"")

    monkeypatch.setattr("forge.tools.git.subprocess.run", fake_run)
    assert _run_git(tmp_path.resolve(), ("status", "--porcelain=v1")) == b"output"
    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == "git"
    assert "core.fsmonitor=false" in command
    assert command[-2:] == ["status", "--porcelain=v1"]
    assert captured["shell"] is False
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["GIT_PAGER"] == "cat"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"

    def oversized_run(command: list[str], **_options: object):
        return subprocess.CompletedProcess(
            command, 0, stdout=b"x" * (MAX_GIT_OUTPUT_BYTES + 1), stderr=b""
        )

    monkeypatch.setattr("forge.tools.git.subprocess.run", oversized_run)
    with pytest.raises(Exception, match="exceeds"):
        _run_git(tmp_path.resolve(), ("diff",))


def test_git_unavailable_and_process_errors_are_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*_args: object, **_kwargs: object):
        raise FileNotFoundError

    monkeypatch.setattr("forge.tools.git.subprocess.run", unavailable)
    with pytest.raises(Exception, match="unavailable"):
        _run_git(tmp_path.resolve(), ("status",))


def test_non_git_workspace_fails_through_executor(tmp_path: Path) -> None:
    result = ToolExecutor(
        create_readonly_repository_registry(), AllowAllPolicy()
    ).execute(
        ToolInvocation("not-git", "git.status", {}),
        ExecutionContext(tmp_path.resolve()),
    )
    assert result.status is ToolResultStatus.FAILURE
    assert "Git command failed" in (result.error_message or "")


def test_real_git_status_and_diff_pipeline_do_not_mutate_workspace_state() -> None:
    workspace = Path(__file__).resolve().parents[1]

    def porcelain() -> bytes:
        return subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            shell=False,
        ).stdout

    before = porcelain()
    executor = ToolExecutor(create_readonly_repository_registry(), AllowAllPolicy())
    context = ExecutionContext(workspace)
    status = executor.execute(ToolInvocation("status-real", "git.status", {}), context)
    diff = executor.execute(
        ToolInvocation("diff-real", "git.diff", {"staged": False}), context
    )
    after = porcelain()

    assert status.status is ToolResultStatus.SUCCESS
    assert diff.status is ToolResultStatus.SUCCESS
    assert status.metadata.permission_decision is PermissionDecision.ALLOW
    assert diff.metadata.permission_decision is PermissionDecision.ALLOW
    assert before == after
