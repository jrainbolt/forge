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


def run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        shell=False,
    )


@pytest.fixture
def git_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    run_git(repository, "init", "--quiet")
    run_git(repository, "config", "user.name", "Forge Test")
    run_git(repository, "config", "user.email", "forge-test@example.invalid")
    for name, content in (
        ("modified.txt", "original modified\n"),
        ("deleted.txt", "original deleted\n"),
        ("working-one.txt", "original one\n"),
        ("working-two.txt", "original two\n"),
        ("staged.txt", "original staged\n"),
    ):
        (repository / name).write_text(content)
    run_git(repository, "add", ".")
    run_git(repository, "commit", "--quiet", "-m", "test fixture")
    return repository


def execute_git(
    repository: Path, name: str, arguments: Mapping[str, object]
) -> ToolResult:
    return ToolExecutor(
        create_readonly_repository_registry(), AllowAllPolicy()
    ).execute(
        ToolInvocation(f"real-{name}", name, arguments),
        ExecutionContext(repository.resolve()),
    )


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


def test_real_git_status_reports_clean_repository(git_repository: Path) -> None:
    result = execute_git(git_repository, "git.status", {})
    output = output_mapping(result)

    assert result.status is ToolResultStatus.SUCCESS
    assert output["clean"] is True
    assert output["entries"] == ()
    assert isinstance(output["branch"], str)


def test_real_git_status_distinguishes_mixed_repository_states(
    git_repository: Path,
) -> None:
    (git_repository / "modified.txt").write_text("worktree modified\n")
    (git_repository / "untracked.txt").write_text("untracked\n")
    (git_repository / "staged-new.txt").write_text("staged new\n")
    (git_repository / "deleted.txt").unlink()
    run_git(git_repository, "add", "staged-new.txt")

    result = execute_git(git_repository, "git.status", {})
    output = output_mapping(result)
    entries = {entry["path"]: entry for entry in output["entries"]}  # type: ignore[index]

    assert result.status is ToolResultStatus.SUCCESS
    assert output["clean"] is False
    assert entries["modified.txt"] == {
        "path": "modified.txt",
        "kind": "modified",
        "index_status": None,
        "worktree_status": "M",
        "staged": False,
        "worktree": True,
    }
    assert entries["untracked.txt"] == {
        "path": "untracked.txt",
        "kind": "untracked",
        "index_status": None,
        "worktree_status": "?",
        "staged": False,
        "worktree": True,
    }
    assert entries["staged-new.txt"] == {
        "path": "staged-new.txt",
        "kind": "added",
        "index_status": "A",
        "worktree_status": None,
        "staged": True,
        "worktree": False,
    }
    assert entries["deleted.txt"] == {
        "path": "deleted.txt",
        "kind": "deleted",
        "index_status": None,
        "worktree_status": "D",
        "staged": False,
        "worktree": True,
    }


def test_real_git_diff_is_empty_for_clean_repository(git_repository: Path) -> None:
    working = execute_git(git_repository, "git.diff", {"staged": False})
    staged = execute_git(git_repository, "git.diff", {"staged": True})

    assert output_mapping(working) == {
        "staged": False,
        "diff": "",
        "truncated": False,
    }
    assert output_mapping(staged) == {
        "staged": True,
        "diff": "",
        "truncated": False,
    }


def test_real_git_diff_keeps_worktree_and_staged_scopes_distinct(
    git_repository: Path,
) -> None:
    (git_repository / "working-one.txt").write_text("working one changed\n")
    (git_repository / "working-two.txt").write_text("working two changed\n")
    (git_repository / "staged.txt").write_text("staged changed\n")
    run_git(git_repository, "add", "staged.txt")

    working = execute_git(git_repository, "git.diff", {"staged": False})
    staged = execute_git(git_repository, "git.diff", {"staged": True})
    working_output = output_mapping(working)
    staged_output = output_mapping(staged)
    working_diff = working_output["diff"]
    staged_diff = staged_output["diff"]

    assert working.status is ToolResultStatus.SUCCESS
    assert staged.status is ToolResultStatus.SUCCESS
    assert isinstance(working_diff, str)
    assert isinstance(staged_diff, str)
    assert working_output["staged"] is False
    assert staged_output["staged"] is True
    assert working_output["truncated"] is False
    assert staged_output["truncated"] is False
    assert "diff --git a/working-one.txt b/working-one.txt" in working_diff
    assert "diff --git a/working-two.txt b/working-two.txt" in working_diff
    assert "+working one changed" in working_diff
    assert "+working two changed" in working_diff
    assert "staged.txt" not in working_diff
    assert "diff --git a/staged.txt b/staged.txt" in staged_diff
    assert "+staged changed" in staged_diff
    assert "working-one.txt" not in staged_diff
    assert "working-two.txt" not in staged_diff


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
