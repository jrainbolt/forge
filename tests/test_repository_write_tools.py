from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from forge.tools import (
    ApplyPatchTool,
    ExecutionContext,
    InvocationApproval,
    PermissionDecision,
    RuleBasedPolicy,
    ToolExecutor,
    ToolInvocation,
    ToolRegistry,
    ToolResultStatus,
    WriteFileTool,
    create_assist_repository_policy,
    create_assist_repository_registry,
    create_readonly_repository_policy,
    preview_repository_mutation,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    return root


def invoke(
    executor: ToolExecutor,
    workspace: Path,
    tool: str,
    arguments: dict[str, object],
    *,
    approval: bool = True,
    invocation_id: str = "write-1",
):
    request = ToolInvocation(invocation_id, tool, arguments)
    return executor.execute(
        request,
        ExecutionContext(workspace.resolve()),
        approval=InvocationApproval.for_invocation(request) if approval else None,
    )


def write_executor(
    decision: PermissionDecision = PermissionDecision.ASK,
) -> ToolExecutor:
    return ToolExecutor(
        ToolRegistry((WriteFileTool(), ApplyPatchTool())),
        RuleBasedPolicy(
            {
                "repository.write_file": decision,
                "repository.apply_patch": decision,
            }
        ),
    )


def test_create_requires_approval_and_exact_approval_creates_utf8_file(
    workspace: Path,
) -> None:
    arguments = {"path": "src/new.py", "content": "name = 'café'\n", "mode": "create"}
    executor = write_executor()
    pending = invoke(
        executor, workspace, "repository.write_file", arguments, approval=False
    )
    assert pending.status is ToolResultStatus.APPROVAL_REQUIRED
    assert not (workspace / "src/new.py").exists()

    changed = ToolInvocation(
        "write-1",
        "repository.write_file",
        {**arguments, "content": "different\n"},
    )
    original = ToolInvocation("write-1", "repository.write_file", arguments)
    denied = executor.execute(
        changed,
        ExecutionContext(workspace.resolve()),
        approval=InvocationApproval.for_invocation(original),
    )
    assert denied.status is ToolResultStatus.APPROVAL_REQUIRED
    assert not (workspace / "src/new.py").exists()

    result = invoke(executor, workspace, "repository.write_file", arguments)
    expected = "name = 'café'\n".encode()
    assert result.status is ToolResultStatus.SUCCESS
    assert (workspace / "src/new.py").read_bytes() == expected
    assert result.output["path"] == "src/new.py"  # type: ignore[index]
    assert result.output["new_sha256"] == digest(expected)  # type: ignore[index]
    assert result.output["verified"] is True  # type: ignore[index]


def test_nested_patch_arguments_are_frozen_for_exact_approval(workspace: Path) -> None:
    target = workspace / "src/value.py"
    original = b"VALUE = 1\n"
    target.write_bytes(original)
    edits = [{"old": "VALUE = 1", "new": "VALUE = 2"}]
    invocation = ToolInvocation(
        "patch",
        "repository.apply_patch",
        {
            "path": "src/value.py",
            "expected_sha256": digest(original),
            "edits": edits,
        },
    )
    approval = InvocationApproval.for_invocation(invocation)
    edits[0]["new"] = "VALUE = 99"
    executor = write_executor()
    result = executor.execute(
        invocation, ExecutionContext(workspace.resolve()), approval=approval
    )
    assert result.status is ToolResultStatus.SUCCESS
    assert target.read_bytes() == b"VALUE = 2\n"


def test_create_rejects_existing_missing_parent_oversize_and_expected_hash(
    workspace: Path,
) -> None:
    target = workspace / "src/existing.py"
    target.write_text("old")
    cases = (
        {"path": "src/existing.py", "content": "new", "mode": "create"},
        {"path": "missing/new.py", "content": "new", "mode": "create"},
        {"path": "src/large.py", "content": "x" * (256 * 1024 + 1), "mode": "create"},
        {
            "path": "src/new.py",
            "content": "new",
            "mode": "create",
            "expected_sha256": "0" * 64,
        },
    )
    for index, arguments in enumerate(cases):
        result = invoke(
            write_executor(),
            workspace,
            "repository.write_file",
            arguments,
            invocation_id=f"case-{index}",
        )
        assert result.status is ToolResultStatus.FAILURE
    assert target.read_text() == "old"
    assert not (workspace / "src/new.py").exists()


def test_replace_requires_matching_hash_is_atomic_and_preserves_mode(
    workspace: Path,
) -> None:
    target = workspace / "src/tool.py"
    target.write_bytes(b"old\r\n")
    target.chmod(0o755)
    arguments = {
        "path": "src/tool.py",
        "content": "new\n",
        "mode": "replace",
        "expected_sha256": digest(b"old\r\n"),
    }
    result = invoke(write_executor(), workspace, "repository.write_file", arguments)
    assert result.status is ToolResultStatus.SUCCESS
    assert target.read_bytes() == b"new\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert not list(target.parent.glob(".forge-write-*"))


def test_replace_stale_hash_and_changed_after_observation_leave_file_unchanged(
    workspace: Path,
) -> None:
    target = workspace / "src/tool.py"
    target.write_bytes(b"human change")
    arguments = {
        "path": "src/tool.py",
        "content": "model change",
        "mode": "replace",
        "expected_sha256": digest(b"old observed"),
    }
    result = invoke(write_executor(), workspace, "repository.write_file", arguments)
    assert result.status is ToolResultStatus.FAILURE
    assert "precondition" in (result.error_message or "")
    assert target.read_bytes() == b"human change"


@pytest.mark.parametrize("requested", ("../outside.txt", "/tmp/outside.txt"))
def test_write_rejects_traversal_and_absolute_paths(
    workspace: Path, requested: str, tmp_path: Path
) -> None:
    result = invoke(
        write_executor(),
        workspace,
        "repository.write_file",
        {"path": requested, "content": "bad", "mode": "create"},
    )
    assert result.status is ToolResultStatus.FAILURE
    assert not (tmp_path / "outside.txt").exists()


def test_write_rejects_similar_prefix_and_external_symlink_paths(
    workspace: Path, tmp_path: Path
) -> None:
    sibling = tmp_path / "workspace-other"
    sibling.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "real.py").write_text("outside")
    (workspace / "external-file.py").symlink_to(outside / "real.py")
    (workspace / "external-dir").symlink_to(outside, target_is_directory=True)
    (workspace / "src/nested").symlink_to(
        workspace / "external-dir", target_is_directory=True
    )
    cases = (
        "../workspace-other/new.py",
        "external-file.py",
        "external-dir/new.py",
        "src/nested/new.py",
    )
    for index, requested in enumerate(cases):
        result = invoke(
            write_executor(),
            workspace,
            "repository.write_file",
            {"path": requested, "content": "bad", "mode": "create"},
            invocation_id=f"escape-{index}",
        )
        assert result.status is ToolResultStatus.FAILURE
    assert (outside / "real.py").read_text() == "outside"
    assert not (outside / "new.py").exists()
    assert not (sibling / "new.py").exists()


def test_internal_symlink_parent_is_confined_but_final_symlink_is_rejected(
    workspace: Path,
) -> None:
    (workspace / "alias").symlink_to(workspace / "src", target_is_directory=True)
    created = invoke(
        write_executor(),
        workspace,
        "repository.write_file",
        {"path": "alias/new.py", "content": "safe", "mode": "create"},
    )
    assert created.status is ToolResultStatus.SUCCESS
    assert (workspace / "src/new.py").read_text() == "safe"
    (workspace / "link.py").symlink_to(workspace / "src/new.py")
    rejected = invoke(
        write_executor(),
        workspace,
        "repository.write_file",
        {
            "path": "link.py",
            "content": "changed",
            "mode": "replace",
            "expected_sha256": digest(b"safe"),
        },
        invocation_id="link",
    )
    assert rejected.status is ToolResultStatus.FAILURE
    assert (workspace / "src/new.py").read_text() == "safe"


def test_git_metadata_and_special_files_are_never_written(workspace: Path) -> None:
    (workspace / ".git").mkdir()
    (workspace / ".git/config").write_text("safe")
    git_result = invoke(
        write_executor(),
        workspace,
        "repository.write_file",
        {
            "path": ".git/config",
            "content": "bad",
            "mode": "replace",
            "expected_sha256": digest(b"safe"),
        },
    )
    assert git_result.status is ToolResultStatus.FAILURE
    assert (workspace / ".git/config").read_text() == "safe"
    if hasattr(os, "mkfifo"):
        fifo = workspace / "src/channel"
        os.mkfifo(fifo)
        special = invoke(
            write_executor(),
            workspace,
            "repository.write_file",
            {
                "path": "src/channel",
                "content": "bad",
                "mode": "replace",
                "expected_sha256": "0" * 64,
            },
            invocation_id="fifo",
        )
        assert special.status is ToolResultStatus.FAILURE


def test_patch_applies_multiple_exact_edits_and_reports_hash(workspace: Path) -> None:
    target = workspace / "src/value.py"
    original = b"FIRST = 1\nSECOND = 2\n"
    target.write_bytes(original)
    arguments = {
        "path": "src/value.py",
        "expected_sha256": digest(original),
        "edits": [
            {"old": "FIRST = 1", "new": "FIRST = 10"},
            {"old": "SECOND = 2", "new": "SECOND = 20"},
        ],
    }
    result = invoke(write_executor(), workspace, "repository.apply_patch", arguments)
    expected = b"FIRST = 10\nSECOND = 20\n"
    assert result.status is ToolResultStatus.SUCCESS
    assert target.read_bytes() == expected
    assert result.output["new_sha256"] == digest(expected)  # type: ignore[index]
    assert not list(target.parent.glob(".forge-write-*"))


@pytest.mark.parametrize(
    "edits",
    (
        [{"old": "missing", "new": "new"}],
        [{"old": "same", "new": "same"}],
        [{"old": "", "new": "new"}],
        [],
    ),
)
def test_patch_conflicts_and_noops_are_atomic(
    workspace: Path, edits: list[dict[str, str]]
) -> None:
    target = workspace / "src/value.py"
    original = b"same same\n"
    target.write_bytes(original)
    result = invoke(
        write_executor(),
        workspace,
        "repository.apply_patch",
        {
            "path": "src/value.py",
            "expected_sha256": digest(original),
            "edits": edits,
        },
    )
    assert result.status is ToolResultStatus.FAILURE
    assert target.read_bytes() == original
    assert not list(target.parent.glob(".forge-write-*"))


def test_patch_rejects_ambiguous_text_stale_hash_and_escape(workspace: Path) -> None:
    target = workspace / "src/value.py"
    original = b"repeat repeat"
    target.write_bytes(original)
    for index, arguments in enumerate(
        (
            {
                "path": "src/value.py",
                "expected_sha256": digest(original),
                "edits": [{"old": "repeat", "new": "changed"}],
            },
            {
                "path": "src/value.py",
                "expected_sha256": "0" * 64,
                "edits": [{"old": "repeat repeat", "new": "changed"}],
            },
            {
                "path": "../outside.py",
                "expected_sha256": digest(original),
                "edits": [{"old": "repeat", "new": "changed"}],
            },
        )
    ):
        result = invoke(
            write_executor(),
            workspace,
            "repository.apply_patch",
            arguments,
            invocation_id=f"conflict-{index}",
        )
        assert result.status is ToolResultStatus.FAILURE
    assert target.read_bytes() == original


def test_oversized_patch_result_is_rejected_atomically(workspace: Path) -> None:
    target = workspace / "src/value.py"
    original = b"small"
    target.write_bytes(original)
    result = invoke(
        write_executor(),
        workspace,
        "repository.apply_patch",
        {
            "path": "src/value.py",
            "expected_sha256": digest(original),
            "edits": [{"old": "small", "new": "x" * (256 * 1024 + 1)}],
        },
    )
    assert result.status is ToolResultStatus.FAILURE
    assert target.read_bytes() == original


def test_preview_is_deterministic_contains_diff_and_never_mutates(
    workspace: Path,
) -> None:
    target = workspace / "src/value.py"
    original = b"VALUE = 1\n"
    target.write_bytes(original)
    context = ExecutionContext(workspace.resolve())
    arguments = {
        "path": "src/value.py",
        "expected_sha256": digest(original),
        "edits": [{"old": "VALUE = 1", "new": "VALUE = 2"}],
    }
    first = preview_repository_mutation("repository.apply_patch", arguments, context)
    second = preview_repository_mutation("repository.apply_patch", arguments, context)
    assert first == second
    assert "-VALUE = 1" in first.diff
    assert "+VALUE = 2" in first.diff
    assert target.read_bytes() == original

    create_arguments = {
        "path": "src/empty.py",
        "content": "",
        "mode": "create",
    }
    create_preview = preview_repository_mutation(
        "repository.write_file", create_arguments, context
    )
    assert create_preview.operation == "create"
    assert "empty content" in create_preview.diff
    assert not (workspace / "src/empty.py").exists()


def test_readonly_deny_and_assist_policies_separate_read_from_write(
    workspace: Path,
) -> None:
    target = workspace / "src/value.py"
    target.write_text("safe")
    registry = create_assist_repository_registry()
    invocation = ToolInvocation(
        "write",
        "repository.write_file",
        {
            "path": "src/value.py",
            "content": "bad",
            "mode": "replace",
            "expected_sha256": digest(b"safe"),
        },
    )
    context = ExecutionContext(workspace.resolve())
    readonly = ToolExecutor(registry, create_readonly_repository_policy())
    assert readonly.execute(invocation, context).status is ToolResultStatus.DENIED
    assert target.read_text() == "safe"
    assist = ToolExecutor(registry, create_assist_repository_policy())
    assert (
        assist.execute(invocation, context).status is ToolResultStatus.APPROVAL_REQUIRED
    )
    approved = assist.execute(
        invocation, context, approval=InvocationApproval.for_invocation(invocation)
    )
    assert approved.status is ToolResultStatus.SUCCESS
    assert target.read_text() == "bad"
