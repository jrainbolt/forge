from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from forge.tools import (
    READ_ONLY_TOOL_NAMES,
    AllowAllPolicy,
    ExecutionContext,
    InvocationApproval,
    PermissionDecision,
    PermissionPolicy,
    ReadFileTool,
    RuleBasedPolicy,
    ToolExecutor,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    create_readonly_repository_registry,
    resolve_workspace_path,
)
from forge.tools.paths import WorkspacePathError
from forge.tools.repository import MAX_READ_BYTES, MAX_SEARCH_FILE_BYTES


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def invoke(
    workspace: Path,
    tool_name: str,
    arguments: Mapping[str, object],
    *,
    policy: PermissionPolicy | None = None,
    invocation_id: str = "repository-test",
    approval: InvocationApproval | None = None,
) -> ToolResult:
    executor = ToolExecutor(
        create_readonly_repository_registry(),
        policy or AllowAllPolicy(),
    )
    return executor.execute(
        ToolInvocation(invocation_id, tool_name, arguments),
        ExecutionContext(workspace.resolve()),
        approval=approval,
    )


def output_mapping(result: ToolResult) -> Mapping[str, object]:
    assert isinstance(result.output, Mapping)
    return result.output


def make_symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")


def test_workspace_resolver_root_nested_and_dot_segments(workspace: Path) -> None:
    nested = workspace / "src"
    nested.mkdir()
    file_path = workspace / "file.txt"
    file_path.write_text("content")
    assert resolve_workspace_path(workspace.resolve(), ".") == workspace.resolve()
    assert resolve_workspace_path(workspace.resolve(), "src") == nested.resolve()
    assert (
        resolve_workspace_path(workspace.resolve(), "src/../file.txt")
        == file_path.resolve()
    )


def test_workspace_resolver_rejects_missing_and_absolute_paths(workspace: Path) -> None:
    with pytest.raises(WorkspacePathError, match="does not exist"):
        resolve_workspace_path(workspace.resolve(), "missing")
    with pytest.raises(WorkspacePathError, match="absolute"):
        resolve_workspace_path(workspace.resolve(), str((workspace / "x").resolve()))


def test_workspace_resolver_blocks_traversal_and_similar_prefix_sibling(
    workspace: Path,
) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("secret")
    sibling = workspace.parent / "workspace-secret"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("secret")
    with pytest.raises(WorkspacePathError, match="outside"):
        resolve_workspace_path(workspace.resolve(), "../outside.txt")
    with pytest.raises(WorkspacePathError, match="outside"):
        resolve_workspace_path(workspace.resolve(), "../workspace-secret/secret.txt")


def test_workspace_resolver_allows_internal_file_and_directory_symlinks(
    workspace: Path,
) -> None:
    target_directory = workspace / "target"
    target_directory.mkdir()
    target_file = target_directory / "file.txt"
    target_file.write_text("inside")
    file_link = workspace / "file-link"
    directory_link = workspace / "directory-link"
    make_symlink(file_link, target_file)
    make_symlink(directory_link, target_directory, directory=True)
    assert resolve_workspace_path(workspace.resolve(), "file-link") == target_file
    assert (
        resolve_workspace_path(workspace.resolve(), "directory-link/file.txt")
        == target_file
    )


def test_workspace_resolver_blocks_external_and_nested_symlink_escape(
    workspace: Path,
) -> None:
    outside_directory = workspace.parent / "private"
    outside_directory.mkdir()
    outside_file = outside_directory / "secret.txt"
    outside_file.write_text("secret")
    file_link = workspace / "external-file"
    directory_link = workspace / "external-directory"
    chained_link = workspace / "chained"
    make_symlink(file_link, outside_file)
    make_symlink(directory_link, outside_directory, directory=True)
    make_symlink(chained_link, directory_link, directory=True)
    for requested in (
        "external-file",
        "external-directory/secret.txt",
        "chained/secret.txt",
    ):
        with pytest.raises(WorkspacePathError, match="outside"):
            resolve_workspace_path(workspace.resolve(), requested)


def test_list_directory_is_sorted_structured_and_includes_hidden_files(
    workspace: Path,
) -> None:
    (workspace / "z.txt").write_text("z")
    (workspace / ".hidden").write_text("hidden")
    (workspace / "directory").mkdir()
    make_symlink(workspace / "link", workspace / "z.txt")
    result = invoke(workspace, "repository.list_directory", {"path": "."})
    assert result.status is ToolResultStatus.SUCCESS
    output = output_mapping(result)
    entries = output["entries"]
    assert isinstance(entries, tuple)
    assert [entry["name"] for entry in entries] == [  # type: ignore[index]
        ".hidden",
        "directory",
        "link",
        "z.txt",
    ]
    assert [entry["type"] for entry in entries] == [  # type: ignore[index]
        "file",
        "directory",
        "symlink",
        "file",
    ]
    assert output["path"] == "."


def test_list_directory_nested_internal_symlink_and_failures(workspace: Path) -> None:
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "file.txt").write_text("content")
    make_symlink(workspace / "nested-link", nested, directory=True)
    result = invoke(workspace, "repository.list_directory", {"path": "nested-link"})
    assert output_mapping(result)["path"] == "nested"
    assert (
        invoke(
            workspace, "repository.list_directory", {"path": "nested/file.txt"}
        ).status
        is ToolResultStatus.FAILURE
    )
    assert (
        invoke(workspace, "repository.list_directory", {"path": "missing"}).status
        is ToolResultStatus.FAILURE
    )
    assert (
        invoke(workspace, "repository.list_directory", {"path": "../outside"}).status
        is ToolResultStatus.FAILURE
    )


def test_list_directory_rejects_external_directory_symlink(workspace: Path) -> None:
    outside = workspace.parent / "outside-directory"
    outside.mkdir()
    make_symlink(workspace / "escape", outside, directory=True)
    assert (
        invoke(workspace, "repository.list_directory", {"path": "escape"}).status
        is ToolResultStatus.FAILURE
    )


def test_read_file_handles_utf8_empty_bom_nested_and_internal_symlink(
    workspace: Path,
) -> None:
    nested = workspace / "src"
    nested.mkdir()
    source = nested / "main.py"
    source.write_text("print('héllo')\n", encoding="utf-8")
    empty = workspace / "empty.txt"
    empty.write_bytes(b"")
    bom = workspace / "bom.txt"
    bom.write_bytes(b"\xef\xbb\xbfcontent")
    make_symlink(workspace / "source-link", source)
    for requested, expected in (
        ("src/main.py", "print('héllo')\n"),
        ("empty.txt", ""),
        ("bom.txt", "content"),
        ("source-link", "print('héllo')\n"),
    ):
        result = invoke(workspace, "repository.read_file", {"path": requested})
        assert result.status is ToolResultStatus.SUCCESS
        assert output_mapping(result)["content"] == expected
    assert (
        output_mapping(
            invoke(workspace, "repository.read_file", {"path": "source-link"})
        )["path"]
        == "src/main.py"
    )


def test_read_file_rejects_missing_directory_binary_and_oversized(
    workspace: Path,
) -> None:
    directory = workspace / "directory"
    directory.mkdir()
    (workspace / "binary").write_bytes(b"\xff\xfe")
    (workspace / "large").write_bytes(b"x" * (MAX_READ_BYTES + 1))
    for requested in ("missing", "directory", "binary", "large"):
        assert (
            invoke(workspace, "repository.read_file", {"path": requested}).status
            is ToolResultStatus.FAILURE
        )


def test_read_file_security_regressions(workspace: Path) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("secret")
    make_symlink(workspace / "escape", outside)
    for requested in ("../outside.txt", str(outside.resolve()), "escape"):
        result = invoke(workspace, "repository.read_file", {"path": requested})
        assert result.status is ToolResultStatus.FAILURE
        assert "secret" not in str(result.output)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation unavailable")
def test_read_file_rejects_special_file_without_opening_it(workspace: Path) -> None:
    fifo = workspace / "fifo"
    os.mkfifo(fifo)
    result = invoke(workspace, "repository.read_file", {"path": "fifo"})
    assert result.status is ToolResultStatus.FAILURE


def test_search_returns_deterministic_paths_lines_and_case_behavior(
    workspace: Path,
) -> None:
    (workspace / "b.txt").write_text("Needle second\n")
    directory = workspace / "a"
    directory.mkdir()
    (directory / "a.txt").write_text("needle first\nother\nNeedle third\n")
    sensitive = invoke(
        workspace,
        "repository.search_files",
        {"query": "Needle", "case_sensitive": True},
    )
    sensitive_matches = output_mapping(sensitive)["matches"]
    assert [(item["path"], item["line_number"]) for item in sensitive_matches] == [  # type: ignore[index]
        ("b.txt", 1),
        ("a/a.txt", 3),
    ]
    insensitive = invoke(
        workspace,
        "repository.search_files",
        {"query": "needle", "case_sensitive": False},
    )
    assert len(output_mapping(insensitive)["matches"]) == 3  # type: ignore[arg-type]


def test_search_scope_limit_hidden_and_ignored_directories(workspace: Path) -> None:
    src = workspace / "src"
    src.mkdir()
    (src / "one.txt").write_text("match\nmatch\n")
    (workspace / "root.txt").write_text("match\n")
    hidden = workspace / ".github"
    hidden.mkdir()
    (hidden / "workflow.yml").write_text("match\n")
    ignored = workspace / ".git"
    ignored.mkdir()
    (ignored / "secret").write_text("match\n")
    result = invoke(
        workspace,
        "repository.search_files",
        {"query": "match", "max_results": 2},
    )
    output = output_mapping(result)
    assert len(output["matches"]) == 2  # type: ignore[arg-type]
    assert output["limit_reached"] is True
    scoped = invoke(
        workspace,
        "repository.search_files",
        {"query": "match", "path": "src"},
    )
    scoped_matches = output_mapping(scoped)["matches"]
    assert {item["path"] for item in scoped_matches} == {"src/one.txt"}  # type: ignore[index]
    all_result = invoke(workspace, "repository.search_files", {"query": "match"})
    paths = {item["path"] for item in output_mapping(all_result)["matches"]}  # type: ignore[index]
    assert ".github/workflow.yml" in paths
    assert ".git/secret" not in paths


def test_search_skips_binary_oversized_and_external_symlink(workspace: Path) -> None:
    (workspace / "binary").write_bytes(b"match\xff")
    (workspace / "large").write_bytes(b"match" + b"x" * MAX_SEARCH_FILE_BYTES)
    outside = workspace.parent / "outside.txt"
    outside.write_text("match")
    make_symlink(workspace / "escape", outside)
    result = invoke(workspace, "repository.search_files", {"query": "match"})
    output = output_mapping(result)
    assert output["matches"] == ()
    assert output["skipped_files"] == 2


def test_search_rejects_traversal_absolute_and_external_symlink_root(
    workspace: Path,
) -> None:
    outside = workspace.parent / "outside"
    outside.mkdir()
    (outside / "secret").write_text("match")
    make_symlink(workspace / "escape", outside, directory=True)
    for requested in ("../outside", str(outside.resolve()), "escape"):
        assert (
            invoke(
                workspace,
                "repository.search_files",
                {"query": "match", "path": requested},
            ).status
            is ToolResultStatus.FAILURE
        )


def test_search_does_not_recurse_forever_through_internal_symlink_loop(
    workspace: Path,
) -> None:
    directory = workspace / "directory"
    directory.mkdir()
    (directory / "file.txt").write_text("match")
    make_symlink(directory / "loop", workspace, directory=True)
    result = invoke(workspace, "repository.search_files", {"query": "match"})
    assert len(output_mapping(result)["matches"]) == 1  # type: ignore[arg-type]


def test_search_validates_result_limit(workspace: Path) -> None:
    (workspace / "file.txt").write_text("match")
    for limit in (0, 101):
        assert (
            invoke(
                workspace,
                "repository.search_files",
                {"query": "match", "max_results": limit},
            ).status
            is ToolResultStatus.FAILURE
        )


def test_every_repository_tool_executes_through_real_pipeline(workspace: Path) -> None:
    (workspace / "file.txt").write_text("needle")
    for name, arguments in (
        ("repository.list_directory", {"path": "."}),
        ("repository.read_file", {"path": "file.txt"}),
        ("repository.search_files", {"query": "needle"}),
    ):
        result = invoke(workspace, name, arguments)
        assert result.status is ToolResultStatus.SUCCESS
        assert result.metadata.permission_decision is PermissionDecision.ALLOW


def test_builtin_registry_contains_exactly_the_read_only_capabilities() -> None:
    registry = create_readonly_repository_registry()
    assert tuple(tool.name for tool in registry.metadata) == READ_ONLY_TOOL_NAMES
    assert all(tool.risk is ToolRisk.READ_ONLY for tool in registry.metadata)


def test_denied_real_repository_tool_is_not_executed(workspace: Path) -> None:
    class SpyReadFileTool(ReadFileTool):
        called = False

        def execute(self, arguments, context):  # type: ignore[no-untyped-def]
            self.called = True
            return super().execute(arguments, context)

    (workspace / "file.txt").write_text("content")
    tool = SpyReadFileTool()
    result = ToolExecutor(
        ToolRegistry((tool,)),
        RuleBasedPolicy({"repository.read_file": PermissionDecision.DENY}),
    ).execute(
        ToolInvocation("denied-read", "repository.read_file", {"path": "file.txt"}),
        ExecutionContext(workspace.resolve()),
    )
    assert result.status is ToolResultStatus.DENIED
    assert tool.called is False


def test_real_repository_tool_deny_and_ask_preserve_approval_integrity(
    workspace: Path,
) -> None:
    (workspace / "file.txt").write_text("content")
    denied = invoke(
        workspace,
        "repository.read_file",
        {"path": "file.txt"},
        policy=RuleBasedPolicy({"repository.read_file": PermissionDecision.DENY}),
    )
    assert denied.status is ToolResultStatus.DENIED

    policy = RuleBasedPolicy({"repository.read_file": PermissionDecision.ASK})
    request = ToolInvocation(
        "approved-read", "repository.read_file", {"path": "file.txt"}
    )
    unapproved = invoke(
        workspace,
        request.tool_name,
        request.arguments,
        policy=policy,
        invocation_id=request.invocation_id,
    )
    assert unapproved.status is ToolResultStatus.APPROVAL_REQUIRED
    approved = invoke(
        workspace,
        request.tool_name,
        request.arguments,
        policy=policy,
        invocation_id=request.invocation_id,
        approval=InvocationApproval.for_invocation(request),
    )
    assert approved.status is ToolResultStatus.SUCCESS
    changed = invoke(
        workspace,
        request.tool_name,
        {"path": "."},
        policy=policy,
        invocation_id=request.invocation_id,
        approval=InvocationApproval.for_invocation(request),
    )
    assert changed.status is ToolResultStatus.APPROVAL_REQUIRED
