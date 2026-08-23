"""Explicit composition for Forge's built-in repository tool modes."""

from __future__ import annotations

from forge.tools.git import GitDiffTool, GitStatusTool
from forge.tools.permissions import RuleBasedPolicy
from forge.tools.registry import ToolRegistry
from forge.tools.repository import ListDirectoryTool, ReadFileTool, SearchFilesTool
from forge.tools.repository_write import ApplyPatchTool, WriteFileTool
from forge.tools.types import PermissionDecision

READ_ONLY_TOOL_NAMES = (
    "git.diff",
    "git.status",
    "repository.list_directory",
    "repository.read_file",
    "repository.search_files",
)
WRITE_TOOL_NAMES = ("repository.apply_patch", "repository.write_file")


def create_readonly_repository_registry() -> ToolRegistry:
    """Create a fresh registry containing only A6 read-only capabilities."""
    return ToolRegistry(
        (
            ListDirectoryTool(),
            ReadFileTool(),
            SearchFilesTool(),
            GitStatusTool(),
            GitDiffTool(),
        )
    )


def create_readonly_repository_policy(
    decision: PermissionDecision = PermissionDecision.ALLOW,
) -> RuleBasedPolicy:
    """Apply one explicit decision to exactly the built-in read-only tools."""
    return RuleBasedPolicy({name: decision for name in READ_ONLY_TOOL_NAMES})


def create_assist_repository_registry() -> ToolRegistry:
    """Create the explicit A6 read plus A9 controlled-write registry."""
    return ToolRegistry(
        (
            ListDirectoryTool(),
            ReadFileTool(),
            SearchFilesTool(),
            GitStatusTool(),
            GitDiffTool(),
            WriteFileTool(),
            ApplyPatchTool(),
        )
    )


def create_assist_repository_policy() -> RuleBasedPolicy:
    """Allow reads, require approval for writes, and deny unknown tools."""
    rules = {name: PermissionDecision.ALLOW for name in READ_ONLY_TOOL_NAMES}
    rules.update({name: PermissionDecision.ASK for name in WRITE_TOOL_NAMES})
    return RuleBasedPolicy(rules)
