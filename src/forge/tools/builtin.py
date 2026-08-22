"""Explicit composition for Forge's built-in read-only tools."""

from __future__ import annotations

from forge.tools.git import GitDiffTool, GitStatusTool
from forge.tools.permissions import RuleBasedPolicy
from forge.tools.registry import ToolRegistry
from forge.tools.repository import ListDirectoryTool, ReadFileTool, SearchFilesTool
from forge.tools.types import PermissionDecision

READ_ONLY_TOOL_NAMES = (
    "git.diff",
    "git.status",
    "repository.list_directory",
    "repository.read_file",
    "repository.search_files",
)


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
