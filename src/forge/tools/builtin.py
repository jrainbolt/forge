"""Explicit composition for Forge's built-in repository tool modes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.interaction import InteractionPolicy
    from forge.lexical_index import RepositoryLexicalIndex
    from forge.repository_index import RepositoryIndex
    from forge.semantic_index import SemanticIndex
from forge.project_config import ProjectCommands
from forge.tools.git import GitDiffTool, GitStatusTool
from forge.tools.lexical import LexicalSearchTool
from forge.tools.permissions import RuleBasedPolicy
from forge.tools.project import ProjectCommandTool
from forge.tools.registry import ToolRegistry
from forge.tools.repository import ListDirectoryTool, ReadFileTool, SearchFilesTool
from forge.tools.repository_analysis import (
    FileOutlineTool,
    FindReferencesTool,
    FindSymbolTool,
    ReadRangeTool,
)
from forge.tools.repository_write import ApplyPatchTool, WriteFileTool
from forge.tools.semantic import SemanticSearchTool
from forge.tools.types import PermissionDecision

READ_ONLY_TOOL_NAMES = (
    "git.diff",
    "git.status",
    "repository.file_outline",
    "repository.find_references",
    "repository.find_symbol",
    "repository.list_directory",
    "repository.read_file",
    "repository.read_range",
    "repository.search_files",
)
SEMANTIC_TOOL_NAME = "repository.semantic_search"
WRITE_TOOL_NAMES = ("repository.apply_patch", "repository.write_file")
PROJECT_TOOL_NAMES = ("project.build", "project.test")


def create_readonly_repository_registry(
    index: RepositoryIndex | None = None,
    semantic_index: SemanticIndex | None = None,
    lexical_index: RepositoryLexicalIndex | None = None,
) -> ToolRegistry:
    """Create a fresh registry containing only A6 read-only capabilities."""
    candidates = [
        ListDirectoryTool(),
        ReadFileTool(),
        SearchFilesTool(),
        FileOutlineTool(index),
        FindSymbolTool(index),
        FindReferencesTool(index),
        ReadRangeTool(),
        GitStatusTool(),
        GitDiffTool(),
    ]
    if semantic_index is not None:
        candidates.append(SemanticSearchTool(semantic_index))
    if lexical_index is not None:
        candidates.append(LexicalSearchTool(lexical_index))
    return ToolRegistry(candidates)


def create_readonly_repository_policy(
    decision: PermissionDecision = PermissionDecision.ALLOW,
) -> RuleBasedPolicy:
    """Apply one explicit decision to exactly the built-in read-only tools."""
    rules = {name: decision for name in READ_ONLY_TOOL_NAMES}
    rules["repository.lexical_search"] = decision
    rules[SEMANTIC_TOOL_NAME] = decision
    return RuleBasedPolicy(rules)


def create_assist_repository_registry(
    commands: ProjectCommands | None = None,
    index: RepositoryIndex | None = None,
    semantic_index: SemanticIndex | None = None,
    lexical_index: RepositoryLexicalIndex | None = None,
) -> ToolRegistry:
    """Create the explicit A6 read plus A9 controlled-write registry."""
    configured = commands or ProjectCommands()
    candidates = [
        ListDirectoryTool(),
        ReadFileTool(),
        SearchFilesTool(),
        FileOutlineTool(index),
        FindSymbolTool(index),
        FindReferencesTool(index),
        ReadRangeTool(),
        GitStatusTool(),
        GitDiffTool(),
        WriteFileTool(),
        ApplyPatchTool(),
        ProjectCommandTool("build", configured.build),
        ProjectCommandTool("test", configured.test),
    ]
    if semantic_index is not None:
        candidates.append(SemanticSearchTool(semantic_index))
    if lexical_index is not None:
        candidates.append(LexicalSearchTool(lexical_index))
    return ToolRegistry(candidates)


def create_assist_repository_policy() -> RuleBasedPolicy:
    """Allow reads, require approval for writes, and deny unknown tools."""
    rules = {name: PermissionDecision.ALLOW for name in READ_ONLY_TOOL_NAMES}
    rules["repository.lexical_search"] = PermissionDecision.ALLOW
    rules[SEMANTIC_TOOL_NAME] = PermissionDecision.ALLOW
    rules.update({name: PermissionDecision.ASK for name in WRITE_TOOL_NAMES})
    rules.update({name: PermissionDecision.ASK for name in PROJECT_TOOL_NAMES})
    return RuleBasedPolicy(rules)


def create_repository_registry(
    interaction: InteractionPolicy,
    commands: ProjectCommands | None = None,
    index: RepositoryIndex | None = None,
    semantic_index: SemanticIndex | None = None,
    lexical_index: RepositoryLexicalIndex | None = None,
) -> ToolRegistry:
    """Create one fixed registry under autonomy ceiling and permanent DENYs."""
    configured = commands or ProjectCommands()
    candidates = [
        ListDirectoryTool(),
        ReadFileTool(),
        SearchFilesTool(),
        FileOutlineTool(index),
        FindSymbolTool(index),
        FindReferencesTool(index),
        ReadRangeTool(),
        GitStatusTool(),
        GitDiffTool(),
        WriteFileTool(),
        ApplyPatchTool(),
        ProjectCommandTool("build", configured.build),
        ProjectCommandTool("test", configured.test),
    ]
    if semantic_index is not None:
        candidates.append(SemanticSearchTool(semantic_index))
    if lexical_index is not None:
        candidates.append(LexicalSearchTool(lexical_index))
    return ToolRegistry(
        tool for tool in candidates if interaction.exposes(tool.metadata)
    )
