"""Explicit permission policies for tool invocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType

from forge.tools.tool import Tool
from forge.tools.types import (
    ExecutionContext,
    PermissionDecision,
    ToolInvocation,
    validate_tool_name,
)


class PermissionPolicy(ABC):
    @abstractmethod
    def evaluate(
        self, tool: Tool, invocation: ToolInvocation, context: ExecutionContext
    ) -> PermissionDecision:
        """Return an explicit decision without executing the tool."""


class AllowAllPolicy(PermissionPolicy):
    def evaluate(
        self, tool: Tool, invocation: ToolInvocation, context: ExecutionContext
    ) -> PermissionDecision:
        return PermissionDecision.ALLOW


class DenyAllPolicy(PermissionPolicy):
    def evaluate(
        self, tool: Tool, invocation: ToolInvocation, context: ExecutionContext
    ) -> PermissionDecision:
        return PermissionDecision.DENY


class RuleBasedPolicy(PermissionPolicy):
    """Map explicit tool names to decisions and deny everything else."""

    def __init__(self, rules: Mapping[str, PermissionDecision]) -> None:
        copied = dict(rules)
        for name in copied:
            validate_tool_name(name)
        if not all(isinstance(value, PermissionDecision) for value in copied.values()):
            raise TypeError("permission rules must contain PermissionDecision values")
        self._rules = MappingProxyType(copied)

    def evaluate(
        self, tool: Tool, invocation: ToolInvocation, context: ExecutionContext
    ) -> PermissionDecision:
        return self._rules.get(tool.metadata.name, PermissionDecision.DENY)
