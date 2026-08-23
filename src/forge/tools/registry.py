"""Explicit deterministic tool registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from forge.tools.tool import Tool
from forge.tools.types import ToolMetadata, validate_tool_name


class ToolRegistrationError(ValueError):
    """A tool cannot be safely registered or located."""


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("tool must implement Tool")
        name = tool.metadata.name
        validate_tool_name(name)
        if name in self._tools:
            raise ToolRegistrationError(f"duplicate tool name: {name!r}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        validate_tool_name(name)
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolRegistrationError(f"unknown tool: {name!r}") from error

    @property
    def metadata(self) -> tuple[ToolMetadata, ...]:
        return tuple(self._tools[name].metadata for name in sorted(self._tools))

    def filtered(self, predicate: Callable[[ToolMetadata], bool]) -> ToolRegistry:
        """Return a fresh immutable-composition view selected by safe metadata."""
        return ToolRegistry(
            tool for tool in self._tools.values() if predicate(tool.metadata)
        )
