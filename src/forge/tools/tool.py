"""Generic Forge tool abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from forge.tools.types import ExecutionContext, StructuredValue, ToolMetadata


class ToolError(Exception):
    """An expected failure reported by a tool implementation."""


class Tool(ABC):
    """A synchronous capability with explicit metadata and arguments."""

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Return immutable safe metadata without executing the tool."""

    @abstractmethod
    def execute(
        self,
        arguments: Mapping[str, object],
        context: ExecutionContext,
    ) -> StructuredValue:
        """Execute using validated arguments and an explicit context."""
