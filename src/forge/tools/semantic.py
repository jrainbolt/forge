"""Conceptual repository discovery backed by a configured semantic index."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.semantic_index import SemanticIndex

from forge.tools.paths import resolve_workspace_path, workspace_relative_path
from forge.tools.tool import Tool, ToolError
from forge.tools.types import (
    ArgumentSchema,
    ArgumentSpec,
    ArgumentType,
    ExecutionContext,
    StructuredValue,
    ToolCapability,
    ToolEvidence,
    ToolMetadata,
    ToolRisk,
)


class SemanticSearchTool(Tool):
    _metadata = ToolMetadata(
        "repository.semantic_search",
        "Find conceptual code candidates; results are discovery hints and must be "
        "followed by repository.read_range or repository.read_file before reasoning "
        "about implementation.",
        ArgumentSchema(
            (
                ArgumentSpec(
                    "query", ArgumentType.STRING, "Conceptual repository question."
                ),
                ArgumentSpec(
                    "path",
                    ArgumentType.STRING,
                    "Optional workspace-relative scope.",
                    False,
                ),
                ArgumentSpec(
                    "limit",
                    ArgumentType.INTEGER,
                    "Optional result count, from 1 through 20.",
                    False,
                ),
            )
        ),
        ToolRisk.READ_ONLY,
        ToolEvidence.DISCOVERY,
        ToolCapability.READ,
    )

    def __init__(self, index: SemanticIndex) -> None:
        self._index = index

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        query = str(arguments["query"])
        limit = int(arguments.get("limit", 8))
        requested = str(arguments.get("path", "."))
        try:
            scope = workspace_relative_path(
                context.workspace, resolve_workspace_path(context.workspace, requested)
            )
            matches = self._index.search(query, path=scope, limit=limit)
        except (OSError, RuntimeError, ValueError) as error:
            raise ToolError(str(error)) from error
        return {
            "query": query,
            "path": requested,
            "evidence": "discovery_only",
            "requires_source_read": True,
            "match_count": len(matches),
            "matches": tuple(
                {
                    "path": match.path,
                    "line_start": match.line_start,
                    "line_end": match.line_end,
                    "symbol": match.symbol,
                    "qualified_name": match.qualified_name,
                    "similarity": match.similarity,
                    "semantic_similarity": match.similarity,
                    "source_kind": match.source_kind.value,
                    "language": match.language,
                    "chunk_kind": match.chunk_kind,
                    "recommended_range": {
                        "start_line": max(1, match.line_start - 20),
                        "end_line": min(match.line_end + 20, match.line_start + 399),
                    },
                }
                for match in matches
            ),
        }
