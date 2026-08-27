"""Ranked language-agnostic repository discovery tool."""

from __future__ import annotations

from collections.abc import Mapping

from forge.lexical_index import RepositoryLexicalIndex
from forge.retrieval import SourceKind
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


class LexicalSearchTool(Tool):
    _metadata = ToolMetadata(
        "repository.lexical_search",
        "Find ranked language-agnostic source candidates from path and bounded token "
        "metadata. Results are discovery only and require a source read.",
        ArgumentSchema(
            (
                ArgumentSpec("query", ArgumentType.STRING, "Repository concept."),
                ArgumentSpec(
                    "path", ArgumentType.STRING, "Optional repository scope.", False
                ),
                ArgumentSpec(
                    "limit", ArgumentType.INTEGER, "Result count from 1 to 20.", False
                ),
                ArgumentSpec(
                    "preferred_source_kind",
                    ArgumentType.STRING,
                    "Optional implementation, test, configuration, or documentation.",
                    False,
                ),
            )
        ),
        ToolRisk.READ_ONLY,
        ToolEvidence.DISCOVERY,
        ToolCapability.READ,
    )

    def __init__(self, index: RepositoryLexicalIndex) -> None:
        self._index = index

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        query = str(arguments["query"])
        requested = str(arguments.get("path", "."))
        limit = int(arguments.get("limit", 8))
        raw_kind = arguments.get("preferred_source_kind")
        try:
            scope = workspace_relative_path(
                context.workspace, resolve_workspace_path(context.workspace, requested)
            )
            preferred = SourceKind(str(raw_kind)) if raw_kind is not None else None
            matches = self._index.search(
                query, path=scope, limit=limit, preferred_source_kind=preferred
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ToolError(str(error)) from error
        return {
            "query": query,
            "path": requested,
            "evidence": "discovery_only",
            "requires_source_read": True,
            "match_count": len(matches),
            "ranking_version": 1,
            "matches": tuple(
                {
                    "path": match.path,
                    "line_start": match.line_start,
                    "line_end": match.line_end,
                    "source_kind": match.source_kind.value,
                    "language": match.language,
                    "matched_tokens": match.matched_tokens,
                    "lexical_score": match.score.final,
                    "recommended_range": {
                        "start_line": match.line_start,
                        "end_line": match.line_end,
                    },
                }
                for match in matches
            ),
        }
