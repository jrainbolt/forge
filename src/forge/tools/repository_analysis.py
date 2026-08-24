"""Workspace-confined structural repository tools."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.repository_index import RepositoryIndex

from forge.repository_analysis import (
    FileLanguage,
    PythonAnalyzer,
    PythonParseError,
    classify_file,
)
from forge.tools.paths import (
    WorkspacePathError,
    resolve_workspace_path,
    workspace_relative_path,
)
from forge.tools.repository import _iter_search_files, _read_bounded
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

MAX_ANALYSIS_FILE_BYTES = 512 * 1024
MAX_ANALYSIS_FILES = 2_000
MAX_OUTLINE_SYMBOLS = 500
MAX_SYMBOL_RESULTS = 100
MAX_REFERENCE_RESULTS = 200
MAX_REFERENCE_SNIPPET_CHARS = 300
MAX_RANGE_LINES = 400
MAX_RANGE_BYTES = 128 * 1024


class FileOutlineTool(Tool):
    _metadata = ToolMetadata(
        "repository.file_outline",
        "Return bounded structural definitions and source ranges from a Python file; "
        "this is discovery evidence, not source content.",
        ArgumentSchema(
            (ArgumentSpec("path", ArgumentType.STRING, "Workspace-relative file."),)
        ),
        ToolRisk.READ_ONLY,
        ToolEvidence.DISCOVERY,
        ToolCapability.READ,
    )

    def __init__(self, index: RepositoryIndex | None = None) -> None:
        self._index = index

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        requested = str(arguments["path"])
        path, data, text = _read_analysis_file(context, requested)
        language = classify_file(path.name)
        if language is not FileLanguage.PYTHON:
            return {
                "path": workspace_relative_path(context.workspace, path),
                "language": language.value,
                "symbols": (),
                "truncated": False,
                "structural_support": False,
            }
        try:
            if self._index is not None:
                try:
                    symbols = self._index.file_symbols(
                        workspace_relative_path(context.workspace, path)
                    )
                except RuntimeError:
                    symbols = PythonAnalyzer().outline(text)
            else:
                symbols = PythonAnalyzer().outline(text)
        except PythonParseError as error:
            raise ToolError(
                str(error),
                output={
                    "path": workspace_relative_path(context.workspace, path),
                    "language": "python",
                    "parse_error": True,
                },
            ) from error
        truncated = len(symbols) > MAX_OUTLINE_SYMBOLS
        return {
            "path": workspace_relative_path(context.workspace, path),
            "language": "python",
            "symbols": tuple(
                _symbol_output(item) for item in symbols[:MAX_OUTLINE_SYMBOLS]
            ),
            "symbol_count": min(len(symbols), MAX_OUTLINE_SYMBOLS),
            "truncated": truncated,
            "sha256": hashlib.sha256(data).hexdigest(),
        }


class FindSymbolTool(Tool):
    _metadata = ToolMetadata(
        "repository.find_symbol",
        "Find exact Python structural symbol definitions; supports simple and "
        "qualified names.",
        ArgumentSchema(
            (
                ArgumentSpec(
                    "symbol",
                    ArgumentType.STRING,
                    "Exact simple or qualified symbol name.",
                ),
                ArgumentSpec(
                    "path",
                    ArgumentType.STRING,
                    "Optional workspace-relative file or directory.",
                    False,
                ),
            )
        ),
        ToolRisk.READ_ONLY,
        ToolEvidence.DISCOVERY,
        ToolCapability.READ,
    )

    def __init__(self, index: RepositoryIndex | None = None) -> None:
        self._index = index

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        symbol = str(arguments["symbol"])
        if not symbol:
            raise ToolError("symbol must not be empty")
        requested = str(arguments.get("path", "."))
        root = _resolve(context, requested)
        if self._index is not None:
            try:
                rows = self._index.find_symbols(
                    symbol, workspace_relative_path(context.workspace, root)
                )
                scanned, parse_failures, oversized = self._index.file_counts(
                    workspace_relative_path(context.workspace, root)
                )
                matches = [
                    {"path": row["path"], **_symbol_output(row)}
                    for row in rows[:MAX_SYMBOL_RESULTS]
                ]
                return _structural_result(
                    symbol,
                    requested,
                    matches,
                    scanned,
                    parse_failures,
                    oversized,
                    len(rows) > MAX_SYMBOL_RESULTS,
                )
            except RuntimeError:
                pass
        matches: list[dict[str, StructuredValue]] = []
        scanned = parse_failures = oversized = 0
        incomplete = False
        for path in _python_files(context, root):
            if scanned == MAX_ANALYSIS_FILES:
                incomplete = True
                break
            scanned += 1
            try:
                _, _, text = _read_analysis_file(
                    context, workspace_relative_path(context.workspace, path)
                )
                definitions = PythonAnalyzer().outline(text)
            except PythonParseError:
                parse_failures += 1
                continue
            except ToolError:
                oversized += 1
                continue
            for definition in definitions:
                if definition.name == symbol or definition.qualified_name == symbol:
                    matches.append(
                        {
                            "path": workspace_relative_path(context.workspace, path),
                            **_symbol_output(definition),
                        }
                    )
                    if len(matches) == MAX_SYMBOL_RESULTS:
                        incomplete = True
                        return _structural_result(
                            symbol,
                            requested,
                            matches,
                            scanned,
                            parse_failures,
                            oversized,
                            incomplete,
                        )
        return _structural_result(
            symbol, requested, matches, scanned, parse_failures, oversized, incomplete
        )


class FindReferencesTool(Tool):
    _metadata = ToolMetadata(
        "repository.find_references",
        "Find bounded Python structural reference candidates, not a complete "
        "semantic call graph.",
        ArgumentSchema(
            (
                ArgumentSpec(
                    "symbol",
                    ArgumentType.STRING,
                    "Exact symbol name to find reference candidates for.",
                ),
                ArgumentSpec(
                    "path",
                    ArgumentType.STRING,
                    "Optional workspace-relative file or directory.",
                    False,
                ),
            )
        ),
        ToolRisk.READ_ONLY,
        ToolEvidence.DISCOVERY,
        ToolCapability.READ,
    )

    def __init__(self, index: RepositoryIndex | None = None) -> None:
        self._index = index

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        symbol = str(arguments["symbol"])
        if not symbol:
            raise ToolError("symbol must not be empty")
        requested = str(arguments.get("path", "."))
        root = _resolve(context, requested)
        if self._index is not None:
            try:
                rows = self._index.find_references(
                    symbol, workspace_relative_path(context.workspace, root)
                )
                scanned, parse_failures, oversized = self._index.file_counts(
                    workspace_relative_path(context.workspace, root)
                )
                matches = []
                for row in rows[:MAX_REFERENCE_RESULTS]:
                    display = row["path"]
                    _, _, text = _read_analysis_file(context, display)
                    lines = text.splitlines()
                    line_text = (
                        lines[row["line"] - 1] if row["line"] <= len(lines) else ""
                    )
                    matches.append(
                        {
                            "path": display,
                            "line": row["line"],
                            "column": row["column_no"],
                            "kind": row["kind"],
                            "containing_symbol": row["containing_symbol"],
                            "line_text": line_text[:MAX_REFERENCE_SNIPPET_CHARS],
                            "line_truncated": len(line_text)
                            > MAX_REFERENCE_SNIPPET_CHARS,
                        }
                    )
                return _structural_result(
                    symbol,
                    requested,
                    matches,
                    scanned,
                    parse_failures,
                    oversized,
                    len(rows) > MAX_REFERENCE_RESULTS,
                    key="references",
                )
            except RuntimeError:
                pass
        matches: list[dict[str, StructuredValue]] = []
        scanned = parse_failures = oversized = 0
        incomplete = False
        for path in _python_files(context, root):
            if scanned == MAX_ANALYSIS_FILES:
                incomplete = True
                break
            scanned += 1
            display = workspace_relative_path(context.workspace, path)
            try:
                _, _, text = _read_analysis_file(context, display)
                references = PythonAnalyzer().references(text, symbol)
            except PythonParseError:
                parse_failures += 1
                continue
            except ToolError:
                oversized += 1
                continue
            lines = text.splitlines()
            for reference in references:
                line_text = (
                    lines[reference.line - 1] if reference.line <= len(lines) else ""
                )
                matches.append(
                    {
                        "path": display,
                        "line": reference.line,
                        "column": reference.column,
                        "kind": reference.kind,
                        "containing_symbol": reference.containing_symbol,
                        "line_text": line_text[:MAX_REFERENCE_SNIPPET_CHARS],
                        "line_truncated": len(line_text) > MAX_REFERENCE_SNIPPET_CHARS,
                    }
                )
                if len(matches) == MAX_REFERENCE_RESULTS:
                    incomplete = True
                    return _structural_result(
                        symbol,
                        requested,
                        matches,
                        scanned,
                        parse_failures,
                        oversized,
                        incomplete,
                        key="references",
                    )
        return _structural_result(
            symbol,
            requested,
            matches,
            scanned,
            parse_failures,
            oversized,
            incomplete,
            key="references",
        )


class ReadRangeTool(Tool):
    _metadata = ToolMetadata(
        "repository.read_range",
        "Read a bounded line range from a workspace UTF-8 text file and return the "
        "current full-file hash.",
        ArgumentSchema(
            (
                ArgumentSpec("path", ArgumentType.STRING, "Workspace-relative file."),
                ArgumentSpec("start_line", ArgumentType.INTEGER, "First 1-based line."),
                ArgumentSpec(
                    "end_line", ArgumentType.INTEGER, "Last 1-based line, inclusive."
                ),
            )
        ),
        ToolRisk.READ_ONLY,
        ToolEvidence.SOURCE_CONTENT,
        ToolCapability.READ,
    )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self, arguments: Mapping[str, object], context: ExecutionContext
    ) -> StructuredValue:
        requested = str(arguments["path"])
        start = int(arguments["start_line"])
        end = int(arguments["end_line"])
        if start < 1 or end < 1:
            raise ToolError("start_line and end_line must be positive")
        if start > end:
            raise ToolError("start_line must not exceed end_line")
        if end - start + 1 > MAX_RANGE_LINES:
            raise ToolError(f"requested range exceeds the {MAX_RANGE_LINES}-line limit")
        path, data, text = _read_analysis_file(context, requested)
        lines = text.splitlines(keepends=True)
        if start > len(lines):
            raise ToolError(f"start_line exceeds the file's {len(lines)} lines")
        actual_end = min(end, len(lines))
        selected = "".join(lines[start - 1 : actual_end])
        if len(selected.encode("utf-8")) > MAX_RANGE_BYTES:
            raise ToolError(f"range exceeds the {MAX_RANGE_BYTES}-byte output limit")
        return {
            "path": workspace_relative_path(context.workspace, path),
            "start_line": start,
            "end_line": end,
            "actual_start_line": start,
            "actual_end_line": actual_end,
            "file_line_count": len(lines),
            "text": selected,
            "size_bytes": len(selected.encode("utf-8")),
            "sha256": hashlib.sha256(data).hexdigest(),
        }


def _resolve(context: ExecutionContext, requested: str) -> Path:
    try:
        return resolve_workspace_path(context.workspace, requested)
    except WorkspacePathError as error:
        raise ToolError(str(error)) from error


def _read_analysis_file(
    context: ExecutionContext, requested: str
) -> tuple[Path, bytes, str]:
    path = _resolve(context, requested)
    try:
        file_stat = path.stat()
    except OSError as error:
        raise ToolError(f"cannot inspect file: {requested}") from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise ToolError(f"path is not a regular file: {requested}")
    if file_stat.st_size > MAX_ANALYSIS_FILE_BYTES:
        raise ToolError(
            "file exceeds the "
            f"{MAX_ANALYSIS_FILE_BYTES}-byte analysis limit: {requested}"
        )
    data = _read_bounded(path, MAX_ANALYSIS_FILE_BYTES, requested)
    try:
        return path, data, data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ToolError(f"file is not valid UTF-8 text: {requested}") from error


def _python_files(context: ExecutionContext, root: Path):  # type: ignore[no-untyped-def]
    if root.is_file():
        if root.suffix.lower() == ".py":
            yield root
        return
    if not root.is_dir():
        raise ToolError("analysis path is not a file or directory")
    for path in _iter_search_files(context.workspace, root):
        if path.suffix.lower() == ".py":
            yield path


def _symbol_output(symbol) -> dict[str, StructuredValue]:  # type: ignore[no-untyped-def]
    if hasattr(symbol, "keys"):
        return {
            "kind": symbol["kind"],
            "name": symbol["name"],
            "qualified_name": symbol["qualified_name"],
            "line_start": symbol["line_start"],
            "line_end": symbol["line_end"],
        }
    return {
        "kind": symbol.kind,
        "name": symbol.name,
        "qualified_name": symbol.qualified_name,
        "line_start": symbol.line_start,
        "line_end": symbol.line_end,
    }


def _structural_result(
    symbol: str,
    scope: str,
    matches: list[dict[str, StructuredValue]],
    scanned: int,
    parse_failures: int,
    oversized: int,
    incomplete: bool,
    *,
    key: str = "matches",
) -> StructuredValue:
    return {
        "symbol": symbol,
        "scope": scope,
        key: tuple(matches),
        "files_scanned": scanned,
        "parse_failures": parse_failures,
        "oversized_files": oversized,
        "truncated": incomplete,
        "complete": not incomplete,
    }
