"""Bounded read-only repository tools."""

from __future__ import annotations

import logging
import stat
from collections.abc import Iterator, Mapping
from pathlib import Path

from forge.tools.paths import (
    WorkspacePathError,
    resolve_workspace_path,
    workspace_relative_path,
)
from forge.tools.tool import Tool, ToolError
from forge.tools.types import (
    ArgumentSchema,
    ArgumentSpec,
    ArgumentType,
    ExecutionContext,
    StructuredValue,
    ToolMetadata,
    ToolRisk,
)

LOGGER = logging.getLogger(__name__)
MAX_READ_BYTES = 256 * 1024
MAX_SEARCH_FILE_BYTES = 256 * 1024
DEFAULT_SEARCH_RESULTS = 100
MAX_SEARCH_RESULTS = 100
MAX_MATCH_LINE_CHARS = 500
IGNORED_SEARCH_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
    }
)


class ListDirectoryTool(Tool):
    _metadata = ToolMetadata(
        "repository.list_directory",
        "List entries in one directory inside the active workspace.",
        ArgumentSchema(
            (
                ArgumentSpec(
                    "path", ArgumentType.STRING, "Workspace-relative directory."
                ),
            )
        ),
        ToolRisk.READ_ONLY,
    )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self,
        arguments: Mapping[str, str | int | bool],
        context: ExecutionContext,
    ) -> StructuredValue:
        requested = str(arguments["path"])
        directory = _resolve(context, requested)
        if not directory.is_dir():
            raise ToolError(f"path is not a directory: {requested}")
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as error:
            raise ToolError(f"cannot list directory: {requested}") from error
        entries = []
        for child in children:
            entries.append(
                {
                    "name": child.name,
                    "path": workspace_relative_path(context.workspace, child),
                    "type": _entry_type(child),
                }
            )
        return {
            "path": workspace_relative_path(context.workspace, directory),
            "entries": entries,
        }


class ReadFileTool(Tool):
    _metadata = ToolMetadata(
        "repository.read_file",
        "Read bounded UTF-8 text from a regular file inside the active workspace.",
        ArgumentSchema(
            (ArgumentSpec("path", ArgumentType.STRING, "Workspace-relative file."),)
        ),
        ToolRisk.READ_ONLY,
    )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self,
        arguments: Mapping[str, str | int | bool],
        context: ExecutionContext,
    ) -> StructuredValue:
        requested = str(arguments["path"])
        path = _resolve(context, requested)
        try:
            file_stat = path.stat()
        except OSError as error:
            raise ToolError(f"cannot inspect file: {requested}") from error
        if not stat.S_ISREG(file_stat.st_mode):
            raise ToolError(f"path is not a regular file: {requested}")
        if file_stat.st_size > MAX_READ_BYTES:
            raise ToolError(
                f"file exceeds the {MAX_READ_BYTES}-byte read limit: {requested}"
            )
        data = _read_bounded(path, MAX_READ_BYTES, requested)
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ToolError(f"file is not valid UTF-8 text: {requested}") from error
        return {
            "path": workspace_relative_path(context.workspace, path),
            "content": text,
            "size_bytes": len(data),
        }


class SearchFilesTool(Tool):
    _metadata = ToolMetadata(
        "repository.search_files",
        "Search UTF-8 repository text for a bounded lexical query.",
        ArgumentSchema(
            (
                ArgumentSpec("query", ArgumentType.STRING, "Text to search for."),
                ArgumentSpec(
                    "path",
                    ArgumentType.STRING,
                    "Optional workspace-relative search directory.",
                    False,
                ),
                ArgumentSpec(
                    "case_sensitive",
                    ArgumentType.BOOLEAN,
                    "Whether matching preserves case.",
                    False,
                ),
                ArgumentSpec(
                    "max_results",
                    ArgumentType.INTEGER,
                    "Maximum matches, from 1 through 100.",
                    False,
                ),
            )
        ),
        ToolRisk.READ_ONLY,
    )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(
        self,
        arguments: Mapping[str, str | int | bool],
        context: ExecutionContext,
    ) -> StructuredValue:
        query = str(arguments["query"])
        if not query:
            raise ToolError("search query must not be empty")
        requested = str(arguments.get("path", "."))
        case_sensitive = bool(arguments.get("case_sensitive", True))
        max_results = int(arguments.get("max_results", DEFAULT_SEARCH_RESULTS))
        if max_results < 1 or max_results > MAX_SEARCH_RESULTS:
            raise ToolError(f"max_results must be between 1 and {MAX_SEARCH_RESULTS}")
        root = _resolve(context, requested)
        if not root.is_dir():
            raise ToolError(f"search path is not a directory: {requested}")
        root_display = workspace_relative_path(context.workspace, root)

        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, StructuredValue]] = []
        skipped_files = 0
        for path in _iter_search_files(context.workspace, root):
            try:
                file_stat = path.stat()
                if (
                    not stat.S_ISREG(file_stat.st_mode)
                    or file_stat.st_size > MAX_SEARCH_FILE_BYTES
                ):
                    skipped_files += 1
                    continue
                text = _read_bounded(
                    path,
                    MAX_SEARCH_FILE_BYTES,
                    workspace_relative_path(context.workspace, path),
                ).decode("utf-8-sig")
            except (OSError, ToolError, UnicodeDecodeError):
                skipped_files += 1
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle in haystack:
                    matches.append(
                        {
                            "path": workspace_relative_path(context.workspace, path),
                            "line_number": line_number,
                            "line_text": line[:MAX_MATCH_LINE_CHARS],
                            "line_truncated": len(line) > MAX_MATCH_LINE_CHARS,
                        }
                    )
                    if len(matches) == max_results:
                        LOGGER.info("Repository search reached result limit")
                        return _search_result(
                            query,
                            root_display,
                            case_sensitive,
                            matches,
                            skipped_files,
                            True,
                        )
        return _search_result(
            query,
            root_display,
            case_sensitive,
            matches,
            skipped_files,
            False,
        )


def _resolve(context: ExecutionContext, requested: str) -> Path:
    try:
        return resolve_workspace_path(context.workspace, requested)
    except WorkspacePathError as error:
        raise ToolError(str(error)) from error


def _entry_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    try:
        mode = path.stat().st_mode
    except OSError:
        return "other"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _read_bounded(path: Path, limit: int, display_path: str) -> bytes:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError as error:
        raise ToolError(f"cannot read file: {display_path}") from error
    if len(data) > limit:
        raise ToolError(f"file exceeds the {limit}-byte limit: {display_path}")
    return data


def _iter_search_files(workspace: Path, root: Path) -> Iterator[Path]:
    pending = [root]
    visited: set[Path] = set()
    while pending:
        directory = pending.pop()
        if directory in visited:
            continue
        visited.add(directory)
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        directories: list[Path] = []
        for entry in entries:
            try:
                resolved = resolve_workspace_path(
                    workspace, workspace_relative_path(workspace, entry)
                )
            except WorkspacePathError:
                continue
            if resolved.is_dir():
                if entry.name not in IGNORED_SEARCH_DIRECTORIES:
                    directories.append(resolved)
            else:
                yield resolved
        pending.extend(reversed(directories))


def _search_result(
    query: str,
    requested: str,
    case_sensitive: bool,
    matches: list[dict[str, StructuredValue]],
    skipped_files: int,
    limit_reached: bool,
) -> StructuredValue:
    return {
        "query": query,
        "path": requested,
        "case_sensitive": case_sensitive,
        "matches": matches,
        "skipped_files": skipped_files,
        "limit_reached": limit_reached,
    }
