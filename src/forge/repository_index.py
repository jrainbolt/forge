"""Disposable persistent structural index for a repository workspace."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from forge.repository_analysis import PythonAnalyzer, PythonParseError
from forge.tools.paths import workspace_relative_path
from forge.tools.repository import _iter_search_files

MAX_ANALYSIS_FILE_BYTES = 512 * 1024
MAX_ANALYSIS_FILES = 2_000

SCHEMA_VERSION = 1


class RepositoryIndexError(RuntimeError):
    """The derived index could not be used safely."""


@dataclass(frozen=True, slots=True)
class IndexMetrics:
    files_considered: int = 0
    files_parsed: int = 0
    files_reused: int = 0
    files_added: int = 0
    files_changed: int = 0
    files_deleted: int = 0
    symbols: int = 0
    references: int = 0
    duration_seconds: float = 0.0


def default_cache_root() -> Path:
    override = os.environ.get("FORGE_CACHE_HOME")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        return (
            Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
            / "Forge"
        )
    if sys_platform() == "darwin":
        return Path.home() / "Library" / "Caches" / "forge"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "forge"


def sys_platform() -> str:
    import sys

    return sys.platform


class RepositoryIndex:
    """SQLite cache containing only derived locations and structural metadata."""

    def __init__(
        self,
        workspace: Path,
        *,
        cache_root: Path | None = None,
        analyzer: PythonAnalyzer | None = None,
    ) -> None:
        try:
            self.workspace = workspace.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"workspace does not exist: {workspace}") from error
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
        identity = hashlib.sha256(str(self.workspace).encode()).hexdigest()
        self.database_path = (
            (cache_root or default_cache_root())
            / "indexes"
            / identity
            / "index.sqlite3"
        )
        self.analyzer = analyzer or PythonAnalyzer()
        self.last_metrics = IndexMetrics()

    def status(self) -> dict[str, object]:
        if not self.database_path.exists():
            return {
                "state": "missing",
                "exists": False,
                "path": str(self.database_path),
            }
        try:
            with self._connect() as connection:
                version = self._schema_version(connection)
                files = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                symbols = connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[
                    0
                ]
                references = connection.execute("SELECT COUNT(*) FROM refs").fetchone()[
                    0
                ]
            state = "ready" if version == SCHEMA_VERSION else "incompatible"
            return {
                "state": state,
                "exists": True,
                "path": str(self.database_path),
                "schema_version": version,
                "files": files,
                "symbols": symbols,
                "references": references,
            }
        except (OSError, sqlite3.Error):
            return {"state": "corrupt", "exists": True, "path": str(self.database_path)}

    def build(self) -> IndexMetrics:
        started = time.perf_counter()
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix="index-", suffix=".sqlite3", dir=self.database_path.parent
            )
            os.close(fd)
            temp_path = Path(temporary)
            try:
                connection = sqlite3.connect(temp_path)
                self._create_schema(connection)
                metrics = self._populate(connection, {})
                connection.close()
                os.replace(temp_path, self.database_path)
            finally:
                temp_path.unlink(missing_ok=True)
        except (OSError, sqlite3.Error) as error:
            raise RepositoryIndexError(
                f"cannot build repository index: {error}"
            ) from error
        self.last_metrics = dataclass_replace_duration(metrics, started)
        return self.last_metrics

    def refresh(self) -> IndexMetrics:
        state = self.status()["state"]
        if state != "ready":
            return self.build()
        started = time.perf_counter()
        try:
            with self._connect() as connection:
                old = {
                    row[0]: row[1]
                    for row in connection.execute("SELECT path, sha256 FROM files")
                }
                metrics = self._populate(connection, old)
        except (OSError, sqlite3.Error) as error:
            raise RepositoryIndexError(
                f"cannot refresh repository index: {error}"
            ) from error
        self.last_metrics = dataclass_replace_duration(metrics, started)
        return self.last_metrics

    def file_symbols(self, path: str) -> list[sqlite3.Row]:
        self.refresh()
        status_rows = self._query("SELECT status FROM files WHERE path=?", (path,))
        if status_rows and status_rows[0]["status"] == "parse_error":
            raise PythonParseError("Python parse failed")
        return self._query(
            "SELECT * FROM symbols WHERE path=? ORDER BY ordinal", (path,)
        )

    def file_counts(self, scope: str) -> tuple[int, int, int]:
        clause, values = self._scope(scope)
        rows = self._query(
            "SELECT COUNT(*) AS total, "
            "SUM(status='parse_error') AS parse_failures, "
            f"SUM(status='oversized') AS oversized FROM files WHERE {clause}",
            values,
        )
        row = rows[0]
        return (
            int(row["total"] or 0),
            int(row["parse_failures"] or 0),
            int(row["oversized"] or 0),
        )

    def find_symbols(self, symbol: str, scope: str) -> list[sqlite3.Row]:
        self.refresh()
        clause, values = self._scope(scope)
        return self._query(
            "SELECT * FROM symbols WHERE (name=? OR qualified_name=?) "
            f"AND {clause} ORDER BY path, ordinal",
            (symbol, symbol, *values),
        )

    def find_references(self, symbol: str, scope: str) -> list[sqlite3.Row]:
        self.refresh()
        simple = symbol.rsplit(".", 1)[-1]
        clause, values = self._scope(scope)
        if symbol == simple:
            match = "simple_name=?"
            args: tuple[object, ...] = (simple,)
        else:
            match = "(name=? OR (simple_name=? AND supports_qualified=1))"
            args = (symbol, simple)
        return self._query(
            f"SELECT * FROM refs WHERE {match} AND {clause} ORDER BY path, ordinal",
            (*args, *values),
        )

    def invalidate(self, relative_path: str) -> None:
        if not self.database_path.exists():
            return
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM files WHERE path=?", (relative_path,))
                connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise RepositoryIndexError(
                f"cannot invalidate repository index: {error}"
            ) from error

    def clear(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)
            except OSError as error:
                raise RepositoryIndexError(
                    f"cannot clear repository index: {error}"
                ) from error

    def _populate(
        self, connection: sqlite3.Connection, old: dict[str, str]
    ) -> IndexMetrics:
        considered = parsed = reused = added = changed = symbols_total = refs_total = 0
        seen: set[str] = set()
        for path in _iter_search_files(self.workspace, self.workspace):
            if path.suffix.lower() != ".py" or len(seen) >= MAX_ANALYSIS_FILES:
                continue
            relative = workspace_relative_path(self.workspace, path)
            if relative in seen:
                continue
            seen.add(relative)
            considered += 1
            try:
                data = path.read_bytes()
                file_stat = path.stat()
            except OSError:
                continue
            digest = hashlib.sha256(data).hexdigest()
            if old.get(relative) == digest:
                reused += 1
                continue
            connection.execute("DELETE FROM files WHERE path=?", (relative,))
            status = "ok"
            definitions = references = ()
            if len(data) > MAX_ANALYSIS_FILE_BYTES:
                status = "oversized"
            else:
                try:
                    source = data.decode("utf-8-sig")
                    definitions = self.analyzer.outline(source)
                    references = self.analyzer.all_references(source)
                    parsed += 1
                except UnicodeDecodeError:
                    status = "invalid_utf8"
                except PythonParseError:
                    status = "parse_error"
            connection.execute(
                "INSERT INTO files(path, sha256, size, mtime_ns, status) "
                "VALUES(?,?,?,?,?)",
                (relative, digest, file_stat.st_size, file_stat.st_mtime_ns, status),
            )
            for ordinal, item in enumerate(definitions):
                connection.execute(
                    "INSERT INTO symbols VALUES(?,?,?,?,?,?,?)",
                    (
                        relative,
                        ordinal,
                        item.kind,
                        item.name,
                        item.qualified_name,
                        item.line_start,
                        item.line_end,
                    ),
                )
            for ordinal, item in enumerate(references):
                connection.execute(
                    "INSERT INTO refs VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        relative,
                        ordinal,
                        item.kind,
                        item.name,
                        item.name.rsplit(".", 1)[-1],
                        int(item.supports_qualified),
                        item.line,
                        item.column,
                        item.containing_symbol,
                    ),
                )
            symbols_total += len(definitions)
            refs_total += len(references)
            if relative in old:
                changed += 1
            else:
                added += 1
        deleted_paths = set(old) - seen
        for relative in deleted_paths:
            connection.execute("DELETE FROM files WHERE path=?", (relative,))
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES('updated_at', ?)",
            (str(time.time()),),
        )
        connection.commit()
        return IndexMetrics(
            considered,
            parsed,
            reused,
            added,
            changed,
            len(deleted_paths),
            symbols_total,
            refs_total,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=0.5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _query(self, sql: str, values: tuple[object, ...]) -> list[sqlite3.Row]:
        try:
            with self._connect() as connection:
                return list(connection.execute(sql, values))
        except sqlite3.Error as error:
            raise RepositoryIndexError(
                f"cannot query repository index: {error}"
            ) from error

    @staticmethod
    def _scope(scope: str) -> tuple[str, tuple[str, ...]]:
        if scope in ("", "."):
            return "1=1", ()
        escaped = scope.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return "(path=? OR path LIKE ? ESCAPE '\\')", (scope, f"{escaped}/%")

    @staticmethod
    def _schema_version(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        return int(row[0]) if row else -1

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES('schema_version', '1');
            CREATE TABLE files(
                path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL, status TEXT NOT NULL
            );
            CREATE TABLE symbols(
                path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,
                qualified_name TEXT NOT NULL, line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL, PRIMARY KEY(path, ordinal)
            );
            CREATE INDEX symbol_names ON symbols(name, qualified_name);
            CREATE TABLE refs(
                path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,
                simple_name TEXT NOT NULL, supports_qualified INTEGER NOT NULL,
                line INTEGER NOT NULL, column_no INTEGER NOT NULL,
                containing_symbol TEXT, PRIMARY KEY(path, ordinal)
            );
            CREATE INDEX reference_names ON refs(simple_name, name);
        """)
        connection.commit()


def dataclass_replace_duration(metrics: IndexMetrics, started: float) -> IndexMetrics:
    return replace(metrics, duration_seconds=time.perf_counter() - started)
