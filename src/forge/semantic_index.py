"""Disposable, source-free semantic repository index."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import sqlite3
import struct
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from forge.embeddings import EmbeddingModel, EmbeddingPurpose, EmbeddingVector
from forge.repository_analysis import PythonAnalyzer, PythonParseError
from forge.repository_index import default_cache_root
from forge.retrieval import (
    RETRIEVAL_RANKING_VERSION,
    RetrievalCandidate,
    RetrievalRankingError,
    SourceKind,
    classify_source,
    rank_candidates,
)
from forge.tools.paths import workspace_relative_path
from forge.tools.repository import _iter_search_files

SCHEMA_VERSION = 1
CHUNKER_VERSION = "semantic-chunks-v1"
MAX_FILE_BYTES = 512 * 1024
MAX_FILES = 2_000
MAX_CHUNKS = 20_000
WINDOW_LINES = 80
WINDOW_OVERLAP = 10
MAX_QUERY_BYTES = 8 * 1024
MIN_RAW_CANDIDATES = 20
MAX_RAW_CANDIDATES = 80
LOGGER = logging.getLogger(__name__)


class SemanticIndexError(RuntimeError):
    """The semantic cache could not be built or queried safely."""


@dataclass(frozen=True, slots=True)
class SemanticChunk:
    chunk_id: str
    path: str
    line_start: int
    line_end: int
    language: str
    symbol: str | None
    qualified_name: str | None
    chunk_kind: str
    file_sha256: str
    chunk_sha256: str
    embedding_text: str


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    path: str
    line_start: int
    line_end: int
    language: str
    symbol: str | None
    qualified_name: str | None
    chunk_kind: str
    similarity: float
    source_kind: SourceKind = SourceKind.OTHER_TEXT
    chunk_sha256: str = ""


@dataclass(frozen=True, slots=True)
class SemanticMetrics:
    files_considered: int = 0
    files_added: int = 0
    files_changed: int = 0
    files_deleted: int = 0
    files_reused: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    failures: int = 0
    duration_seconds: float = 0.0


class SemanticIndex:
    """A model-bound SQLite vector cache independent from structural state."""

    def __init__(
        self, workspace: Path, model: EmbeddingModel, *, cache_root: Path | None = None
    ) -> None:
        self.workspace = workspace.resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        if model.dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        identity = hashlib.sha256(str(self.workspace).encode()).hexdigest()
        self.database_path = (
            (cache_root or default_cache_root())
            / "indexes"
            / identity
            / "semantic.sqlite3"
        )
        self.model = model
        self.last_metrics = SemanticMetrics()

    def status(self) -> dict[str, object]:
        if not self.database_path.exists():
            return {
                "state": "missing",
                "exists": False,
                "path": str(self.database_path),
            }
        try:
            with self._connect() as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                files = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                invalid_vectors = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE length(vector) != ?",
                    (int(metadata.get("dimensions", 0)) * 4,),
                ).fetchone()[0]
                if invalid_vectors:
                    raise ValueError("invalid semantic vector payload")
            compatible = self._compatible(metadata)
            return {
                "state": "ready" if compatible else "incompatible",
                "exists": True,
                "path": str(self.database_path),
                "schema_version": int(metadata.get("schema_version", -1)),
                "backend": metadata.get("backend"),
                "model": metadata.get("model"),
                "dimensions": int(metadata.get("dimensions", 0)),
                "chunker_version": metadata.get("chunker_version"),
                "retrieval_ranking_version": int(
                    metadata.get("retrieval_ranking_version", 0)
                ),
                "files": files,
                "chunks": chunks,
            }
        except (OSError, sqlite3.Error, ValueError):
            return {"state": "corrupt", "exists": True, "path": str(self.database_path)}

    def build(self) -> SemanticMetrics:
        started = time.perf_counter()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix="semantic-", suffix=".sqlite3", dir=self.database_path.parent
        )
        os.close(fd)
        temp_path = Path(temporary)
        try:
            connection = sqlite3.connect(temp_path)
            self._create_schema(connection)
            metrics = self._populate(connection, {})
            connection.close()
            os.replace(temp_path, self.database_path)
        except Exception as error:
            raise SemanticIndexError(f"cannot build semantic index: {error}") from error
        finally:
            temp_path.unlink(missing_ok=True)
        self.last_metrics = replace(
            metrics, duration_seconds=time.perf_counter() - started
        )
        return self.last_metrics

    def refresh(self) -> SemanticMetrics:
        if self.status()["state"] != "ready":
            return self.build()
        started = time.perf_counter()
        try:
            with self._connect() as connection:
                old = dict(connection.execute("SELECT path, sha256 FROM files"))
                metrics = self._populate(connection, old)
        except Exception as error:
            raise SemanticIndexError(
                f"cannot refresh semantic index: {error}"
            ) from error
        self.last_metrics = replace(
            metrics, duration_seconds=time.perf_counter() - started
        )
        return self.last_metrics

    def search(
        self, query: str, *, path: str = ".", limit: int = 8
    ) -> tuple[SemanticMatch, ...]:
        matches = self._semantic_matches(query, path=path, limit=limit)
        raw_limit = min(MAX_RAW_CANDIDATES, max(MIN_RAW_CANDIDATES, limit * 4))
        raw = matches[:raw_limit]
        candidates = tuple(self._ranking_candidate(match) for match in raw)
        try:
            ranked = rank_candidates(query, candidates, limit=limit)
        except RetrievalRankingError:
            LOGGER.warning(
                "Hybrid reranking rejected candidate data; using semantic order"
            )
            return tuple(raw[:limit])
        return tuple(
            SemanticMatch(
                candidate.path,
                candidate.line_start,
                candidate.line_end,
                candidate.language,
                candidate.symbol,
                candidate.qualified_name,
                candidate.chunk_kind,
                candidate.semantic_similarity,
                candidate.source_kind,
                candidate.chunk_sha256,
            )
            for candidate in ranked
        )

    def search_raw(
        self, query: str, *, path: str = ".", limit: int = 8
    ) -> tuple[SemanticMatch, ...]:
        """Return semantic-only order for deterministic evaluation comparisons."""
        return self._semantic_matches(query, path=path, limit=limit)[:limit]

    def _semantic_matches(
        self, query: str, *, path: str, limit: int
    ) -> tuple[SemanticMatch, ...]:
        if not query.strip() or len(query.encode("utf-8")) > MAX_QUERY_BYTES:
            raise ValueError("query must be non-empty and at most 8192 UTF-8 bytes")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        self.refresh()
        query_vector = self.model.embed_text(
            query, purpose=EmbeddingPurpose.QUERY
        ).values
        clause, values = self._scope(path)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"SELECT * FROM chunks WHERE {clause}", values
                )
                matches = []
                for row in rows:
                    vector = _unpack_vector(row["vector"], self.model.dimensions)
                    matches.append(
                        SemanticMatch(
                            row["path"],
                            row["line_start"],
                            row["line_end"],
                            row["language"],
                            row["symbol"],
                            row["qualified_name"],
                            row["chunk_kind"],
                            _cosine(query_vector, vector),
                            classify_source(row["path"]),
                            row["chunk_sha256"],
                        )
                    )
        except sqlite3.Error as error:
            raise SemanticIndexError(f"cannot query semantic index: {error}") from error
        matches.sort(
            key=lambda item: (
                -item.similarity,
                item.path,
                item.line_start,
                item.line_end,
            )
        )
        return tuple(matches)

    def _ranking_candidate(self, match: SemanticMatch) -> RetrievalCandidate:
        source = ""
        try:
            lines = (
                (self.workspace / match.path)
                .read_text(encoding="utf-8-sig")
                .splitlines()
            )
            source = "\n".join(lines[match.line_start - 1 : match.line_end])
        except (OSError, UnicodeDecodeError):
            pass
        return RetrievalCandidate(
            match.path,
            match.line_start,
            match.line_end,
            match.language,
            match.symbol,
            match.qualified_name,
            match.chunk_kind,
            match.chunk_sha256,
            match.similarity,
            source,
        )

    def invalidate(self, relative_path: str) -> None:
        if self.database_path.exists():
            try:
                with self._connect() as connection:
                    connection.execute(
                        "DELETE FROM files WHERE path=?", (relative_path,)
                    )
                    connection.commit()
            except sqlite3.Error as error:
                raise SemanticIndexError(
                    f"cannot invalidate semantic index: {error}"
                ) from error

    def clear(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    def close(self) -> None:
        self.model.close()

    def _populate(
        self, connection: sqlite3.Connection, old: dict[str, str]
    ) -> SemanticMetrics:
        considered = added = changed = reused = embedded = reused_chunks = failures = 0
        seen: set[str] = set()
        for file_path in _iter_search_files(self.workspace, self.workspace):
            if len(seen) >= MAX_FILES:
                break
            relative = workspace_relative_path(self.workspace, file_path)
            if relative in seen:
                continue
            seen.add(relative)
            considered += 1
            try:
                data = file_path.read_bytes()
                if len(data) > MAX_FILE_BYTES:
                    continue
                text = data.decode("utf-8-sig")
            except (OSError, UnicodeDecodeError):
                failures += 1
                continue
            digest = hashlib.sha256(data).hexdigest()
            if old.get(relative) == digest:
                reused += 1
                reused_chunks += connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE path=?", (relative,)
                ).fetchone()[0]
                continue
            connection.execute("DELETE FROM files WHERE path=?", (relative,))
            chunks = chunk_file(relative, text, digest)
            if embedded + len(chunks) > MAX_CHUNKS:
                failures += 1
                break
            vectors = (
                self.model.embed_batch(
                    tuple(chunk.embedding_text for chunk in chunks),
                    purpose=EmbeddingPurpose.DOCUMENT,
                )
                if chunks
                else ()
            )
            connection.execute(
                "INSERT INTO files(path, sha256) VALUES(?,?)", (relative, digest)
            )
            for chunk, vector in zip(chunks, vectors, strict=True):
                if len(vector.values) != self.model.dimensions:
                    raise SemanticIndexError("embedding dimensions changed")
                connection.execute(
                    "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        chunk.chunk_id,
                        chunk.path,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.language,
                        chunk.symbol,
                        chunk.qualified_name,
                        chunk.chunk_kind,
                        chunk.file_sha256,
                        chunk.chunk_sha256,
                        _pack_vector(vector),
                    ),
                )
            embedded += len(chunks)
            if relative in old:
                changed += 1
            else:
                added += 1
        deleted = set(old) - seen
        for relative in deleted:
            connection.execute("DELETE FROM files WHERE path=?", (relative,))
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES('updated_at', ?)",
            (str(time.time()),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES('retrieval_ranking_version', ?)",
            (str(RETRIEVAL_RANKING_VERSION),),
        )
        connection.commit()
        return SemanticMetrics(
            considered,
            added,
            changed,
            len(deleted),
            reused,
            embedded,
            reused_chunks,
            failures,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=0.5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _compatible(self, metadata: dict[str, str]) -> bool:
        identity = self.model.identity
        return (
            metadata.get("schema_version") == str(SCHEMA_VERSION)
            and metadata.get("chunker_version") == CHUNKER_VERSION
            and metadata.get("backend") == identity.backend
            and metadata.get("model") == identity.model
            and metadata.get("dimensions") == str(self.model.dimensions)
        )

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        identity = self.model.identity
        connection.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE files(path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
            CREATE TABLE chunks(
                chunk_id TEXT PRIMARY KEY,
                path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                line_start INTEGER NOT NULL, line_end INTEGER NOT NULL,
                language TEXT NOT NULL, symbol TEXT, qualified_name TEXT,
                chunk_kind TEXT NOT NULL, file_sha256 TEXT NOT NULL,
                chunk_sha256 TEXT NOT NULL, vector BLOB NOT NULL
            );
            CREATE INDEX semantic_paths ON chunks(path, line_start);
        """)
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            (
                ("schema_version", str(SCHEMA_VERSION)),
                ("chunker_version", CHUNKER_VERSION),
                ("backend", identity.backend),
                ("model", identity.model),
                ("dimensions", str(self.model.dimensions)),
                ("retrieval_ranking_version", str(RETRIEVAL_RANKING_VERSION)),
            ),
        )
        connection.commit()

    @staticmethod
    def _scope(scope: str) -> tuple[str, tuple[str, ...]]:
        if scope in ("", "."):
            return "1=1", ()
        escaped = scope.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return "(path=? OR path LIKE ? ESCAPE '\\')", (scope, f"{escaped}/%")


def chunk_file(path: str, text: str, file_sha256: str) -> tuple[SemanticChunk, ...]:
    lines = text.splitlines()
    language = "python" if Path(path).suffix.lower() == ".py" else "text"
    specs: list[tuple[int, int, str | None, str | None, str]] = []
    if language == "python":
        try:
            definitions = PythonAnalyzer().outline(text)
        except PythonParseError:
            definitions = ()
        for item in definitions:
            for start, end in _windows(item.line_start, item.line_end):
                specs.append((start, end, item.name, item.qualified_name, item.kind))
        covered = {
            line
            for item in definitions
            for line in range(item.line_start, item.line_end + 1)
        }
        module_lines = [
            line for line in range(1, len(lines) + 1) if line not in covered
        ]
        for first, last in _runs(module_lines):
            specs.extend(
                (start, end, None, None, "module")
                for start, end in _windows(first, last)
            )
    else:
        specs.extend(
            (start, end, None, None, "text") for start, end in _windows(1, len(lines))
        )
    result = []
    for start, end, symbol, qualified, kind in specs:
        source = "\n".join(lines[start - 1 : end])
        if not source.strip():
            continue
        embedding_text = (
            f"Path: {path}\n"
            + (f"Symbol: {qualified or symbol}\n" if symbol else "")
            + f"Source:\n{source}"
        )
        digest = hashlib.sha256(embedding_text.encode()).hexdigest()
        chunk_id = hashlib.sha256(
            f"{CHUNKER_VERSION}\0{path}\0{start}\0{end}\0{digest}".encode()
        ).hexdigest()
        result.append(
            SemanticChunk(
                chunk_id,
                path,
                start,
                end,
                language,
                symbol,
                qualified,
                kind,
                file_sha256,
                digest,
                embedding_text,
            )
        )
    return tuple(result)


def _windows(start: int, end: int) -> tuple[tuple[int, int], ...]:
    if end < start:
        return ()
    result = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + WINDOW_LINES - 1)
        result.append((cursor, stop))
        if stop == end:
            break
        cursor = stop - WINDOW_OVERLAP + 1
    return tuple(result)


def _runs(lines: list[int]) -> tuple[tuple[int, int], ...]:
    if not lines:
        return ()
    result, start, previous = [], lines[0], lines[0]
    for line in lines[1:]:
        if line != previous + 1:
            result.append((start, previous))
            start = line
        previous = line
    result.append((start, previous))
    return tuple(result)


def _pack_vector(vector: EmbeddingVector) -> bytes:
    return struct.pack(f"<{len(vector.values)}f", *vector.values)


def _unpack_vector(data: bytes, dimensions: int) -> tuple[float, ...]:
    if len(data) != dimensions * 4:
        raise SemanticIndexError("invalid vector payload")
    values = struct.unpack(f"<{dimensions}f", data)
    if not all(math.isfinite(value) for value in values):
        raise SemanticIndexError("non-finite vector payload")
    return values


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return (
        sum(a * b for a, b in zip(left, right, strict=True)) / denominator
        if denominator
        else 0.0
    )
