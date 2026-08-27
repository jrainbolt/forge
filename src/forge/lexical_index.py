"""Fast persistent language-agnostic repository discovery index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from forge.retrieval import SourceKind, classify_source, tokenize

LEXICAL_SCHEMA_VERSION = 1
REPOSITORY_TOKENIZER_VERSION = 2
LEXICAL_RETRIEVAL_VERSION = 1
MAX_LEXICAL_FILE_BYTES = 512 * 1024
MAX_LEXICAL_FILES = 5_000
MAX_FILE_TOKENS = 8_192
MAX_TOKEN_LENGTH = 80
MAX_POSITIONS_PER_TOKEN_FILE = 8
DEFAULT_LEXICAL_RESULTS = 8
MAX_LEXICAL_RESULTS = 20
RECOMMENDED_RANGE_LINES = 120

_TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".scala",
        ".sh",
        ".swift",
        ".ts",
        ".tsx",
        ".cfg",
        ".cmake",
        ".gradle",
        ".ini",
        ".json",
        ".lock",
        ".md",
        ".rst",
        ".adoc",
        ".toml",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_TEXT_NAMES = frozenset(
    {"cmakelists.txt", "dockerfile", "makefile", "pom.xml", "build.gradle"}
)
_LANGUAGES = {
    ".c": "c",
    ".h": "c_header",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp_header",
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
}
_CONTENT_STOPWORDS = frozenset(
    {
        "and",
        "are",
        "bool",
        "char",
        "class",
        "const",
        "def",
        "else",
        "false",
        "for",
        "from",
        "function",
        "if",
        "import",
        "include",
        "int",
        "let",
        "new",
        "none",
        "null",
        "public",
        "return",
        "self",
        "static",
        "string",
        "struct",
        "this",
        "true",
        "use",
        "var",
        "void",
        "while",
    }
)
_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        "target",
        "cmakefiles",
    }
)


class LexicalIndexError(RuntimeError):
    """The rebuildable lexical index could not be used safely."""


@dataclass(frozen=True, slots=True)
class LexicalIndexMetrics:
    files_scanned: int = 0
    files_indexed: int = 0
    files_retokenized: int = 0
    files_reused: int = 0
    files_added: int = 0
    files_changed: int = 0
    files_deleted: int = 0
    tokens: int = 0
    postings: int = 0
    duration_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class LexicalScore:
    path: float
    basename: float
    content: float
    rarity: float
    source_kind: float
    coverage: float
    final: float


@dataclass(frozen=True, slots=True)
class LexicalMatch:
    path: str
    line_start: int
    line_end: int
    extension: str
    language: str
    source_kind: SourceKind
    matched_tokens: tuple[str, ...]
    score: LexicalScore


class RepositoryLexicalIndex:
    """SQLite path/token map that never persists repository source text."""

    def __init__(self, workspace: Path, *, cache_root: Path | None = None) -> None:
        self.workspace = workspace.resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        identity = hashlib.sha256(str(self.workspace).encode()).hexdigest()
        self.database_path = (
            (cache_root or _default_cache_root())
            / "indexes"
            / identity
            / "lexical.sqlite3"
        )
        self.last_metrics = LexicalIndexMetrics()
        self.builds = 0
        self.refreshes = 0
        self.total_duration_seconds = 0.0
        self.total_files_retokenized = 0

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
                tokens = connection.execute("SELECT COUNT(*) FROM token_df").fetchone()[
                    0
                ]
                postings = connection.execute(
                    "SELECT COUNT(*) FROM postings"
                ).fetchone()[0]
            compatible = all(
                int(metadata.get(key, -1)) == expected
                for key, expected in (
                    ("schema_version", LEXICAL_SCHEMA_VERSION),
                    ("tokenizer_version", REPOSITORY_TOKENIZER_VERSION),
                    ("ranking_version", LEXICAL_RETRIEVAL_VERSION),
                )
            )
            return {
                "state": "ready" if compatible else "incompatible",
                "exists": True,
                "path": str(self.database_path),
                "schema_version": int(metadata.get("schema_version", -1)),
                "tokenizer_version": int(metadata.get("tokenizer_version", -1)),
                "ranking_version": int(metadata.get("ranking_version", -1)),
                "files": int(files),
                "tokens": int(tokens),
                "postings": int(postings),
            }
        except (OSError, sqlite3.Error, ValueError):
            return {"state": "corrupt", "exists": True, "path": str(self.database_path)}

    def build(self) -> LexicalIndexMetrics:
        started = time.perf_counter()
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix="lexical-", suffix=".sqlite3", dir=self.database_path.parent
            )
            os.close(fd)
            temporary_path = Path(temporary)
            try:
                connection = sqlite3.connect(temporary_path)
                self._create_schema(connection)
                metrics = self._populate(connection, {})
                connection.close()
                os.replace(temporary_path, self.database_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        except (OSError, sqlite3.Error) as error:
            raise LexicalIndexError(f"cannot build lexical index: {error}") from error
        self.last_metrics = replace(
            metrics, duration_seconds=time.perf_counter() - started
        )
        self.builds += 1
        self.total_duration_seconds += self.last_metrics.duration_seconds
        self.total_files_retokenized += self.last_metrics.files_retokenized
        return self.last_metrics

    def refresh(self) -> LexicalIndexMetrics:
        self.refreshes += 1
        if self.status()["state"] != "ready":
            return self.build()
        started = time.perf_counter()
        try:
            with self._connect() as connection:
                old = dict(connection.execute("SELECT path, sha256 FROM files"))
                metrics = self._populate(connection, old)
        except (OSError, sqlite3.Error) as error:
            raise LexicalIndexError(f"cannot refresh lexical index: {error}") from error
        self.last_metrics = replace(
            metrics, duration_seconds=time.perf_counter() - started
        )
        self.total_duration_seconds += self.last_metrics.duration_seconds
        self.total_files_retokenized += self.last_metrics.files_retokenized
        return self.last_metrics

    def search(
        self,
        query: str,
        *,
        path: str = ".",
        limit: int = DEFAULT_LEXICAL_RESULTS,
        preferred_source_kind: SourceKind | None = None,
    ) -> tuple[LexicalMatch, ...]:
        if not query.strip():
            raise ValueError("lexical query must not be empty")
        if not 1 <= limit <= MAX_LEXICAL_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_LEXICAL_RESULTS}")
        query_tokens = tuple(sorted(lexical_tokenize(query)))
        if not query_tokens:
            return ()
        self.refresh()
        scope, scope_values = _scope(path)
        placeholders = ",".join("?" for _ in query_tokens)
        with self._connect() as connection:
            file_count = int(
                connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            )
            rows = list(
                connection.execute(
                    "SELECT f.path,f.extension,f.language,f.source_kind,f.path_tokens,"
                    "f.basename_tokens,p.token,p.count,p.lines,d.document_frequency "
                    "FROM postings p JOIN files f ON f.path=p.path "
                    "JOIN token_df d ON d.token=p.token "
                    f"WHERE p.token IN ({placeholders}) AND {scope}",
                    (*query_tokens, *scope_values),
                )
            )
        grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            grouped[row[0]].append(row)
        matches = [
            _rank_file(query_tokens, file_count, file_rows, preferred_source_kind)
            for file_rows in grouped.values()
        ]
        matches.sort(key=lambda item: (-item.score.final, item.path, item.line_start))
        basename_tokens = {
            file_rows[0][0]: frozenset(json.loads(file_rows[0][5]))
            for file_rows in grouped.values()
        }
        return _diversify(matches, basename_tokens, query_tokens, limit)

    def invalidate(self, relative_path: str) -> None:
        if not self.database_path.exists():
            return
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM files WHERE path=?", (relative_path,))
                _refresh_document_frequency(connection)
                connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise LexicalIndexError(
                f"cannot invalidate lexical index: {error}"
            ) from error

    def clear(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)
            except OSError as error:
                raise LexicalIndexError(
                    f"cannot clear lexical index: {error}"
                ) from error

    def _populate(
        self, connection: sqlite3.Connection, old: dict[str, str]
    ) -> LexicalIndexMetrics:
        scanned = indexed = retokenized = reused = added = changed = 0
        seen: set[str] = set()
        for file_path in _iter_files(self.workspace):
            if len(seen) >= MAX_LEXICAL_FILES:
                break
            relative = file_path.relative_to(self.workspace).as_posix()
            if not _eligible(relative):
                continue
            scanned += 1
            try:
                info = file_path.stat()
                if not file_path.is_file() or info.st_size > MAX_LEXICAL_FILE_BYTES:
                    continue
                data = file_path.read_bytes()
                text = data.decode("utf-8-sig")
                if "\x00" in text:
                    continue
            except (OSError, UnicodeDecodeError):
                continue
            digest = hashlib.sha256(data).hexdigest()
            seen.add(relative)
            indexed += 1
            if old.get(relative) == digest:
                reused += 1
                continue
            connection.execute("DELETE FROM files WHERE path=?", (relative,))
            path_tokens = tuple(sorted(lexical_tokenize(relative)))
            basename_tokens = tuple(
                sorted(lexical_tokenize(PurePosixPath(relative).stem))
            )
            postings = _content_postings(text)
            kind = classify_source(relative)
            suffix = PurePosixPath(relative).suffix.casefold()
            connection.execute(
                "INSERT INTO files VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    relative,
                    digest,
                    len(data),
                    suffix,
                    _LANGUAGES.get(suffix, "text"),
                    kind.value,
                    json.dumps(path_tokens),
                    json.dumps(basename_tokens),
                    int(info.st_mtime_ns),
                ),
            )
            all_tokens = set(postings) | set(path_tokens) | set(basename_tokens)
            for token in sorted(all_tokens):
                count, positions = postings.get(token, (0, ()))
                connection.execute(
                    "INSERT INTO postings VALUES(?,?,?,?)",
                    (relative, token, count, json.dumps(positions)),
                )
            retokenized += 1
            if relative in old:
                changed += 1
            else:
                added += 1
        deleted = set(old) - seen
        for relative in deleted:
            connection.execute("DELETE FROM files WHERE path=?", (relative,))
        _refresh_document_frequency(connection)
        connection.commit()
        tokens = int(connection.execute("SELECT COUNT(*) FROM token_df").fetchone()[0])
        postings_count = int(
            connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        )
        return LexicalIndexMetrics(
            scanned,
            indexed,
            retokenized,
            reused,
            added,
            changed,
            len(deleted),
            tokens,
            postings_count,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=1.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            f"""
            PRAGMA foreign_keys=ON;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES('schema_version','{LEXICAL_SCHEMA_VERSION}');
            INSERT INTO metadata VALUES(
                'tokenizer_version','{REPOSITORY_TOKENIZER_VERSION}'
            );
            INSERT INTO metadata VALUES(
                'ranking_version','{LEXICAL_RETRIEVAL_VERSION}'
            );
            CREATE TABLE files(
                path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, size INTEGER NOT NULL,
                extension TEXT NOT NULL, language TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                path_tokens TEXT NOT NULL, basename_tokens TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL
            );
            CREATE TABLE postings(
                path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                token TEXT NOT NULL, count INTEGER NOT NULL, lines TEXT NOT NULL,
                PRIMARY KEY(path,token)
            );
            CREATE INDEX posting_tokens ON postings(token,path);
            CREATE TABLE token_df(
                token TEXT PRIMARY KEY, document_frequency INTEGER NOT NULL
            );
        """
        )
        connection.commit()


def _eligible(path: str) -> bool:
    pure = PurePosixPath(path.casefold())
    return pure.suffix in _TEXT_SUFFIXES or pure.name in _TEXT_NAMES


def _default_cache_root() -> Path:
    override = os.environ.get("FORGE_CACHE_HOME")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        return (
            Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
            / "Forge"
        )
    import sys

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "forge"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "forge"


def _iter_files(workspace: Path):  # type: ignore[no-untyped-def]
    pending = [workspace]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        children = []
        for entry in entries:
            if entry.is_symlink():
                continue
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                continue
            if workspace != resolved and workspace not in resolved.parents:
                continue
            if resolved.is_dir():
                if entry.name.casefold() not in _IGNORED_DIRECTORIES and not any(
                    part.casefold().endswith((".egg-info", ".dist-info"))
                    for part in resolved.relative_to(workspace).parts
                ):
                    children.append(resolved)
            elif resolved.is_file():
                yield resolved
        pending.extend(reversed(children))


def _content_postings(text: str) -> dict[str, tuple[int, tuple[int, ...]]]:
    counts: Counter[str] = Counter()
    lines: dict[str, list[int]] = defaultdict(list)
    accepted = 0
    for number, line in enumerate(text.splitlines(), start=1):
        for token in sorted(lexical_tokenize(line)):
            if token in _CONTENT_STOPWORDS or len(token) > MAX_TOKEN_LENGTH:
                continue
            if len(token) >= 24 and all(
                character in "0123456789abcdef" for character in token
            ):
                continue
            counts[token] += 1
            if len(lines[token]) < MAX_POSITIONS_PER_TOKEN_FILE:
                lines[token].append(number)
            accepted += 1
            if accepted >= MAX_FILE_TOKENS:
                return {
                    key: (count, tuple(lines[key])) for key, count in counts.items()
                }
    return {key: (count, tuple(lines[key])) for key, count in counts.items()}


def lexical_tokenize(value: str) -> frozenset[str]:
    """Return identifier tokens plus conservative singular lookup variants."""
    tokens = set(tokenize(value))
    for token in tuple(tokens):
        if len(token) > 4 and token.endswith("ies"):
            tokens.add(f"{token[:-3]}y")
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            tokens.add(token[:-1])
    return frozenset(tokens)


def _refresh_document_frequency(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM token_df")
    connection.execute(
        "INSERT INTO token_df SELECT token,COUNT(*) FROM postings GROUP BY token"
    )


def _rank_file(
    query_tokens: tuple[str, ...],
    file_count: int,
    rows: list[sqlite3.Row],
    preferred: SourceKind | None,
) -> LexicalMatch:
    first = rows[0]
    path_tokens = set(json.loads(first[4]))
    basename_tokens = set(json.loads(first[5]))
    posting = {row[6]: row for row in rows}
    matched = tuple(sorted(posting))
    path_overlap = len(set(query_tokens) & path_tokens) / len(query_tokens)
    basename_overlap = len(set(query_tokens) & basename_tokens) / len(query_tokens)
    content = rarity = 0.0
    line_weights: Counter[int] = Counter()
    for row in posting.values():
        count = int(row[7])
        document_frequency = int(row[9])
        weight = math.log((file_count + 1) / (document_frequency + 1)) + 1.0
        rarity += weight
        if count:
            content += weight * min(1.0, count / 3.0)
            for line in json.loads(row[8]):
                line_weights[int(line)] += weight
    coverage = len(matched) / len(query_tokens)
    kind = SourceKind(first[3])
    source_adjustment = _source_adjustment(kind, preferred, str(first[0]))
    final = (
        30.0 * basename_overlap
        + 2.0 * path_overlap
        + 1.2 * coverage
        + 0.35 * content
        + 0.15 * rarity
        + source_adjustment
    )
    center = (
        min(line_weights, key=lambda line: (-line_weights[line], line))
        if line_weights
        else 1
    )
    start = max(1, center - RECOMMENDED_RANGE_LINES // 3)
    end = start + RECOMMENDED_RANGE_LINES - 1
    return LexicalMatch(
        first[0],
        start,
        end,
        first[1],
        first[2],
        kind,
        matched,
        LexicalScore(
            path_overlap,
            basename_overlap,
            content,
            rarity,
            source_adjustment,
            coverage,
            final,
        ),
    )


def _source_adjustment(
    kind: SourceKind, preferred: SourceKind | None, path: str
) -> float:
    root = PurePosixPath(path).parts[0].casefold()
    layout_adjustment = 5.0 if root in {"src", "source", "lib", "app"} else 0.0
    if preferred is not None:
        if kind is preferred:
            return 1.5 + layout_adjustment
        if kind is SourceKind.THIRD_PARTY:
            return -0.7
        return -0.4
    if kind is SourceKind.IMPLEMENTATION:
        return 0.25 + layout_adjustment
    return -0.25 if kind is SourceKind.THIRD_PARTY else 0.0


def _diversify(
    ranked: list[LexicalMatch],
    basename_tokens: dict[str, frozenset[str]],
    query_tokens: tuple[str, ...],
    limit: int,
) -> tuple[LexicalMatch, ...]:
    """Retain top scores while representing distinct basename concepts."""
    if len(ranked) <= limit:
        return tuple(ranked)
    selected = list(ranked[: max(1, limit // 2)])
    covered = set().union(*(basename_tokens[item.path] for item in selected)) & set(
        query_tokens
    )
    positions = {item.path: position for position, item in enumerate(ranked)}
    source_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".java",
        ".js",
        ".kt",
        ".php",
        ".py",
        ".rb",
        ".rs",
        ".swift",
        ".ts",
    }
    for token in query_tokens:
        if token in covered or len(selected) >= limit:
            continue
        candidates = [
            item
            for item in ranked
            if item not in selected and token in basename_tokens[item.path]
        ]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda item: (
                item.extension not in source_suffixes,
                positions[item.path],
                item.path,
            ),
        )
        selected.append(chosen)
        covered.update(basename_tokens[chosen.path] & set(query_tokens))
    selected.extend(item for item in ranked if item not in selected)
    return tuple(selected[:limit])


def _scope(path: str) -> tuple[str, tuple[str, ...]]:
    if path in {"", "."}:
        return "1=1", ()
    normalized = path.strip("/")
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return "(f.path=? OR f.path LIKE ? ESCAPE '\\')", (normalized, f"{escaped}/%")
