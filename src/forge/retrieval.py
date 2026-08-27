"""Deterministic, model-independent hybrid retrieval ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import PurePosixPath

RETRIEVAL_RANKING_VERSION = 1
SEMANTIC_WEIGHT = 0.86
LEXICAL_WEIGHT = 0.10
STRUCTURAL_WEIGHT = 0.025
MAX_RESULTS_PER_FILE = 3

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "does",
        "for",
        "how",
        "in",
        "is",
        "of",
        "or",
        "the",
        "to",
        "what",
        "where",
        "which",
    }
)
_CODE_SUFFIXES = frozenset(
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
    }
)
_CONFIG_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".lock", ".toml", ".yaml", ".yml"}
)
_STRUCTURAL_KINDS = frozenset(
    {"class", "function", "method", "async_function", "async_method"}
)


class SourceKind(StrEnum):
    IMPLEMENTATION = "implementation"
    TEST = "test"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    GENERATED_METADATA = "generated_metadata"
    THIRD_PARTY = "third_party"
    OTHER_TEXT = "other_text"


class RetrievalRankingError(RuntimeError):
    """Candidate data cannot be ranked safely."""


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    path: str
    line_start: int
    line_end: int
    language: str
    symbol: str | None
    qualified_name: str | None
    chunk_kind: str
    chunk_sha256: str
    semantic_similarity: float
    source_text: str = ""
    source_kind: SourceKind = SourceKind.OTHER_TEXT
    lexical_score: float = 0.0
    structural_score: float = 0.0
    final_score: float = 0.0


def classify_source(path: str) -> SourceKind:
    """Classify a repository path with one centralized, deterministic policy."""
    normalized = path.replace("\\", "/")
    parts = tuple(part.casefold() for part in PurePosixPath(normalized).parts)
    name = parts[-1] if parts else ""
    suffix = PurePosixPath(name).suffix
    if any(part in {"vendor", "third_party", "external"} for part in parts):
        return SourceKind.THIRD_PARTY
    if any(part.endswith((".egg-info", ".dist-info")) for part in parts) or name in {
        "pkg-info",
        "sources.txt",
    }:
        return SourceKind.GENERATED_METADATA
    if (
        "tests" in parts
        or "test" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
    ):
        return SourceKind.TEST
    if (
        "docs" in parts
        or name.startswith("readme")
        or suffix in {".md", ".rst", ".adoc"}
    ):
        return SourceKind.DOCUMENTATION
    if suffix in _CONFIG_SUFFIXES or name in {
        "build.gradle",
        "cmakelists.txt",
        "dockerfile",
        "makefile",
        "pom.xml",
        "pyproject.toml",
    }:
        return SourceKind.CONFIGURATION
    if suffix in _CODE_SUFFIXES:
        return SourceKind.IMPLEMENTATION
    return SourceKind.OTHER_TEXT


def is_generated_metadata_path(path: str) -> bool:
    return classify_source(path) is SourceKind.GENERATED_METADATA


def tokenize(value: str) -> frozenset[str]:
    """Tokenize identifiers, dotted names, and paths without NLP dependencies."""
    split_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    split_acronym = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", split_camel)
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", split_acronym.casefold())
        if token not in _STOPWORDS and len(token) > 1
    )


def rank_candidates(
    query: str, candidates: tuple[RetrievalCandidate, ...], *, limit: int
) -> tuple[RetrievalCandidate, ...]:
    """Score, deduplicate, sort, and diversify a bounded candidate collection."""
    if limit < 1:
        raise RetrievalRankingError("limit must be positive")
    query_tokens = tokenize(query)
    scored = []
    for candidate in candidates:
        if (
            not candidate.path
            or candidate.line_start < 1
            or candidate.line_end < candidate.line_start
        ):
            raise RetrievalRankingError("invalid retrieval candidate")
        kind = classify_source(candidate.path)
        metadata = " ".join(
            filter(None, (candidate.path, candidate.symbol, candidate.qualified_name))
        )
        metadata_overlap = _coverage(query_tokens, tokenize(metadata))
        source_overlap = _coverage(query_tokens, tokenize(candidate.source_text))
        lexical = 0.7 * metadata_overlap + 0.3 * source_overlap
        structural = 1.0 if candidate.chunk_kind in _STRUCTURAL_KINDS else 0.25
        prior = {
            SourceKind.IMPLEMENTATION: 0.025,
            SourceKind.TEST: -0.01,
            SourceKind.CONFIGURATION: 0.0,
            SourceKind.DOCUMENTATION: -0.01,
            SourceKind.OTHER_TEXT: -0.01,
            SourceKind.GENERATED_METADATA: -0.08,
            SourceKind.THIRD_PARTY: -0.035,
        }[kind]
        final = (
            SEMANTIC_WEIGHT * candidate.semantic_similarity
            + LEXICAL_WEIGHT * lexical
            + STRUCTURAL_WEIGHT * structural
            + prior
        )
        scored.append(
            replace(
                candidate,
                source_kind=kind,
                lexical_score=lexical,
                structural_score=structural,
                final_score=final,
            )
        )
    scored.sort(
        key=lambda item: (
            -item.final_score,
            -item.semantic_similarity,
            item.path,
            item.line_start,
            item.line_end,
        )
    )

    unique: list[RetrievalCandidate] = []
    seen_locations: set[tuple[str, int, int]] = set()
    seen_hashes: set[str] = set()
    for candidate in scored:
        location = (candidate.path, candidate.line_start, candidate.line_end)
        if location in seen_locations or (
            candidate.chunk_sha256 and candidate.chunk_sha256 in seen_hashes
        ):
            continue
        seen_locations.add(location)
        if candidate.chunk_sha256:
            seen_hashes.add(candidate.chunk_sha256)
        unique.append(candidate)

    selected: list[RetrievalCandidate] = []
    deferred: list[RetrievalCandidate] = []
    counts: dict[str, int] = {}
    for candidate in unique:
        if counts.get(candidate.path, 0) >= MAX_RESULTS_PER_FILE:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        counts[candidate.path] = counts.get(candidate.path, 0) + 1
        if len(selected) == limit:
            return tuple(selected)
    selected.extend(deferred[: limit - len(selected)])
    return tuple(selected[:limit])


def _coverage(query: frozenset[str], candidate: frozenset[str]) -> float:
    return len(query & candidate) / len(query) if query else 0.0
