"""Deterministic retrieval-v1 comparison harness (no generation model required)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from forge.retrieval import SourceKind
from forge.semantic_index import SemanticMatch

RETRIEVAL_V1 = "retrieval-v1"
RETRIEVAL_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class RetrievalTask:
    task_id: str
    query: str
    expected_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalTaskResult:
    task_id: str
    raw_rank: int | None
    reranked_rank: int | None
    raw_paths: tuple[str, ...]
    reranked_paths: tuple[str, ...]
    raw_generated_metadata: int
    reranked_generated_metadata: int
    raw_docs_before_implementation: bool
    reranked_docs_before_implementation: bool
    raw_distinct_files: int
    reranked_distinct_files: int


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    tasks: int
    raw_top1: float
    raw_top3: float
    raw_top5: float
    reranked_top1: float
    reranked_top3: float
    reranked_top5: float
    raw_generated_metadata: int
    reranked_generated_metadata: int
    raw_docs_before_implementation: int
    reranked_docs_before_implementation: int
    raw_mean_distinct_files: float
    reranked_mean_distinct_files: float


RETRIEVAL_V1_TASKS = (
    RetrievalTask(
        "H01",
        "What determines whether failed verification permits another repair?",
        ("src/forge/orchestration/coding_task.py",),
    ),
    RetrievalTask(
        "H02",
        "Where does Forge compact old observations to fit the context budget?",
        ("src/forge/context_planner.py",),
    ),
    RetrievalTask(
        "H03",
        "What code decides whether a tool action requires user approval?",
        ("src/forge/interaction.py",),
    ),
    RetrievalTask(
        "H04",
        "Where is the persistent repository symbol and reference index managed?",
        ("src/forge/repository_index.py",),
    ),
    RetrievalTask(
        "H05",
        "Where are configured project build and test commands executed?",
        ("src/forge/tools/project.py", "src/forge/project_config.py"),
    ),
    RetrievalTask(
        "H06",
        "How are repository tool paths confined to the configured workspace?",
        ("src/forge/tools/paths.py",),
    ),
)


class _SearchIndex(Protocol):
    def search_raw(self, query: str, *, limit: int) -> tuple[SemanticMatch, ...]: ...

    def search(self, query: str, *, limit: int) -> tuple[SemanticMatch, ...]: ...


def evaluate_retrieval(
    index: _SearchIndex,
    tasks: tuple[RetrievalTask, ...] = RETRIEVAL_V1_TASKS,
    *,
    limit: int = 5,
) -> tuple[tuple[RetrievalTaskResult, ...], RetrievalMetrics]:
    """Compare semantic-only and hybrid ranks over a fixed bounded result set."""
    results = []
    for task in tasks:
        raw = index.search_raw(task.query, limit=limit)
        reranked = index.search(task.query, limit=limit)
        results.append(_task_result(task, raw, reranked))
    frozen = tuple(results)
    return frozen, _metrics(frozen)


def _task_result(
    task: RetrievalTask,
    raw: tuple[SemanticMatch, ...],
    reranked: tuple[SemanticMatch, ...],
) -> RetrievalTaskResult:
    return RetrievalTaskResult(
        task.task_id,
        _rank(raw, task.expected_paths),
        _rank(reranked, task.expected_paths),
        tuple(item.path for item in raw),
        tuple(item.path for item in reranked),
        _count_kind(raw, SourceKind.GENERATED_METADATA),
        _count_kind(reranked, SourceKind.GENERATED_METADATA),
        _docs_before_implementation(raw),
        _docs_before_implementation(reranked),
        len({item.path for item in raw}),
        len({item.path for item in reranked}),
    )


def _rank(matches: tuple[SemanticMatch, ...], paths: tuple[str, ...]) -> int | None:
    return next(
        (index for index, item in enumerate(matches, 1) if item.path in paths), None
    )


def _count_kind(matches: tuple[SemanticMatch, ...], kind: SourceKind) -> int:
    return sum(item.source_kind is kind for item in matches)


def _docs_before_implementation(matches: tuple[SemanticMatch, ...]) -> bool:
    first_docs = next(
        (
            i
            for i, item in enumerate(matches)
            if item.source_kind is SourceKind.DOCUMENTATION
        ),
        None,
    )
    first_code = next(
        (
            i
            for i, item in enumerate(matches)
            if item.source_kind is SourceKind.IMPLEMENTATION
        ),
        None,
    )
    return first_docs is not None and (first_code is None or first_docs < first_code)


def _metrics(results: tuple[RetrievalTaskResult, ...]) -> RetrievalMetrics:
    count = len(results)

    def rate(attribute: str, cutoff: int) -> float:
        return (
            sum(
                (rank := getattr(item, attribute)) is not None and rank <= cutoff
                for item in results
            )
            / count
            if count
            else 0.0
        )

    def mean(attribute: str) -> float:
        return (
            sum(getattr(item, attribute) for item in results) / count if count else 0.0
        )

    return RetrievalMetrics(
        count,
        rate("raw_rank", 1),
        rate("raw_rank", 3),
        rate("raw_rank", 5),
        rate("reranked_rank", 1),
        rate("reranked_rank", 3),
        rate("reranked_rank", 5),
        sum(item.raw_generated_metadata for item in results),
        sum(item.reranked_generated_metadata for item in results),
        sum(item.raw_docs_before_implementation for item in results),
        sum(item.reranked_docs_before_implementation for item in results),
        mean("raw_distinct_files"),
        mean("reranked_distinct_files"),
    )
