"""Deterministic repository retrieval workflow state and tool routing."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType

from forge.tools import ToolResult, ToolResultStatus

LOGGER = logging.getLogger(__name__)
MAX_RETRIEVAL_CANDIDATES = 32
BROAD_DISCOVERY_TOOLS = frozenset(
    {
        "repository.semantic_search",
        "repository.search_files",
        "repository.list_directory",
    }
)
TARGETED_TOOLS = frozenset(
    {
        "repository.read_range",
        "repository.read_file",
        "repository.file_outline",
        "repository.find_symbol",
        "repository.find_references",
    }
)


class RetrievalState(StrEnum):
    UNSTARTED = "unstarted"
    DISCOVERING = "discovering"
    CANDIDATES_AVAILABLE = "candidates_available"
    TARGET_IDENTIFIED = "target_identified"
    SOURCE_ACQUIRED = "source_acquired"
    EXHAUSTED = "exhausted"


class CandidateSource(StrEnum):
    SEMANTIC = "semantic"
    SYMBOL = "symbol"
    REFERENCE = "reference"
    LEXICAL = "lexical"
    OUTLINE = "outline"
    DIRECTORY = "directory"


class InspectionState(StrEnum):
    UNINSPECTED = "uninspected"
    INSPECTED = "inspected"
    FAILED = "failed"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class RetrievalCandidateState:
    path: str
    start_line: int | None
    end_line: int | None
    symbol: str | None
    source: CandidateSource
    priority: int
    generation: int
    inspection: InspectionState = InspectionState.UNINSPECTED

    @property
    def identity(self) -> tuple[str, int | None, int | None, str | None, int]:
        return (self.path, self.start_line, self.end_line, self.symbol, self.generation)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    transitions: int = 0
    candidates_discovered: int = 0
    candidates_inspected: int = 0
    candidate_failures: int = 0
    broad_discovery_calls: int = 0
    targeted_inspections: int = 0
    suppressed_discovery_attempts: int = 0
    candidate_set_repeats: int = 0
    source_acquired_transitions: int = 0
    candidates_truncated: int = 0


class RetrievalStrategy:
    """Own bounded candidate state; it can narrow but never expand a registry."""

    def __init__(self, *, candidate_limit: int = MAX_RETRIEVAL_CANDIDATES) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self._limit = candidate_limit
        self._state = RetrievalState.UNSTARTED
        self._candidates: tuple[RetrievalCandidateState, ...] = ()
        self._signatures: set[frozenset[tuple[object, ...]]] = set()
        self._metrics = RetrievalMetrics()

    @property
    def state(self) -> RetrievalState:
        return self._state

    @property
    def candidates(self) -> tuple[RetrievalCandidateState, ...]:
        return self._candidates

    @property
    def metrics(self) -> RetrievalMetrics:
        return self._metrics

    @property
    def unresolved(self) -> tuple[RetrievalCandidateState, ...]:
        return tuple(
            c for c in self._candidates if c.inspection is InspectionState.UNINSPECTED
        )

    def allowed_tools(
        self, available: set[str], *, evidence_sufficient: bool
    ) -> set[str]:
        allowed = set(available)
        if self.unresolved or (
            self._state is RetrievalState.SOURCE_ACQUIRED and evidence_sufficient
        ):
            allowed -= BROAD_DISCOVERY_TOOLS
        return allowed

    def observe(self, result: ToolResult, *, generation: int) -> bool:
        """Consume one trusted tool result and report candidate-set novelty."""
        tool = result.tool_name
        if tool in BROAD_DISCOVERY_TOOLS:
            self._metrics = replace(
                self._metrics,
                broad_discovery_calls=self._metrics.broad_discovery_calls + 1,
            )
        if tool in {"repository.read_file", "repository.read_range"}:
            self._metrics = replace(
                self._metrics,
                targeted_inspections=self._metrics.targeted_inspections + 1,
            )
            return self._observe_read(result, generation)
        produced = _result_candidates(result, generation)
        if not produced:
            if tool in BROAD_DISCOVERY_TOOLS or tool in TARGETED_TOOLS:
                self._transition(RetrievalState.DISCOVERING)
            return False
        signature = frozenset(candidate.identity for candidate in produced)
        repeated = signature in self._signatures
        self._signatures.add(signature)
        if repeated:
            self._metrics = replace(
                self._metrics,
                candidate_set_repeats=self._metrics.candidate_set_repeats + 1,
            )
        added = self._merge(produced)
        exact = tool == "repository.find_symbol" and len(produced) == 1
        self._transition(
            RetrievalState.TARGET_IDENTIFIED
            if exact
            else RetrievalState.CANDIDATES_AVAILABLE
        )
        return bool(added) and not repeated

    def invalidate_path(self, path: str, *, generation: int) -> None:
        self._candidates = tuple(
            replace(candidate, inspection=InspectionState.STALE)
            if candidate.path == path and candidate.generation < generation
            else candidate
            for candidate in self._candidates
        )
        if not self.unresolved:
            self._transition(RetrievalState.DISCOVERING)

    def note_suppressed_discovery(self) -> None:
        self._metrics = replace(
            self._metrics,
            suppressed_discovery_attempts=self._metrics.suppressed_discovery_attempts
            + 1,
        )

    def _observe_read(self, result: ToolResult, generation: int) -> bool:
        path = _result_path(result)
        if path is None:
            return False
        context_rejection = (
            isinstance(result.output, Mapping)
            and result.output.get("context_admission") == "reject_too_large"
        )
        if result.status is ToolResultStatus.SUCCESS:
            inspection = InspectionState.INSPECTED
            self._metrics = replace(
                self._metrics,
                candidates_inspected=self._metrics.candidates_inspected + 1,
            )
            self._transition(RetrievalState.SOURCE_ACQUIRED)
        elif context_rejection:
            return False
        else:
            inspection = InspectionState.FAILED
            self._metrics = replace(
                self._metrics, candidate_failures=self._metrics.candidate_failures + 1
            )
        self._candidates = tuple(
            replace(candidate, inspection=inspection)
            if candidate.path == path and candidate.generation == generation
            else candidate
            for candidate in self._candidates
        )
        if inspection is InspectionState.FAILED and not self.unresolved:
            self._transition(RetrievalState.DISCOVERING)
        return inspection is InspectionState.INSPECTED

    def _merge(self, produced: tuple[RetrievalCandidateState, ...]) -> int:
        existing = {candidate.identity for candidate in self._candidates}
        additions = [
            candidate for candidate in produced if candidate.identity not in existing
        ]
        combined = sorted(
            (*self._candidates, *additions),
            key=lambda c: (
                c.inspection is not InspectionState.UNINSPECTED,
                c.priority,
                c.path,
                c.start_line or 0,
            ),
        )
        truncated = max(0, len(combined) - self._limit)
        self._candidates = tuple(combined[: self._limit])
        self._metrics = replace(
            self._metrics,
            candidates_discovered=self._metrics.candidates_discovered + len(additions),
            candidates_truncated=self._metrics.candidates_truncated + truncated,
        )
        return len(additions)

    def _transition(self, state: RetrievalState) -> None:
        if state is self._state:
            return
        LOGGER.debug(
            "Retrieval state transition %s -> %s", self._state.value, state.value
        )
        source = int(
            state is RetrievalState.SOURCE_ACQUIRED and self._state is not state
        )
        self._state = state
        self._metrics = replace(
            self._metrics,
            transitions=self._metrics.transitions + 1,
            source_acquired_transitions=self._metrics.source_acquired_transitions
            + source,
        )


def _result_path(result: ToolResult) -> str | None:
    if isinstance(result.output, Mapping) and isinstance(
        result.output.get("path"), str
    ):
        return result.output["path"]
    return None


def _result_candidates(
    result: ToolResult, generation: int
) -> tuple[RetrievalCandidateState, ...]:
    if result.status is not ToolResultStatus.SUCCESS or not isinstance(
        result.output, Mapping
    ):
        return ()
    source = {
        "repository.semantic_search": CandidateSource.SEMANTIC,
        "repository.search_files": CandidateSource.LEXICAL,
        "repository.find_symbol": CandidateSource.SYMBOL,
        "repository.find_references": CandidateSource.REFERENCE,
        "repository.file_outline": CandidateSource.OUTLINE,
        "repository.list_directory": CandidateSource.DIRECTORY,
    }.get(result.tool_name)
    if source is None:
        return ()
    values = result.output.get(
        "matches", result.output.get("references", result.output.get("entries", ()))
    )
    if result.tool_name == "repository.file_outline":
        values = (MappingProxyType({"path": result.output.get("path")}),)
    if not isinstance(values, tuple):
        return ()
    candidates = []
    for priority, item in enumerate(values):
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or item.get("type") == "directory"
        ):
            continue
        start = item.get("line_start", item.get("line_number"))
        end = item.get("line_end", start)
        recommended = item.get("recommended_range")
        if isinstance(recommended, Mapping):
            start, end = (
                recommended.get("start_line", start),
                recommended.get("end_line", end),
            )
        candidates.append(
            RetrievalCandidateState(
                item["path"],
                start if isinstance(start, int) else None,
                end if isinstance(end, int) else None,
                item.get("qualified_name", item.get("symbol"))
                if isinstance(item.get("qualified_name", item.get("symbol")), str)
                else None,
                source,
                priority,
                generation,
            )
        )
    return tuple(candidates)
