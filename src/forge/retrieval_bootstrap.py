"""Bounded Forge-owned discovery bootstrap for active evidence goals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from forge.evidence_coverage import EvidenceGoal, EvidenceGoalKind
from forge.retrieval_strategy import RetrievalState
from forge.tools import PermissionDecision, ToolInvocation, ToolResult, ToolResultStatus

SEMANTIC_TOOL = "repository.semantic_search"
LEXICAL_TOOL = "repository.lexical_search"


class BootstrapProvider(StrEnum):
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    NONE = "none"


class BootstrapReason(StrEnum):
    ELIGIBLE = "eligible"
    NO_GOAL = "no_goal"
    RELATIONSHIP = "relationship"
    CANDIDATES_AVAILABLE = "candidates_available"
    ALREADY_USED = "already_used"
    UNAVAILABLE = "unavailable"
    PERMISSION = "permission"


@dataclass(frozen=True, slots=True)
class BootstrapRequest:
    goal_id: str
    generation: int
    provider: BootstrapProvider
    query: str
    invocation: ToolInvocation


@dataclass(frozen=True, slots=True)
class BootstrapMetrics:
    attempts: int = 0
    executions: int = 0
    successes: int = 0
    empty_results: int = 0
    failures: int = 0
    candidates: int = 0
    tool_executions: int = 0
    model_discovery_calls_after_bootstrap: int = 0
    semantic_ready_bootstraps: int = 0
    lexical_fallback_bootstraps: int = 0


class RetrievalBootstrap:
    """Own one automatic discovery attempt per goal and generation."""

    def __init__(self) -> None:
        self._used: set[tuple[str, int]] = set()
        self._sequence = 0
        self._metrics = BootstrapMetrics()

    @property
    def metrics(self) -> BootstrapMetrics:
        return self._metrics

    def prepare(
        self,
        goal: EvidenceGoal | None,
        *,
        generation: int,
        retrieval_state: RetrievalState,
        actionable_candidates: int,
        semantic_available: bool,
        permission: PermissionDecision,
        semantic_ready: bool | None = None,
        lexical_available: bool = False,
    ) -> tuple[BootstrapRequest | None, BootstrapReason]:
        self._metrics = replace(self._metrics, attempts=self._metrics.attempts + 1)
        if goal is None:
            return None, BootstrapReason.NO_GOAL
        if goal.kind is EvidenceGoalKind.RELATIONSHIP:
            return None, BootstrapReason.RELATIONSHIP
        if actionable_candidates or retrieval_state in {
            RetrievalState.CANDIDATES_AVAILABLE,
            RetrievalState.TARGET_IDENTIFIED,
        }:
            return None, BootstrapReason.CANDIDATES_AVAILABLE
        key = (goal.goal_id, generation)
        if key in self._used:
            return None, BootstrapReason.ALREADY_USED
        semantic_is_ready = (
            semantic_available if semantic_ready is None else semantic_ready
        )
        if semantic_available and semantic_is_ready:
            provider = BootstrapProvider.SEMANTIC
            tool = SEMANTIC_TOOL
        elif lexical_available:
            provider = BootstrapProvider.LEXICAL
            tool = LEXICAL_TOOL
        else:
            return None, BootstrapReason.UNAVAILABLE
        if permission is not PermissionDecision.ALLOW:
            return None, BootstrapReason.PERMISSION
        self._used.add(key)
        self._sequence += 1
        invocation = ToolInvocation(
            f"forge-bootstrap-{self._sequence}-{goal.goal_id}-g{generation}",
            tool,
            {
                "query": goal.description,
                **(
                    {"preferred_source_kind": goal.kind.value}
                    if provider is BootstrapProvider.LEXICAL
                    and goal.kind.value in {"implementation", "test", "configuration"}
                    else {}
                ),
            },
        )
        return (
            BootstrapRequest(
                goal.goal_id,
                generation,
                provider,
                goal.description,
                invocation,
            ),
            BootstrapReason.ELIGIBLE,
        )

    def record(
        self,
        result: ToolResult,
        candidate_count: int,
        provider: BootstrapProvider | None = None,
    ) -> None:
        success = result.status is ToolResultStatus.SUCCESS
        self._metrics = replace(
            self._metrics,
            executions=self._metrics.executions + 1,
            successes=self._metrics.successes + int(success and candidate_count > 0),
            empty_results=self._metrics.empty_results
            + int(success and candidate_count == 0),
            failures=self._metrics.failures + int(not success),
            candidates=self._metrics.candidates + candidate_count,
            tool_executions=self._metrics.tool_executions + 1,
            semantic_ready_bootstraps=self._metrics.semantic_ready_bootstraps
            + int(provider is BootstrapProvider.SEMANTIC),
            lexical_fallback_bootstraps=self._metrics.lexical_fallback_bootstraps
            + int(provider is BootstrapProvider.LEXICAL),
        )

    def note_model_discovery(self) -> None:
        if self._used:
            self._metrics = replace(
                self._metrics,
                model_discovery_calls_after_bootstrap=(
                    self._metrics.model_discovery_calls_after_bootstrap + 1
                ),
            )
