"""Bounded deterministic task evidence plans and trusted coverage state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from forge.retrieval import SourceKind, classify_source

MAX_EVIDENCE_GOALS = 4
MAX_GOAL_DESCRIPTION = 240


class EvidenceGoalKind(StrEnum):
    IMPLEMENTATION = "implementation"
    TEST = "test"
    CONFIGURATION = "configuration"
    RELATIONSHIP = "relationship"
    OTHER = "other"


class EvidenceGoalStatus(StrEnum):
    UNRESOLVED = "unresolved"
    DISCOVERY_ONLY = "discovery_only"
    SOURCE_COVERED = "source_covered"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EvidenceGoal:
    goal_id: str
    description: str
    kind: EvidenceGoalKind = EvidenceGoalKind.IMPLEMENTATION
    required: bool = True
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.goal_id or not self.goal_id.isascii():
            raise ValueError("goal_id must be non-empty ASCII")
        if not self.description.strip() or len(self.description) > MAX_GOAL_DESCRIPTION:
            raise ValueError(
                "goal description must be non-empty and at most 240 characters"
            )
        if not isinstance(self.kind, EvidenceGoalKind):
            raise TypeError("goal kind must be EvidenceGoalKind")


@dataclass(frozen=True, slots=True)
class TaskEvidencePlan:
    goals: tuple[EvidenceGoal, ...]

    def __post_init__(self) -> None:
        goals = tuple(self.goals)
        if not goals or len(goals) > MAX_EVIDENCE_GOALS:
            raise ValueError("evidence plans require between 1 and 4 goals")
        ids = {goal.goal_id for goal in goals}
        if len(ids) != len(goals):
            raise ValueError("evidence goal IDs must be unique")
        if any(
            dependency not in ids for goal in goals for dependency in goal.depends_on
        ):
            raise ValueError("evidence goal dependency is unknown")
        _reject_cycles(goals)
        object.__setattr__(self, "goals", goals)


@dataclass(frozen=True, slots=True)
class EvidenceCoverage:
    goal_id: str
    path: str
    generation: int
    evidence_type: str
    source_kind: SourceKind
    observation_id: str


@dataclass(frozen=True, slots=True)
class EvidenceGoalResult:
    goal_id: str
    description: str
    required: bool
    status: EvidenceGoalStatus
    source_paths: tuple[str, ...]


class EvidenceCoverageState:
    def __init__(self, plan: TaskEvidencePlan) -> None:
        self.plan = plan
        self._statuses = {
            goal.goal_id: EvidenceGoalStatus.UNRESOLVED for goal in plan.goals
        }
        self._coverage: list[EvidenceCoverage] = []
        self.premature_finals = 0
        self.goal_transitions = 0

    @property
    def active_goal(self) -> EvidenceGoal | None:
        return next(
            (
                goal
                for goal in self.plan.goals
                if goal.required
                and self._statuses[goal.goal_id]
                not in {EvidenceGoalStatus.SOURCE_COVERED, EvidenceGoalStatus.FAILED}
            ),
            None,
        )

    @property
    def complete(self) -> bool:
        return all(
            not goal.required
            or self._statuses[goal.goal_id] is EvidenceGoalStatus.SOURCE_COVERED
            for goal in self.plan.goals
        )

    def note_discovery(self, goal_id: str) -> None:
        if self._statuses[goal_id] is EvidenceGoalStatus.UNRESOLVED:
            self._statuses[goal_id] = EvidenceGoalStatus.DISCOVERY_ONLY

    def register_source(
        self, goal_id: str, path: str, generation: int, observation_id: str
    ) -> bool:
        goal = next(goal for goal in self.plan.goals if goal.goal_id == goal_id)
        kind = classify_source(path)
        if not _kind_satisfies(goal.kind, kind) or any(
            self._statuses[dep] is not EvidenceGoalStatus.SOURCE_COVERED
            for dep in goal.depends_on
        ):
            return False
        self._coverage.append(
            EvidenceCoverage(
                goal_id, path, generation, "source_content", kind, observation_id
            )
        )
        if self._statuses[goal_id] is not EvidenceGoalStatus.SOURCE_COVERED:
            self._statuses[goal_id] = EvidenceGoalStatus.SOURCE_COVERED
            self.goal_transitions += 1
        self._cover_relationships()
        return True

    def invalidate_path(self, path: str) -> None:
        affected = {item.goal_id for item in self._coverage if item.path == path}
        self._coverage = [item for item in self._coverage if item.path != path]
        for goal_id in affected:
            if not any(item.goal_id == goal_id for item in self._coverage):
                self._statuses[goal_id] = EvidenceGoalStatus.UNRESOLVED

    def results(self) -> tuple[EvidenceGoalResult, ...]:
        return tuple(
            EvidenceGoalResult(
                goal.goal_id,
                goal.description,
                goal.required,
                self._statuses[goal.goal_id],
                tuple(
                    sorted(
                        {
                            item.path
                            for item in self._coverage
                            if item.goal_id == goal.goal_id
                        }
                    )
                ),
            )
            for goal in self.plan.goals
        )

    def _cover_relationships(self) -> None:
        for goal in self.plan.goals:
            if (
                goal.kind is EvidenceGoalKind.RELATIONSHIP
                and goal.depends_on
                and all(
                    self._statuses[dep] is EvidenceGoalStatus.SOURCE_COVERED
                    for dep in goal.depends_on
                )
            ):
                self._statuses[goal.goal_id] = EvidenceGoalStatus.SOURCE_COVERED


def default_evidence_plan(task: str) -> TaskEvidencePlan:
    return TaskEvidencePlan(
        (
            EvidenceGoal(
                "G1", task.strip()[:MAX_GOAL_DESCRIPTION], EvidenceGoalKind.OTHER
            ),
        )
    )


def _kind_satisfies(goal: EvidenceGoalKind, source: SourceKind) -> bool:
    if goal is EvidenceGoalKind.TEST:
        return source is SourceKind.TEST
    if goal is EvidenceGoalKind.CONFIGURATION:
        return source in {SourceKind.CONFIGURATION, SourceKind.IMPLEMENTATION}
    if goal is EvidenceGoalKind.IMPLEMENTATION:
        return source is SourceKind.IMPLEMENTATION
    return source is not SourceKind.GENERATED_METADATA


def _reject_cycles(goals: tuple[EvidenceGoal, ...]) -> None:
    graph = {goal.goal_id: goal.depends_on for goal in goals}

    def visit(node: str, active: set[str], done: set[str]) -> None:
        if node in active:
            raise ValueError("evidence goal dependencies must be acyclic")
        if node in done:
            return
        active.add(node)
        for child in graph[node]:
            visit(child, active, done)
        active.remove(node)
        done.add(node)

    done: set[str] = set()
    for node in graph:
        visit(node, set(), done)
