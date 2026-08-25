"""Bounded deterministic task evidence plans and trusted coverage state."""

from __future__ import annotations

import re
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

    @property
    def has_required_failure(self) -> bool:
        return any(
            goal.required and self._statuses[goal.goal_id] is EvidenceGoalStatus.FAILED
            for goal in self.plan.goals
        )

    def note_discovery(self, goal_id: str) -> None:
        if self._statuses[goal_id] is EvidenceGoalStatus.UNRESOLVED:
            self._statuses[goal_id] = EvidenceGoalStatus.DISCOVERY_ONLY

    def mark_failed(self, goal_id: str) -> None:
        if self._statuses[goal_id] is not EvidenceGoalStatus.SOURCE_COVERED:
            self._statuses[goal_id] = EvidenceGoalStatus.FAILED
            self.goal_transitions += 1

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

    def required_observation_ids(self) -> tuple[tuple[str, str, str], ...]:
        """Select one trusted current source observation per covered source goal."""
        selected = []
        for goal in self.plan.goals:
            if goal.kind is EvidenceGoalKind.RELATIONSHIP or not goal.required:
                continue
            match = next(
                (item for item in self._coverage if item.goal_id == goal.goal_id),
                None,
            )
            if match is not None:
                selected.append((goal.goal_id, match.observation_id, match.path))
        return tuple(selected)

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


_LIST_ITEM = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)(\S.*)$")
_RELATIONSHIP = re.compile(
    r"^\s*how\s+do\s+(.+?)\s+and\s+(.+?)\s+"
    r"(work\s+together|interact|connect)(?:\s+.*)?[?.!]?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def decompose_evidence_plan(task: str) -> TaskEvidencePlan:
    """Build a conservative, deterministic evidence plan from request structure."""
    normalized = " ".join(task.strip().split())
    if not normalized:
        raise ValueError("task must be non-empty")

    relationship = _RELATIONSHIP.match(normalized)
    if relationship is not None:
        left = _clean_description(relationship.group(1))
        right = _clean_description(relationship.group(2))
        if (
            left
            and right
            and len(left) <= MAX_GOAL_DESCRIPTION
            and len(right) <= MAX_GOAL_DESCRIPTION
        ):
            return TaskEvidencePlan(
                (
                    EvidenceGoal("G1", left, EvidenceGoalKind.IMPLEMENTATION),
                    EvidenceGoal("G2", right, EvidenceGoalKind.IMPLEMENTATION),
                    EvidenceGoal(
                        "G3",
                        f"relationship between {left} and {right}"[
                            :MAX_GOAL_DESCRIPTION
                        ],
                        EvidenceGoalKind.RELATIONSHIP,
                        depends_on=("G1", "G2"),
                    ),
                )
            )

    listed = [
        match.group(1).strip()
        for line in task.splitlines()
        if (match := _LIST_ITEM.match(line)) is not None
    ]
    if len(listed) >= 2:
        return _facet_plan(listed, task)

    separated = _split_semicolons(task)
    if len(separated) >= 2:
        return _facet_plan(separated, task)
    return default_evidence_plan(normalized)


def _facet_plan(facets: list[str], fallback: str) -> TaskEvidencePlan:
    cleaned = [_clean_description(item) for item in facets]
    if len(cleaned) > MAX_EVIDENCE_GOALS or any(
        not item or len(item) > MAX_GOAL_DESCRIPTION for item in cleaned
    ):
        return default_evidence_plan(" ".join(fallback.strip().split()))
    return TaskEvidencePlan(
        tuple(
            EvidenceGoal(f"G{index}", item, EvidenceGoalKind.OTHER)
            for index, item in enumerate(cleaned, 1)
        )
    )


def _clean_description(value: str) -> str:
    return " ".join(value.strip().rstrip("?.!;").split())


def _split_semicolons(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    in_code = False
    for character in value:
        if character == "`":
            in_code = not in_code
        elif character in {'"', "'"} and not in_code:
            quote = (
                None if quote == character else character if quote is None else quote
            )
        if character == ";" and quote is None and not in_code:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


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
