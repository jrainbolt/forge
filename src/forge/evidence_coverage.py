"""Bounded deterministic task evidence plans and trusted coverage state."""

from __future__ import annotations

import re
from collections import Counter
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


class EvidenceReasonKind(StrEnum):
    LEXICAL_MATCH = "lexical_match"
    SYMBOL_DEFINITION = "symbol_definition"
    SYMBOL_REFERENCE = "symbol_reference"
    INCLUDE_OR_IMPORT = "include_or_import"
    DECLARATION_IMPLEMENTATION = "declaration_implementation"
    DEPENDENCY_RELATION = "dependency_relation"


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
    semantic_matching: bool = False

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
class _SourceStructure:
    path: str
    definitions: frozenset[str]
    declarations: frozenset[str]
    references: frozenset[str]
    includes: frozenset[str]


@dataclass(frozen=True, slots=True)
class EvidenceGoalResult:
    goal_id: str
    description: str
    required: bool
    status: EvidenceGoalStatus
    source_paths: tuple[str, ...]
    evidence_reasons: tuple[str, ...] = ()


class EvidenceCoverageState:
    def __init__(self, plan: TaskEvidencePlan) -> None:
        self.plan = plan
        self._statuses = {
            goal.goal_id: EvidenceGoalStatus.UNRESOLVED for goal in plan.goals
        }
        self._coverage: list[EvidenceCoverage] = []
        self._reasons: dict[str, list[str]] = {goal.goal_id: [] for goal in plan.goals}
        self._structures: dict[str, _SourceStructure] = {}
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

    @property
    def unresolved_reference_symbols(self) -> tuple[str, ...]:
        """Bounded structural leads relevant to the active relationship goal."""
        active = self.active_goal
        if active is None or active.kind is not EvidenceGoalKind.RELATIONSHIP:
            return ()
        tokens = _semantic_tokens(active.description)
        defined = {
            symbol
            for structure in self._structures.values()
            for symbol in structure.definitions
        }
        return tuple(
            sorted(
                {
                    symbol
                    for structure in self._structures.values()
                    for symbol in structure.references
                    if symbol not in defined
                    and tokens.intersection(_semantic_tokens(symbol))
                }
            )[:4]
        )

    def note_discovery(self, goal_id: str) -> None:
        if self._statuses[goal_id] is EvidenceGoalStatus.UNRESOLVED:
            self._statuses[goal_id] = EvidenceGoalStatus.DISCOVERY_ONLY

    def mark_failed(self, goal_id: str) -> None:
        if self._statuses[goal_id] is not EvidenceGoalStatus.SOURCE_COVERED:
            self._statuses[goal_id] = EvidenceGoalStatus.FAILED
            self.goal_transitions += 1

    def register_source(
        self,
        goal_id: str,
        path: str,
        generation: int,
        observation_id: str,
        *,
        content: str | None = None,
        reason: EvidenceReasonKind = EvidenceReasonKind.LEXICAL_MATCH,
        detail: str = "task/source token overlap",
    ) -> bool:
        goal = next(goal for goal in self.plan.goals if goal.goal_id == goal_id)
        kind = classify_source(path)
        if (
            not _kind_satisfies(goal.kind, kind)
            or (
                self.plan.semantic_matching
                and goal.kind is not EvidenceGoalKind.RELATIONSHIP
                and content is not None
                and not _semantic_match(goal.description, path, content or "")
            )
            or any(
                self._statuses[dep] is not EvidenceGoalStatus.SOURCE_COVERED
                for dep in goal.depends_on
            )
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
        explanation = f"{reason.value}:{path}:{detail}"
        if explanation not in self._reasons[goal_id]:
            self._reasons[goal_id].append(explanation)
        if not self.plan.semantic_matching:
            self._cover_dependency_relationships()
        return True

    def register_matching_source(
        self,
        path: str,
        content: str,
        generation: int,
        observation_id: str,
    ) -> tuple[str, ...]:
        """Apply one trusted source to every semantically supported open goal."""
        self._structures[path] = _source_structure(path, content)
        covered = []
        for goal in self.plan.goals:
            if goal.kind is EvidenceGoalKind.RELATIONSHIP:
                continue
            structural = self._structural_owner_match(goal, path)
            if structural is not None:
                reason, detail = structural
                matched = self.register_source(
                    goal.goal_id,
                    path,
                    generation,
                    observation_id,
                    content=None,
                    reason=reason,
                    detail=detail,
                )
            else:
                matched = self.register_source(
                    goal.goal_id,
                    path,
                    generation,
                    observation_id,
                    content=content,
                )
            if matched:
                covered.append(goal.goal_id)
        for goal in self.plan.goals:
            if (
                goal.kind is EvidenceGoalKind.RELATIONSHIP
                and self._statuses[goal.goal_id]
                is not EvidenceGoalStatus.SOURCE_COVERED
                and all(
                    self._statuses[dependency] is EvidenceGoalStatus.SOURCE_COVERED
                    for dependency in goal.depends_on
                )
                and _semantic_match(goal.description, path, content)
                and self.register_source(
                    goal.goal_id,
                    path,
                    generation,
                    observation_id,
                    content=content,
                )
            ):
                covered.append(goal.goal_id)
        covered.extend(self._cover_structural_relationships(generation, observation_id))
        return tuple(covered)

    def _cover_dependency_relationships(self) -> None:
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
                reason = "dependency_relation:plan:covered dependencies"
                if reason not in self._reasons[goal.goal_id]:
                    self._reasons[goal.goal_id].append(reason)

    def invalidate_path(self, path: str) -> None:
        affected = {item.goal_id for item in self._coverage if item.path == path}
        self._coverage = [item for item in self._coverage if item.path != path]
        self._structures.pop(path, None)
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
                tuple(self._reasons[goal.goal_id]),
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

    def _structural_owner_match(
        self, goal: EvidenceGoal, path: str
    ) -> tuple[EvidenceReasonKind, str] | None:
        goal_tokens = _expanded_semantic_tokens(goal.description)
        other_goal_tokens = frozenset(
            token
            for item in self.plan.goals
            if item.goal_id != goal.goal_id
            for token in _semantic_tokens(item.description)
        )
        structure = self._structures[path]
        relevant_references = sorted(
            reference
            for reference in structure.references
            if other_goal_tokens.intersection(_semantic_tokens(reference))
        )
        for symbol in sorted(structure.definitions):
            support = set(goal_tokens.intersection(_semantic_tokens(symbol)))
            support.update(
                goal_tokens.intersection(
                    _semantic_tokens(" ".join(relevant_references))
                )
            )
            required = min(2, len(_semantic_tokens(goal.description)))
            if len(support) >= required and relevant_references:
                return (
                    EvidenceReasonKind.SYMBOL_DEFINITION,
                    f"{symbol} references {relevant_references[0]}",
                )
        for symbol in sorted(structure.declarations):
            if not goal_tokens.intersection(_semantic_tokens(symbol)):
                continue
            if any(
                symbol in other.definitions
                for other in self._structures.values()
                if other.path != path
            ):
                return EvidenceReasonKind.DECLARATION_IMPLEMENTATION, symbol
        return None

    def _cover_structural_relationships(
        self, generation: int, observation_id: str
    ) -> tuple[str, ...]:
        covered = []
        for goal in self.plan.goals:
            if (
                goal.kind is EvidenceGoalKind.RELATIONSHIP
                and goal.depends_on
                and self._statuses[goal.goal_id]
                is not EvidenceGoalStatus.SOURCE_COVERED
                and all(
                    self._statuses[dep] is EvidenceGoalStatus.SOURCE_COVERED
                    for dep in goal.depends_on
                )
            ):
                relation = _find_structural_relation(
                    tuple(self._structures.values()), goal.description
                )
                if relation is None:
                    continue
                kind, source_path, detail = relation
                self._coverage.append(
                    EvidenceCoverage(
                        goal.goal_id,
                        source_path,
                        generation,
                        "structural_relation",
                        classify_source(source_path),
                        observation_id,
                    )
                )
                self._statuses[goal.goal_id] = EvidenceGoalStatus.SOURCE_COVERED
                self._reasons[goal.goal_id].append(
                    f"{kind.value}:{source_path}:{detail}"
                )
                self.goal_transitions += 1
                covered.append(goal.goal_id)
        return tuple(covered)


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

    semantic = _semantic_facets(normalized)
    if semantic is not None:
        return semantic

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


_BEHAVIOR_TASK = re.compile(
    r"^(?:fix|repair|correct|restore|reject|prevent|ensure|change|update)\b",
    re.IGNORECASE,
)
_FACET_SPLIT = re.compile(r"\s+(?:so(?:\s+that)?|while)\s+", re.IGNORECASE)


def _semantic_facets(task: str) -> TaskEvidencePlan | None:
    """Conservatively decompose behavioral work without model or repository hints."""
    if not _BEHAVIOR_TASK.match(task):
        return None
    clauses = [part.strip(" .") for part in _FACET_SPLIT.split(task) if part.strip()]
    if not clauses:
        return None
    owners = [clauses[0]]
    first = clauses[0]
    both = re.search(r"\bboth\s+(.+?)\s+and\s+(.+)$", first, re.IGNORECASE)
    paired_for = re.search(r"^(.+?\bfor\s+.+?)\s+and\s+(.+?\bfor\s+.+)$", first)
    if both is not None:
        prefix = first[: both.start()].strip()
        owners = [f"{prefix} {both.group(1)}", both.group(2)]
    elif paired_for is not None:
        owners = [paired_for.group(1), paired_for.group(2)]
    owners = [item for item in owners if len(_semantic_tokens(item)) >= 2]
    if not owners:
        return None
    owners = owners[:3]
    goals = [
        EvidenceGoal(f"G{index}", item, EvidenceGoalKind.IMPLEMENTATION)
        for index, item in enumerate(owners, 1)
    ]
    if len(clauses) > 1 and len(goals) < MAX_EVIDENCE_GOALS:
        goals.append(
            EvidenceGoal(
                f"G{len(goals) + 1}",
                " ".join(clauses[1:])[:MAX_GOAL_DESCRIPTION],
                EvidenceGoalKind.RELATIONSHIP,
                depends_on=tuple(goal.goal_id for goal in goals),
            )
        )
    return TaskEvidencePlan(tuple(goals), semantic_matching=True)


_SEMANTIC_STOPWORDS = frozenset(
    {
        "a",
        "all",
        "and",
        "are",
        "both",
        "correct",
        "every",
        "fix",
        "for",
        "from",
        "it",
        "of",
        "only",
        "repair",
        "restore",
        "so",
        "the",
        "then",
        "to",
        "valid",
        "verify",
        "while",
        "whose",
    }
)


def _stem(value: str) -> str:
    semantic_forms = {
        "creation": "create",
        "destruction": "destroy",
        "rejection": "reject",
    }
    if value in semantic_forms:
        return semantic_forms[value]
    for suffix in (
        "ization",
        "ation",
        "ments",
        "ment",
        "ness",
        "ing",
        "ers",
        "es",
        "s",
    ):
        if value.endswith(suffix) and len(value) > len(suffix) + 3:
            return value[: -len(suffix)]
    return value


def _semantic_tokens(value: str) -> frozenset[str]:
    return frozenset(
        _stem(token.casefold())
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", value)
        if token.casefold() not in _SEMANTIC_STOPWORDS
    )


_SEMANTIC_EQUIVALENTS = {
    "boundary": frozenset({"edge", "limit", "threshold", "intensity"}),
}


def _expanded_semantic_tokens(value: str) -> frozenset[str]:
    tokens = set(_semantic_tokens(value))
    for token in tuple(tokens):
        tokens.update(_SEMANTIC_EQUIVALENTS.get(token, ()))
    return frozenset(tokens)


def _semantic_match(description: str, path: str, content: str) -> bool:
    goal = _semantic_tokens(description)
    evidence = _semantic_tokens(f"{path} {content}")
    overlap = goal.intersection(evidence)
    required = 1 if len(goal) == 1 else 2 if len(goal) <= 5 else 3
    return len(overlap) >= required


_C_DEFINITION = re.compile(
    r"(?m)^\s*(?:[A-Za-z_]\w*[\s*]+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)
_C_DECLARATION = re.compile(
    r"(?m)^\s*(?:[A-Za-z_]\w*[\s*]+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*;"
)
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_INCLUDE = re.compile(r'(?m)^\s*#\s*include\s*[<"]([^>"]+)[>"]')
_PY_DEFINITION = re.compile(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
_PY_IMPORT = re.compile(
    r"(?m)^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))"
)
_CONTROL_WORDS = frozenset({"if", "for", "while", "switch", "return", "sizeof"})


def _source_structure(path: str, content: str) -> _SourceStructure:
    definitions = frozenset(
        {match.group(1) for match in _C_DEFINITION.finditer(content)}
        | {match.group(1) for match in _PY_DEFINITION.finditer(content)}
    )
    declarations = frozenset(
        match.group(1) for match in _C_DECLARATION.finditer(content)
    )
    calls = Counter(match.group(1) for match in _CALL.finditer(content))
    references = frozenset(
        symbol
        for symbol, count in calls.items()
        if symbol not in _CONTROL_WORDS and (symbol not in definitions or count > 1)
    )
    includes = {
        match.group(1).replace("\\", "/") for match in _INCLUDE.finditer(content)
    }
    includes.update(
        next(group for group in match.groups() if group is not None)
        for match in _PY_IMPORT.finditer(content)
    )
    return _SourceStructure(
        path, definitions, declarations, references, frozenset(includes)
    )


def _find_structural_relation(
    structures: tuple[_SourceStructure, ...], description: str
) -> tuple[EvidenceReasonKind, str, str] | None:
    relationship_tokens = _semantic_tokens(description)
    for implementation in structures:
        for symbol in sorted(implementation.definitions):
            if not relationship_tokens.intersection(_semantic_tokens(symbol)):
                continue
            for caller in structures:
                if symbol in caller.references:
                    return (
                        EvidenceReasonKind.SYMBOL_REFERENCE,
                        implementation.path,
                        f"{caller.path} references definition {symbol}",
                    )
            for declaration in structures:
                if (
                    declaration.path != implementation.path
                    and symbol in declaration.declarations
                ):
                    return (
                        EvidenceReasonKind.DECLARATION_IMPLEMENTATION,
                        implementation.path,
                        f"{declaration.path} declares {symbol}",
                    )
    for importer in structures:
        for included in sorted(importer.includes):
            if not relationship_tokens.intersection(_semantic_tokens(included)):
                continue
            target = included.rsplit("/", 1)[-1]
            for dependency in structures:
                if dependency.path == importer.path:
                    continue
                dependency_name = dependency.path.rsplit("/", 1)[-1].split(".", 1)[0]
                if dependency.path.endswith(target) or dependency_name == target:
                    return (
                        EvidenceReasonKind.INCLUDE_OR_IMPORT,
                        dependency.path,
                        f"{importer.path} includes/imports {included}",
                    )
    return None


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
