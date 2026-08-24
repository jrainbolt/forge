"""Immutable values for deterministic repository coding evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from forge.context_planner import ContextPlannerMetrics
from forge.models import ModelUsage

SCHEMA_VERSION = 1
SUITE_VERSION = 1
MAX_CAPTURED_ANSWER_CHARS = 8_000


class TaskCategory(Enum):
    LOCALIZATION = "localization"
    EXPLANATION = "explanation"
    TRACE = "trace"
    BUG_FIND = "bug_find"
    BUG_EXPLAIN = "bug_explain"
    TEST_COVERAGE = "test_coverage"
    ARCHITECTURE = "architecture"
    CONTEXT = "context"


class FailureReason(Enum):
    MODEL_ERROR = "model_error"
    PROTOCOL_ERROR = "protocol_error"
    CONTEXT_LIMIT = "context_limit"
    TOOL_LIMIT = "tool_limit"
    TOOL_FAILURE = "tool_failure"
    GROUNDING_FAILURE = "grounding_failure"
    ANSWER_INCORRECT = "answer_incorrect"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RequiredFact:
    """One fact satisfied by any normalized phrase alternative."""

    alternatives: tuple[str, ...]

    def __post_init__(self) -> None:
        values = tuple(self.alternatives)
        if not values or any(not value.strip() for value in values):
            raise ValueError("fact alternatives must contain non-empty text")
        object.__setattr__(self, "alternatives", values)


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    task_id: str
    category: TaskCategory
    prompt: str
    required_files: tuple[str, ...]
    required_facts: tuple[RequiredFact, ...]
    expected_answer_files: tuple[str, ...] = ()
    expected_symbols: tuple[str, ...] = ()
    max_tool_calls: int = 4
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.prompt.strip():
            raise ValueError("task ID and prompt must be non-empty")
        if not isinstance(self.category, TaskCategory):
            raise TypeError("category must be a TaskCategory")
        for name in ("required_files", "expected_answer_files", "expected_symbols"):
            values = tuple(getattr(self, name))
            if any(not value.strip() for value in values):
                raise ValueError(f"{name} must contain non-empty text")
            object.__setattr__(self, name, values)
        facts = tuple(self.required_facts)
        if not facts or not all(isinstance(fact, RequiredFact) for fact in facts):
            raise ValueError("required_facts must contain RequiredFact values")
        object.__setattr__(self, "required_facts", facts)
        if isinstance(self.max_tool_calls, bool) or self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")


@dataclass(frozen=True, slots=True)
class DimensionScore:
    earned: int
    maximum: int

    def __post_init__(self) -> None:
        if self.maximum < 0 or self.earned < 0 or self.earned > self.maximum:
            raise ValueError("invalid dimension score")


@dataclass(frozen=True, slots=True)
class TaskScores:
    correctness: DimensionScore
    grounding: DimensionScore
    localization: DimensionScore
    efficiency: DimensionScore
    completion: DimensionScore

    @property
    def total(self) -> int:
        return sum(score.earned for score in self.dimensions)

    @property
    def maximum(self) -> int:
        return sum(score.maximum for score in self.dimensions)

    @property
    def dimensions(self) -> tuple[DimensionScore, ...]:
        return (
            self.correctness,
            self.grounding,
            self.localization,
            self.efficiency,
            self.completion,
        )


@dataclass(frozen=True, slots=True)
class ToolRecord:
    name: str
    status: str
    evidence: str
    path: str | None = None
    returned_bytes: int | None = None
    returned_lines: int | None = None


@dataclass(frozen=True, slots=True)
class TaskResult:
    task_id: str
    category: TaskCategory
    model_profile: str
    success: bool
    answer: str
    files_inspected: tuple[str, ...]
    tools: tuple[ToolRecord, ...]
    orchestration_steps: int | None
    protocol_corrections: int
    usage: ModelUsage
    elapsed_seconds: float
    scores: TaskScores
    failure_reason: FailureReason | None = None
    failure_message: str | None = None
    context_metrics: ContextPlannerMetrics = ContextPlannerMetrics()

    @property
    def tool_count(self) -> int:
        return len(self.tools)


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    suite: str
    suite_version: int
    model_profile: str
    task_results: tuple[TaskResult, ...]
    elapsed_seconds: float
    schema_version: int = SCHEMA_VERSION

    @property
    def completed(self) -> int:
        return sum(result.success for result in self.task_results)

    @property
    def failed(self) -> int:
        return len(self.task_results) - self.completed

    @property
    def score(self) -> int:
        return sum(result.scores.total for result in self.task_results)

    @property
    def maximum_score(self) -> int:
        return sum(result.scores.maximum for result in self.task_results)

    @property
    def tool_calls(self) -> int:
        return sum(result.tool_count for result in self.task_results)

    @property
    def protocol_corrections(self) -> int:
        return sum(result.protocol_corrections for result in self.task_results)

    @property
    def usage(self) -> ModelUsage:
        inputs = [result.usage.input_tokens for result in self.task_results]
        outputs = [result.usage.output_tokens for result in self.task_results]
        return ModelUsage(
            input_tokens=(
                sum(inputs) if all(value is not None for value in inputs) else None
            ),
            output_tokens=(
                sum(outputs) if all(value is not None for value in outputs) else None
            ),
        )
