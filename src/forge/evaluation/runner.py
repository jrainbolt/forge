"""Model-neutral runner over production repository-aware orchestration."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from forge.context_planner import ContextPlannerMetrics
from forge.conversation import ContextBudgetError
from forge.evaluation.scoring import score_task
from forge.evaluation.types import (
    MAX_CAPTURED_ANSWER_CHARS,
    EvaluationRun,
    EvaluationTask,
    FailureReason,
    TaskResult,
    ToolRecord,
)
from forge.lexical_index import RepositoryLexicalIndex
from forge.models import GenerationConfig, Model, ModelError, ModelUsage
from forge.orchestration import (
    RepositoryChatSession,
    RepositoryOrchestrationError,
    ToolActivity,
)
from forge.repository_index import RepositoryIndex
from forge.semantic_index import SemanticIndex
from forge.tools import create_readonly_repository_registry

Clock = Callable[[], float]


class EvaluationRunner:
    """Run isolated tasks while reusing one already-loaded generic model."""

    def __init__(
        self,
        model_profile: str,
        model: Model,
        workspace: Path,
        *,
        generation: GenerationConfig | None = None,
        clock: Clock = time.perf_counter,
        semantic_index: SemanticIndex | None = None,
    ) -> None:
        self._profile = model_profile
        self._model = model
        self._workspace = workspace
        self._generation = generation or GenerationConfig(
            max_tokens=512, temperature=0.0
        )
        self._clock = clock
        self._semantic_index = semantic_index

    def run(
        self,
        suite: str,
        tasks: Iterable[EvaluationTask],
        *,
        suite_version: int = 1,
    ) -> EvaluationRun:
        task_values = tuple(tasks)
        active: list[ToolActivity] = []

        def record(activity: ToolActivity) -> None:
            active.append(activity)

        lexical_index = RepositoryLexicalIndex(
            self._workspace,
            cache_root=Path(tempfile.gettempdir()) / "forge-evaluation-cache",
        )
        session = RepositoryChatSession(
            self._profile,
            self._model,
            self._workspace,
            generation=self._generation,
            activity_callback=record,
            registry=create_readonly_repository_registry(
                RepositoryIndex(self._workspace),
                self._semantic_index,
                lexical_index,
            ),
            repository_index=RepositoryIndex(self._workspace),
            semantic_index=self._semantic_index,
            lexical_index=lexical_index,
        )
        run_started = self._clock()
        results = []
        for task in task_values:
            active.clear()
            started = self._clock()
            answer = ""
            completed = False
            corrections = 0
            steps: int | None = None
            usage = ModelUsage()
            failure_reason = None
            failure_message = None
            context_metrics = ContextPlannerMetrics()
            try:
                response = session.ask(task.prompt)
                answer = response.text[:MAX_CAPTURED_ANSWER_CHARS]
                completed = True
                corrections = response.protocol_corrections
                steps = response.orchestration_steps
                usage = response.usage
                context_metrics = response.context_metrics
            except Exception as error:  # one task must not abort the evaluation run
                failure_reason = _classify_failure(error)
                failure_message = str(error)[:1_000]
            tools = tuple(_tool_record(activity) for activity in active)
            files = tuple(
                dict.fromkeys(
                    record.path
                    for record in tools
                    if record.name in {"repository.read_file", "repository.read_range"}
                    and record.status == "success"
                    and record.path is not None
                )
            )
            scores = score_task(task, answer, files, tools, completed=completed)
            if completed and scores.total < scores.maximum:
                failure_reason = (
                    FailureReason.GROUNDING_FAILURE
                    if scores.grounding.earned < scores.grounding.maximum
                    else FailureReason.ANSWER_INCORRECT
                )
            results.append(
                TaskResult(
                    task.task_id,
                    task.category,
                    self._profile,
                    completed,
                    answer,
                    files,
                    tools,
                    steps,
                    corrections,
                    usage,
                    max(0.0, self._clock() - started),
                    scores,
                    failure_reason,
                    failure_message,
                    context_metrics,
                )
            )
            session.clear()
        return EvaluationRun(
            suite,
            suite_version,
            self._profile,
            tuple(results),
            max(0.0, self._clock() - run_started),
        )


def _tool_record(activity: ToolActivity) -> ToolRecord:
    return ToolRecord(
        activity.tool_name,
        activity.status,
        activity.evidence,
        activity.path,
        activity.returned_bytes,
        activity.returned_lines,
    )


def _classify_failure(error: Exception) -> FailureReason:
    if isinstance(error, ContextBudgetError):
        return FailureReason.CONTEXT_LIMIT
    if isinstance(error, ModelError):
        return FailureReason.MODEL_ERROR
    if isinstance(error, RepositoryOrchestrationError):
        message = str(error).casefold()
        if "protocol" in message or "json" in message or "duplicate" in message:
            return FailureReason.PROTOCOL_ERROR
        if "limit" in message or "repeated" in message:
            return FailureReason.TOOL_LIMIT
        if "evidence" in message:
            return FailureReason.GROUNDING_FAILURE
        return FailureReason.TOOL_FAILURE
    return FailureReason.UNKNOWN
