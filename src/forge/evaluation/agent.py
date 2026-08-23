"""Deterministic A12 agent-loop evaluation on fresh fixture copies."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from shutil import copytree

from forge.models import GenerationConfig, Model
from forge.orchestration import AgentStopReason, RepositoryChatSession
from forge.project_config import ProjectCommands
from forge.tools import (
    MutationPreview,
    PreparedProjectCommand,
    ToolInvocation,
    create_assist_repository_policy,
    create_assist_repository_registry,
)

AGENT_V1 = "agent-v1"
AGENT_SUITE_VERSION = 1
AgentApproval = Callable[
    [ToolInvocation, MutationPreview | PreparedProjectCommand], bool
]


@dataclass(frozen=True, slots=True)
class AgentEvaluationTask:
    task_id: str
    prompt: str
    expected_path: str | None
    required_text: str | None
    minimum_tools: int
    expected_stop: AgentStopReason


@dataclass(frozen=True, slots=True)
class AgentTaskScore:
    multi_step: bool
    expected_artifact: bool
    single_mutation: bool
    bounded: bool
    correct_stop: bool
    workspace_confined: bool

    @property
    def total(self) -> int:
        return sum(
            (
                self.multi_step,
                self.expected_artifact,
                self.single_mutation,
                self.bounded,
                self.correct_stop,
                self.workspace_confined,
            )
        )


@dataclass(frozen=True, slots=True)
class AgentEvaluationResult:
    task_id: str
    status: str
    stop_reason: str
    tool_calls: int
    mutation_count: int
    changed_files: tuple[str, ...]
    score: AgentTaskScore
    elapsed_seconds: float


AGENT_V1_TASKS = (
    AgentEvaluationTask(
        "G01",
        "Trace how RetryPolicy and QueueService cooperate without changing files.",
        None,
        None,
        3,
        AgentStopReason.COMPLETED,
    ),
    AgentEvaluationTask(
        "G02",
        "Inspect the retry implementation, fix its boundary, and verify the change.",
        "src/tinyqueue/retry.py",
        "task.attempts < self.max_attempts",
        3,
        AgentStopReason.COMPLETED,
    ),
    AgentEvaluationTask(
        "G03",
        "Recover from an irrelevant search and explain where Task attempts advance.",
        None,
        None,
        3,
        AgentStopReason.COMPLETED,
    ),
    AgentEvaluationTask(
        "G04",
        "Make the retry boundary fix, run verification, and explain any failure.",
        "src/tinyqueue/retry.py",
        "task.attempts < self.max_attempts",
        4,
        AgentStopReason.VERIFICATION_FAILED,
    ),
)


def load_agent_suite(name: str) -> tuple[AgentEvaluationTask, ...]:
    if name != AGENT_V1:
        raise ValueError(
            f"unknown agent evaluation suite {name!r}; available: {AGENT_V1}"
        )
    return AGENT_V1_TASKS


class AgentEvaluationRunner:
    """Exercise autonomous multi-step behavior in an isolated copy per task."""

    def __init__(
        self,
        model_profile: str,
        model: Model,
        fixture: Path,
        *,
        approval_callback: AgentApproval,
        commands: ProjectCommands | None = None,
        generation: GenerationConfig | None = None,
    ) -> None:
        self._profile = model_profile
        self._model = model
        self._fixture = fixture
        self._approval = approval_callback
        self._commands = commands or ProjectCommands()
        self._generation = generation or GenerationConfig(
            max_tokens=512, temperature=0.0
        )

    def run(
        self, tasks: Iterable[AgentEvaluationTask]
    ) -> tuple[AgentEvaluationResult, ...]:
        results = []
        for task in tasks:
            with tempfile.TemporaryDirectory(prefix="forge-agent-eval-") as name:
                workspace = Path(copytree(self._fixture, Path(name) / "fixture"))
                before = _file_bytes(workspace)
                started = time.perf_counter()
                session = RepositoryChatSession(
                    self._profile,
                    self._model,
                    workspace,
                    generation=self._generation,
                    registry=create_assist_repository_registry(self._commands),
                    policy=create_assist_repository_policy(),
                    approval_callback=self._approval,
                    require_relevant_source=False,
                    agent_mode=True,
                )
                response = session.run_agent_task(task.prompt)
                agent = response.agent_task
                assert agent is not None
                after = _file_bytes(workspace)
                changed = tuple(
                    sorted(
                        set(before) | set(after),
                        key=str,
                    )
                )
                changed = tuple(
                    path for path in changed if before.get(path) != after.get(path)
                )
                expected_artifact = task.expected_path is None or (
                    task.expected_path in changed
                    and task.required_text is not None
                    and task.required_text
                    in after.get(task.expected_path, b"").decode(
                        "utf-8", errors="replace"
                    )
                )
                expected_changed = (
                    set() if task.expected_path is None else {task.expected_path}
                )
                score = AgentTaskScore(
                    agent.tool_calls >= task.minimum_tools,
                    expected_artifact,
                    agent.mutation_count == (0 if task.expected_path is None else 1),
                    agent.model_calls <= 16 and agent.tool_calls <= 12,
                    agent.stop_reason is task.expected_stop,
                    set(changed) == expected_changed,
                )
                results.append(
                    AgentEvaluationResult(
                        task.task_id,
                        agent.status,
                        agent.stop_reason.value,
                        agent.tool_calls,
                        agent.mutation_count,
                        changed,
                        score,
                        time.perf_counter() - started,
                    )
                )
        return tuple(results)


def _file_bytes(workspace: Path) -> dict[str, bytes]:
    return {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
