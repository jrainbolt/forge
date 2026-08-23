"""Deterministic A13 repair-loop evaluation on isolated fixture copies."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
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

REPAIR_V1 = "repair-v1"
REPAIR_SUITE_VERSION = 1
RepairApproval = Callable[
    [ToolInvocation, MutationPreview | PreparedProjectCommand], bool
]


@dataclass(frozen=True, slots=True)
class RepairEvaluationTask:
    task_id: str
    prompt: str
    expected_mutations: int
    expected_attempts: int
    expected_status: str
    expected_stop: AgentStopReason


@dataclass(frozen=True, slots=True)
class RepairTaskScore:
    initial_mutation: bool
    initial_failure_observed: bool
    repair_decision_correct: bool
    mutation_budget: bool
    verification_budget: bool
    final_status_truthful: bool
    workspace_confined: bool

    @property
    def total(self) -> int:
        return sum(
            (
                self.initial_mutation,
                self.initial_failure_observed,
                self.repair_decision_correct,
                self.mutation_budget,
                self.verification_budget,
                self.final_status_truthful,
                self.workspace_confined,
            )
        )


@dataclass(frozen=True, slots=True)
class RepairEvaluationResult:
    task_id: str
    status: str
    stop_reason: str
    mutation_count: int
    verification_attempts: int
    repair_eligible: bool
    repair_attempted: bool
    repair_succeeded: bool
    score: RepairTaskScore
    elapsed_seconds: float


REPAIR_V1_TASKS = (
    RepairEvaluationTask(
        "R01",
        "Apply the retry fix, diagnose the failing test, repair it, and retest.",
        2,
        2,
        "completed_repaired_verified",
        AgentStopReason.COMPLETED,
    ),
    RepairEvaluationTask(
        "R02",
        "Implement the retry fix, repair any compile failure, and rebuild.",
        2,
        2,
        "completed_repaired_verified",
        AgentStopReason.COMPLETED,
    ),
    RepairEvaluationTask(
        "R03",
        "Attempt one repair and report truthfully if tests still fail.",
        2,
        2,
        "repair_verification_failed",
        AgentStopReason.REPAIR_VERIFICATION_FAILED,
    ),
    RepairEvaluationTask(
        "R04",
        "Verify the change and do not repair an execution-environment failure.",
        1,
        1,
        "mutated_verification_failed",
        AgentStopReason.VERIFICATION_FAILED,
    ),
)


def load_repair_suite(name: str) -> tuple[RepairEvaluationTask, ...]:
    if name != REPAIR_V1:
        raise ValueError(
            f"unknown repair evaluation suite {name!r}; available: {REPAIR_V1}"
        )
    return REPAIR_V1_TASKS


class RepairEvaluationRunner:
    """Run each repair scenario through production state in a fresh copy."""

    def __init__(
        self,
        model_profile: str,
        model: Model,
        fixture: Path,
        *,
        approval_callback: RepairApproval,
        commands: Mapping[str, ProjectCommands],
        generation: GenerationConfig | None = None,
    ) -> None:
        self._profile = model_profile
        self._model = model
        self._fixture = fixture
        self._approval = approval_callback
        self._commands = dict(commands)
        self._generation = generation or GenerationConfig(
            max_tokens=512, temperature=0.0
        )

    def run(
        self, tasks: Iterable[RepairEvaluationTask]
    ) -> tuple[RepairEvaluationResult, ...]:
        results = []
        for task in tasks:
            with tempfile.TemporaryDirectory(prefix="forge-repair-eval-") as name:
                workspace = Path(copytree(self._fixture, Path(name) / "fixture"))
                before_paths = _paths(workspace)
                started = time.perf_counter()
                session = RepositoryChatSession(
                    self._profile,
                    self._model,
                    workspace,
                    generation=self._generation,
                    registry=create_assist_repository_registry(
                        self._commands[task.task_id]
                    ),
                    policy=create_assist_repository_policy(),
                    approval_callback=self._approval,
                    require_relevant_source=False,
                    agent_mode=True,
                    repair_enabled=True,
                )
                response = session.run_agent_task(task.prompt)
                agent = response.agent_task
                assert agent is not None
                attempts = len(agent.build_attempts) + len(agent.test_attempts)
                initial_failed = any(
                    record.status == "failed"
                    for record in (*agent.build_attempts, *agent.test_attempts)[:1]
                )
                score = RepairTaskScore(
                    agent.mutation_count >= 1,
                    initial_failed,
                    agent.mutation_count == task.expected_mutations,
                    agent.mutation_count <= 2,
                    len(agent.build_attempts) <= 2 and len(agent.test_attempts) <= 2,
                    agent.status == task.expected_status
                    and agent.stop_reason is task.expected_stop
                    and attempts == task.expected_attempts,
                    _paths(workspace) == before_paths,
                )
                results.append(
                    RepairEvaluationResult(
                        task.task_id,
                        agent.status,
                        agent.stop_reason.value,
                        agent.mutation_count,
                        attempts,
                        agent.repair_eligible,
                        agent.repair_attempted,
                        agent.repair_succeeded,
                        score,
                        time.perf_counter() - started,
                    )
                )
        return tuple(results)


def _paths(workspace: Path) -> frozenset[str]:
    return frozenset(
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    )
