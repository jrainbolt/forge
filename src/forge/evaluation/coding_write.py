"""Minimal deterministic A11 write-capable evaluation suite and isolated runner."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from shutil import copytree

from forge.models import GenerationConfig, Model
from forge.orchestration import CodingTaskStatus, RepositoryChatSession
from forge.project_config import ProjectCommands
from forge.tools import (
    MutationPreview,
    PreparedProjectCommand,
    ToolInvocation,
    create_assist_repository_policy,
    create_assist_repository_registry,
)

CODING_WRITE_V1 = "coding-write-v1"
CODING_WRITE_SUITE_VERSION = 1
ApprovalCallback = Callable[
    [ToolInvocation, MutationPreview | PreparedProjectCommand], bool
]


@dataclass(frozen=True, slots=True)
class WriteEvaluationTask:
    task_id: str
    prompt: str
    expected_path: str
    required_text: str


@dataclass(frozen=True, slots=True)
class WriteTaskScore:
    expected_file_changed: bool
    expected_content: bool
    unexpected_files_unchanged: bool
    single_mutation: bool
    verification: bool
    completion: bool

    @property
    def total(self) -> int:
        return sum(
            (
                self.expected_file_changed,
                self.expected_content,
                self.unexpected_files_unchanged,
                self.single_mutation,
                self.verification,
                self.completion,
            )
        )


@dataclass(frozen=True, slots=True)
class WriteTaskResult:
    task_id: str
    status: str
    changed_files: tuple[str, ...]
    mutation_count: int
    score: WriteTaskScore
    elapsed_seconds: float


CODING_WRITE_V1_TASKS = (
    WriteEvaluationTask(
        "W01",
        "Fix RetryPolicy.should_retry so retries stop at max_attempts.",
        "src/tinyqueue/retry.py",
        "task.attempts < self.max_attempts",
    ),
    WriteEvaluationTask(
        "W02",
        "Add a unit test covering a zero-attempt retry policy.",
        "tests/test_retry.py",
        "test_zero_attempt_policy",
    ),
    WriteEvaluationTask(
        "W03",
        "Add a small is_exhausted helper to RetryPolicy.",
        "src/tinyqueue/retry.py",
        "def is_exhausted",
    ),
)


def load_write_suite(name: str) -> tuple[WriteEvaluationTask, ...]:
    if name != CODING_WRITE_V1:
        raise ValueError(
            f"unknown write evaluation suite {name!r}; available: {CODING_WRITE_V1}"
        )
    return CODING_WRITE_V1_TASKS


class CodingWriteEvaluationRunner:
    """Run every task in a fresh copied fixture with injected human/test approval."""

    def __init__(
        self,
        model_profile: str,
        model: Model,
        fixture: Path,
        *,
        approval_callback: ApprovalCallback,
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

    def run(self, tasks: Iterable[WriteEvaluationTask]) -> tuple[WriteTaskResult, ...]:
        results = []
        for task in tasks:
            with tempfile.TemporaryDirectory(prefix="forge-write-eval-") as name:
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
                )
                response = session.execute_task(task.prompt)
                coding = response.coding_task
                assert coding is not None
                after = _file_bytes(workspace)
                changed = tuple(
                    sorted(path for path in after if before.get(path) != after[path])
                )
                unexpected_unchanged = all(
                    before.get(path) == data
                    for path, data in after.items()
                    if path != task.expected_path
                ) and all(path in after for path in before)
                expected = after.get(task.expected_path, b"").decode(
                    "utf-8", errors="replace"
                )
                score = WriteTaskScore(
                    task.expected_path in changed,
                    task.required_text in expected,
                    unexpected_unchanged,
                    coding.mutation_count == 1,
                    coding.status is CodingTaskStatus.COMPLETED_VERIFIED,
                    coding.status
                    in {
                        CodingTaskStatus.COMPLETED_VERIFIED,
                        CodingTaskStatus.COMPLETED_UNVERIFIED,
                    },
                )
                results.append(
                    WriteTaskResult(
                        task.task_id,
                        coding.status.value,
                        coding.changed_files,
                        coding.mutation_count,
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
