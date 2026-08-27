"""Fixed deterministic bootstrap-v1 production orchestration evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forge.embeddings import MockEmbeddingModel
from forge.lexical_index import RepositoryLexicalIndex
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession, ToolActivity
from forge.semantic_index import SemanticIndex
from forge.tools import create_readonly_repository_registry

BOOTSTRAP_V1 = "bootstrap-v1"
BOOTSTRAP_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class BootstrapTaskResult:
    task_id: str
    completed: bool
    bootstrap_attempts: int
    bootstrap_executions: int
    bootstrap_successes: int
    bootstrap_empty_results: int
    bootstrap_failures: int
    bootstrap_candidates: int
    model_discovery_calls: int
    tool_count: int


@dataclass(frozen=True, slots=True)
class BootstrapEvaluationResult:
    tasks: tuple[BootstrapTaskResult, ...]
    tasks_passed: int
    tasks_total: int
    bootstrap_attempts: int
    bootstrap_executions: int
    bootstrap_successes: int
    bootstrap_empty_results: int
    bootstrap_failures: int
    bootstrap_candidates: int
    bootstrap_tool_executions: int
    model_discovery_calls_after_bootstrap: int


def run_bootstrap_v1(root: Path) -> BootstrapEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(_run(task, root / task.lower()) for task in _TASK_IDS)
    return BootstrapEvaluationResult(
        tasks,
        sum(task.completed for task in tasks),
        len(tasks),
        sum(task.bootstrap_attempts for task in tasks),
        sum(task.bootstrap_executions for task in tasks),
        sum(task.bootstrap_successes for task in tasks),
        sum(task.bootstrap_empty_results for task in tasks),
        sum(task.bootstrap_failures for task in tasks),
        sum(task.bootstrap_candidates for task in tasks),
        sum(task.bootstrap_executions for task in tasks),
        sum(task.model_discovery_calls for task in tasks),
    )


_TASK_IDS = ("B01", "B02", "B03", "B04", "B05", "B06")


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _final() -> str:
    return json.dumps({"type": "final", "answer": "Grounded."})


def _run(task: str, workspace: Path) -> BootstrapTaskResult:
    workspace.mkdir()
    (workspace / "a.py").write_text('def alpha():\n    return "ALPHA"\n')
    (workspace / "b.py").write_text('def beta():\n    return "BETA"\n')
    semantic = task != "B05"
    index = (
        SemanticIndex(
            workspace,
            MockEmbeddingModel(32),
            cache_root=workspace / ".semantic-cache",
        )
        if semantic
        else None
    )
    lexical = (
        RepositoryLexicalIndex(workspace, cache_root=workspace / ".lexical-cache")
        if semantic
        else None
    )
    registry = create_readonly_repository_registry(
        semantic_index=index, lexical_index=lexical
    )
    question = "Where is alpha implemented?"
    scripted: list[str]
    callback = None
    if task == "B02":
        question = "How do alpha and beta work together?"
        scripted = [
            _call("r1", "repository.read_file", {"path": "a.py"}),
            _call("r2", "repository.read_file", {"path": "b.py"}),
            _final(),
        ]
    elif task == "B03":
        scripted = [
            _call("wander", "repository.search_files", {"query": "ALPHA"}),
            _call("r1", "repository.read_file", {"path": "a.py"}),
            _final(),
        ]
    elif task == "B04":
        (workspace / "a.py").unlink()
        (workspace / "b.py").unlink()

        def create_after_empty(activity: ToolActivity) -> None:
            if activity.invocation_id.startswith("forge-bootstrap"):
                (workspace / "a.py").write_text("ALPHA = 1\n")

        callback = create_after_empty
        scripted = [
            _call("search", "repository.search_files", {"query": "ALPHA"}),
            _call("read", "repository.read_file", {"path": "a.py"}),
            _final(),
        ]
    elif task == "B05":
        scripted = [
            _call("search", "repository.search_files", {"query": "ALPHA"}),
            _call("read", "repository.read_file", {"path": "a.py"}),
            _final(),
        ]
    elif task == "B06":
        removed = False

        def exhaust(activity: ToolActivity) -> None:
            nonlocal removed
            if activity.invocation_id.startswith("forge-bootstrap") and not removed:
                (workspace / "a.py").unlink()
                (workspace / "b.py").unlink()
                removed = True
            elif activity.invocation_id == "bad2":
                (workspace / "c.py").write_text("ALPHA = 2\n")

        callback = exhaust
        scripted = [
            _call("bad1", "repository.read_file", {"path": "a.py"}),
            _call("bad2", "repository.read_file", {"path": "b.py"}),
            _call("search", "repository.search_files", {"query": "ALPHA"}),
            _call("read", "repository.read_file", {"path": "c.py"}),
            _final(),
        ]
    else:
        scripted = [
            _call("read", "repository.read_file", {"path": "a.py"}),
            _final(),
        ]
    response = RepositoryChatSession(
        "bootstrap-v1",
        MockModel(tuple(scripted)),
        workspace,
        registry=registry,
        semantic_index=index,
        lexical_index=lexical,
        activity_callback=callback,
        require_relevant_source=False,
        minimum_source_files=1,
        enforce_retrieval_routing=True,
    ).ask(question)
    metrics = response.bootstrap_metrics
    expected = 2 if task == "B02" else 0 if task == "B05" else 1
    completed = response.coverage_complete and metrics.executions == expected
    if task == "B03":
        completed = (
            completed and response.retrieval_metrics.suppressed_discovery_attempts == 1
        )
    if task == "B04":
        completed = completed and metrics.empty_results == 1
    if task == "B06":
        completed = completed and metrics.executions == 1
    return BootstrapTaskResult(
        task,
        completed,
        metrics.attempts,
        metrics.executions,
        metrics.successes,
        metrics.empty_results,
        metrics.failures,
        metrics.candidates,
        metrics.model_discovery_calls_after_bootstrap,
        len(response.tool_activity),
    )
