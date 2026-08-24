"""Deterministic routing-v1 production-orchestration evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forge.embeddings import MockEmbeddingModel
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession, ToolActivity
from forge.semantic_index import SemanticIndex
from forge.tools import create_readonly_repository_registry

ROUTING_V1 = "routing-v1"
ROUTING_SUITE_VERSION = 1
BROAD = frozenset(
    {
        "repository.semantic_search",
        "repository.search_files",
        "repository.list_directory",
    }
)


@dataclass(frozen=True, slots=True)
class RoutingTask:
    task_id: str
    description: str
    expected_transition: str


@dataclass(frozen=True, slots=True)
class RoutingTaskResult:
    task_id: str
    completed: bool
    attempted_tool_calls: int
    executed_tool_calls: int
    broad_discovery_attempts: int
    broad_discovery_executions: int
    suppressed_discovery: int
    candidate_count: int
    candidate_inspections: int
    candidate_failures: int
    candidate_set_repetitions: int
    source_files_acquired: tuple[str, ...]
    retrieval_state_final: str
    tool_count: int


@dataclass(frozen=True, slots=True)
class RoutingEvaluationResult:
    tasks: tuple[RoutingTaskResult, ...]
    tasks_passed: int
    tasks_total: int
    broad_discoveries_attempted: int
    broad_discoveries_executed: int
    suppressed_broad_discoveries: int
    candidate_inspections: int
    candidate_failures: int
    candidate_set_repetitions: int
    source_files_acquired: int
    mean_tool_count: float


ROUTING_V1_TASKS = tuple(
    RoutingTask(f"R0{i}", text, state)
    for i, (text, state) in enumerate(
        (
            ("semantic candidate then targeted read", "source_acquired"),
            ("exact symbol suppresses broad search", "source_acquired"),
            ("repeated candidate set has no novelty", "source_acquired"),
            ("failed candidate advances to alternate", "source_acquired"),
            ("multi-file final gate", "source_acquired"),
            ("post-source broad search suppressed", "source_acquired"),
        ),
        1,
    )
)


def load_routing_suite(name: str) -> tuple[RoutingTask, ...]:
    if name != ROUTING_V1:
        raise ValueError(f"unknown routing suite {name!r}; available: {ROUTING_V1}")
    return ROUTING_V1_TASKS


def run_routing_v1(root: Path) -> RoutingEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(
        _run_task(task, root / task.task_id.lower()) for task in ROUTING_V1_TASKS
    )
    total = len(tasks)
    return RoutingEvaluationResult(
        tasks,
        sum(x.completed for x in tasks),
        total,
        sum(x.broad_discovery_attempts for x in tasks),
        sum(x.broad_discovery_executions for x in tasks),
        sum(x.suppressed_discovery for x in tasks),
        sum(x.candidate_inspections for x in tasks),
        sum(x.candidate_failures for x in tasks),
        sum(x.candidate_set_repetitions for x in tasks),
        sum(len(x.source_files_acquired) for x in tasks),
        sum(x.tool_count for x in tasks) / total,
    )


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _final(text: str = "Grounded.") -> str:
    return json.dumps({"type": "final", "answer": text})


def _run_task(task: RoutingTask, workspace: Path) -> RoutingTaskResult:
    workspace.mkdir()
    (workspace / "a.py").write_text(
        "def target():\n    return 'AVALUE routing source'\n"
    )
    (workspace / "b.py").write_text(
        "def other():\n    return 'BVALUE routing source'\n"
    )
    semantic = task.task_id in {"R01", "R06"}
    index = SemanticIndex(
        workspace, MockEmbeddingModel(32), cache_root=workspace / ".cache"
    )
    registry = (
        create_readonly_repository_registry(semantic_index=index)
        if semantic
        else create_readonly_repository_registry()
    )
    scripted, question, minimum = _script(task.task_id)
    model = MockModel(scripted)
    removed = False

    def callback(activity: ToolActivity) -> None:
        nonlocal removed
        if (
            task.task_id in {"R03", "R04"}
            and activity.tool_name == "repository.search_files"
            and activity.status == "success"
            and not removed
        ):
            (workspace / "a.py").unlink()
            removed = True
        elif (
            task.task_id == "R03"
            and activity.tool_name == "repository.read_range"
            and activity.status == "failure"
        ):
            (workspace / "a.py").write_text(
                "def target():\n    return 'AVALUE routing source'\n"
            )

    response = RepositoryChatSession(
        "routing-v1",
        model,
        workspace,
        registry=registry,
        semantic_index=index if semantic else None,
        minimum_source_files=minimum,
        require_relevant_source=False,
        activity_callback=callback,
        enforce_retrieval_routing=True,
    ).ask(question)
    activities = response.tool_activity
    attempted = sum('"type": "tool_call"' in x for x in scripted)
    broad_attempted = sum(
        any(f'"tool": "{name}"' in x for name in BROAD) for x in scripted
    )
    broad_executed = sum(
        x.tool_name in BROAD and x.status == "success" for x in activities
    )
    metrics = response.retrieval_metrics
    sources = tuple(
        sorted(
            {
                x.path
                for x in activities
                if x.evidence == "source_content" and x.status == "success" and x.path
            }
        )
    )
    return RoutingTaskResult(
        task.task_id,
        bool(response.text),
        attempted,
        len(activities),
        broad_attempted,
        broad_executed,
        broad_attempted - broad_executed,
        response.retrieval_candidate_count,
        metrics.candidates_inspected,
        metrics.candidate_failures,
        metrics.candidate_set_repeats,
        sources,
        response.retrieval_state.value,
        len(activities),
    )


def _script(task: str) -> tuple[tuple[str, ...], str, int]:
    read_a = _call(
        "ra", "repository.read_range", {"path": "a.py", "start_line": 1, "end_line": 2}
    )
    read_b = _call(
        "rb", "repository.read_range", {"path": "b.py", "start_line": 1, "end_line": 2}
    )
    if task == "R01":
        return (
            (
                _call("s1", "repository.semantic_search", {"query": "routing source"}),
                _call("s2", "repository.semantic_search", {"query": "routing source"}),
                read_a,
                _final(),
            ),
            "routing source",
            1,
        )
    if task == "R02":
        return (
            (
                _call("f1", "repository.find_symbol", {"symbol": "target"}),
                _call("s1", "repository.search_files", {"query": "target"}),
                read_a,
                _final(),
            ),
            "target",
            1,
        )
    if task == "R03":
        return (
            (
                _call("s1", "repository.search_files", {"query": "AVALUE"}),
                read_a,
                _call("s2", "repository.search_files", {"query": "AVALUE"}),
                _call("s3", "repository.search_files", {"query": "BVALUE"}),
                read_b,
                _final(),
            ),
            "Find AVALUE BVALUE",
            1,
        )
    if task == "R04":
        return (
            (
                _call("s1", "repository.search_files", {"query": "routing source"}),
                read_a,
                read_b,
                _final(),
            ),
            "routing source",
            1,
        )
    if task == "R05":
        return (
            (
                _call("s1", "repository.search_files", {"query": "routing source"}),
                read_a,
                _final("early"),
                read_b,
                _final(),
            ),
            "routing source",
            2,
        )
    return (
        (
            _call("s1", "repository.semantic_search", {"query": "routing source"}),
            read_a,
            _call("s2", "repository.semantic_search", {"query": "routing source"}),
            _final(),
        ),
        "routing source",
        1,
    )
