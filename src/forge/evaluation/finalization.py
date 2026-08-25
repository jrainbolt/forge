"""Fixed deterministic finalization-v1 full-orchestration evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forge.models import MockModel
from forge.orchestration import RepositoryChatSession, ToolActivity

FINALIZATION_V1 = "finalization-v1"
FINALIZATION_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class FinalizationTaskResult:
    task_id: str
    completed: bool
    entries: int
    model_calls: int
    corrections: int
    prevented_tools: int
    tools_executed: int
    goals_in_snapshot: int


@dataclass(frozen=True, slots=True)
class FinalizationEvaluationResult:
    tasks: tuple[FinalizationTaskResult, ...]
    tasks_passed: int
    tasks_total: int
    finalization_entries: int
    final_model_calls: int
    protocol_corrections: int
    post_coverage_tool_attempts: int
    post_coverage_tools_executed: int
    required_goals_represented: int
    grounded_completions: int


def run_finalization_v1(root: Path) -> FinalizationEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(_run(task, root / task.lower()) for task in _TASK_IDS)
    return FinalizationEvaluationResult(
        tasks,
        sum(task.completed for task in tasks),
        len(tasks),
        sum(task.entries for task in tasks),
        sum(task.model_calls for task in tasks),
        sum(task.corrections for task in tasks),
        sum(task.prevented_tools for task in tasks),
        0,
        sum(task.goals_in_snapshot for task in tasks),
        sum(task.completed and task.entries > 0 for task in tasks),
    )


_TASK_IDS = ("F01", "F02", "F03", "F04", "F05", "F06")


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _final(answer: str = "Grounded.") -> str:
    return json.dumps({"type": "final", "answer": answer})


def _source_flow(prefix: str, path: str, query: str) -> list[str]:
    return [
        _call(f"s{prefix}", "repository.search_files", {"query": query}),
        _call(f"r{prefix}", "repository.read_file", {"path": path}),
    ]


def _run(task: str, workspace: Path) -> FinalizationTaskResult:
    workspace.mkdir()
    (workspace / "a.py").write_text('def alpha():\n    return "ALPHA"\n')
    (workspace / "b.py").write_text('def beta():\n    return "BETA"\n')
    question = "Where is ALPHA implemented?"
    callback = None
    if task in {"F02", "F03", "F04"}:
        question = "How do ALPHA and BETA work together?"
        if task == "F04":
            (workspace / "a.py").write_text("# ALPHA\n" + "x = 1\n" * 200)
        scripted = [
            *_source_flow("1", "a.py", "ALPHA"),
            *_source_flow("2", "b.py", "BETA"),
        ]
        if task == "F03":
            scripted.append(_call("blocked", "repository.read_file", {"path": "a.py"}))
        scripted.append(_final())
    elif task == "F05":
        changed = False

        def mutate_after_read(activity: ToolActivity) -> None:
            nonlocal changed
            if activity.invocation_id == "r1" and not changed:
                (workspace / "a.py").write_text("ALPHA = 2\n")
                changed = True

        callback = mutate_after_read
        scripted = [
            *_source_flow("1", "a.py", "ALPHA"),
            _final("early"),
            *_source_flow("2", "a.py", "ALPHA = 2"),
            _final(),
        ]
    elif task == "F06":
        question = "1. Find ALPHA.\n2. Find MISSING."
        scripted = [
            *_source_flow("1", "a.py", "ALPHA"),
            _call("e1", "repository.search_files", {"query": "MISSING"}),
            _call("e2", "repository.search_files", {"query": "STILL_MISSING"}),
            _final("No required evidence was found."),
        ]
    else:
        scripted = [*_source_flow("1", "a.py", "ALPHA"), _final()]
    model = MockModel(tuple(scripted), context_capacity=8192)
    response = RepositoryChatSession(
        "finalization-v1",
        model,
        workspace,
        activity_callback=callback,
        require_relevant_source=False,
        minimum_source_files=1,
    ).ask(question)
    metrics = response.finalization_metrics
    final_schema = model.requests[-1].output.schema
    final_only = (
        final_schema.get("properties", {}).get("type", {}).get("const") == "final"
        and "oneOf" not in final_schema
    )
    completed = final_only and response.text
    if task == "F03":
        completed = completed and metrics.post_coverage_tool_calls_prevented == 1
    if task == "F05":
        completed = completed and len(response.tool_activity) == 4
    if task == "F06":
        completed = not response.coverage_complete and metrics.entries == 0
    return FinalizationTaskResult(
        task,
        bool(completed),
        metrics.entries,
        metrics.model_calls,
        metrics.protocol_corrections,
        metrics.post_coverage_tool_calls_prevented,
        len(response.tool_activity),
        metrics.required_goals_in_snapshot,
    )
