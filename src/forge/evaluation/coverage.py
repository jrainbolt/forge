"""Fixed deterministic coverage-v1 evidence evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forge.evidence_coverage import (
    EvidenceCoverageState,
    EvidenceGoal,
    EvidenceGoalKind,
    TaskEvidencePlan,
)
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession

COVERAGE_V1 = "coverage-v1"
COVERAGE_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class CoverageTaskResult:
    task_id: str
    completed: bool
    required_goals: int
    covered_goals: int
    premature_finals: int
    source_reads: int
    distinct_source_paths: int


@dataclass(frozen=True, slots=True)
class CoverageEvaluationResult:
    tasks: tuple[CoverageTaskResult, ...]
    tasks_passed: int
    tasks_total: int
    required_goals: int
    covered_goals: int
    premature_finals: int
    source_reads: int


@dataclass(frozen=True, slots=True)
class ProductionDecompositionResult:
    tasks_passed: int
    tasks_total: int
    production_plans_created: int
    single_goal_plans: int
    multi_goal_plans: int
    goals_total: int
    goals_covered: int
    goals_failed: int
    premature_finals: int
    goal_transitions: int
    wrong_goal_reads: int
    coverage_complete_tasks: int


def run_coverage_v1(root: Path) -> CoverageEvaluationResult:
    root.mkdir(parents=True, exist_ok=True)
    results = tuple(
        _run(task, root / task.lower())
        for task in ("COV01", "COV02", "COV03", "COV04", "COV05", "COV06")
    )
    return CoverageEvaluationResult(
        results,
        sum(x.completed for x in results),
        len(results),
        sum(x.required_goals for x in results),
        sum(x.covered_goals for x in results),
        sum(x.premature_finals for x in results),
        sum(x.source_reads for x in results),
    )


def run_production_decomposition_v1(root: Path) -> ProductionDecompositionResult:
    root.mkdir(parents=True, exist_ok=True)
    responses = (
        _production_case(root / "dec01", "relationship"),
        _production_case(root / "dec02", "explicit"),
        _production_case(root / "dec03", "wrong_goal"),
        _production_case(root / "dec04", "exhausted"),
    )
    goals = tuple(goal for response in responses for goal in response.evidence_goals)
    passed = (
        responses[0].coverage_complete,
        responses[1].coverage_complete and responses[1].premature_finals == 1,
        responses[2].coverage_complete and responses[2].wrong_goal_reads == 1,
        not responses[3].coverage_complete
        and any(goal.status.value == "failed" for goal in responses[3].evidence_goals),
    )
    return ProductionDecompositionResult(
        sum(passed),
        len(passed),
        len(responses),
        sum(len(response.evidence_goals) == 1 for response in responses),
        sum(len(response.evidence_goals) > 1 for response in responses),
        len(goals),
        sum(goal.status.value == "source_covered" for goal in goals),
        sum(goal.status.value == "failed" for goal in goals),
        sum(response.premature_finals for response in responses),
        sum(response.goal_transitions for response in responses),
        sum(response.wrong_goal_reads for response in responses),
        sum(response.coverage_complete for response in responses),
    )


def _production_case(workspace: Path, case: str):
    workspace.mkdir()
    (workspace / "a.py").write_text('def alpha():\n    return "ALPHA"\n')
    (workspace / "b.py").write_text('def beta():\n    return "BETA"\n')
    request = (
        "How do subsystem alpha and subsystem beta work together?"
        if case == "relationship"
        else "1. Inspect subsystem alpha.\n2. Inspect subsystem beta."
    )
    scripted = [
        _call("s1", "repository.search_files", {"query": "ALPHA"}),
        _call("r1", "repository.read_file", {"path": "a.py"}),
    ]
    if case == "explicit":
        scripted.append(json.dumps({"type": "final", "answer": "early"}))
    if case == "wrong_goal":
        scripted.append(
            _call(
                "wrong",
                "repository.read_range",
                {"path": "a.py", "start_line": 1, "end_line": 2},
            )
        )
    if case == "exhausted":
        scripted.extend(
            (
                _call("e1", "repository.search_files", {"query": "MISSING"}),
                _call("e2", "repository.search_files", {"query": "STILL_MISSING"}),
                json.dumps({"type": "final", "answer": "No beta evidence."}),
            )
        )
    else:
        scripted.extend(
            (
                _call("s2", "repository.search_files", {"query": "BETA"}),
                _call("r2", "repository.read_file", {"path": "b.py"}),
                json.dumps({"type": "final", "answer": "Grounded."}),
            )
        )
    return RepositoryChatSession(
        "production-decomposition-v1",
        MockModel(tuple(scripted)),
        workspace,
        require_relevant_source=False,
        minimum_source_files=1,
    ).ask(request)


def _call(identifier: str, tool: str, args: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": args}
    )


def _run(task: str, workspace: Path) -> CoverageTaskResult:
    workspace.mkdir()
    (workspace / "a.py").write_text("def alpha():\n    return 'ALPHA'\n")
    (workspace / "b.py").write_text("def beta():\n    return 'BETA'\n")
    if task == "COV06":
        plan = TaskEvidencePlan((EvidenceGoal("G1", "changed implementation"),))
        state = EvidenceCoverageState(plan)
        state.register_source("G1", "a.py", 0, "r1")
        state.invalidate_path("a.py")
        state.register_source("G1", "a.py", 1, "r2")
        return CoverageTaskResult(task, state.complete, 1, 1, 0, 2, 1)
    relationship = task == "COV03"
    two = task in {"COV02", "COV03", "COV04", "COV05"}
    goals = [EvidenceGoal("G1", "ALPHA implementation")]
    if two:
        goals.append(EvidenceGoal("G2", "BETA implementation"))
    if relationship:
        goals.append(
            EvidenceGoal(
                "G3",
                "relationship",
                EvidenceGoalKind.RELATIONSHIP,
                depends_on=("G1", "G2"),
            )
        )
    plan = TaskEvidencePlan(tuple(goals))
    responses = [
        _call("s1", "repository.search_files", {"query": "ALPHA"}),
        _call(
            "r1",
            "repository.read_range",
            {"path": "a.py", "start_line": 1, "end_line": 2},
        ),
    ]
    if two:
        responses.extend(
            (
                json.dumps({"type": "final", "answer": "early"}),
                _call("s2", "repository.search_files", {"query": "BETA"}),
                _call(
                    "r2",
                    "repository.read_range",
                    {"path": "b.py", "start_line": 1, "end_line": 2},
                ),
            )
        )
    responses.append(json.dumps({"type": "final", "answer": "grounded"}))
    response = RepositoryChatSession(
        "coverage-v1",
        MockModel(tuple(responses)),
        workspace,
        evidence_plan=plan,
        require_relevant_source=False,
        minimum_source_files=1,
    ).ask("Inspect ALPHA and BETA")
    covered = sum(
        item.status.value == "source_covered" for item in response.evidence_goals
    )
    reads = [
        item
        for item in response.tool_activity
        if item.evidence == "source_content" and item.status == "success"
    ]
    return CoverageTaskResult(
        task,
        response.coverage_complete,
        len(goals),
        covered,
        response.premature_finals,
        len(reads),
        len({item.path for item in reads}),
    )
