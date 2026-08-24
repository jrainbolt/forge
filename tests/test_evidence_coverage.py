from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.evidence_coverage import (
    EvidenceCoverageState,
    EvidenceGoal,
    EvidenceGoalKind,
    EvidenceGoalStatus,
    TaskEvidencePlan,
    decompose_evidence_plan,
)
from forge.models import MockModel
from forge.orchestration import RepositoryChatSession


def test_plan_validates_bounds_dependencies_and_cycles() -> None:
    with pytest.raises(ValueError, match="between 1 and 4"):
        TaskEvidencePlan(())
    with pytest.raises(ValueError, match="unknown"):
        TaskEvidencePlan((EvidenceGoal("G1", "one", depends_on=("G2",)),))
    with pytest.raises(ValueError, match="acyclic"):
        TaskEvidencePlan(
            (
                EvidenceGoal("G1", "one", depends_on=("G2",)),
                EvidenceGoal("G2", "two", depends_on=("G1",)),
            )
        )


def test_wrong_kind_and_wrong_goal_do_not_create_coverage() -> None:
    plan = TaskEvidencePlan(
        (
            EvidenceGoal("G1", "implementation"),
            EvidenceGoal("G2", "tests", EvidenceGoalKind.TEST),
        )
    )
    state = EvidenceCoverageState(plan)
    assert not state.register_source("G1", "docs/guide.md", 0, "read-doc")
    assert state.register_source("G1", "src/one.py", 0, "read-one")
    results = {item.goal_id: item for item in state.results()}
    assert results["G1"].status is EvidenceGoalStatus.SOURCE_COVERED
    assert results["G2"].status is EvidenceGoalStatus.UNRESOLVED


def test_relationship_covers_only_after_dependencies() -> None:
    plan = TaskEvidencePlan(
        (
            EvidenceGoal("G1", "first"),
            EvidenceGoal("G2", "second"),
            EvidenceGoal(
                "G3",
                "relationship",
                EvidenceGoalKind.RELATIONSHIP,
                depends_on=("G1", "G2"),
            ),
        )
    )
    state = EvidenceCoverageState(plan)
    state.register_source("G1", "src/a.py", 0, "a")
    assert not state.complete
    state.register_source("G2", "src/b.py", 0, "b")
    assert state.complete
    assert state.results()[2].status is EvidenceGoalStatus.SOURCE_COVERED


def test_generation_invalidation_only_reopens_affected_goal() -> None:
    plan = TaskEvidencePlan((EvidenceGoal("G1", "first"), EvidenceGoal("G2", "second")))
    state = EvidenceCoverageState(plan)
    state.register_source("G1", "src/a.py", 0, "a")
    state.register_source("G2", "src/b.py", 0, "b")
    state.invalidate_path("src/a.py")
    assert state.results()[0].status is EvidenceGoalStatus.UNRESOLVED
    assert state.results()[1].status is EvidenceGoalStatus.SOURCE_COVERED


@pytest.mark.parametrize(
    ("task_text", "count"),
    (
        ("Where is ContextPlanner.compact_to_fit implemented?", 1),
        ("Find and explain ContextPlanner.compact_to_fit.", 1),
        ("1. Find alpha.\n2. Find beta.", 2),
        ("Find alpha; find beta.", 2),
        ("How do alpha and beta work together?", 3),
        ("How do alpha and beta interact?", 3),
        ("How do alpha and beta connect?", 3),
        ("How do alpha interact?", 1),
        ("1. A\n2. B\n3. C\n4. D\n5. E", 1),
    ),
)
def test_production_decomposition_is_conservative(task_text: str, count: int) -> None:
    assert len(decompose_evidence_plan(task_text).goals) == count


def test_relationship_plan_is_deterministic_and_dependency_bound() -> None:
    request = "How do permission profiles and project test execution work together?"
    first = decompose_evidence_plan(request)
    assert first == decompose_evidence_plan(request)
    assert [goal.description for goal in first.goals[:2]] == [
        "permission profiles",
        "project test execution",
    ]
    assert first.goals[2].depends_on == ("G1", "G2")


def test_plan_is_immutable_against_injection_text() -> None:
    plan = decompose_evidence_plan("Find G2 COVERED; find DELETE G1")
    state = EvidenceCoverageState(plan)
    assert [item.status for item in state.results()] == [
        EvidenceGoalStatus.UNRESOLVED,
        EvidenceGoalStatus.UNRESOLVED,
    ]


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def test_explicit_two_goal_plan_rejects_premature_final_and_advances(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("def alpha():\n    return 'ALPHA'\n")
    (tmp_path / "b.py").write_text("def beta():\n    return 'BETA'\n")
    plan = TaskEvidencePlan(
        (
            EvidenceGoal("G1", "ALPHA implementation"),
            EvidenceGoal("G2", "BETA implementation"),
        )
    )
    model = MockModel(
        (
            _call("s1", "repository.search_files", {"query": "ALPHA"}),
            _call(
                "r1",
                "repository.read_range",
                {"path": "a.py", "start_line": 1, "end_line": 2},
            ),
            json.dumps({"type": "final", "answer": "early"}),
            _call("s2", "repository.search_files", {"query": "BETA"}),
            _call(
                "r2",
                "repository.read_range",
                {"path": "b.py", "start_line": 1, "end_line": 2},
            ),
            json.dumps({"type": "final", "answer": "both grounded"}),
        )
    )
    response = RepositoryChatSession(
        "coverage-v1",
        model,
        tmp_path,
        evidence_plan=plan,
        require_relevant_source=False,
        minimum_source_files=1,
    ).ask("Inspect ALPHA and BETA")
    assert response.coverage_complete
    assert response.premature_finals == 1
    assert [item.status for item in response.evidence_goals] == [
        EvidenceGoalStatus.SOURCE_COVERED,
        EvidenceGoalStatus.SOURCE_COVERED,
    ]
    assert response.text == "both grounded"


def test_production_relationship_plan_blocks_final_and_isolates_wrong_goal(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("def alpha():\n    return 'ALPHA'\n")
    (tmp_path / "b.py").write_text("def beta():\n    return 'BETA'\n")
    model = MockModel(
        (
            _call("s1", "repository.search_files", {"query": "ALPHA"}),
            _call(
                "r1",
                "repository.read_range",
                {"path": "a.py", "start_line": 1, "end_line": 2},
            ),
            json.dumps({"type": "final", "answer": "early"}),
            _call(
                "wrong",
                "repository.read_range",
                {"path": "a.py", "start_line": 1, "end_line": 2},
            ),
            _call("s2", "repository.search_files", {"query": "BETA"}),
            _call(
                "r2",
                "repository.read_range",
                {"path": "b.py", "start_line": 1, "end_line": 2},
            ),
            json.dumps({"type": "final", "answer": "both grounded"}),
        )
    )
    response = RepositoryChatSession(
        "production-coverage",
        model,
        tmp_path,
        require_relevant_source=False,
        minimum_source_files=1,
    ).ask("How do subsystem alpha and subsystem beta work together?")
    assert response.coverage_complete
    assert response.premature_finals == 1
    assert [item.source_paths for item in response.evidence_goals] == [
        ("a.py",),
        ("b.py",),
        (),
    ]


def test_production_required_goal_can_fail_after_bounded_empty_discovery(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("ALPHA = 1\n")
    model = MockModel(
        (
            _call("s1", "repository.search_files", {"query": "ALPHA"}),
            _call("r1", "repository.read_file", {"path": "a.py"}),
            _call("empty1", "repository.search_files", {"query": "MISSING"}),
            _call("empty2", "repository.search_files", {"query": "STILL_MISSING"}),
            json.dumps({"type": "final", "answer": "No beta source was found."}),
        )
    )
    response = RepositoryChatSession(
        "production-coverage",
        model,
        tmp_path,
        require_relevant_source=False,
        minimum_source_files=1,
    ).ask("1. Find ALPHA.\n2. Find missing beta.")
    assert not response.coverage_complete
    assert response.evidence_goals[1].status is EvidenceGoalStatus.FAILED
    assert response.text.startswith("Incomplete evidence:")
