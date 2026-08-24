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
