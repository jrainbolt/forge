"""A63 deterministic grounding and coding-diagnosis controls."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.grounding_diagnosis_v1.coding import (
    CodingCondition,
    CodingDiagnostic,
)
from benchmarks.grounding_diagnosis_v1.grounding import GroundingDiagnostic
from benchmarks.grounding_diagnosis_v1.suite import (
    CODING_CASES,
    GROUNDING_CASES,
    SUITE,
)
from benchmarks.real_repository_pilot_v1.suite import (
    create_snapshot,
    repository_identity,
)
from benchmarks.real_repository_pilot_v2.runner import (
    atomic_json,
    read_checkpoint,
)
from benchmarks.real_repository_pilot_v2.suite import tasks
from forge.evidence_coverage import EvidenceCoverageState, decompose_evidence_plan


def test_hidden_expected_paths_remain_evaluator_only() -> None:
    definitions = {item.task_id: item for item in tasks()}
    for case in GROUNDING_CASES:
        task = definitions[case.task_id]
        if task.authority_mode == "DISCOVERY_REQUIRED":
            assert not task.production_task.required_candidate_paths
            assert not task.production_task.expected_changed_paths


def test_edit_ready_independent_from_semantic_sufficiency() -> None:
    assert any(case.context_quality == "MISDIRECTED" for case in GROUNDING_CASES)


def test_misdirected_source_can_be_authority_valid() -> None:
    case = GROUNDING_CASES[0]
    assert case.acquired_paths and case.context_quality == "MISDIRECTED"


def test_evidence_coverage_uses_production_visible_task_and_read() -> None:
    plan = decompose_evidence_plan("Repair entity liveness after create and destroy.")
    state = EvidenceCoverageState(plan)
    goal = state.active_goal
    assert goal is not None
    state.register_source(goal.goal_id, "src/power.c", 0, "trusted-read")
    assert state.complete


def test_g1_g2_cases_contain_no_expected_paths() -> None:
    payload = repr(GROUNDING_CASES)
    assert "expected" not in payload.lower()


def test_additional_discovery_is_bounded() -> None:
    diagnostic = GroundingDiagnostic(
        "G",
        "model",
        "F",
        "task",
        ("task",),
        (),
        ("a.c",),
        ("a.c",),
        True,
        (),
        100,
        "MODEL_ACCEPTED_FIRST_PLAUSIBLE_SOURCE",
        False,
        ("b.c",),
        False,
        False,
        (),
        False,
        1,
    )
    assert diagnostic.maximum_additional_rounds == 1
    assert len(diagnostic.g1_additional_reads) <= 1


def _condition(structural: bool, semantic: bool, failure: str) -> CodingCondition:
    return CodingCondition(
        structural, structural, semantic, False, semantic, failure, 1, 10, 5, 0.1
    )


def test_c0_c1_same_task_source_and_representation() -> None:
    diagnostic = CodingDiagnostic(
        "C",
        "model",
        "F",
        ("a.c",),
        "exact_text",
        True,
        True,
        True,
        _condition(False, False, "PROTOCOL_SCHEMA"),
        _condition(True, False, "SEMANTIC_IMPLEMENTATION_ERROR"),
        "PRODUCTION_CONTINUITY_OR_FRAMING_ISSUE",
    )
    assert diagnostic.same_task and diagnostic.same_source
    assert diagnostic.same_representation


def test_structural_and_semantic_failure_are_separate() -> None:
    structural = _condition(False, False, "PROTOCOL_SCHEMA")
    semantic = _condition(True, False, "SEMANTIC_IMPLEMENTATION_ERROR")
    assert structural.failure != semantic.failure


def test_durable_results_are_source_free() -> None:
    keys = set(asdict(_condition(False, False, "PROTOCOL_SCHEMA")))
    assert not keys.intersection({"source", "content", "new_text", "old_text"})


def test_checkpoint_resume(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    expected = {"suite": SUITE, "case_id": "G01"}
    atomic_json(path, expected)
    assert read_checkpoint(path, expected) == expected
    with pytest.raises(FileExistsError):
        atomic_json(path, expected)


def test_canonical_foundation_safety(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "a.c").write_text("int a;\n")
    before = repository_identity(canonical)
    snapshot = create_snapshot(canonical, tmp_path / "snapshot")
    (snapshot / "a.c").write_text("changed\n")
    assert repository_identity(canonical) == before


def test_package_exclusion() -> None:
    assert "grounding_diagnosis_v1" not in Path("pyproject.toml").read_text()


def test_historical_statuses_preserved() -> None:
    roadmap = Path("docs/ROADMAP.md").read_text()
    assert "A59 remains blocked" in roadmap
    assert "A60 remains blocked" in roadmap


def test_corpus_sizes() -> None:
    assert len(GROUNDING_CASES) == 8
    assert len(CODING_CASES) == 6
