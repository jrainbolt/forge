from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.semantic_obligation_completeness_v1.checks import check
from benchmarks.semantic_obligation_completeness_v1.runner import (
    SATISFIED,
    UNAVAILABLE,
    UNSATISFIED,
    ObligationResult,
    align_verification_coverage,
    classify_completeness,
    classify_create,
    classify_mixed,
    classify_relationship,
    compare_repair,
)
from benchmarks.semantic_obligation_completeness_v1.suite import (
    OBLIGATIONS,
    TASK_IDS,
    ObligationOrigin,
    corpus_identity,
    tasks,
    validate_corpus,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint


def _result(
    obligation_id: str,
    satisfaction: str = SATISFIED,
    *,
    component: str = "CREATE",
    origin: str = "REQUIRED_BEHAVIOR_GOAL",
    relationship: bool = False,
) -> ObligationResult:
    return ObligationResult(
        obligation_id,
        "a" * 64,
        origin,
        component,
        "OBLIGATION_PRESENT_IN_PRODUCTION_STATE",
        satisfaction,
        "VERIFICATION_DOES_NOT_COVER_OBLIGATION",
        relationship,
    )


def test_obligations_are_frozen_and_production_visible() -> None:
    validate_corpus()
    assert tuple(OBLIGATIONS) == TASK_IDS
    assert len(tasks()) == 8
    assert len(corpus_identity()) == 64
    assert all(
        isinstance(item.origin, ObligationOrigin)
        for values in OBLIGATIONS.values()
        for item in values
    )


def test_diagnostic_wording_is_not_added_to_model_prompt() -> None:
    prompts = {task.task_id: task.production_task.prompt for task in tasks()}
    assert all("OBLIGATION_" not in prompt for prompt in prompts.values())
    assert all(
        "semantic obligation" not in prompt.lower() for prompt in prompts.values()
    )


def test_checksum_interface_probe_accepts_nonzero_checksum(tmp_path: Path) -> None:
    root = tmp_path / "cengine"
    root.mkdir()
    (root / "checksum.h").write_text(
        "unsigned checksum_bytes(const unsigned char *data, unsigned size);\n"
    )
    (root / "checksum.c").write_text(
        '#include "checksum.h"\n'
        "unsigned checksum_bytes(const unsigned char *data, unsigned size) {\n"
        "  unsigned value = 0;\n"
        "  for (unsigned i = 0; i < size; ++i) value = value * 33U ^ data[i];\n"
        "  return value;\n"
        "}\n"
    )
    assert check("H08-header-interface", tmp_path)


@pytest.mark.parametrize(
    ("results", "semantic", "expected"),
    [
        ((_result("x"),), True, "ALL_OBLIGATIONS_SATISFIED"),
        ((_result("x", UNSATISFIED),), False, "ONE_OBLIGATION_MISSED"),
        (
            (_result("x", UNSATISFIED), _result("y", UNSATISFIED)),
            False,
            "MULTIPLE_OBLIGATIONS_MISSED",
        ),
        ((_result("x", UNAVAILABLE),), False, "UNKNOWN"),
    ],
)
def test_completeness_classification(results, semantic: bool, expected: str) -> None:  # type: ignore[no-untyped-def]
    assert classify_completeness(results, semantic) == expected


def test_verification_pass_demonstrates_noncoverage_of_missed_obligation() -> None:
    missed = _result("x", UNSATISFIED)
    covered = ObligationResult(
        missed.obligation_id,
        missed.obligation_identity,
        missed.origin,
        missed.component,
        missed.production_state,
        missed.satisfaction,
        "VERIFICATION_COVERS_OBLIGATION",
        missed.relationship,
    )
    aligned = align_verification_coverage((covered,), True)
    assert aligned[0].verification_coverage == (
        "VERIFICATION_DOES_NOT_COVER_OBLIGATION"
    )


def test_create_taxonomy_distinguishes_interface_edge_and_behavior() -> None:
    assert (
        classify_create((_result("H08-header-interface", UNSATISFIED),))
        == "CREATE_INTERFACE_WRONG"
    )
    assert (
        classify_create((_result("H07-reject-negative", UNSATISFIED),))
        == "CREATE_EDGE_CASE_MISSED"
    )
    assert (
        classify_create((_result("H07-accumulate", UNSATISFIED),))
        == "CREATE_BEHAVIOR_WRONG"
    )


def test_mixed_and_relationship_taxonomy() -> None:
    relationship = _result(
        "H11-checksum-interface",
        UNSATISFIED,
        component="RELATIONSHIP",
        origin="REQUIRED_RELATIONSHIP",
        relationship=True,
    )
    assert classify_mixed((relationship,)) == "EDIT_CREATE_RELATIONSHIP_MISSING"
    assert (
        classify_relationship((relationship,)) == "RELATIONSHIP_OBLIGATION_UNSATISFIED"
    )
    assert (
        classify_relationship((_result("x"),))
        == "NO_REQUIRED_RELATIONSHIP_IN_PRODUCTION_STATE"
    )


def test_repair_comparison_identifies_fix_and_regression() -> None:
    before = (_result("x", UNSATISFIED), _result("y"))
    assert (
        compare_repair(before, (_result("x"), _result("y")))
        == "REPAIR_FIXES_MISSED_OBLIGATION"
    )
    assert (
        compare_repair(before, (_result("x"), _result("y", UNSATISFIED)))
        == "REPAIR_FIXES_AND_REGRESSES_OBLIGATIONS"
    )


def test_source_free_checkpoint_exact_once_and_package_exclusion(
    tmp_path: Path,
) -> None:
    payload = {
        "obligation_id": "H07-initial-zero",
        "obligation_identity": "a" * 64,
        "satisfaction": SATISFIED,
        "production_state": "OBLIGATION_PRESENT_IN_PRODUCTION_STATE",
    }
    assert standard_result_is_source_free(payload)
    path = tmp_path / "cell.json"
    atomic_checkpoint(path, payload)
    assert resume_checkpoint(path) == json.loads(json.dumps(payload))
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, payload)
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "semantic_obligation_completeness_v1" not in configuration
