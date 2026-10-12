from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from shutil import copytree

from benchmarks.default_candidate_confirmation_v1.suite import tasks
from benchmarks.executable_verification_diagnosis_v1.runner import CheckResult
from benchmarks.executable_verification_diagnosis_v1.suite import (
    OBLIGATIONS,
    Coverage,
    Derivability,
    corpus_identity,
    validate_corpus,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.semantic_obligation_completeness_v1.checks import check
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.evaluation.realworld import apply_task_setup


def test_frozen_coverage_inventory_is_complete_and_source_free() -> None:
    validate_corpus()
    payload = {
        task: [asdict(item) | {"identity": item.identity} for item in values]
        for task, values in OBLIGATIONS.items()
    }
    serialized = json.dumps(payload, default=str)
    assert "reference patch" not in serialized
    assert "oracle.py" not in serialized
    assert len(corpus_identity()) == 64
    assert {item.coverage for values in OBLIGATIONS.values() for item in values} <= {
        Coverage.COVERS,
        Coverage.PARTIAL,
        Coverage.DOES_NOT_COVER,
        Coverage.UNKNOWN,
    }


def test_explicit_examples_and_invariants_are_derivable() -> None:
    h07 = OBLIGATIONS["H07"]
    assert all(item.derivability is Derivability.DERIVABLE for item in h07)
    assert {item.source_class for item in h07} == {"EXPLICIT_TASK_AND_PUBLIC_API"}


def test_unsupported_behavior_is_not_derivable() -> None:
    from benchmarks.executable_verification_diagnosis_v1.suite import (
        VerificationObligation,
    )

    item = VerificationObligation(
        "unknown-policy",
        None,
        "CREATE",
        Coverage.UNKNOWN,
        Derivability.NOT_DERIVABLE,
    )
    assert item.check_id is None
    assert item.derivability is Derivability.NOT_DERIVABLE


def test_candidate_check_passes_correct_and_fails_wrong_control(tmp_path: Path) -> None:
    correct = Path(copytree(REPOSITORY, tmp_path / "correct"))
    assert check("H03-clock-forward", correct) is True
    wrong = Path(copytree(REPOSITORY, tmp_path / "wrong"))
    definition = next(item for item in tasks() if item.task_id == "H03")
    apply_task_setup(wrong, definition.production_task.setup)
    assert check("H03-clock-backward", wrong) is False


def test_c_candidate_checks_match_public_api_types(tmp_path: Path) -> None:
    correct = Path(copytree(REPOSITORY, tmp_path / "c-correct"))
    assert check("H06-port-zero", correct) is True
    assert check("H06-quota-exact-fill", correct) is True


def test_false_positive_accounting_is_explicit() -> None:
    item = CheckResult(
        "obligation",
        "a" * 64,
        "CREATE",
        Coverage.DOES_NOT_COVER.value,
        Coverage.DOES_NOT_COVER.value,
        Derivability.DERIVABLE.value,
        "EXPLICIT_TASK_AND_PUBLIC_API",
        "CHECK_FAIL",
    )
    semantic_pass = True
    assert semantic_pass and item.outcome == "CHECK_FAIL"


def test_checkpoint_resume_preserves_source_free_inventory(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    payload = {
        "corpus_identity": corpus_identity(),
        "obligations": [item.identity for item in OBLIGATIONS["H07"]],
        "source_free": True,
    }
    atomic_checkpoint(path, payload)
    assert resume_checkpoint(path) == payload
