from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from benchmarks.production_detectable_completeness_v1.suite import (
    CheckKind,
    CompletenessCheck,
)
from benchmarks.structural_completeness_shadow_v1 import runner
from benchmarks.structural_completeness_shadow_v1.runner import (
    ShadowCheckResult,
    ShadowSnapshot,
    compare_repair,
    decide,
    observe,
)
from benchmarks.structural_completeness_shadow_v1.suite import (
    CASES,
    TASK_IDS,
    ShadowCase,
    corpus_identity,
    tasks,
    validate_corpus,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint


def _result(status: str) -> ShadowCheckResult:
    return ShadowCheckResult(
        "check",
        "a" * 64,
        "REQUIRED_SYMBOL_PRESENCE",
        "b" * 64,
        "c" * 64,
        status,
        0.001,
        False,
    )


def _case(*, deterministic: bool = False) -> ShadowCase:
    check = CompletenessCheck(
        "check",
        CheckKind.REQUIRED_SYMBOL_PRESENCE,
        "module.py",
        None,
        "serve",
        None,
        ("module.py",),
    )
    return ShadowCase("T", "CONTROL", (check,), deterministic)


def test_frozen_corpus_shape_and_all_six_kinds() -> None:
    validate_corpus()
    assert len(CASES) == 14
    assert tuple(case.task_id for case in CASES) == TASK_IDS
    assert len(tasks()) == 14
    assert len(corpus_identity()) == 64
    assert {check.kind for case in CASES for check in case.checks} == set(CheckKind)


def test_shadow_mode_never_blocks_on_checker_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "module.py").write_text("def serve():\n    return 1\n")

    def fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("observer failure")

    monkeypatch.setattr(runner, "evaluate", fail)
    snapshot = observe(_case(), tmp_path)
    assert snapshot.decision == "STRUCTURAL_SHADOW_PARTIAL"


def test_shadow_observer_has_no_repair_triggering_path() -> None:
    source = inspect.getsource(observe)
    assert "repair" not in source.lower()
    primary = ShadowSnapshot(
        (_result("COMPLETE"),), "STRUCTURAL_SHADOW_PASS", 0.1, 0, None, None
    )
    assert compare_repair(primary, None) == "REPAIR_NOT_RUN"


def test_shadow_decisions_include_fail_pass_partial_and_not_applicable() -> None:
    assert decide((_result("COMPLETE"),)) == "STRUCTURAL_SHADOW_PASS"
    assert decide((_result("INCOMPLETE"),)) == "STRUCTURAL_SHADOW_FAIL"
    assert decide((_result("NOT_CHECKABLE"),)) == "STRUCTURAL_SHADOW_PARTIAL"
    assert decide(()) == "STRUCTURAL_SHADOW_NOT_APPLICABLE"


def test_known_structural_miss_fails(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def other():\n    return 1\n")
    snapshot = observe(_case(), tmp_path)
    assert snapshot.decision == "STRUCTURAL_SHADOW_FAIL"


def test_correct_and_semantic_negative_structure_pass(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def serve():\n    return 1\n")
    snapshot = observe(_case(), tmp_path)
    semantic_pass = False
    assert snapshot.decision == "STRUCTURAL_SHADOW_PASS"
    assert not semantic_pass


def test_repeated_checking_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def serve():\n    return 1\n")
    snapshot = observe(_case(deterministic=True), tmp_path)
    assert snapshot.deterministic_repeat is True


def test_no_hidden_oracle_or_reference_dependencies() -> None:
    for case in CASES:
        for check in case.checks:
            values = (check.check_id, check.source_path, check.target_path or "")
            assert all("oracle" not in value.lower() for value in values)
            assert all("reference" not in value.lower() for value in values)


def test_source_free_checkpoint_resume_and_package_exclusion(tmp_path: Path) -> None:
    payload = {
        "check_identity": "a" * 64,
        "decision": "STRUCTURAL_SHADOW_PASS",
        "latency_seconds": 0.001,
        "workflow_attempt_id": "w1",
    }
    assert standard_result_is_source_free(payload)
    path = tmp_path / "cell.json"
    atomic_checkpoint(path, payload)
    assert resume_checkpoint(path) == json.loads(json.dumps(payload))
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, payload)
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "structural_completeness_shadow_v1" not in configuration
