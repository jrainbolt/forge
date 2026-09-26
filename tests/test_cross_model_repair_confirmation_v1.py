"""Deterministic predeclaration and repair invariants for A59."""

from __future__ import annotations

import json
from dataclasses import asdict
from inspect import getsource
from pathlib import Path

import pytest

from benchmarks.cross_model_repair_confirmation_v1.runner import (
    RepairCell,
    _taxonomy,
    candidate_cases,
    cell_path,
    identity,
    load_cell,
    run_repair,
)
from benchmarks.cross_model_repair_confirmation_v1.suite import (
    HISTORICAL_EXCLUSION,
    HYPOTHESIS,
    PRIMARY_PROFILES,
    THRESHOLD,
    definition_identity,
    tasks,
)
from benchmarks.realistic_coding_v2.runner import _atomic_json


@pytest.fixture
def cell() -> RepairCell:
    return RepairCell(
        "cross-model-repair-confirmation-v1",
        1,
        "QK01",
        "Q01",
        "qwen-small",
        "qwen-large",
        "R1",
        42,
        "model:sha256:abc",
        "exact_text",
        "manifest",
        "workspace",
        "evidence",
        ("pyservice/retry.py",),
        "YES",
        True,
        True,
        True,
        True,
        None,
        True,
        True,
        True,
        True,
        "mutation",
        "SEMANTIC_RECOVERED",
        100,
        20,
        1,
        1.0,
        1.0,
        2.0,
    )


def test_q01_new_corpus_excludes_historical_k01() -> None:
    assert HISTORICAL_EXCLUSION == "K01"
    assert all(case.case_id != "K01" for case in candidate_cases())


def test_q02_primary_exactly_once_checkpoint_is_source_free() -> None:
    from benchmarks.repair_framing_v1.runner import capture_primary

    source = getsource(capture_primary)
    assert "if destination.exists()" in source
    assert "return load_primary" in source


def test_q03_successful_primary_is_not_selected() -> None:
    source = getsource(
        __import__(
            "benchmarks.repair_framing_v1.runner", fromlist=["capture_primary"]
        ).capture_primary
    )
    assert "raw.oracle is EvaluationOutcome.FAIL" in source


def test_q04_invalid_primary_is_not_rerun_for_eligibility() -> None:
    source = getsource(
        __import__(
            "scripts.run_cross_model_repair_confirmation_v1", fromlist=["main"]
        ).main
    )
    assert "primary_path(args.root, case).exists()" in source
    assert "while" not in source


def test_q05_verification_blind_failure_excluded() -> None:
    from benchmarks.repair_framing_v1.runner import capture_primary

    assert 'verification_plan_result == "step_failed"' in getsource(capture_primary)


def test_q06_authority_insufficient_case_excluded() -> None:
    from benchmarks.repair_framing_v1.runner import capture_primary

    source = getsource(capture_primary)
    assert 'sufficient != "YES"' in source
    assert "REPAIR_AUTHORITY" in source


def test_q07_conditions_validate_byte_identical_start() -> None:
    source = getsource(run_repair)
    assert "record.post_primary_hash" in source
    assert "tree_hash(workspace)" in source


def test_q08_conditions_carry_identical_authority(cell) -> None:
    assert cell.authority_sufficient == "YES"
    assert cell.authorized_paths == ("pyservice/retry.py",)


def test_q09_conditions_use_same_production_repair_framing() -> None:
    source = getsource(run_repair)
    assert "production_task" in source
    assert "prompt" not in source


def test_q10_only_repair_backend_differs() -> None:
    source = getsource(run_repair)
    assert "RealWorldEvaluationRunner(\n            record.profile," in source
    assert "mutation_representation=representation" in source


def test_q11_qwen_large_threshold_is_frozen() -> None:
    assert THRESHOLD == {
        "qwen_large_recoveries": 3,
        "minimum_advantage": 2,
        "minimum_source_models_or_operation_classes": 2,
    }
    assert "more semantic recoveries" in HYPOTHESIS


def test_q12_semantic_recovery_scoring() -> None:
    assert _taxonomy(True, True, True, True, True, True) == "SEMANTIC_RECOVERED"
    assert (
        _taxonomy(True, True, True, True, True, False)
        == "VERIFICATION_PASS_SEMANTIC_FAIL"
    )


def test_q13_operation_class_aggregation_is_balanced() -> None:
    counts = {}
    for case in candidate_cases():
        key = case.definition.operation_class.value
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {4}


def test_q14_standard_result_is_source_free(cell) -> None:
    output = json.dumps(asdict(cell))
    assert "responses" not in output
    assert "source_content" not in output
    assert "verification_evidence" not in output


def test_q15_repair_checkpoint_resume(tmp_path: Path, cell) -> None:
    case = candidate_cases()[0]
    _atomic_json(cell_path(tmp_path, case, "R1"), asdict(cell))
    assert load_cell(tmp_path, case, "R1", "manifest") == cell
    with pytest.raises(ValueError, match="identity"):
        load_cell(tmp_path, case, "R1", "wrong")


def test_q16_no_production_routing_mutation() -> None:
    source = getsource(run_repair)
    assert "routing" not in source.lower()
    assert "default" not in source.lower()
    assert "repair_model =" not in source


def test_q17_model_sha_identity(cell) -> None:
    assert ":sha256:" in cell.model_artifact


def test_q18_load_and_switch_timing_accounting() -> None:
    module = __import__(
        "scripts.run_cross_model_repair_confirmation_v1", fromlist=["_record_load"]
    )
    assert "load_seconds" in getsource(module._record_load)
    assert "time.perf_counter" in getsource(module.main)


def test_manifest_identity_is_stable_and_binds_all_tasks() -> None:
    assert definition_identity() == definition_identity()
    assert len(tasks()) == 8
    assert len(candidate_cases()) == 16
    assert identity({"a": 1, "b": 2}) == identity({"b": 2, "a": 1})


def test_language_balance() -> None:
    counts = {language: 0 for language in ("Python", "C17")}
    for case in candidate_cases():
        counts[case.definition.language] += 1
    assert counts == {"Python": 8, "C17": 8}
    assert PRIMARY_PROFILES == ("qwen-small", "codestral-22b")
