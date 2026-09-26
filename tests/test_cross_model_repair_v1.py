"""Deterministic invariants for the A58 evaluator-only repair handoff."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from inspect import getsource
from pathlib import Path

import pytest

from benchmarks.cross_model_repair_v1.runner import (
    Cell,
    _taxonomy,
    cell_path,
    load_cell,
    manifest_hash,
    run_cross_cell,
    summarize_case,
)
from benchmarks.cross_model_repair_v1.suite import (
    REPAIR_PROFILES,
    classify_case,
    pairs,
)
from benchmarks.realistic_coding_v2.runner import _atomic_json
from benchmarks.repair_framing_v1.suite import cases


@pytest.fixture
def pair_values():
    return pairs(cases())


@pytest.fixture
def cell(pair_values):
    pair = pair_values[0]
    return Cell(
        "cross-model-repair-v1",
        1,
        pair.case.case_id,
        pair.case.task_id,
        pair.case.profile,
        pair.repair_profile,
        pair.same_model,
        pair.same_model,
        "model:sha256:abc",
        "exact_text",
        "manifest",
        "post-primary",
        "primary",
        "evidence",
        ("cengine/parser.c",),
        "YES",
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        None,
        False,
        False,
        False,
        "test",
        "VERIFICATION_STILL_FAIL",
        "mutation",
        "TRANSACTION_SEMANTIC_FAIL",
        100,
        20,
        1,
        1.0,
        1.0,
        2.0,
    )


def test_x01_same_post_primary_state_across_profiles(pair_values) -> None:
    grouped = {}
    for pair in pair_values:
        grouped.setdefault(pair.case.case_id, set()).add(pair.case.task_id)
    assert all(len(task_ids) == 1 for task_ids in grouped.values())
    assert len(pair_values) == 18


def test_x02_primary_model_identity_not_injected(cell) -> None:
    assert not cell.model_identity_injected
    assert "which model" not in getsource(run_cross_cell).lower()


def test_x03_path_authority_identical(pair_values) -> None:
    for case_id in {pair.case.case_id for pair in pair_values}:
        assert (
            len(
                {
                    pair.case.definition.production_task.allowed_paths
                    for pair in pair_values
                    if pair.case.case_id == case_id
                }
            )
            == 1
        )


def test_x04_representation_is_case_controlled() -> None:
    source = getsource(run_cross_cell)
    assert "mutation_representation=representation" in source
    assert "mutation_representation=pair.repair_profile" not in source


def test_x05_verification_evidence_identity_is_frozen(cell) -> None:
    assert cell.evidence_identical
    assert cell.evidence_hash == "evidence"


def test_x06_same_model_reuse_identity_is_explicit(cell) -> None:
    assert cell.same_model and cell.reused_a57
    assert cell.start_hash == "post-primary"


def test_x07_backend_substitution_preserves_primary_profile() -> None:
    source = getsource(run_cross_cell)
    assert "RealWorldEvaluationRunner(\n            record.profile," in source
    assert "backend," in source


def test_x08_hidden_oracle_excluded_from_model_material() -> None:
    source = getsource(run_cross_cell)
    assert "oracle_commands" not in source
    assert "semantic" not in source.split(".run(", 1)[0]


def test_x09_full_production_verification_reruns() -> None:
    source = getsource(run_cross_cell)
    assert "production_task" in source
    assert "reverification_executed" in source


def test_x10_semantic_and_verification_are_separate(cell) -> None:
    assert cell.full_verification_executed
    assert not cell.verification_passed
    assert not cell.semantic_passed


def test_x11_pair_matrix_has_all_nine_profile_pairs(pair_values) -> None:
    matrix = {(pair.case.profile, pair.repair_profile) for pair in pair_values}
    assert matrix == {
        (left, right) for left in REPAIR_PROFILES for right in REPAIR_PROFILES
    }


def test_x12_cross_model_recovery_classification() -> None:
    assert (
        classify_case(frozenset({"qwen-large"}), "qwen-small") == "CROSS_MODEL_RECOVERY"
    )
    assert (
        classify_case(frozenset({"qwen-large", "codestral-22b"}), "qwen-small")
        == "ANY_MODEL_RECOVERY"
    )


def test_x13_no_model_recovery_classification() -> None:
    assert classify_case(frozenset(), "qwen-small") == "NO_MODEL_RECOVERY"


def test_x14_standard_result_is_source_free(cell) -> None:
    output = json.dumps(asdict(cell))
    assert "responses" not in output
    assert "verification_evidence" not in output
    assert "source_content" not in output


def test_x15_checkpoint_resume_skips_committed_cell(
    tmp_path: Path, pair_values, cell
) -> None:
    pair = pair_values[0]
    _atomic_json(cell_path(tmp_path, pair), asdict(cell))
    assert load_cell(tmp_path, pair, "manifest") == cell
    with pytest.raises(ValueError, match="identity"):
        load_cell(tmp_path, pair, "wrong")


def test_x16_no_production_routing_or_config_mutation() -> None:
    source = getsource(run_cross_cell)
    assert "load_model_catalog" not in source
    assert "default" not in source.lower()
    assert "routing" not in source.lower()


def test_x17_exact_model_sha_recorded(cell) -> None:
    assert ":sha256:" in cell.repair_model_artifact


def test_x18_authority_sufficiency_carried_forward(cell) -> None:
    assert cell.authority_sufficient == "YES"
    assert cell.authorized_path_set


def test_failure_taxonomy_covers_required_boundaries() -> None:
    base = dict(
        schema=True,
        authorized=True,
        material=True,
        transaction=True,
        verification_executed=True,
        verification_passed=False,
        semantic=False,
    )
    assert _taxonomy(**{**base, "schema": False}) == "NO_VALID_PROPOSAL"
    assert _taxonomy(**{**base, "authorized": False}) == "UNAUTHORIZED_PATH"
    assert _taxonomy(**{**base, "material": False}) == "NO_CHANGE"
    assert _taxonomy(**{**base, "transaction": False}) == "TRANSACTION_FAILURE"
    assert _taxonomy(**{**base, "verification_executed": False}) == "STRUCTURAL_FAILURE"
    assert (
        _taxonomy(**{**base, "verification_passed": True})
        == "VERIFICATION_PASS_SEMANTIC_FAIL"
    )
    assert (
        _taxonomy(**{**base, "verification_passed": True, "semantic": True})
        == "SEMANTIC_RECOVERED"
    )


def test_corrective_diversity_uses_hashes_not_source(cell) -> None:
    values = tuple(
        replace(
            cell,
            repair_profile=profile,
            same_model=profile == cell.primary_profile,
            mutation_hash=f"hash-{profile}",
        )
        for profile in REPAIR_PROFILES
    )
    summary = summarize_case(values)
    assert summary["corrective_diversity"] == "DIVERGENT_WRONG_FIX"
    assert "mutation_hash" not in summary


def test_manifest_hash_is_order_stable() -> None:
    assert manifest_hash({"a": 1, "b": 2}) == manifest_hash({"b": 2, "a": 1})
