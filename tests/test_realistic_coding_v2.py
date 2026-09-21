"""Model-free integrity and durability tests for realistic-coding-v2."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.realistic_coding_v2.runner import (
    Failure,
    _atomic_json,
    _proposal_shape,
    aggregate_cells,
    checkpoint_path,
    classify_failure,
    envelope_shape,
    freeze_manifest,
    read_checkpoint,
    repair_recovered,
)
from benchmarks.realistic_coding_v2.suite import (
    REPOSITORY,
    OperationClass,
    baseline_workspace,
    manifest_identity,
    tasks,
    validate_integrity,
)
from forge.evaluation import (
    EvaluationOutcome,
    RealWorldMetrics,
    RealWorldStatus,
    RealWorldTaskResult,
)
from forge.evaluation.realworld import hash_workspace
from forge.models import ModelUsage


@pytest.fixture(scope="module")
def definitions():  # type: ignore[no-untyped-def]
    return tasks()


@pytest.fixture(scope="module")
def integrity(definitions):  # type: ignore[no-untyped-def]
    return validate_integrity(definitions)


def _raw(**changes):  # type: ignore[no-untyped-def]
    values = dict(
        expected_implementation_acquired=True,
        mutation_ready_reached=True,
        mutation_proposed=True,
        structured_mutation_valid=1,
        preview_created=1,
        mutations=1,
        verification_result="pass",
    )
    values.update(changes)
    metrics = replace(RealWorldMetrics(), **values)
    return RealWorldTaskResult(
        "C01",
        "repair",
        "bounded_repair",
        42,
        RealWorldStatus.PASS,
        EvaluationOutcome.PASS,
        EvaluationOutcome.PASS,
        EvaluationOutcome.FAIL,
        None,
        None,
        "failed",
        (),
        (),
        (),
        (),
        (),
        metrics,
        ModelUsage(),
        1.0,
    )


def test_v01_task_versioning(definitions) -> None:  # type: ignore[no-untyped-def]
    assert [item.versioned_id for item in definitions] == [
        f"C{number:02d}-v1" for number in range(1, 13)
    ]
    assert len(manifest_identity(definitions)) == 64


def test_v02_operation_class_integrity(definitions) -> None:  # type: ignore[no-untyped-def]
    assert {
        kind: sum(item.operation_class is kind for item in definitions)
        for kind in OperationClass
    } == {kind: 3 for kind in OperationClass}
    assert sum(item.language == "Python" for item in definitions) == 6
    assert sum(item.language == "C17" for item in definitions) == 6


def test_v03_baseline_reference_wrong_oracles(integrity) -> None:  # type: ignore[no-untyped-def]
    assert len(integrity) == 12
    assert all(
        not item.baseline_pass
        and item.reference_pass
        and not item.wrong_pass
        and item.eligible
        for item in integrity
    )


def test_v04_create_only_needs_absent_path(definitions) -> None:  # type: ignore[no-untyped-def]
    for item in definitions[6:9]:
        assert item.edit_paths == ()
        assert len(item.create_paths) == 1
        assert item.production_task.setup_absent_paths == item.create_paths
        assert not item.production_task.mixed_file_operations


def test_v05_mixed_requires_both_operation_types(definitions, integrity) -> None:  # type: ignore[no-untyped-def]
    assert all(
        item.edit_paths
        and item.create_paths
        and item.production_task.mixed_file_operations
        for item in definitions[9:]
    )
    assert all(not item.partial_pass for item in integrity[9:])


def test_v06_reference_and_oracle_not_leaked(definitions) -> None:  # type: ignore[no-untyped-def]
    for item in definitions:
        prompt = item.production_task.prompt.lower()
        assert "oracle.py" not in prompt and "reference" not in prompt
        assert "wrong_create_source" not in prompt
        assert (
            item.wrong_create_source is None or item.wrong_create_source not in prompt
        )


def test_v07_wrong_states_are_predeclared(definitions, integrity) -> None:  # type: ignore[no-untyped-def]
    assert all(
        item.wrong_replacements
        or item.wrong_create_source
        or (
            item.operation_class is OperationClass.EDIT_MULTI
            and len(item.edit_paths) > 1
        )
        for item in definitions
    )
    assert all(not item.wrong_pass for item in integrity)


def test_v08_disposable_workspace_safety(definitions, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    before = hash_workspace(REPOSITORY)
    item = definitions[6]
    workspace = baseline_workspace(item, tmp_path / "workspace")
    assert workspace != REPOSITORY
    assert not (workspace / item.create_paths[0]).exists()
    assert (REPOSITORY / item.create_paths[0]).exists()
    assert hash_workspace(REPOSITORY) == before


def test_v09_checkpoint_atomic_write(tmp_path: Path) -> None:
    target = tmp_path / "cell.json"
    _atomic_json(target, {"result": "failed"})
    assert json.loads(target.read_text()) == {"result": "failed"}
    with pytest.raises(FileExistsError):
        _atomic_json(target, {"result": "pass"})
    assert not list(tmp_path.glob("*.tmp"))


def test_v10_committed_failed_cell_resumes_without_rerun(
    definitions, integrity, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    manifest = freeze_manifest(tmp_path, definitions, integrity)
    item = definitions[0]
    path = checkpoint_path(tmp_path, "qwen-small", 42, item.task_id)
    payload = {
        "suite": "realistic-coding-v2",
        "schema_version": 1,
        "task_id": item.task_id,
        "task_version": 1,
        "operation_class": item.operation_class.value,
        "model_profile": "qwen-small",
        "model_artifact": "model:sha256:abc",
        "seed": 42,
        "repository_identity": manifest["repository_identity"],
        "manifest_identity": manifest["manifest_identity"],
        "failure_layer": "PROTOCOL_FAILED",
        "final_semantic": False,
    }
    _atomic_json(path, payload)
    assert (
        read_checkpoint(
            path,
            profile="qwen-small",
            seed=42,
            definition=item,
            artifact="model:sha256:abc",
            manifest=manifest,
        )
        == payload
    )
    with pytest.raises(ValueError):
        read_checkpoint(
            path,
            profile="qwen-large",
            seed=42,
            definition=item,
            artifact="model:sha256:abc",
            manifest=manifest,
        )


def test_v11_failure_taxonomy(definitions) -> None:  # type: ignore[no-untyped-def]
    item = definitions[0]
    assert (
        classify_failure(
            item,
            _raw(mutation_ready_reached=False),
            proposal_role_correct=True,
            first_pass=None,
        )
        == Failure.AUTHORITY_NOT_READY
    )
    assert (
        classify_failure(item, _raw(), proposal_role_correct=False, first_pass=False)
        == Failure.EDIT_CONSTRUCTION_FAILED
    )
    assert (
        classify_failure(item, _raw(), proposal_role_correct=True, first_pass=False)
        == Failure.SEMANTIC_ORACLE_FAILED
    )
    misleading = replace(_raw(), oracle=EvaluationOutcome.PASS)
    assert (
        classify_failure(item, misleading, proposal_role_correct=False, first_pass=True)
        == Failure.EDIT_CONSTRUCTION_FAILED
    )
    post_mutation_refresh_failure = _raw(
        source_acquisition_failures=1, repair_attempts=1, verification_result="fail"
    )
    assert (
        classify_failure(
            item,
            post_mutation_refresh_failure,
            proposal_role_correct=True,
            first_pass=False,
        )
        == Failure.REPAIR_FAILED
    )
    rejected_edit = _raw(mutation_proposed=False)
    assert (
        classify_failure(
            item,
            rejected_edit,
            proposal_role_correct=True,
            first_pass=None,
            mutation_envelope_seen=True,
        )
        == Failure.EDIT_CONSTRUCTION_FAILED
    )
    repaired = replace(
        _raw(
            repair_attempts=1,
            verification_result="fail",
            reverification_result="pass",
        ),
        oracle=EvaluationOutcome.PASS,
    )
    assert (
        classify_failure(item, repaired, proposal_role_correct=True, first_pass=False)
        == Failure.PASS
    )


def test_v12_first_pass_and_final_are_separate() -> None:
    assert repair_recovered(False, True, True)
    assert not repair_recovered(True, True, True)
    assert not repair_recovered(False, True, False)


def test_v13_operation_class_aggregation(definitions) -> None:  # type: ignore[no-untyped-def]
    from benchmarks.realistic_coding_v2.runner import CellResult

    base = {name: False for name in CellResult.__dataclass_fields__}
    base.update(
        operation_class=OperationClass.CREATE.value,
        proposal_role_correct=True,
        transaction_executed=True,
        final_semantic=False,
    )
    create = CellResult(**base)
    edit = replace(
        create, operation_class=OperationClass.EDIT_SINGLE.value, final_semantic=True
    )
    totals = aggregate_cells((create, edit))
    assert totals["CREATE"] == {
        "cells": 1,
        "role_correct": 1,
        "transactions": 1,
        "semantic": 0,
    }
    assert totals["EDIT_SINGLE"]["semantic"] == 1


def test_v14_verification_alignment_is_explicit(integrity) -> None:  # type: ignore[no-untyped-def]
    assert all(item.verification_reference_pass for item in integrity)
    assert any(item.verification_baseline_pass for item in integrity)
    assert all(isinstance(item.verification_wrong_pass, bool) for item in integrity)


def test_v15_result_envelope_is_source_free(definitions) -> None:  # type: ignore[no-untyped-def]
    item = definitions[6]
    path = item.create_paths[0]
    secret = "highly_sensitive_generated_source"
    shaped = envelope_shape(
        json.dumps({"type": "create_file", "path": path, "content": secret}),
        frozenset({path}),
    )
    assert shaped == {
        "type": "create_file",
        "children": [{"type": "create_file", "path": path}],
    }
    assert secret not in json.dumps(shaped)
    assert _proposal_shape(item, (shaped,))[0]
    assert "<unauthorized>" in json.dumps(
        envelope_shape(
            json.dumps({"type": "create_file", "path": secret, "content": secret}),
            frozenset({path}),
        )
    )
    mixed = definitions[9]
    duplicate = envelope_shape(
        json.dumps(
            {
                "type": "multi_file_change",
                "operations": [
                    {"type": "line_range_edit", "path": mixed.edit_paths[0]},
                    {"type": "create_file", "path": mixed.create_paths[0]},
                    {"type": "create_file", "path": mixed.create_paths[0]},
                ],
            }
        ),
        frozenset((*mixed.edit_paths, *mixed.create_paths)),
    )
    correct, duplicates, *_ = _proposal_shape(mixed, (duplicate,))
    assert not correct and duplicates == 1


def test_v16_packaging_excludes_a56_fixture_and_results() -> None:
    project = (REPOSITORY.parents[2] / "pyproject.toml").read_text()
    assert '"forge.evaluation"' in project
    package_data = project.split("[tool.setuptools.package-data]", 1)[1]
    assert "realistic_coding_v2" not in package_data
    assert "eval-results" not in package_data
    assert "checkpoints" not in package_data
