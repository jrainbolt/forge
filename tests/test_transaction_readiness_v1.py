from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.realistic_coding_v2.suite import REPOSITORY, OperationClass
from benchmarks.transaction_readiness_v1.protocol import (
    FailureLayer,
    TransitionStage,
    evaluate_funnel,
    source_free_funnel,
)
from benchmarks.transaction_readiness_v1.runner import (
    checkpoint_path,
    read_cell,
    standard_result_is_source_free,
)
from benchmarks.transaction_readiness_v1.suite import tasks, validate_corpus
from forge.evaluation.mutation_ready import (
    CandidateMetadata,
    ChildMetadata,
    MutationReadyMetadata,
    ObservationMetadataClassification,
    atomic_checkpoint,
    classify_observation_metadata,
    evaluate_mutation_ready_v1,
    source_free_metadata,
)
from forge.evaluation.replay import source_state_identity


def _metadata(operation: str = "edit") -> dict[str, object]:
    representation = "create_text" if operation == "create" else "line_range"
    candidate = CandidateMetadata(
        "src/a.py",
        "observation",
        "a" * 64,
        None if operation == "create" else 1,
        None if operation == "create" else 20,
        3,
        "trusted",
        "create_parent" if operation == "create" else "existing_source",
        "a" * 64 if operation == "create" else None,
    )
    child = ChildMetadata(
        "src/a.py",
        operation,
        representation,
        "a" * 64,
        None if operation == "create" else 4,
        None if operation == "create" else 6,
        3,
        "observation",
    )
    return source_free_metadata(
        MutationReadyMetadata(
            1,
            3,
            ("src/a.py",),
            (candidate,),
            representation,
            (child,),
            1,
            ("src/a.py",),
            "group",
            3,
            True,
            "ready",
        )
    )


def _funnel(metadata: dict[str, object] | None = None):  # type: ignore[no-untyped-def]
    value = metadata or _metadata()
    ready = evaluate_mutation_ready_v1(value).transaction_ready
    return evaluate_funnel(
        model_output=True,
        schema_valid=True,
        metadata=value,
        transaction_applied=ready,
        verification_pass=ready,
        semantic_pass=ready,
    )


def test_exact_transition_order_and_pass() -> None:
    result = _funnel()
    assert tuple(name for name, _value in result.transitions) == tuple(
        stage.value for stage in TransitionStage
    )
    assert all(value for _name, value in result.transitions)
    assert result.failure_layer == FailureLayer.PASS.value


def test_impossible_state_skipping_is_rejected() -> None:
    with pytest.raises(ValueError, match="impossible transition skip"):
        evaluate_funnel(
            model_output=False,
            schema_valid=False,
            metadata=None,
            transaction_applied=True,
            verification_pass=False,
            semantic_pass=False,
        )


def test_metadata_is_evaluated_before_downstream_outcome() -> None:
    result = evaluate_funnel(
        model_output=True,
        schema_valid=True,
        metadata=_metadata(),
        transaction_applied=False,
        verification_pass=False,
        semantic_pass=False,
    )
    assert result.first_failed_stage == TransitionStage.TRANSACTION_APPLIED.value
    assert dict(result.transitions)[TransitionStage.TRANSACTION_READY.value]


def test_missing_metadata_fails_closed() -> None:
    result = evaluate_funnel(
        model_output=True,
        schema_valid=True,
        metadata=None,
        transaction_applied=False,
        verification_pass=False,
        semantic_pass=False,
    )
    assert result.failure_layer == FailureLayer.MUTATION_READY_METADATA_INCOMPLETE.value


def test_instrumentation_failure_is_separate_from_production_unavailability() -> None:
    complete = {
        **_metadata(),
        "proposal_observation_id": "proposal-observation-1",
        "observation_metadata_status": "complete",
        "production_metadata_status": "available",
    }
    assert (
        classify_observation_metadata(complete)
        is ObservationMetadataClassification.OBSERVATION_METADATA_COMPLETE
    )
    assert (
        classify_observation_metadata(_metadata())
        is ObservationMetadataClassification.OBSERVATION_METADATA_INCOMPLETE
    )
    unavailable = {**complete, "production_metadata_status": "unavailable"}
    assert (
        classify_observation_metadata(unavailable)
        is ObservationMetadataClassification.PRODUCTION_METADATA_UNAVAILABLE
    )


@pytest.mark.parametrize("operation", ["edit", "create"])
def test_single_and_create_readiness(operation: str) -> None:
    assert _funnel(_metadata(operation)).failure_layer == FailureLayer.PASS.value


def test_grouped_and_mixed_readiness() -> None:
    base = _metadata()
    first_candidate = base["candidates"][0]  # type: ignore[index]
    first_child = base["children"][0]  # type: ignore[index]
    second_candidate = {
        **first_candidate,
        "path": "src/b.py",
        "observation_id": "second",
    }  # type: ignore[arg-type]
    second_child = {
        **first_child,
        "path": "src/b.py",
        "candidate_observation_id": "second",
    }  # type: ignore[arg-type]
    grouped = {
        **base,
        "authorized_paths": ("src/a.py", "src/b.py"),
        "candidates": (first_candidate, second_candidate),
        "children": (first_child, second_child),
        "normalized_operation_count": 2,
        "canonical_child_order": ("src/a.py", "src/b.py"),
    }
    assert _funnel(grouped).failure_layer == FailureLayer.PASS.value
    create_candidate = {
        **second_candidate,
        "authorized_start_line": None,
        "authorized_end_line": None,
        "authority_kind": "create_parent",
        "creation_parent_identity": "b" * 64,
    }
    create_child = {
        **second_child,
        "operation_type": "create",
        "representation": "create_text",
        "start_line": None,
        "end_line": None,
    }
    mixed = {
        **grouped,
        "mutation_representation": "mixed",
        "candidates": (first_candidate, create_candidate),
        "children": (first_child, create_child),
    }
    assert _funnel(mixed).failure_layer == FailureLayer.PASS.value


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"candidate_observation_id": "bad"}, FailureLayer.PROVENANCE_FAILURE),
        ({"generation": 4}, FailureLayer.GENERATION_FAILURE),
        ({"end_line": 21}, FailureLayer.RANGE_IDENTITY_FAILURE),
    ],
)
def test_identity_failure_layers(
    change: dict[str, object], expected: FailureLayer
) -> None:
    value = _metadata()
    value["children"] = ({**value["children"][0], **change},)  # type: ignore[index]
    assert _funnel(value).failure_layer == expected.value


def test_authority_and_group_binding_failures() -> None:
    authority = _metadata()
    authority["authorized_paths"] = ("src/other.py",)
    assert _funnel(authority).failure_layer == FailureLayer.AUTHORITY_FAILURE.value
    group = _metadata()
    group["group_generation"] = 4
    assert _funnel(group).failure_layer == FailureLayer.GROUP_BINDING_FAILURE.value


def test_checkpoint_resume_exactness_source_free_and_package_exclusion(
    tmp_path: Path,
) -> None:
    result = source_free_funnel(_funnel())
    metadata = {
        **_metadata(),
        "proposal_observation_id": "proposal-observation-1",
        "observation_metadata_status": "complete",
        "production_metadata_status": "available",
    }
    payload = {
        "task_id": "C01",
        "model_profile": "qwen-small",
        "result": result,
        "metadata": metadata,
    }
    path = checkpoint_path(tmp_path, "qwen-small", "C01")
    atomic_checkpoint(path, payload)
    expected = {"task_id": "C01", "model_profile": "qwen-small"}
    assert read_cell(path, expected) == json.loads(json.dumps(payload))
    assert read_cell(path, expected)["metadata"]["proposal_observation_id"] == (
        "proposal-observation-1"
    )
    assert standard_result_is_source_free(payload)
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration


def test_frozen_corpus_coverage_and_canonical_source_identity() -> None:
    definitions = tasks()
    validate_corpus(definitions)
    counts = {
        operation: sum(item.operation_class is operation for item in definitions)
        for operation in OperationClass
    }
    assert counts == {
        OperationClass.EDIT_SINGLE: 4,
        OperationClass.EDIT_MULTI: 4,
        OperationClass.CREATE: 3,
        OperationClass.MIXED_EDIT_CREATE: 3,
    }
    before = source_state_identity(REPOSITORY)
    assert before == source_state_identity(REPOSITORY)
