from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.transaction_readiness_v1.protocol import TransitionStage
from benchmarks.transaction_readiness_v2.protocol import (
    FailureLayer,
    TransactionFailureSubtype,
    evaluate_funnel,
    transaction_failure_subtype,
)
from benchmarks.transaction_readiness_v2.runner import (
    InstrumentationError,
    _assert_observations,
    checkpoint_path,
    read_cell,
)
from benchmarks.transaction_readiness_v2.suite import (
    SUITE,
    frozen_matrix_identity,
    tasks,
)
from forge.evaluation.mutation_ready import (
    CandidateMetadata,
    ChildMetadata,
    MutationReadyMetadata,
    atomic_checkpoint,
    source_free_metadata,
)


def _metadata() -> dict[str, object]:
    value = source_free_metadata(
        MutationReadyMetadata(
            1,
            2,
            ("src/a.py",),
            (CandidateMetadata("src/a.py", "obs", "a" * 64, 1, 9, 2, "trusted"),),
            "line_range",
            (
                ChildMetadata(
                    "src/a.py", "edit", "line_range", "a" * 64, 2, 3, 2, "obs"
                ),
            ),
            1,
            ("src/a.py",),
            "group",
            2,
            True,
            "ready",
        )
    )
    return {
        **value,
        "proposal_observation_id": "proposal-observation-1",
        "observation_metadata_status": "complete",
        "production_metadata_status": "available",
    }


def test_corrected_observation_boundary_and_exactly_once() -> None:
    metadata = (_metadata(),)
    _assert_observations(metadata, ("OBSERVATION_METADATA_COMPLETE",))
    with pytest.raises(InstrumentationError, match="duplicate"):
        _assert_observations(metadata * 2, ("OBSERVATION_METADATA_COMPLETE",) * 2)
    with pytest.raises(InstrumentationError, match="INCOMPLETE"):
        _assert_observations(metadata, ("OBSERVATION_METADATA_INCOMPLETE",))


def test_state_order_first_failure_and_no_output() -> None:
    passed = evaluate_funnel(
        model_output=True,
        schema_valid=True,
        metadata=_metadata(),
        observation_classification="OBSERVATION_METADATA_COMPLETE",
        transaction_applied=True,
        verification_pass=True,
        semantic_pass=True,
    )
    assert tuple(name for name, _ in passed.transitions) == tuple(
        stage.value for stage in TransitionStage
    )
    assert passed.failure_layer == FailureLayer.PASS.value
    no_output = evaluate_funnel(
        model_output=False,
        schema_valid=False,
        metadata=None,
        observation_classification=None,
        transaction_applied=False,
        verification_pass=False,
        semantic_pass=False,
    )
    assert no_output.failure_layer == FailureLayer.NO_MODEL_OUTPUT.value


def test_transaction_rejection_subtypes() -> None:
    assert (
        transaction_failure_subtype(apply_result="failed", failure_message=None)
        == TransactionFailureSubtype.TRANSACTION_APPLICATION_FAILURE.value
    )
    assert (
        transaction_failure_subtype(
            apply_result="failed", failure_message="source changed"
        )
        == TransactionFailureSubtype.SOURCE_CHANGED.value
    )
    assert (
        transaction_failure_subtype(
            apply_result="failed", failure_message="workspace integrity failure"
        )
        == TransactionFailureSubtype.WORKSPACE_INTEGRITY_FAILURE.value
    )


def test_checkpoint_resume_source_free_and_package_exclusion(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "qwen-small", "C01")
    payload = {
        "suite": SUITE,
        "task_id": "C01",
        "model_profile": "qwen-small",
        "proposals": [{"mutation_ready_metadata": _metadata()}],
    }
    atomic_checkpoint(path, payload)
    assert read_cell(path, {"suite": SUITE, "task_id": "C01"}) == json.loads(
        json.dumps(payload)
    )
    assert "/Users/" not in path.read_text()
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration


def test_frozen_matrix_is_stable_and_uses_unchanged_a71_corpus() -> None:
    assert len(tasks()) == 14
    assert len(frozen_matrix_identity()) == 64
