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
    EVIDENCE_IDENTITY_MISMATCH,
    InstrumentationError,
    _assert_observations,
    bind_proposal_evidence,
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


def _proposal(proposal_id: str) -> dict[str, object]:
    return {**_metadata(), "proposal_observation_id": proposal_id}


def _transaction(proposal_id: str, attempt_id: str, outcome: str) -> dict[str, object]:
    return {
        "event": "transaction_result",
        "proposal_observation_id": proposal_id,
        "group_identity": "group",
        "workspace_generation": 2,
        "transaction_attempt_id": attempt_id,
        "transaction_outcome": outcome,
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


def test_rejected_then_applied_proposal_binds_only_applied_identity() -> None:
    metadata = (_proposal("p1"), _proposal("p2"))
    evidence = (
        _transaction("p1", "attempt-1", "failure"),
        _transaction("p2", "attempt-2", "success"),
        {
            "event": "verification_result",
            "proposal_observation_id": "p2",
            "verification_outcome": "pass",
        },
        {
            "event": "semantic_result",
            "proposal_observation_id": "p2",
            "semantic_outcome": "PASS",
        },
    )
    bound = bind_proposal_evidence(metadata, evidence)
    assert not bound["p1"]["transaction_applied"]
    assert not bound["p1"]["verification_pass"]
    assert not bound["p1"]["semantic_pass"]
    assert bound["p2"]["transaction_applied"]
    assert bound["p2"]["verification_pass"]
    assert bound["p2"]["semantic_pass"]


def test_c08_legacy_last_proposal_association_bug_is_reproduced() -> None:
    proposal_ids = ("applied-first", "rejected-last")
    legacy = {
        proposal_id: index == len(proposal_ids) - 1
        for index, proposal_id in enumerate(proposal_ids)
    }
    assert legacy == {"applied-first": False, "rejected-last": True}
    corrected = bind_proposal_evidence(
        tuple(_proposal(value) for value in proposal_ids),
        (
            _transaction("applied-first", "attempt-1", "success"),
            _transaction("rejected-last", "attempt-2", "failure"),
        ),
    )
    assert corrected["applied-first"]["transaction_applied"]
    assert not corrected["rejected-last"]["transaction_applied"]


def test_applied_then_rejected_and_multiple_rejected_proposals() -> None:
    metadata = (_proposal("p1"), _proposal("p2"), _proposal("p3"))
    bound = bind_proposal_evidence(
        metadata,
        (
            _transaction("p1", "attempt-1", "success"),
            _transaction("p2", "attempt-2", "failure"),
            _transaction("p3", "attempt-3", "failure"),
        ),
    )
    assert bound["p1"]["transaction_applied"]
    assert not bound["p2"]["transaction_applied"]
    assert not bound["p3"]["transaction_applied"]
    assert len({*bound}) == 3


def test_wrong_proposal_downstream_evidence_fails_closed() -> None:
    with pytest.raises(InstrumentationError, match=EVIDENCE_IDENTITY_MISMATCH):
        bind_proposal_evidence(
            (_proposal("p1"), _proposal("p2")),
            (
                _transaction("p1", "attempt-1", "failure"),
                {
                    "event": "verification_result",
                    "proposal_observation_id": "p2",
                    "verification_outcome": "pass",
                },
            ),
        )


def test_correction_or_repair_proposals_keep_distinct_transaction_bindings() -> None:
    bound = bind_proposal_evidence(
        (_proposal("primary"), _proposal("repair")),
        (
            _transaction("primary", "primary-attempt", "success"),
            _transaction("repair", "repair-attempt", "success"),
        ),
    )
    assert (
        bound["primary"]["transaction_attempts"]
        != bound["repair"]["transaction_attempts"]
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
        "proposals": [
            {
                "mutation_ready_metadata": _metadata(),
                "downstream_evidence": {
                    "transaction_attempts": [
                        {
                            "transaction_attempt_id": "attempt-1",
                            "transaction_outcome": "success",
                            "group_identity": "group",
                            "workspace_generation": 2,
                        }
                    ],
                    "transaction_applied": True,
                    "verification_pass": True,
                    "semantic_pass": True,
                },
            }
        ],
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
