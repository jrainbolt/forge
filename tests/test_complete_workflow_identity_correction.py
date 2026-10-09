from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.default_candidate_confirmation_v1.runner import (
    classify_workflow_outcome,
)
from benchmarks.default_candidate_confirmation_v1.suite import tasks
from benchmarks.default_candidate_confirmation_v1.workflow_identity import (
    WorkflowOutcome,
    begin_workflow_attempt,
    complete_workflow_attempt,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint


def _begin(profile: str = "profile-a", index: int = 8):  # type: ignore[no-untyped-def]
    return begin_workflow_attempt(
        tasks()[index],
        evaluator_corpus_identity="c" * 64,
        repository_identity="r" * 64,
        model_profile_identity=profile,
        seed=42,
        temperature=0,
        context_size=8192,
        output_budget=512,
        representation="line_range",
    )


def test_workflow_identity_exists_before_any_child_activity() -> None:
    attempt = _begin()
    assert attempt.workflow_attempt_id
    assert attempt.outcome is WorkflowOutcome.IN_FLIGHT
    assert attempt.mutation_request_ids == ()
    assert attempt.proposal_lineage == ()
    assert attempt.transaction_lineage == ()


@pytest.mark.parametrize("child_count", [0, 1, 3])
def test_workflow_accepts_zero_one_or_multiple_requests(child_count: int) -> None:
    attempt = _begin()
    request_ids = tuple(f"request-{index}" for index in range(child_count))
    completed = complete_workflow_attempt(
        attempt,
        outcome=(
            WorkflowOutcome.COMPLETED_WITH_MUTATION_REQUEST
            if child_count
            else WorkflowOutcome.COMPLETED_NO_MUTATION_REQUEST
        ),
        mutation_request_ids=request_ids,
    )
    assert completed.mutation_request_ids == request_ids


def test_request_proposal_transaction_and_repair_lineage_remain_distinct() -> None:
    attempt = complete_workflow_attempt(
        _begin(),
        outcome=WorkflowOutcome.COMPLETED_WITH_MUTATION_REQUEST,
        mutation_request_ids=("primary-request", "repair-request"),
        proposal_lineage=(("proposal", "primary-request"),),
        transaction_lineage=(("proposal", "transaction"),),
    )
    assert (
        len({attempt.workflow_attempt_id, "primary-request", "proposal", "transaction"})
        == 4
    )
    assert "repair-request" in attempt.mutation_request_ids


def test_h09_can_end_before_mutation_construction_without_fabrication() -> None:
    completed = complete_workflow_attempt(
        _begin(), outcome=WorkflowOutcome.COMPLETED_NO_MUTATION_REQUEST
    )
    assert completed.outcome is WorkflowOutcome.COMPLETED_NO_MUTATION_REQUEST
    assert not completed.mutation_request_ids
    assert not completed.proposal_lineage
    assert (
        classify_workflow_outcome({"failure": "INFRASTRUCTURE"}, ())
        is WorkflowOutcome.TOOL_FAILURE
    )


def test_representation_independent_identity_applies_to_all_operation_classes() -> None:
    identities = []
    for index in (0, 3, 6, 9):
        identities.append(_begin(index=index).representation_identity)
    assert len(set(identities)) == 1


def test_cross_profile_equivalence_excludes_model_profile() -> None:
    first = _begin("profile-a")
    second = _begin("profile-b")
    assert first.equivalence_identity == second.equivalence_identity
    assert first.workflow_attempt_id != second.workflow_attempt_id
    assert first.model_profile_identity != second.model_profile_identity


def test_exactly_one_terminal_outcome() -> None:
    completed = complete_workflow_attempt(
        _begin(), outcome=WorkflowOutcome.MODEL_PROTOCOL_FAILURE
    )
    with pytest.raises(ValueError, match="already"):
        complete_workflow_attempt(completed, outcome=WorkflowOutcome.OTHER_FAILURE)
    with pytest.raises(ValueError, match="terminal"):
        complete_workflow_attempt(_begin(), outcome=WorkflowOutcome.IN_FLIGHT)


def test_source_free_checkpoint_resume_and_absolute_paths_excluded(
    tmp_path: Path,
) -> None:
    payload = asdict(
        complete_workflow_attempt(
            _begin(), outcome=WorkflowOutcome.COMPLETED_NO_MUTATION_REQUEST
        )
    )
    encoded = json.dumps(payload)
    assert "/Users/" not in encoded
    assert standard_result_is_source_free(payload)
    path = tmp_path / "workflow.json"
    atomic_checkpoint(path, payload)
    assert resume_checkpoint(path) == json.loads(encoded)
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, payload)


def test_workflow_evaluator_and_results_are_package_excluded() -> None:
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "default_candidate_confirmation_v1" not in configuration
    assert "eval-results" not in configuration
