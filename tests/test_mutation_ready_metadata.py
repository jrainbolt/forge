from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from forge.evaluation.mutation_ready import (
    CandidateMetadata,
    ChildMetadata,
    MutationEvaluationState,
    MutationReadyClassification,
    MutationReadyMetadata,
    atomic_checkpoint,
    evaluate_mutation_ready_v1,
    load_mutation_ready_metadata,
    resume_checkpoint,
    source_free_evaluation,
    source_free_metadata,
)

PATHS = ("src/a.py", "src/b.py")


def _metadata() -> MutationReadyMetadata:
    candidates = tuple(
        CandidateMetadata(
            path,
            f"observation-{index}",
            str(index) * 64,
            1,
            20,
            7,
            "trusted_task_metadata",
        )
        for index, path in enumerate(PATHS, 1)
    )
    children = tuple(
        ChildMetadata(
            path,
            "edit",
            "line_range",
            str(index) * 64,
            4,
            6,
            7,
            f"observation-{index}",
        )
        for index, path in enumerate(PATHS, 1)
    )
    return MutationReadyMetadata(
        1,
        7,
        PATHS,
        candidates,
        "line_range",
        children,
        2,
        PATHS,
        "group-identity",
        7,
        True,
        "ready",
    )


def test_proposal_metadata_capture_preserves_every_required_identity() -> None:
    metadata = _metadata()
    restored = load_mutation_ready_metadata(source_free_metadata(metadata))
    assert restored == metadata
    assert restored.candidates[0].observation_id == "observation-1"
    assert restored.workspace_generation == restored.candidates[0].generation == 7
    assert (
        restored.candidates[0].authorized_start_line,
        restored.candidates[0].authorized_end_line,
    ) == (1, 20)
    assert restored.candidates[0].authority_provenance_class == "trusted_task_metadata"
    assert restored.mutation_representation == "line_range"
    assert restored.canonical_child_order == PATHS
    assert restored.group_identity == "group-identity"
    assert restored.preview_eligible
    assert restored.transaction_readiness_state == "ready"


def test_exact_state_transitions_are_not_collapsed() -> None:
    ready = evaluate_mutation_ready_v1(_metadata())
    assert ready.highest_state is MutationEvaluationState.TRANSACTION_READY
    preview_blocked = evaluate_mutation_ready_v1(
        replace(_metadata(), preview_eligible=False)
    )
    assert preview_blocked.mechanically_materializable
    assert preview_blocked.production_validatable
    assert not preview_blocked.transaction_ready
    provenance_blocked = evaluate_mutation_ready_v1(
        replace(
            _metadata(),
            children=(
                replace(_metadata().children[0], candidate_observation_id="untrusted"),
                _metadata().children[1],
            ),
        )
    )
    assert (
        provenance_blocked.highest_state
        is MutationEvaluationState.MECHANICALLY_MATERIALIZABLE
    )
    assert not provenance_blocked.production_validatable
    malformed = evaluate_mutation_ready_v1(
        replace(_metadata(), normalized_operation_count=3)
    )
    assert malformed.highest_state is None
    assert not malformed.mechanically_materializable


def test_exact_text_terminal_newline_range_matches_production_semantics() -> None:
    metadata = _metadata()
    candidate = replace(metadata.candidates[0], authorized_end_line=6)
    child = replace(metadata.children[0], representation="exact_text", end_line=7)
    value = replace(
        metadata,
        candidates=(candidate, metadata.candidates[1]),
        children=(child, metadata.children[1]),
        mutation_representation="exact_text",
    )
    value = replace(
        value,
        children=(child, replace(value.children[1], representation="exact_text")),
    )
    assert evaluate_mutation_ready_v1(value).transaction_ready


@pytest.mark.parametrize(
    ("metadata", "classification", "valid"),
    [
        (_metadata(), MutationReadyClassification.PASS, True),
        (
            replace(_metadata(), group_identity="valid-group-2"),
            MutationReadyClassification.PASS,
            True,
        ),
        (
            replace(
                _metadata(),
                children=(
                    replace(_metadata().children[0], candidate_observation_id="bad"),
                    _metadata().children[1],
                ),
            ),
            MutationReadyClassification.PROVENANCE_MISMATCH,
            False,
        ),
        (
            replace(
                _metadata(),
                candidates=(
                    replace(_metadata().candidates[0], authority_provenance_class=""),
                    _metadata().candidates[1],
                ),
            ),
            MutationReadyClassification.PROVENANCE_MISMATCH,
            False,
        ),
        (
            replace(
                _metadata(),
                children=(
                    replace(_metadata().children[0], generation=8),
                    _metadata().children[1],
                ),
            ),
            MutationReadyClassification.GENERATION_MISMATCH,
            False,
        ),
        (
            replace(
                _metadata(),
                children=(
                    replace(_metadata().children[0], end_line=21),
                    _metadata().children[1],
                ),
            ),
            MutationReadyClassification.RANGE_IDENTITY_MISMATCH,
            False,
        ),
    ],
)
def test_six_case_stored_metadata_control_set(
    metadata: MutationReadyMetadata,
    classification: MutationReadyClassification,
    valid: bool,
) -> None:
    result = evaluate_mutation_ready_v1(source_free_metadata(metadata))
    assert result.classification is classification
    assert result.production_validatable is valid


def test_missing_and_legacy_metadata_fail_closed_without_upgrade() -> None:
    partial = source_free_metadata(_metadata())
    partial.pop("candidates")
    assert (
        evaluate_mutation_ready_v1(partial).classification
        is MutationReadyClassification.MUTATION_READY_METADATA_INCOMPLETE
    )
    historical = {"task_id": "A68-F06", "mutation_ready": True}
    before = dict(historical)
    result = evaluate_mutation_ready_v1(historical)
    assert (
        result.classification
        is MutationReadyClassification.LEGACY_MUTATION_READY_METADATA_INCOMPLETE
    )
    assert historical == before
    invalid_range = source_free_metadata(_metadata())
    invalid_range["candidates"][0]["authorized_end_line"] = None  # type: ignore[index]
    assert (
        evaluate_mutation_ready_v1(invalid_range).classification
        is MutationReadyClassification.MUTATION_READY_METADATA_INCOMPLETE
    )


def test_source_free_checkpoint_resume_is_exact_and_does_not_regenerate(
    tmp_path: Path,
) -> None:
    metadata = source_free_metadata(_metadata())
    payload = {
        "cell_id": "control-1",
        "mutation_ready_metadata": metadata,
        "evaluation": source_free_evaluation(evaluate_mutation_ready_v1(metadata)),
    }
    encoded = json.dumps(payload).casefold()
    for forbidden in (
        "source_content",
        "replacement",
        "new_text",
        "old_text",
        "hidden_reference",
        "expected_implementation",
        "chain_of_thought",
        "/users/",
    ):
        assert forbidden not in encoded
    checkpoint = tmp_path / "checkpoint.json"
    atomic_checkpoint(checkpoint, payload)
    first = resume_checkpoint(checkpoint)
    second = resume_checkpoint(checkpoint)
    assert first == second == json.loads(json.dumps(payload))
    with pytest.raises(FileExistsError):
        atomic_checkpoint(checkpoint, {"regenerated": True})


def test_evaluator_data_and_tinyqueue_packaging_boundaries() -> None:
    root = Path(__file__).parents[1]
    configuration = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'where = ["src"]' in configuration
    assert "eval-results" not in configuration
    assert "fixtures/eval_repo/src/tinyqueue/__init__.py" in configuration
