"""A72 source-free funnel and exact first-failure taxonomy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

from benchmarks.transaction_readiness_v1.protocol import TransitionStage
from forge.evaluation.mutation_ready import (
    MutationReadyClassification,
    evaluate_mutation_ready_v1,
)

STAGES = tuple(TransitionStage)


class FailureLayer(StrEnum):
    NO_MODEL_OUTPUT = "NO_MODEL_OUTPUT"
    MODEL_PROTOCOL_FAILURE = "MODEL_PROTOCOL_FAILURE"
    MATERIALIZATION_FAILURE = "MATERIALIZATION_FAILURE"
    PROVENANCE_FAILURE = "PROVENANCE_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    RANGE_IDENTITY_FAILURE = "RANGE_IDENTITY_FAILURE"
    AUTHORITY_FAILURE = "AUTHORITY_FAILURE"
    GROUP_BINDING_FAILURE = "GROUP_BINDING_FAILURE"
    PRODUCTION_METADATA_UNAVAILABLE = "PRODUCTION_METADATA_UNAVAILABLE"
    OBSERVATION_METADATA_INCOMPLETE = "OBSERVATION_METADATA_INCOMPLETE"
    PROPOSAL_EVIDENCE_IDENTITY_MISMATCH = "PROPOSAL_EVIDENCE_IDENTITY_MISMATCH"
    PREVIEW_OR_APPROVAL_FAILURE = "PREVIEW_OR_APPROVAL_FAILURE"
    TRANSACTION_PRECHECK_FAILURE = "TRANSACTION_PRECHECK_FAILURE"
    TRANSACTION_FAILURE = "TRANSACTION_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    SEMANTIC_FAILURE = "SEMANTIC_FAILURE"
    PASS = "PASS"


class TransactionFailureSubtype(StrEnum):
    TRANSACTION_PRECHECK_REJECTED = "TRANSACTION_PRECHECK_REJECTED"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    PATCH_CONFLICT = "PATCH_CONFLICT"
    INVALID_PATCH = "INVALID_PATCH"
    WORKSPACE_INTEGRITY_FAILURE = "WORKSPACE_INTEGRITY_FAILURE"
    TRANSACTION_APPLICATION_FAILURE = "TRANSACTION_APPLICATION_FAILURE"
    OTHER_TRANSACTION_FAILURE = "OTHER_TRANSACTION_FAILURE"


@dataclass(frozen=True, slots=True)
class FunnelResult:
    transitions: tuple[tuple[str, bool], ...]
    first_failed_stage: str | None
    failure_layer: str
    metadata_classification: str | None


def _metadata_failure(classification: MutationReadyClassification) -> FailureLayer:
    value = classification.value
    if "INCOMPLETE" in value:
        return FailureLayer.OBSERVATION_METADATA_INCOMPLETE
    if classification is MutationReadyClassification.PROVENANCE_MISMATCH:
        return FailureLayer.PROVENANCE_FAILURE
    if classification is MutationReadyClassification.GENERATION_MISMATCH:
        return FailureLayer.GENERATION_FAILURE
    if classification in {
        MutationReadyClassification.RANGE_IDENTITY_MISMATCH,
        MutationReadyClassification.SOURCE_IDENTITY_MISMATCH,
    }:
        return FailureLayer.RANGE_IDENTITY_FAILURE
    if classification in {
        MutationReadyClassification.AUTHORITY_MISMATCH,
        MutationReadyClassification.PATH_SET_MISMATCH,
    }:
        return FailureLayer.AUTHORITY_FAILURE
    if classification is MutationReadyClassification.GROUP_BINDING_MISMATCH:
        return FailureLayer.GROUP_BINDING_FAILURE
    if classification in {
        MutationReadyClassification.PREVIEW_INELIGIBLE,
        MutationReadyClassification.TRANSACTION_NOT_READY,
    }:
        return FailureLayer.PREVIEW_OR_APPROVAL_FAILURE
    return FailureLayer.MATERIALIZATION_FAILURE


def evaluate_funnel(
    *,
    model_output: bool,
    schema_valid: bool,
    metadata: Mapping[str, object] | None,
    observation_classification: str | None,
    transaction_applied: bool,
    verification_pass: bool,
    semantic_pass: bool,
) -> FunnelResult:
    """Evaluate ordered states; instrumentation failures fail closed and separately."""
    evaluated = evaluate_mutation_ready_v1(metadata or {}) if schema_valid else None
    mechanically = bool(evaluated and evaluated.mechanically_materializable)
    production = bool(evaluated and evaluated.production_validatable)
    ready = bool(evaluated and evaluated.transaction_ready)
    reached = (
        model_output,
        model_output and schema_valid,
        schema_valid and mechanically,
        schema_valid and production,
        schema_valid and ready,
        transaction_applied,
        verification_pass,
        semantic_pass,
    )
    for index, value in enumerate(reached):
        if value and not all(reached[:index]):
            raise ValueError(f"impossible transition skip at {STAGES[index].value}")
    failed = next((STAGES[i] for i, value in enumerate(reached) if not value), None)
    if not model_output:
        failure = FailureLayer.NO_MODEL_OUTPUT
    elif not schema_valid:
        failure = FailureLayer.MODEL_PROTOCOL_FAILURE
    elif observation_classification == "OBSERVATION_METADATA_INCOMPLETE":
        failure = FailureLayer.OBSERVATION_METADATA_INCOMPLETE
    elif observation_classification == "PRODUCTION_METADATA_UNAVAILABLE":
        failure = FailureLayer.PRODUCTION_METADATA_UNAVAILABLE
    elif failed in {
        TransitionStage.MECHANICALLY_MATERIALIZABLE,
        TransitionStage.PRODUCTION_VALIDATABLE,
        TransitionStage.TRANSACTION_READY,
    }:
        assert evaluated is not None
        failure = _metadata_failure(evaluated.classification)
    elif failed is TransitionStage.TRANSACTION_APPLIED:
        failure = FailureLayer.TRANSACTION_FAILURE
    elif failed is TransitionStage.VERIFICATION_PASS:
        failure = FailureLayer.VERIFICATION_FAILURE
    elif failed is TransitionStage.SEMANTIC_PASS:
        failure = FailureLayer.SEMANTIC_FAILURE
    else:
        failure = FailureLayer.PASS
    return FunnelResult(
        tuple(
            (stage.value, value) for stage, value in zip(STAGES, reached, strict=True)
        ),
        failed.value if failed else None,
        failure.value,
        evaluated.classification.value if evaluated else None,
    )


def transaction_failure_subtype(
    *, apply_result: str, failure_message: str | None
) -> str:
    """Classify only recorded production evidence, never reconstructed source."""
    evidence = f"{apply_result} {failure_message or ''}".casefold()
    if "source changed" in evidence or "stale" in evidence:
        return TransactionFailureSubtype.SOURCE_CHANGED.value
    if "conflict" in evidence:
        return TransactionFailureSubtype.PATCH_CONFLICT.value
    if "invalid patch" in evidence or "malformed patch" in evidence:
        return TransactionFailureSubtype.INVALID_PATCH.value
    if "integrity" in evidence or "rolled back" in evidence:
        return TransactionFailureSubtype.WORKSPACE_INTEGRITY_FAILURE.value
    if "precheck" in evidence or "pre-check" in evidence:
        return TransactionFailureSubtype.TRANSACTION_PRECHECK_REJECTED.value
    if apply_result == "failed" or "application failed" in evidence:
        return TransactionFailureSubtype.TRANSACTION_APPLICATION_FAILURE.value
    return TransactionFailureSubtype.OTHER_TRANSACTION_FAILURE.value


def source_free_funnel(value: FunnelResult) -> dict[str, object]:
    return asdict(value)
