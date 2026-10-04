"""Source-free A71 readiness funnel and failure classification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

from forge.evaluation.mutation_ready import (
    MutationReadyClassification,
    evaluate_mutation_ready_v1,
    source_free_evaluation,
)


class TransitionStage(StrEnum):
    MODEL_OUTPUT = "MODEL_OUTPUT"
    SCHEMA_VALID = "SCHEMA_VALID"
    MECHANICALLY_MATERIALIZABLE = "MECHANICALLY_MATERIALIZABLE"
    PRODUCTION_VALIDATABLE = "PRODUCTION_VALIDATABLE"
    TRANSACTION_READY = "TRANSACTION_READY"
    TRANSACTION_APPLIED = "TRANSACTION_APPLIED"
    VERIFICATION_PASS = "VERIFICATION_PASS"
    SEMANTIC_PASS = "SEMANTIC_PASS"


STAGES = tuple(TransitionStage)


class FailureLayer(StrEnum):
    MODEL_PROTOCOL_FAILURE = "MODEL_PROTOCOL_FAILURE"
    MATERIALIZATION_FAILURE = "MATERIALIZATION_FAILURE"
    PROVENANCE_FAILURE = "PROVENANCE_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    RANGE_IDENTITY_FAILURE = "RANGE_IDENTITY_FAILURE"
    AUTHORITY_FAILURE = "AUTHORITY_FAILURE"
    GROUP_BINDING_FAILURE = "GROUP_BINDING_FAILURE"
    PREVIEW_OR_APPROVAL_FAILURE = "PREVIEW_OR_APPROVAL_FAILURE"
    TRANSACTION_PRECHECK_FAILURE = "TRANSACTION_PRECHECK_FAILURE"
    TRANSACTION_FAILURE = "TRANSACTION_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    SEMANTIC_FAILURE = "SEMANTIC_FAILURE"
    MUTATION_READY_METADATA_INCOMPLETE = "MUTATION_READY_METADATA_INCOMPLETE"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class FunnelResult:
    transitions: tuple[tuple[str, bool], ...]
    first_failed_stage: str | None
    failure_layer: str
    metadata_classification: str

    def reached(self, stage: TransitionStage) -> bool:
        return dict(self.transitions)[stage.value]


def _metadata_failure(classification: MutationReadyClassification) -> FailureLayer:
    if classification in {
        MutationReadyClassification.MUTATION_READY_METADATA_INCOMPLETE,
        MutationReadyClassification.LEGACY_MUTATION_READY_METADATA_INCOMPLETE,
    }:
        return FailureLayer.MUTATION_READY_METADATA_INCOMPLETE
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
    if classification is MutationReadyClassification.MATERIALIZATION_FAILURE:
        return FailureLayer.MATERIALIZATION_FAILURE
    return FailureLayer.MATERIALIZATION_FAILURE


def evaluate_funnel(
    *,
    model_output: bool,
    schema_valid: bool,
    metadata: Mapping[str, object] | None,
    transaction_applied: bool,
    verification_pass: bool,
    semantic_pass: bool,
) -> FunnelResult:
    metadata_value: Mapping[str, object] = metadata or {}
    evaluated = evaluate_mutation_ready_v1(metadata_value)
    reached = (
        model_output,
        schema_valid,
        schema_valid and evaluated.mechanically_materializable,
        schema_valid and evaluated.production_validatable,
        schema_valid and evaluated.transaction_ready,
        transaction_applied,
        verification_pass,
        semantic_pass,
    )
    for index, value in enumerate(reached):
        if value and not all(reached[:index]):
            raise ValueError(f"impossible transition skip at {STAGES[index].value}")
    failed = next(
        (STAGES[index] for index, value in enumerate(reached) if not value), None
    )
    if failed is TransitionStage.MODEL_OUTPUT or failed is TransitionStage.SCHEMA_VALID:
        failure = FailureLayer.MODEL_PROTOCOL_FAILURE
    elif failed in {
        TransitionStage.MECHANICALLY_MATERIALIZABLE,
        TransitionStage.PRODUCTION_VALIDATABLE,
        TransitionStage.TRANSACTION_READY,
    }:
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
        evaluated.classification.value,
    )


def source_free_funnel(value: FunnelResult) -> dict[str, object]:
    return asdict(value)


def metadata_evaluation(metadata: Mapping[str, object]) -> dict[str, object]:
    return source_free_evaluation(evaluate_mutation_ready_v1(metadata))


def aggregate(cells: tuple[Mapping[str, object], ...], key: str) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for cell in cells:
        groups.setdefault(str(cell[key]), []).append(cell)
    result: dict[str, object] = {}
    for name, selected in sorted(groups.items()):
        counts = {
            stage.value: sum(
                bool(dict(item["transitions"])[stage.value])  # type: ignore[arg-type]
                for item in selected
            )
            for stage in STAGES
        }
        failures = {
            failure: sum(item["failure_layer"] == failure for item in selected)
            for failure in sorted({str(item["failure_layer"]) for item in selected})
        }
        result[name] = {
            "cells": len(selected),
            "transitions": counts,
            "failures": failures,
        }
    return result
