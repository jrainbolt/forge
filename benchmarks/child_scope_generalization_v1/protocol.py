"""Precise production rejection tracing for complete A68 compositions."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class ProductionRejection(StrEnum):
    COMPOSITION_VALID = "COMPOSITION_VALID"
    PREVIEW_REJECTED = "PREVIEW_REJECTED"
    APPROVAL_IDENTITY_REJECTED = "APPROVAL_IDENTITY_REJECTED"
    SOURCE_CURRENTNESS_REJECTED = "SOURCE_CURRENTNESS_REJECTED"
    GROUP_VALIDATION_REJECTED = "GROUP_VALIDATION_REJECTED"
    TRANSACTION_PRECHECK_REJECTED = "TRANSACTION_PRECHECK_REJECTED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    SEMANTIC_FAILED = "SEMANTIC_FAILED"
    PASS = "PASS"


class RejectionMetrics(Protocol):
    structured_mutation_valid: int
    line_range_attempts: int
    line_range_materialized: int
    mutation_group_validation_result: str
    mutation_group_preview_created: int
    mutation_group_apply_result: str
    preview_created: int
    mutations: int
    verification_plan_result: str
    reverification_result: str


def classify_production_rejection(
    metrics: RejectionMetrics,
    *,
    semantic_pass: bool,
) -> ProductionRejection:
    """Return the first exact observable production rejection layer."""
    if metrics.structured_mutation_valid == 0:
        if metrics.line_range_attempts and metrics.line_range_materialized == 0:
            return ProductionRejection.SOURCE_CURRENTNESS_REJECTED
        return ProductionRejection.GROUP_VALIDATION_REJECTED
    if metrics.mutation_group_validation_result == "failed":
        return ProductionRejection.GROUP_VALIDATION_REJECTED
    if metrics.mutation_group_preview_created == 0 and metrics.preview_created == 0:
        return ProductionRejection.PREVIEW_REJECTED
    if metrics.mutations == 0:
        if metrics.mutation_group_apply_result == "failed":
            return ProductionRejection.TRANSACTION_FAILED
        if metrics.mutation_group_validation_result == "passed":
            return ProductionRejection.TRANSACTION_PRECHECK_REJECTED
        return ProductionRejection.APPROVAL_IDENTITY_REJECTED
    verification = (
        metrics.verification_plan_result == "pass"
        or metrics.reverification_result == "pass"
    )
    if not verification:
        return ProductionRejection.VERIFICATION_FAILED
    if not semantic_pass:
        return ProductionRejection.SEMANTIC_FAILED
    return ProductionRejection.PASS
