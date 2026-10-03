"""Source-free V1 mirror of production grouped candidate validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GroupValidationFailure(StrEnum):
    PATH_SET_MISMATCH = "PATH_SET_MISMATCH"
    OPERATION_COUNT_MISMATCH = "OPERATION_COUNT_MISMATCH"
    DUPLICATE_PATH = "DUPLICATE_PATH"
    OPERATION_TYPE_MISMATCH = "OPERATION_TYPE_MISMATCH"
    REPRESENTATION_MISMATCH = "REPRESENTATION_MISMATCH"
    SOURCE_IDENTITY_MISMATCH = "SOURCE_IDENTITY_MISMATCH"
    RANGE_IDENTITY_MISMATCH = "RANGE_IDENTITY_MISMATCH"
    GENERATION_MISMATCH = "GENERATION_MISMATCH"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"
    AUTHORITY_MISMATCH = "AUTHORITY_MISMATCH"
    CANONICALIZATION_MISMATCH = "CANONICALIZATION_MISMATCH"
    OTHER_GROUP_VALIDATION_FAILURE = "OTHER_GROUP_VALIDATION_FAILURE"
    PASS = "PASS"


class ValidationClass(StrEnum):
    EVALUATOR_GAP = "EVALUATOR_GAP"
    PRODUCTION_BUG = "PRODUCTION_BUG"
    EXPECTED_STRICTNESS = "EXPECTED_STRICTNESS"
    REPRESENTATION_MISMATCH = "REPRESENTATION_MISMATCH"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ProductionCandidate:
    path: str
    representation: str
    source_sha256: str
    generation: int
    start_line: int
    end_line: int
    observation_id: str
    authority_provenance: str


@dataclass(frozen=True, slots=True)
class ComposedChild:
    path: str
    operation_type: str
    representation: str
    source_sha256: str
    generation: int
    start_line: int
    end_line: int
    candidate_observation_id: str


@dataclass(frozen=True, slots=True)
class V1Result:
    failure: GroupValidationFailure
    classification: ValidationClass
    mechanically_materializable: bool
    production_validatable: bool
    transaction_ready: bool
    canonical_paths: tuple[str, ...]


def _reject(
    failure: GroupValidationFailure,
    classification: ValidationClass,
    paths: tuple[str, ...],
) -> V1Result:
    return V1Result(failure, classification, True, False, False, paths)


def validate_v1(
    required_paths: tuple[str, ...],
    children: tuple[ComposedChild, ...],
    candidates: tuple[ProductionCandidate, ...],
    *,
    representation: str,
    generation: int,
    mutation_ready: bool,
    canonicalized: bool = True,
) -> V1Result:
    """Mirror intended production invariants without invoking or relaxing production."""
    paths = tuple(child.path for child in children)
    canonical = tuple(sorted(paths))
    if not mutation_ready:
        return _reject(
            GroupValidationFailure.PROVENANCE_MISMATCH,
            ValidationClass.PROVENANCE_MISMATCH,
            canonical,
        )
    if len(children) != len(required_paths) or not 2 <= len(children) <= 4:
        return _reject(
            GroupValidationFailure.OPERATION_COUNT_MISMATCH,
            ValidationClass.EVALUATOR_GAP,
            canonical,
        )
    if len(set(paths)) != len(paths):
        return _reject(
            GroupValidationFailure.DUPLICATE_PATH,
            ValidationClass.EXPECTED_STRICTNESS,
            canonical,
        )
    if set(paths) != set(required_paths):
        return _reject(
            GroupValidationFailure.PATH_SET_MISMATCH,
            ValidationClass.EXPECTED_STRICTNESS,
            canonical,
        )
    if any(child.operation_type != "edit" for child in children):
        return _reject(
            GroupValidationFailure.OPERATION_TYPE_MISMATCH,
            ValidationClass.EXPECTED_STRICTNESS,
            canonical,
        )
    if any(child.representation != representation for child in children):
        return _reject(
            GroupValidationFailure.REPRESENTATION_MISMATCH,
            ValidationClass.REPRESENTATION_MISMATCH,
            canonical,
        )
    if any(child.generation != generation for child in children):
        return _reject(
            GroupValidationFailure.GENERATION_MISMATCH,
            ValidationClass.IDENTITY_MISMATCH,
            canonical,
        )
    authority = {candidate.path: candidate for candidate in candidates}
    if set(authority) != set(required_paths):
        return _reject(
            GroupValidationFailure.AUTHORITY_MISMATCH,
            ValidationClass.PROVENANCE_MISMATCH,
            canonical,
        )
    for child in children:
        candidate = authority[child.path]
        if (
            candidate.generation != generation
            or child.candidate_observation_id != candidate.observation_id
        ):
            return _reject(
                GroupValidationFailure.PROVENANCE_MISMATCH,
                ValidationClass.PROVENANCE_MISMATCH,
                canonical,
            )
        if not candidate.observation_id or not candidate.authority_provenance:
            return _reject(
                GroupValidationFailure.PROVENANCE_MISMATCH,
                ValidationClass.PROVENANCE_MISMATCH,
                canonical,
            )
        if child.source_sha256 != candidate.source_sha256:
            return _reject(
                GroupValidationFailure.SOURCE_IDENTITY_MISMATCH,
                ValidationClass.IDENTITY_MISMATCH,
                canonical,
            )
        if (
            child.start_line < candidate.start_line
            or child.end_line > candidate.end_line
            or child.end_line < child.start_line
        ):
            return _reject(
                GroupValidationFailure.RANGE_IDENTITY_MISMATCH,
                ValidationClass.IDENTITY_MISMATCH,
                canonical,
            )
    if not canonicalized:
        return _reject(
            GroupValidationFailure.CANONICALIZATION_MISMATCH,
            ValidationClass.EVALUATOR_GAP,
            canonical,
        )
    return V1Result(
        GroupValidationFailure.PASS,
        ValidationClass.EXPECTED_STRICTNESS,
        True,
        True,
        True,
        canonical,
    )


def source_free_result(value: V1Result) -> dict[str, object]:
    return {
        "failure": value.failure.value,
        "classification": value.classification.value,
        "mechanically_materializable": value.mechanically_materializable,
        "production_validatable": value.production_validatable,
        "transaction_ready": value.transaction_ready,
        "canonical_paths": value.canonical_paths,
    }
