"""Source-free proposal-boundary metadata and strict V1 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class MutationEvaluationState(StrEnum):
    MECHANICALLY_MATERIALIZABLE = "MECHANICALLY_MATERIALIZABLE"
    PRODUCTION_VALIDATABLE = "PRODUCTION_VALIDATABLE"
    TRANSACTION_READY = "TRANSACTION_READY"


class MutationReadyClassification(StrEnum):
    PASS = "PASS"
    MUTATION_READY_METADATA_INCOMPLETE = "MUTATION_READY_METADATA_INCOMPLETE"
    LEGACY_MUTATION_READY_METADATA_INCOMPLETE = (
        "LEGACY_MUTATION_READY_METADATA_INCOMPLETE"
    )
    OPERATION_COUNT_MISMATCH = "OPERATION_COUNT_MISMATCH"
    PATH_SET_MISMATCH = "PATH_SET_MISMATCH"
    DUPLICATE_PATH = "DUPLICATE_PATH"
    OPERATION_TYPE_MISMATCH = "OPERATION_TYPE_MISMATCH"
    REPRESENTATION_MISMATCH = "REPRESENTATION_MISMATCH"
    GENERATION_MISMATCH = "GENERATION_MISMATCH"
    AUTHORITY_MISMATCH = "AUTHORITY_MISMATCH"
    PROVENANCE_MISMATCH = "PROVENANCE_MISMATCH"
    SOURCE_IDENTITY_MISMATCH = "SOURCE_IDENTITY_MISMATCH"
    RANGE_IDENTITY_MISMATCH = "RANGE_IDENTITY_MISMATCH"
    CANONICALIZATION_MISMATCH = "CANONICALIZATION_MISMATCH"
    GROUP_BINDING_MISMATCH = "GROUP_BINDING_MISMATCH"
    PREVIEW_INELIGIBLE = "PREVIEW_INELIGIBLE"
    TRANSACTION_NOT_READY = "TRANSACTION_NOT_READY"
    MATERIALIZATION_FAILURE = "MATERIALIZATION_FAILURE"


class ObservationMetadataClassification(StrEnum):
    OBSERVATION_METADATA_COMPLETE = "OBSERVATION_METADATA_COMPLETE"
    OBSERVATION_METADATA_INCOMPLETE = "OBSERVATION_METADATA_INCOMPLETE"
    PRODUCTION_METADATA_UNAVAILABLE = "PRODUCTION_METADATA_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class CandidateMetadata:
    path: str
    observation_id: str
    trusted_source_sha256: str
    authorized_start_line: int | None
    authorized_end_line: int | None
    generation: int
    authority_provenance_class: str
    authority_kind: str = "existing_source"
    creation_parent_identity: str | None = None


@dataclass(frozen=True, slots=True)
class ChildMetadata:
    path: str
    operation_type: str
    representation: str
    source_sha256: str
    start_line: int | None
    end_line: int | None
    generation: int
    candidate_observation_id: str


@dataclass(frozen=True, slots=True)
class MutationReadyMetadata:
    schema_version: int
    workspace_generation: int
    authorized_paths: tuple[str, ...]
    candidates: tuple[CandidateMetadata, ...]
    mutation_representation: str
    children: tuple[ChildMetadata, ...]
    normalized_operation_count: int
    canonical_child_order: tuple[str, ...]
    group_identity: str
    group_generation: int
    preview_eligible: bool
    transaction_readiness_state: str


@dataclass(frozen=True, slots=True)
class MutationReadyEvaluation:
    classification: MutationReadyClassification
    highest_state: MutationEvaluationState | None
    mechanically_materializable: bool
    production_validatable: bool
    transaction_ready: bool


def _reject(
    classification: MutationReadyClassification,
    *,
    mechanical: bool = True,
    production: bool = False,
) -> MutationReadyEvaluation:
    highest = (
        MutationEvaluationState.PRODUCTION_VALIDATABLE
        if production
        else MutationEvaluationState.MECHANICALLY_MATERIALIZABLE
        if mechanical
        else None
    )
    return MutationReadyEvaluation(
        classification, highest, mechanical, production, False
    )


def evaluate_mutation_ready_v1(
    metadata: MutationReadyMetadata | Mapping[str, object],
) -> MutationReadyEvaluation:
    """Classify only recorded metadata; never infer absent provenance."""
    if not isinstance(metadata, MutationReadyMetadata):
        loaded = load_mutation_ready_metadata(metadata)
        if isinstance(loaded, MutationReadyEvaluation):
            return loaded
        metadata = loaded
    if metadata.schema_version != SCHEMA_VERSION:
        return _reject(
            MutationReadyClassification.MUTATION_READY_METADATA_INCOMPLETE,
            mechanical=False,
        )
    if (
        isinstance(metadata.workspace_generation, bool)
        or not isinstance(metadata.workspace_generation, int)
        or any(
            candidate.authority_kind == "existing_source"
            and (
                isinstance(candidate.authorized_start_line, bool)
                or not isinstance(candidate.authorized_start_line, int)
                or isinstance(candidate.authorized_end_line, bool)
                or not isinstance(candidate.authorized_end_line, int)
            )
            or candidate.authority_kind == "create_parent"
            and not candidate.creation_parent_identity
            or isinstance(candidate.generation, bool)
            or not isinstance(candidate.generation, int)
            for candidate in metadata.candidates
        )
        or any(
            child.operation_type == "edit"
            and (
                isinstance(child.start_line, bool)
                or not isinstance(child.start_line, int)
                or isinstance(child.end_line, bool)
                or not isinstance(child.end_line, int)
            )
            or isinstance(child.generation, bool)
            or not isinstance(child.generation, int)
            for child in metadata.children
        )
    ):
        return _reject(
            MutationReadyClassification.MUTATION_READY_METADATA_INCOMPLETE,
            mechanical=False,
        )
    paths = tuple(child.path for child in metadata.children)
    canonical = tuple(sorted(paths))
    if (
        metadata.normalized_operation_count != len(metadata.children)
        or not 1 <= len(metadata.children) <= 4
    ):
        return _reject(
            MutationReadyClassification.OPERATION_COUNT_MISMATCH, mechanical=False
        )
    if len(set(paths)) != len(paths):
        return _reject(MutationReadyClassification.DUPLICATE_PATH, mechanical=False)
    if set(paths) != set(metadata.authorized_paths):
        return _reject(MutationReadyClassification.PATH_SET_MISMATCH, mechanical=False)
    if metadata.canonical_child_order != canonical:
        return _reject(
            MutationReadyClassification.CANONICALIZATION_MISMATCH, mechanical=False
        )
    if any(
        child.operation_type not in {"edit", "create"} for child in metadata.children
    ):
        return _reject(
            MutationReadyClassification.OPERATION_TYPE_MISMATCH, mechanical=False
        )
    representations_valid = (
        metadata.mutation_representation == "mixed"
        and {child.operation_type for child in metadata.children} == {"edit", "create"}
        and all(
            child.representation in {"exact_text", "line_range", "create_text"}
            for child in metadata.children
        )
    ) or all(
        child.representation == metadata.mutation_representation
        for child in metadata.children
    )
    if not representations_valid:
        return _reject(
            MutationReadyClassification.REPRESENTATION_MISMATCH, mechanical=False
        )
    if any(
        child.generation != metadata.workspace_generation for child in metadata.children
    ):
        return _reject(MutationReadyClassification.GENERATION_MISMATCH)
    authority = {candidate.path: candidate for candidate in metadata.candidates}
    if len(authority) != len(metadata.candidates) or set(authority) != set(
        metadata.authorized_paths
    ):
        return _reject(MutationReadyClassification.AUTHORITY_MISMATCH)
    for child in metadata.children:
        candidate = authority[child.path]
        if (
            not candidate.observation_id
            or not candidate.authority_provenance_class
            or child.candidate_observation_id != candidate.observation_id
        ):
            return _reject(MutationReadyClassification.PROVENANCE_MISMATCH)
        if candidate.generation != metadata.workspace_generation:
            return _reject(MutationReadyClassification.GENERATION_MISMATCH)
        if child.source_sha256 != candidate.trusted_source_sha256:
            return _reject(MutationReadyClassification.SOURCE_IDENTITY_MISMATCH)
        if child.operation_type == "create":
            if (
                candidate.authority_kind != "create_parent"
                or child.start_line is not None
                or child.end_line is not None
            ):
                return _reject(MutationReadyClassification.AUTHORITY_MISMATCH)
            continue
        assert candidate.authorized_start_line is not None
        assert candidate.authorized_end_line is not None
        assert child.start_line is not None and child.end_line is not None
        authorized_end = candidate.authorized_end_line + (
            1 if child.representation == "exact_text" else 0
        )
        if (
            child.start_line < candidate.authorized_start_line
            or child.end_line > authorized_end
            or child.end_line < child.start_line
        ):
            return _reject(MutationReadyClassification.RANGE_IDENTITY_MISMATCH)
    if (
        not metadata.group_identity
        or metadata.group_generation != metadata.workspace_generation
    ):
        return _reject(MutationReadyClassification.GROUP_BINDING_MISMATCH)
    if metadata.transaction_readiness_state == "materialization_failed":
        return _reject(MutationReadyClassification.MATERIALIZATION_FAILURE)
    if not metadata.preview_eligible:
        return _reject(MutationReadyClassification.PREVIEW_INELIGIBLE, production=True)
    if metadata.transaction_readiness_state != "ready":
        return _reject(
            MutationReadyClassification.TRANSACTION_NOT_READY, production=True
        )
    return MutationReadyEvaluation(
        MutationReadyClassification.PASS,
        MutationEvaluationState.TRANSACTION_READY,
        True,
        True,
        True,
    )


def classify_observation_metadata(
    metadata: MutationReadyMetadata | Mapping[str, object],
) -> ObservationMetadataClassification:
    """Separate observer coverage from production proposal validity."""
    if isinstance(metadata, MutationReadyMetadata):
        return ObservationMetadataClassification.OBSERVATION_METADATA_INCOMPLETE
    observation_id = metadata.get("proposal_observation_id")
    observation_status = metadata.get("observation_metadata_status")
    production_status = metadata.get("production_metadata_status")
    if production_status == "unavailable":
        return ObservationMetadataClassification.PRODUCTION_METADATA_UNAVAILABLE
    if (
        not isinstance(observation_id, str)
        or not observation_id
        or observation_status != "complete"
    ):
        return ObservationMetadataClassification.OBSERVATION_METADATA_INCOMPLETE
    return ObservationMetadataClassification.OBSERVATION_METADATA_COMPLETE


def source_free_metadata(metadata: MutationReadyMetadata) -> dict[str, object]:
    """Return the durable normalized form; all fields are IDs, hashes, ranges/enums."""
    return asdict(metadata)


def source_free_evaluation(value: MutationReadyEvaluation) -> dict[str, object]:
    return {
        "classification": value.classification.value,
        "highest_state": value.highest_state.value if value.highest_state else None,
        "mechanically_materializable": value.mechanically_materializable,
        "production_validatable": value.production_validatable,
        "transaction_ready": value.transaction_ready,
    }


def load_mutation_ready_metadata(
    payload: Mapping[str, object],
) -> MutationReadyMetadata | MutationReadyEvaluation:
    """Read V1 or fail closed for historical/partial results without upgrading."""
    if "mutation_ready_metadata" in payload:
        nested = payload["mutation_ready_metadata"]
        if not isinstance(nested, Mapping):
            return _reject(
                MutationReadyClassification.MUTATION_READY_METADATA_INCOMPLETE,
                mechanical=False,
            )
        payload = nested
    if "schema_version" not in payload:
        return _reject(
            MutationReadyClassification.LEGACY_MUTATION_READY_METADATA_INCOMPLETE,
            mechanical=False,
        )
    required = {
        "schema_version",
        "workspace_generation",
        "authorized_paths",
        "candidates",
        "mutation_representation",
        "children",
        "normalized_operation_count",
        "canonical_child_order",
        "group_identity",
        "group_generation",
        "preview_eligible",
        "transaction_readiness_state",
    }
    if not required.issubset(payload):
        return _reject(
            MutationReadyClassification.MUTATION_READY_METADATA_INCOMPLETE,
            mechanical=False,
        )
    try:
        return MutationReadyMetadata(
            schema_version=int(payload["schema_version"]),
            workspace_generation=int(payload["workspace_generation"]),
            authorized_paths=tuple(payload["authorized_paths"]),  # type: ignore[arg-type]
            candidates=tuple(
                CandidateMetadata(**item)
                for item in payload["candidates"]  # type: ignore[arg-type]
            ),
            mutation_representation=str(payload["mutation_representation"]),
            children=tuple(
                ChildMetadata(**item)
                for item in payload["children"]  # type: ignore[arg-type]
            ),
            normalized_operation_count=int(payload["normalized_operation_count"]),
            canonical_child_order=tuple(payload["canonical_child_order"]),  # type: ignore[arg-type]
            group_identity=str(payload["group_identity"]),
            group_generation=int(payload["group_generation"]),
            preview_eligible=bool(payload["preview_eligible"]),
            transaction_readiness_state=str(payload["transaction_readiness_state"]),
        )
    except (KeyError, TypeError, ValueError):
        return _reject(
            MutationReadyClassification.MUTATION_READY_METADATA_INCOMPLETE,
            mechanical=False,
        )


def proposal_group_identity(payload: Mapping[str, object]) -> str:
    """Hash only normalized source-free proposal identity fields."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    """Commit a proposal boundary exactly once, including its recorded metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resume_checkpoint(path: Path) -> dict[str, Any] | None:
    """Return the committed boundary verbatim; callers must not regenerate it."""
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mutation-ready checkpoint must contain an object")
    return value
