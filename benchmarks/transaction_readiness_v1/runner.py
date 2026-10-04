"""A71 production-path runner with source-free exactly-once checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.realistic_coding_v2.runner import (
    MUTATION_TYPES,
    RecordingModel,
    snapshot,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY, FrozenTask, OperationClass
from benchmarks.transaction_readiness_v1.protocol import evaluate_funnel
from benchmarks.transaction_readiness_v1.suite import (
    SCHEMA_VERSION,
    SEED,
    SUITE,
    VERSION,
)
from forge.evaluation.mutation_ready import (
    atomic_checkpoint,
    evaluate_mutation_ready_v1,
    resume_checkpoint,
)
from forge.evaluation.realworld import EvaluationOutcome, RealWorldEvaluationRunner
from forge.models import Model, MutationRepresentationPolicy


@dataclass(frozen=True, slots=True)
class ProposalResult:
    proposal_index: int
    envelope: dict[str, object]
    mutation_ready_metadata: dict[str, object] | None
    transitions: tuple[tuple[str, bool], ...]
    first_failed_stage: str | None
    failure_layer: str
    metadata_classification: str
    observation_id: str | None
    observation_metadata_classification: str


@dataclass(frozen=True, slots=True)
class CellResult:
    suite: str
    suite_version: int
    schema_version: int
    task_id: str
    task_version: int
    operation_class: str
    model_profile: str
    model_artifact: str
    seed: int
    repository_identity: str
    corpus_identity: str
    proposals: tuple[ProposalResult, ...]
    proposal_metadata_exactly_once: bool
    transaction_applied: bool
    verification_pass: bool
    semantic_pass: bool
    final_status: str


def _schema_valid(definition: FrozenTask, envelope: dict[str, object]) -> bool:
    children = envelope.get("children")
    if not isinstance(children, list):
        return False
    roles = {
        (
            "create" if child.get("type") == "create_file" else "edit",
            child.get("path"),
        )
        for child in children
        if isinstance(child, dict)
    }
    expected = {
        *(("edit", path) for path in definition.edit_paths),
        *(("create", path) for path in definition.create_paths),
    }
    expected_types = {
        OperationClass.EDIT_SINGLE: {"line_range_edit", "structured_edit"},
        OperationClass.EDIT_MULTI: {
            "multi_file_line_range_edit",
            "multi_file_structured_edit",
        },
        OperationClass.CREATE: {"create_file", "multi_file_create"},
        OperationClass.MIXED_EDIT_CREATE: {"multi_file_change"},
    }[definition.operation_class]
    return envelope.get("type") in expected_types and roles == expected


def run_cell(
    definition: FrozenTask,
    backend: Model,
    *,
    profile: str,
    artifact: str,
    repository_identity: str,
    corpus_identity: str,
    representation: MutationRepresentationPolicy,
) -> CellResult:
    recorder = RecordingModel(
        backend, frozenset((*definition.edit_paths, *definition.create_paths))
    )
    task = replace(definition.production_task, seeds=(SEED,))
    raw = (
        RealWorldEvaluationRunner(
            profile,
            recorder,
            REPOSITORY,
            mutation_representation=representation,
        )
        .run((task,), snapshot(repository_identity))
        .results[0]
    )
    envelopes = tuple(
        envelope
        for envelope in recorder.envelopes
        if envelope.get("type") in MUTATION_TYPES
    )
    metadata = raw.mutation_ready_metadata
    count = max(len(envelopes), len(metadata))
    verification = raw.metrics.verification_plan_result == "pass" or (
        raw.metrics.reverification_result == "pass"
    )
    semantic = raw.oracle is EvaluationOutcome.PASS
    proposals = []
    for index in range(count):
        envelope = envelopes[index] if index < len(envelopes) else {"type": "missing"}
        recorded = metadata[index] if index < len(metadata) else None
        observation_classification = (
            raw.mutation_observation_classifications[index]
            if index < len(raw.mutation_observation_classifications)
            else "OBSERVATION_METADATA_INCOMPLETE"
        )
        applied = bool(raw.metrics.mutations) and index == count - 1
        readiness = evaluate_mutation_ready_v1(recorded or {}).transaction_ready
        applied_in_funnel = (
            applied and readiness and _schema_valid(definition, envelope)
        )
        funnel = evaluate_funnel(
            model_output=index < len(envelopes),
            schema_valid=_schema_valid(definition, envelope),
            metadata=recorded,
            transaction_applied=applied_in_funnel,
            verification_pass=applied_in_funnel and verification,
            semantic_pass=applied_in_funnel and semantic,
        )
        proposals.append(
            ProposalResult(
                index + 1,
                envelope,
                recorded,
                funnel.transitions,
                funnel.first_failed_stage,
                funnel.failure_layer,
                funnel.metadata_classification,
                (
                    str(recorded.get("proposal_observation_id"))
                    if recorded is not None
                    and recorded.get("proposal_observation_id") is not None
                    else None
                ),
                observation_classification,
            )
        )
    return CellResult(
        SUITE,
        VERSION,
        SCHEMA_VERSION,
        definition.task_id,
        definition.version,
        definition.operation_class.value,
        profile,
        artifact,
        SEED,
        repository_identity,
        corpus_identity,
        tuple(proposals),
        bool(proposals) and len(metadata) == len(envelopes),
        bool(raw.metrics.mutations),
        verification,
        semantic,
        raw.final_status,
    )


def checkpoint_path(root: Path, profile: str, task_id: str) -> Path:
    return root / "cells" / f"{profile}-seed{SEED}-{task_id}.json"


def commit_cell(path: Path, cell: CellResult) -> None:
    atomic_checkpoint(path, asdict(cell))


def read_cell(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    payload = resume_checkpoint(path)
    if payload is not None and any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"A71 checkpoint identity mismatch: {path.name}")
    return payload


def standard_result_is_source_free(payload: object) -> bool:
    rendered = str(payload).casefold()
    forbidden = (
        "old_text",
        "new_text",
        "replacement",
        "source_content",
        "hidden_reference",
        "expected_implementation",
        "chain_of_thought",
        "/users/",
    )
    return not any(value in rendered for value in forbidden)
