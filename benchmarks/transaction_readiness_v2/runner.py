"""A72 production-path runner with strict observation and checkpoint invariants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.realistic_coding_v2.runner import (
    MUTATION_TYPES,
    RecordingModel,
    snapshot,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY, FrozenTask, OperationClass
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from benchmarks.transaction_readiness_v2.protocol import (
    FailureLayer,
    evaluate_funnel,
    transaction_failure_subtype,
)
from benchmarks.transaction_readiness_v2.suite import (
    SCHEMA_VERSION,
    SEED,
    SUITE,
    VERSION,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.evaluation.realworld import RealWorldEvaluationRunner
from forge.models import Model, MutationRepresentationPolicy


class InstrumentationError(RuntimeError):
    """A fatal matrix-integrity failure; the run must stop."""


EVIDENCE_IDENTITY_MISMATCH = "PROPOSAL_EVIDENCE_IDENTITY_MISMATCH"


@dataclass(frozen=True, slots=True)
class ProposalResult:
    proposal_index: int
    envelope: dict[str, object]
    mutation_ready_metadata: dict[str, object] | None
    transitions: tuple[tuple[str, bool], ...]
    first_failed_stage: str | None
    failure_layer: str
    metadata_classification: str | None
    observation_id: str | None
    observation_metadata_classification: str | None
    transaction_failure_subtype: str | None
    downstream_evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class CellResult:
    suite: str
    suite_version: int
    schema_version: int
    evaluator_identity: str
    task_id: str
    task_version: int
    operation_class: str
    model_profile: str
    model_artifact: str
    model_config_identity: str
    seed: int
    repository_identity: str
    frozen_matrix_identity: str
    proposals: tuple[ProposalResult, ...]
    model_output_received: bool
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
        ("create" if child.get("type") == "create_file" else "edit", child.get("path"))
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


def _assert_observations(
    metadata: tuple[dict[str, object], ...], classifications: tuple[str, ...]
) -> None:
    if len(metadata) != len(classifications):
        raise InstrumentationError("proposal metadata/classification count mismatch")
    ids = [item.get("proposal_observation_id") for item in metadata]
    if any(not isinstance(item, str) or not item for item in ids):
        raise InstrumentationError("proposal observation ID missing")
    if len(ids) != len(set(ids)):
        raise InstrumentationError("duplicate proposal observation ID")
    if "OBSERVATION_METADATA_INCOMPLETE" in classifications:
        raise InstrumentationError("OBSERVATION_METADATA_INCOMPLETE")


def bind_proposal_evidence(
    metadata: tuple[dict[str, object], ...],
    evidence: tuple[dict[str, object], ...],
) -> dict[str, dict[str, object]]:
    """Bind downstream evidence by explicit proposal ID and fail closed."""
    proposal_ids = {str(item["proposal_observation_id"]) for item in metadata}
    bound: dict[str, dict[str, object]] = {
        proposal_id: {
            "transaction_attempts": [],
            "transaction_applied": False,
            "verification_pass": False,
            "semantic_pass": False,
        }
        for proposal_id in proposal_ids
    }
    attempt_ids: set[str] = set()
    for item in evidence:
        proposal_id = item.get("proposal_observation_id")
        if not isinstance(proposal_id, str) or proposal_id not in bound:
            raise InstrumentationError(EVIDENCE_IDENTITY_MISMATCH)
        event = item.get("event")
        target = bound[proposal_id]
        if event == "transaction_result":
            attempt_id = item.get("transaction_attempt_id")
            if not isinstance(attempt_id, str) or attempt_id in attempt_ids:
                raise InstrumentationError(EVIDENCE_IDENTITY_MISMATCH)
            attempt_ids.add(attempt_id)
            attempts = target["transaction_attempts"]
            assert isinstance(attempts, list)
            attempts.append(
                {
                    "transaction_attempt_id": attempt_id,
                    "transaction_outcome": item.get("transaction_outcome"),
                    "group_identity": item.get("group_identity"),
                    "workspace_generation": item.get("workspace_generation"),
                }
            )
            if item.get("transaction_outcome") == "success":
                target["transaction_applied"] = True
        elif event == "verification_result":
            if not target["transaction_applied"]:
                raise InstrumentationError(EVIDENCE_IDENTITY_MISMATCH)
            target["verification_pass"] = item.get("verification_outcome") == "pass"
        elif event == "semantic_result":
            if not target["transaction_applied"]:
                raise InstrumentationError(EVIDENCE_IDENTITY_MISMATCH)
            target["semantic_pass"] = item.get("semantic_outcome") == "PASS"
        else:
            raise InstrumentationError(EVIDENCE_IDENTITY_MISMATCH)
    return bound


def run_cell(
    definition: FrozenTask,
    backend: Model,
    *,
    profile: str,
    artifact: str,
    model_config_identity: str,
    repository_identity: str,
    matrix_identity: str,
    representation: MutationRepresentationPolicy,
) -> CellResult:
    recorder = RecordingModel(
        backend, frozenset((*definition.edit_paths, *definition.create_paths))
    )
    raw = (
        RealWorldEvaluationRunner(
            profile, recorder, REPOSITORY, mutation_representation=representation
        )
        .run(
            (replace(definition.production_task, seeds=(SEED,)),),
            snapshot(repository_identity),
        )
        .results[0]
    )
    envelopes = tuple(
        item for item in recorder.envelopes if item.get("type") in MUTATION_TYPES
    )
    metadata = raw.mutation_ready_metadata
    classifications = raw.mutation_observation_classifications
    _assert_observations(metadata, classifications)
    if len(metadata) != len(envelopes):
        raise InstrumentationError(
            f"proposal/observation count mismatch: {len(envelopes)} != {len(metadata)}"
        )
    binding = bind_proposal_evidence(metadata, raw.proposal_evidence)
    proposals = []
    for index, (envelope, recorded, observation) in enumerate(
        zip(envelopes, metadata, classifications, strict=True)
    ):
        schema_valid = _schema_valid(definition, envelope)
        proposal_id = str(recorded["proposal_observation_id"])
        downstream = binding[proposal_id]
        applied = bool(downstream["transaction_applied"])
        verification = bool(downstream["verification_pass"])
        semantic = bool(downstream["semantic_pass"])
        funnel = evaluate_funnel(
            model_output=True,
            schema_valid=schema_valid,
            metadata=recorded,
            observation_classification=observation,
            transaction_applied=applied,
            verification_pass=verification,
            semantic_pass=verification and semantic,
        )
        subtype = None
        if funnel.failure_layer == FailureLayer.TRANSACTION_FAILURE.value:
            subtype = transaction_failure_subtype(
                apply_result=raw.metrics.mutation_group_apply_result,
                failure_message=raw.failure_message,
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
                str(recorded["proposal_observation_id"]),
                observation,
                subtype,
                downstream,
            )
        )
    return CellResult(
        SUITE,
        VERSION,
        SCHEMA_VERSION,
        SUITE,
        definition.task_id,
        definition.version,
        definition.operation_class.value,
        profile,
        artifact,
        model_config_identity,
        SEED,
        repository_identity,
        matrix_identity,
        tuple(proposals),
        recorder.calls > 0,
        len(metadata) == len(envelopes),
        any(bool(item["transaction_applied"]) for item in binding.values()),
        any(bool(item["verification_pass"]) for item in binding.values()),
        any(bool(item["semantic_pass"]) for item in binding.values()),
        raw.final_status,
    )


def checkpoint_path(root: Path, profile: str, task_id: str) -> Path:
    return root / "cells" / f"{profile}-seed{SEED}-{task_id}.json"


def commit_cell(path: Path, cell: CellResult) -> None:
    payload = asdict(cell)
    if not standard_result_is_source_free(payload):
        raise InstrumentationError("durable cell is not source-free")
    atomic_checkpoint(path, payload)


def read_cell(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    payload = resume_checkpoint(path)
    if payload is not None and any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"A72 checkpoint identity mismatch: {path.name}")
    return payload
