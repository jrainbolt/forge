"""Paired A74 execution with primary and repair outcomes separated."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.grounded_mutation_planning_v1.paired_identity import (
    PairedInputCaptureModel,
)
from benchmarks.grounded_mutation_planning_v1.planning import PlanningModel
from benchmarks.grounded_mutation_planning_v1.suite import (
    CONTEXT_SIZE,
    CORRECTION_RUN_ID,
    SCHEMA_VERSION,
    SEED,
    SUITE,
    VERSION,
    Condition,
    PlanningCase,
)
from benchmarks.realistic_coding_v2.runner import (
    MUTATION_TYPES,
    RecordingModel,
    snapshot,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY, FrozenTask
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from benchmarks.transaction_readiness_v2.runner import (
    _assert_observations,
    _schema_valid,
)
from benchmarks.verification_attribution_v1.protocol import semantic_failure
from forge.evaluation.mutation_ready import (
    atomic_checkpoint,
    evaluate_mutation_ready_v1,
    resume_checkpoint,
)
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    run_oracle,
)
from forge.models import Model, MutationRepresentationPolicy


@dataclass(frozen=True, slots=True)
class Outcome:
    proposal_observation_id: str | None
    schema_valid: bool
    mechanically_materializable: bool
    production_validatable: bool
    transaction_ready: bool
    applied: bool
    verification_pass: bool
    semantic_pass: bool
    semantic_failure: str | None


@dataclass(frozen=True, slots=True)
class CellResult:
    suite: str
    run_identity: str
    suite_version: int
    schema_version: int
    case_id: str
    condition: str
    profile: str
    task_id: str
    operation_class: str
    rationale: str
    seed: int
    corpus_identity: str
    repository_identity: str
    model_artifact: str
    model_config_identity: str
    plan: dict[str, object] | None
    paired_input_identity: dict[str, object]
    grounding_identity: str | None
    authority_identity: str | None
    primary: Outcome
    repair: Outcome | None
    failure: str
    plan_usefulness: str | None
    observation_count: int
    final_status: str


def _identity(metadata: dict[str, object], fields: tuple[str, ...]) -> str:
    payload = {field: metadata.get(field) for field in fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _grounding_identity(metadata: dict[str, object]) -> str:
    candidates = metadata.get("candidates", ())
    payload = {
        "workspace_generation": metadata.get("workspace_generation"),
        "candidates": [
            {
                key: candidate.get(key)
                for key in (
                    "path",
                    "observation_id",
                    "trusted_source_sha256",
                    "generation",
                )
            }
            for candidate in candidates
            if isinstance(candidate, dict)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _authority_identity(metadata: dict[str, object]) -> str:
    candidates = metadata.get("candidates", ())
    payload = {
        "authorized_paths": metadata.get("authorized_paths"),
        "candidates": [
            {
                key: candidate.get(key)
                for key in (
                    "path",
                    "authorized_start_line",
                    "authorized_end_line",
                    "authority_provenance_class",
                )
            }
            for candidate in candidates
            if isinstance(candidate, dict)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_cell(
    case: PlanningCase,
    condition: Condition,
    definition: FrozenTask,
    backend: Model,
    *,
    artifact: str,
    model_config_identity: str,
    repository_identity: str,
    corpus_identity: str,
    representation: MutationRepresentationPolicy,
) -> CellResult:
    trusted = frozenset(definition.production_task.allowed_paths)
    planning = PlanningModel(backend, trusted) if condition is Condition.P1 else None
    task_identity = hashlib.sha256(
        json.dumps(
            {
                "case_id": case.case_id,
                "task_id": definition.task_id,
                "task_version": definition.version,
                "prompt": definition.production_task.prompt,
                "allowed_paths": definition.production_task.allowed_paths,
                "operation_class": definition.operation_class.value,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    paired_capture = PairedInputCaptureModel(
        planning or backend,
        task_identity=task_identity,
        model_config_identity=model_config_identity,
        context_size=CONTEXT_SIZE,
        representation=representation.value,
    )
    recorder = RecordingModel(paired_capture, trusted)
    observed: list[dict[str, object]] = []
    primary_semantic: dict[str, str] = {}

    def evidence_callback(_task, _workspace, payload):  # type: ignore[no-untyped-def]
        observed.append(dict(payload))

    def primary_callback(task, workspace: Path) -> None:  # type: ignore[no-untyped-def]
        applied = [
            item
            for item in observed
            if item.get("event") == "transaction_result"
            and item.get("transaction_outcome") == "success"
        ]
        if applied:
            primary_semantic[str(applied[-1]["proposal_observation_id"])] = run_oracle(
                workspace, task.oracle_commands
            ).value

    raw = (
        RealWorldEvaluationRunner(
            case.profile,
            recorder,
            REPOSITORY,
            mutation_representation=representation,
            primary_mutation_callback=primary_callback,
            proposal_evidence_callback=evidence_callback,
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
    _assert_observations(metadata, raw.mutation_observation_classifications)
    if len(envelopes) != len(metadata):
        raise RuntimeError("A74 proposal/observation mismatch")
    applied_events = [
        item
        for item in raw.proposal_evidence
        if item.get("event") == "transaction_result"
        and item.get("transaction_outcome") == "success"
    ]
    applied_ids = [str(item["proposal_observation_id"]) for item in applied_events]
    verification = {
        str(item["proposal_observation_id"]): item.get("verification_outcome") == "pass"
        for item in raw.proposal_evidence
        if item.get("event") == "verification_result"
    }
    final_semantic = {
        str(item["proposal_observation_id"]): item.get("semantic_outcome")
        == EvaluationOutcome.PASS.value
        for item in raw.proposal_evidence
        if item.get("event") == "semantic_result"
    }
    final_semantic.update(
        {
            proposal_id: outcome == EvaluationOutcome.PASS.value
            for proposal_id, outcome in primary_semantic.items()
        }
    )

    def outcome(index: int | None, proposal_id: str | None) -> Outcome:
        if index is None or proposal_id is None:
            return Outcome(None, False, False, False, False, False, False, False, None)
        evaluated = evaluate_mutation_ready_v1(metadata[index])
        schema = _schema_valid(definition, envelopes[index])
        semantic = bool(final_semantic.get(proposal_id, False))
        return Outcome(
            proposal_id,
            schema,
            schema and evaluated.mechanically_materializable,
            schema and evaluated.production_validatable,
            schema and evaluated.transaction_ready,
            proposal_id in applied_ids,
            bool(verification.get(proposal_id, False)),
            semantic,
            None
            if semantic
            else semantic_failure(definition.operation_class.value, None).value,
        )

    proposal_ids = [str(item["proposal_observation_id"]) for item in metadata]
    primary_id = proposal_ids[0] if proposal_ids else None
    repair_id = proposal_ids[1] if len(proposal_ids) > 1 else None
    primary = outcome(0 if primary_id else None, primary_id)
    repair = outcome(1, repair_id) if repair_id else None
    plan_payload = asdict(planning.record) if planning and planning.record else None
    if paired_capture.record is None:
        raise RuntimeError("A74 request-time paired identity was not captured")
    paired_input = asdict(paired_capture.record)
    plan_valid = plan_payload is not None and plan_payload["classification"] == "VALID"
    if condition is Condition.P1 and not plan_valid:
        failure = str(
            plan_payload["classification"] if plan_payload else "PLAN_INVALID"
        )
    elif not primary.schema_valid:
        failure = "MODEL_PROTOCOL_FAILURE"
    elif not primary.mechanically_materializable:
        failure = "MATERIALIZATION_FAILURE"
    elif not primary.applied:
        failure = (
            "PARTIAL_CHANGE_SET"
            if raw.metrics.mutation_proposed
            else "MODEL_PROTOCOL_FAILURE"
        )
    elif (
        not primary.verification_pass
        and raw.metrics.verification_plan_failed_step == "project.build"
    ):
        failure = "BUILD_FAILURE"
    elif not primary.verification_pass:
        failure = "VERIFICATION_FAILURE"
    elif not primary.semantic_pass:
        failure = "SEMANTIC_FAILURE"
    else:
        failure = "PASS"
    usefulness = None
    if condition is Condition.P1 and failure != "PASS":
        usefulness = "PLAN_INCOMPLETE" if not plan_valid else "UNKNOWN"
    first_metadata = metadata[0] if metadata else None
    grounding_identity = _grounding_identity(first_metadata) if first_metadata else None
    authority_identity = _authority_identity(first_metadata) if first_metadata else None
    return CellResult(
        SUITE,
        CORRECTION_RUN_ID,
        VERSION,
        SCHEMA_VERSION,
        case.case_id,
        condition.value,
        case.profile,
        case.task_id,
        definition.operation_class.value,
        case.rationale,
        SEED,
        corpus_identity,
        repository_identity,
        artifact,
        model_config_identity,
        plan_payload,
        paired_input,
        grounding_identity,
        authority_identity,
        primary,
        repair,
        failure,
        usefulness,
        len(metadata),
        raw.final_status,
    )


def checkpoint_path(root: Path, case_id: str, condition: Condition) -> Path:
    return root / "cells" / f"{case_id}-{condition.name.lower()}.json"


def commit_cell(path: Path, result: CellResult) -> None:
    payload = asdict(result)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A74 result is not source-free")
    atomic_checkpoint(path, payload)


def read_cell(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    payload = resume_checkpoint(path)
    if payload is not None and any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"A74 checkpoint identity mismatch: {path.name}")
    return payload
