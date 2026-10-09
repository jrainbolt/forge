"""A75 paired execution with proposal-local primary and repair outcomes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.grounded_mutation_contract_v1.contract import ContractModel
from benchmarks.grounded_mutation_contract_v1.suite import (
    CONTEXT_SIZE,
    RUN_ID,
    SCHEMA_VERSION,
    SEED,
    SUITE,
    VERSION,
    Condition,
)
from benchmarks.grounded_mutation_planning_v1.paired_identity import (
    PairedInputCaptureModel,
    bind_request_proposals,
)
from benchmarks.grounded_mutation_planning_v1.suite import PlanningCase
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
    model_output: bool
    schema_valid: bool
    mechanically_materializable: bool
    production_validatable: bool
    transaction_ready: bool
    transaction_applied: bool
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
    paired_input_identity: dict[str, object] | None
    mutation_requests: tuple[dict[str, object], ...]
    proposal_request_lineage: dict[str, str]
    proposal_transaction_lineage: dict[str, tuple[str, ...]]
    contract: dict[str, object] | None
    primary: Outcome
    repair: Outcome | None
    failure: str
    contract_diagnosis: str | None
    observation_count: int
    final_status: str
    workflow_trace: dict[str, object]


def _task_identity(case: PlanningCase, definition: FrozenTask) -> str:
    payload = {
        "case_id": case.case_id,
        "task_id": definition.task_id,
        "task_version": definition.version,
        "prompt": definition.production_task.prompt,
        "allowed_paths": definition.production_task.allowed_paths,
        "operation_class": definition.operation_class.value,
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
    primary_workspace_callback: Callable[[Path], None] | None = None,
    final_workspace_callback: Callable[[Path], None] | None = None,
) -> CellResult:
    trusted = frozenset(definition.production_task.allowed_paths)
    contract_model = (
        ContractModel(backend, definition.production_task.prompt, trusted)
        if condition is Condition.M1
        else None
    )
    paired_capture = PairedInputCaptureModel(
        contract_model or backend,
        task_identity=_task_identity(case, definition),
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
            if primary_workspace_callback is not None:
                primary_workspace_callback(workspace)

    def result_callback(_task, workspace: Path, _result) -> None:  # type: ignore[no-untyped-def]
        if final_workspace_callback is not None:
            final_workspace_callback(workspace)

    raw = (
        RealWorldEvaluationRunner(
            case.profile,
            recorder,
            REPOSITORY,
            mutation_representation=representation,
            primary_mutation_callback=primary_callback,
            proposal_evidence_callback=evidence_callback,
            result_callback=result_callback,
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
        raise RuntimeError("A75 proposal/observation mismatch")
    applied_ids = [
        str(item["proposal_observation_id"])
        for item in raw.proposal_evidence
        if item.get("event") == "transaction_result"
        and item.get("transaction_outcome") == "success"
    ]
    verification = {
        str(item["proposal_observation_id"]): item.get("verification_outcome") == "pass"
        for item in raw.proposal_evidence
        if item.get("event") == "verification_result"
    }
    semantics = {
        str(item["proposal_observation_id"]): item.get("semantic_outcome")
        == EvaluationOutcome.PASS.value
        for item in raw.proposal_evidence
        if item.get("event") == "semantic_result"
    }
    semantics.update(
        {
            proposal_id: result == EvaluationOutcome.PASS.value
            for proposal_id, result in primary_semantic.items()
        }
    )

    def outcome(index: int | None, proposal_id: str | None) -> Outcome:
        if index is None or proposal_id is None:
            return Outcome(
                None, False, False, False, False, False, False, False, False, None
            )
        evaluated = evaluate_mutation_ready_v1(metadata[index])
        schema = _schema_valid(definition, envelopes[index])
        semantic = bool(semantics.get(proposal_id, False))
        failure = None
        if not semantic:
            verification_class = (
                "BUILD_FAILURE"
                if raw.metrics.verification_plan_failed_step == "project.build"
                else None
            )
            failure = semantic_failure(
                definition.operation_class.value, verification_class
            ).value
        return Outcome(
            proposal_id,
            True,
            schema,
            schema and evaluated.mechanically_materializable,
            schema and evaluated.production_validatable,
            schema and evaluated.transaction_ready,
            proposal_id in applied_ids,
            bool(verification.get(proposal_id, False)),
            semantic,
            failure,
        )

    proposal_ids = [str(item["proposal_observation_id"]) for item in metadata]
    groups: list[tuple[str, ...]] = []
    for index in range(len(paired_capture.records)):
        if index == len(paired_capture.records) - 1:
            groups.append(tuple(proposal_ids[index:]))
        else:
            groups.append(tuple(proposal_ids[index : index + 1]))
    request_records = bind_request_proposals(
        tuple(paired_capture.records), tuple(groups)
    )
    lineage = {
        proposal_id: record.mutation_request_id
        for record in request_records
        for proposal_id in record.proposal_observation_ids
    }
    transaction_lineage: dict[str, list[str]] = {}
    for item in raw.proposal_evidence:
        proposal_id = item.get("proposal_observation_id")
        attempt_id = item.get("transaction_attempt_id")
        if isinstance(proposal_id, str) and isinstance(attempt_id, str):
            transaction_lineage.setdefault(proposal_id, []).append(attempt_id)
    primary_id = proposal_ids[0] if proposal_ids else None
    repair_id = proposal_ids[1] if len(proposal_ids) > 1 else None
    primary = outcome(0 if primary_id else None, primary_id)
    repair = outcome(1, repair_id) if repair_id else None
    contract = (
        asdict(contract_model.record)
        if contract_model and contract_model.record
        else None
    )
    if condition is Condition.M1 and (
        contract is None or contract["classification"] != "VALID"
    ):
        failure = "CONTRACT_UNAVAILABLE"
    elif not primary.model_output or not primary.schema_valid:
        failure = "MODEL_PROTOCOL_FAILURE"
    elif not primary.mechanically_materializable or not primary.transaction_applied:
        failure = "MATERIALIZATION_FAILURE"
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
    diagnosis = None
    if condition is Condition.M1 and failure != "PASS":
        diagnosis = (
            "CONTRACT_UNAVAILABLE" if failure == "CONTRACT_UNAVAILABLE" else "UNKNOWN"
        )
    return CellResult(
        SUITE,
        RUN_ID,
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
        asdict(paired_capture.record) if paired_capture.record is not None else None,
        tuple(asdict(record) for record in request_records),
        lineage,
        {key: tuple(value) for key, value in transaction_lineage.items()},
        contract,
        primary,
        repair,
        failure,
        diagnosis,
        len(metadata),
        raw.final_status,
        {
            "status": raw.status.value,
            "failure": raw.failure.value if raw.failure is not None else None,
            "model_calls": raw.metrics.model_calls,
            "tool_executions": raw.metrics.tool_executions,
            "discovery_calls": raw.metrics.discovery_calls,
            "source_reads": raw.metrics.source_reads,
            "mutation_ready_reached": raw.metrics.mutation_ready_reached,
            "structured_mutation_attempts": raw.metrics.structured_mutation_attempts,
            "mutation_proposed": raw.metrics.mutation_proposed,
            "final_status": raw.final_status,
        },
    )


def checkpoint_path(root: Path, case_id: str, condition: Condition) -> Path:
    return root / "cells" / f"{case_id}-{condition.name.lower()}.json"


def commit_cell(path: Path, result: CellResult) -> None:
    payload = asdict(result)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A75 result is not source-free")
    atomic_checkpoint(path, payload)


def read_cell(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    payload = resume_checkpoint(path)
    if payload is not None and any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"A75 checkpoint identity mismatch: {path.name}")
    return payload
