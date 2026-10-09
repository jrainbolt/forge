"""A78 complete-workflow task/profile execution and checkpoints."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.atomic_coding_capability_v1.runner import (
    MeasuredModel,
    classify_failure,
)
from benchmarks.behavioral_obligation_diagnosis_v1.runner import (
    run_cell as run_a76_cell,
)
from benchmarks.behavioral_obligation_diagnosis_v1.suite import (
    Condition,
    DiagnosticCase,
)
from benchmarks.default_candidate_confirmation_v1.suite import (
    CONTEXT_SIZE,
    RUN_ID,
    SCHEMA_VERSION,
    SEED,
    SUITE,
    VERSION,
)
from benchmarks.default_candidate_confirmation_v1.workflow_identity import (
    WorkflowOutcome,
    begin_workflow_attempt,
    complete_workflow_attempt,
    workflow_payload,
)
from benchmarks.realistic_coding_v2.suite import FrozenTask
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.models import Model, MutationRepresentationPolicy


@dataclass(frozen=True, slots=True)
class CellResult:
    suite: str
    run_identity: str
    suite_version: int
    schema_version: int
    cell_id: str
    task_id: str
    operation_class: str
    profile: str
    seed: int
    corpus_identity: str
    repository_identity: str
    model_artifact: str
    model_artifact_size: int
    model_config_identity: str
    workflow_attempt: dict[str, object]
    workflow_trace: dict[str, object]
    paired_input_identity: dict[str, object] | None
    mutation_requests: tuple[dict[str, object], ...]
    proposal_request_lineage: dict[str, str]
    proposal_transaction_lineage: dict[str, tuple[str, ...]]
    proposal: dict[str, object]
    primary_semantic_pass: bool
    repair: dict[str, object] | None
    failure: str
    generation_calls: int
    median_generation_latency_seconds: float | None
    input_tokens: int
    output_tokens: int


def classify_workflow_outcome(
    trace: dict[str, object], request_ids: tuple[str, ...]
) -> WorkflowOutcome:
    if request_ids:
        return WorkflowOutcome.COMPLETED_WITH_MUTATION_REQUEST
    failure = trace.get("failure")
    if failure == "INFRASTRUCTURE":
        return WorkflowOutcome.TOOL_FAILURE
    if failure == "GROUNDING":
        return WorkflowOutcome.GROUNDING_FAILURE
    if failure == "CONTEXT_OVERFLOW":
        return WorkflowOutcome.CONTEXT_FAILURE
    if failure == "MODEL":
        return WorkflowOutcome.MODEL_ERROR
    if failure == "PROTOCOL":
        return WorkflowOutcome.MODEL_PROTOCOL_FAILURE
    if failure is not None:
        return WorkflowOutcome.OTHER_FAILURE
    return WorkflowOutcome.COMPLETED_NO_MUTATION_REQUEST


def run_cell(
    definition: FrozenTask,
    profile: str,
    backend: Model,
    *,
    artifact: str,
    artifact_size: int,
    model_config_identity: str,
    repository_identity: str,
    corpus_identity: str,
    representation: MutationRepresentationPolicy,
    run_identity: str = RUN_ID,
    primary_workspace_callback: Callable[[Path], None] | None = None,
    final_workspace_callback: Callable[[Path], None] | None = None,
) -> CellResult:
    workflow = begin_workflow_attempt(
        definition,
        evaluator_corpus_identity=corpus_identity,
        repository_identity=repository_identity,
        model_profile_identity=model_config_identity,
        seed=SEED,
        temperature=0,
        context_size=CONTEXT_SIZE,
        output_budget=512,
        representation=representation.value,
    )
    measured = MeasuredModel(backend)
    inner = run_a76_cell(
        DiagnosticCase(
            f"A78-{definition.task_id}",
            profile,
            definition.task_id,
            definition.operation_class.value,
            "held_out_complete_workflow",
        ),
        Condition.F0,
        definition,
        measured,
        obligation=None,
        artifact=artifact,
        model_config_identity=model_config_identity,
        repository_identity=repository_identity,
        corpus_identity=corpus_identity,
        representation=representation,
        primary_workspace_callback=primary_workspace_callback,
        final_workspace_callback=final_workspace_callback,
    )
    semantic = bool(inner.full_task_semantic_pass)
    failure = classify_failure(
        inner.proposal,
        inner_failure=inner.failure,
        obligation_oracle_pass=semantic,
    )
    requests = tuple(
        {**record, "workflow_attempt_id": workflow.workflow_attempt_id}
        for record in inner.mutation_requests
    )
    request_ids = tuple(str(record["mutation_request_id"]) for record in requests)
    proposal_lineage = tuple(
        sorted(
            (proposal_id, request_id)
            for proposal_id, request_id in inner.proposal_request_lineage.items()
        )
    )
    transaction_lineage = tuple(
        sorted(
            (proposal_id, transaction_id)
            for proposal_id, transaction_ids in (
                inner.proposal_transaction_lineage.items()
            )
            for transaction_id in transaction_ids
        )
    )
    workflow = complete_workflow_attempt(
        workflow,
        outcome=classify_workflow_outcome(inner.workflow_trace, request_ids),
        mutation_request_ids=request_ids,
        proposal_lineage=proposal_lineage,
        transaction_lineage=transaction_lineage,
    )
    return CellResult(
        SUITE,
        run_identity,
        VERSION,
        SCHEMA_VERSION,
        f"A78-{definition.task_id}-{profile}",
        definition.task_id,
        definition.operation_class.value,
        profile,
        SEED,
        corpus_identity,
        repository_identity,
        artifact,
        artifact_size,
        model_config_identity,
        workflow_payload(workflow),
        inner.workflow_trace,
        inner.paired_input_identity,
        requests,
        inner.proposal_request_lineage,
        inner.proposal_transaction_lineage,
        inner.proposal,
        semantic,
        inner.repair,
        failure,
        len(measured.latencies),
        statistics.median(measured.latencies) if measured.latencies else None,
        measured.input_tokens,
        measured.output_tokens,
    )


def checkpoint_path(root: Path, task_id: str, profile: str) -> Path:
    return root / "cells" / f"A78-{task_id}-{profile}.json"


def commit_cell(path: Path, result: CellResult) -> None:
    payload = asdict(result)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A78 result is not source-free")
    atomic_checkpoint(path, payload)


def read_cell(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    payload = resume_checkpoint(path)
    if payload is not None and any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"A78 checkpoint identity mismatch: {path.name}")
    return payload
