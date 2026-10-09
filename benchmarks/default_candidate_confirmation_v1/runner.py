"""A78 complete-workflow task/profile execution and checkpoints."""

from __future__ import annotations

import statistics
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
    RUN_ID,
    SCHEMA_VERSION,
    SEED,
    SUITE,
    VERSION,
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
    paired_input_identity: dict[str, object]
    proposal: dict[str, object]
    primary_semantic_pass: bool
    repair: dict[str, object] | None
    failure: str
    generation_calls: int
    median_generation_latency_seconds: float | None
    input_tokens: int
    output_tokens: int


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
) -> CellResult:
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
    )
    semantic = bool(inner.full_task_semantic_pass)
    failure = classify_failure(
        inner.proposal,
        inner_failure=inner.failure,
        obligation_oracle_pass=semantic,
    )
    return CellResult(
        SUITE,
        RUN_ID,
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
        inner.paired_input_identity,
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
