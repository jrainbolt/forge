"""A77 atomic task/profile execution and source-free checkpoints."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.atomic_coding_capability_v1.suite import (
    RUN_ID,
    SCHEMA_VERSION,
    SEED,
    SUITE,
    VERSION,
    AtomicTask,
    diagnostic_case,
)
from benchmarks.behavioral_obligation_diagnosis_v1.runner import (
    run_cell as run_a76_cell,
)
from benchmarks.behavioral_obligation_diagnosis_v1.suite import Condition
from benchmarks.realistic_coding_v2.suite import FrozenTask
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.models import (
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    MutationRepresentationPolicy,
)


class MeasuredModel(Model):
    def __init__(self, backend: Model) -> None:
        self.backend = backend
        self.latencies: list[float] = []
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def identity(self) -> ModelIdentity:
        return self.backend.identity

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.backend.capabilities

    @property
    def context_capacity(self) -> int | None:
        return self.backend.context_capacity

    def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        response = self.backend.generate(request)
        self.latencies.append(time.perf_counter() - started)
        self.input_tokens += response.usage.input_tokens or 0
        self.output_tokens += response.usage.output_tokens or 0
        return response

    def close(self) -> None:
        self.backend.close()


@dataclass(frozen=True, slots=True)
class CellResult:
    suite: str
    run_identity: str
    suite_version: int
    schema_version: int
    cell_id: str
    atomic_id: str
    source_case_id: str
    task_id: str
    operation_class: str
    control: bool
    profile: str
    seed: int
    corpus_identity: str
    repository_identity: str
    model_artifact: str
    model_artifact_size: int
    model_config_identity: str
    backend_id: str
    model_id: str
    paired_input_identity: dict[str, object]
    proposal: dict[str, object]
    obligation_oracle_pass: bool
    repair: dict[str, object] | None
    failure: str
    generation_calls: int
    median_generation_latency_seconds: float | None
    input_tokens: int
    output_tokens: int


def classify_failure(
    proposal: dict[str, object],
    *,
    inner_failure: str,
    obligation_oracle_pass: bool,
    load_failed: bool = False,
) -> str:
    if load_failed:
        return "MODEL_LOAD_FAILURE"
    if not proposal["model_output"]:
        return "NO_MODEL_OUTPUT"
    if not proposal["schema_valid"]:
        return "MODEL_PROTOCOL_FAILURE"
    if not proposal["mechanically_materializable"]:
        return "MATERIALIZATION_FAILURE"
    if not proposal["production_validatable"]:
        return "PRODUCTION_VALIDATION_FAILURE"
    if not proposal["transaction_applied"]:
        return "MATERIALIZATION_FAILURE"
    if not proposal["verification_pass"]:
        return (
            "BUILD_FAILURE"
            if inner_failure == "BUILD_FAILURE"
            else "VERIFICATION_FAILURE"
        )
    if not obligation_oracle_pass:
        return str(proposal["semantic_failure"] or "WRONG_BEHAVIOR")
    return "PASS"


def run_cell(
    task: AtomicTask,
    profile: str,
    definition: FrozenTask,
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
        diagnostic_case(task, profile),
        Condition.F1,
        definition,
        measured,
        obligation=task.obligation,
        artifact=artifact,
        model_config_identity=model_config_identity,
        repository_identity=repository_identity,
        corpus_identity=corpus_identity,
        representation=representation,
    )
    proposal = inner.proposal
    failure = classify_failure(
        proposal,
        inner_failure=inner.failure,
        obligation_oracle_pass=bool(inner.obligation_oracle_pass),
    )
    return CellResult(
        SUITE,
        RUN_ID,
        VERSION,
        SCHEMA_VERSION,
        f"{task.atomic_id}-{profile}",
        task.atomic_id,
        task.source_case_id,
        task.task_id,
        task.operation_class,
        task.control,
        profile,
        SEED,
        corpus_identity,
        repository_identity,
        artifact,
        artifact_size,
        model_config_identity,
        backend.identity.backend_id,
        backend.identity.model_id,
        inner.paired_input_identity,
        proposal,
        bool(inner.obligation_oracle_pass),
        inner.repair,
        failure,
        len(measured.latencies),
        statistics.median(measured.latencies) if measured.latencies else None,
        measured.input_tokens,
        measured.output_tokens,
    )


def checkpoint_path(root: Path, atomic_id: str, profile: str) -> Path:
    return root / "cells" / f"{atomic_id}-{profile}.json"


def commit_cell(path: Path, result: CellResult) -> None:
    payload = asdict(result)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A77 result is not source-free")
    atomic_checkpoint(path, payload)


def read_cell(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    payload = resume_checkpoint(path)
    if payload is not None and any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"A77 checkpoint identity mismatch: {path.name}")
    return payload
