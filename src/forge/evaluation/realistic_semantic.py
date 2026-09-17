"""A45 realistic semantic benchmark result and production-run machinery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    RealWorldFailure,
    RealWorldTask,
    RealWorldTaskResult,
    RepositorySnapshot,
)
from forge.models import Model, MutationRepresentationPolicy

REALISTIC_SEMANTIC_V1 = "realistic-semantic-v1"
REALISTIC_SEMANTIC_SUITE_VERSION = 1
REALISTIC_SEMANTIC_SCHEMA_VERSION = 1
FORGE_MILESTONE = "A45"


class SemanticFailureLayer(Enum):
    DISCOVERY_FAILED = "DISCOVERY_FAILED"
    SOURCE_ACQUISITION_FAILED = "SOURCE_ACQUISITION_FAILED"
    MUTATION_NOT_READY = "MUTATION_NOT_READY"
    PROTOCOL_FAILED = "PROTOCOL_FAILED"
    EDIT_CONSTRUCTION_FAILED = "EDIT_CONSTRUCTION_FAILED"
    PREVIEW_FAILED = "PREVIEW_FAILED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    SEMANTIC_ORACLE_FAILED = "SEMANTIC_ORACLE_FAILED"
    SEMANTIC_SCORE_INELIGIBLE = "SEMANTIC_SCORE_INELIGIBLE"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class RealisticSemanticTask:
    task_id: str
    task_version: int
    language: str
    difficulty: str
    mutation_kind: str
    path_mode: str
    production_task: RealWorldTask

    def __post_init__(self) -> None:
        if not self.task_id.startswith("R") or self.task_version < 1:
            raise ValueError("realistic semantic tasks require stable Rxx versions")
        if self.mutation_kind not in {"single_file", "multi_file"}:
            raise ValueError("mutation_kind must be single_file or multi_file")
        if self.path_mode not in {"discovery_required", "path_known"}:
            raise ValueError("path_mode must describe source authority")
        if self.production_task.task_id != self.task_id:
            raise ValueError("production and semantic task IDs must agree")

    @property
    def versioned_id(self) -> str:
        return f"{self.task_id}-v{self.task_version}"


@dataclass(frozen=True, slots=True)
class SemanticIntegrityResult:
    task_id: str
    task_version: int
    baseline_oracle_pass: bool
    reference_oracle_pass: bool
    wrong_mutation_oracle_pass: bool
    semantic_score_eligible: bool


@dataclass(frozen=True, slots=True)
class RealisticSemanticResult:
    task_id: str
    task_version: int
    repository_identity: str
    model_profile: str
    model_artifact: str
    seed: int
    context_capacity: int
    output_budget: int
    temperature: float
    path_mode: str
    mutation_kind: str
    discovery_status: str
    source_status: str
    mutation_ready: bool
    schema_valid: bool
    preview_created: bool
    transaction_executed: bool
    verification_status: str
    build_test_status: str
    semantic_oracle: str
    first_pass_semantic: bool
    final_semantic: bool
    repair_used: bool
    repair_success: bool
    failure_layer: str
    discovery_calls: int
    source_reads: int
    required_candidates: int
    required_sources_ready: int
    mutations: int
    tool_calls: int
    model_calls: int
    input_tokens: int | None
    output_tokens: int | None
    context_peak: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RealisticSemanticAggregate:
    model_profile: str
    seed: int
    tasks: int
    first_pass_semantic_passes: int
    final_semantic_passes: int
    protocol_failures: int
    retrieval_failures: int
    verification_failures: int
    semantic_oracle_failures: int
    repairs_attempted: int
    repairs_successful: int
    tool_calls: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    token_measurement_complete: bool
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RealisticSemanticRun:
    suite: str
    suite_version: int
    schema_version: int
    forge_milestone: str
    repository_identity: str
    model_profile: str
    model_artifact: str
    context_capacity: int
    output_budget: int
    temperature: float
    integrity: tuple[SemanticIntegrityResult, ...]
    results: tuple[RealisticSemanticResult, ...]
    aggregates: tuple[RealisticSemanticAggregate, ...]
    canonical_unchanged: bool


def run_realistic_semantic_v1(
    model_profile: str,
    model: Model,
    repository: Path,
    snapshot: RepositorySnapshot,
    tasks: tuple[RealisticSemanticTask, ...],
    integrity: tuple[SemanticIntegrityResult, ...],
    *,
    model_artifact: str,
    mutation_representation: MutationRepresentationPolicy,
) -> RealisticSemanticRun:
    """Run the frozen task set through unchanged production orchestration."""
    eligibility = {item.task_id: item for item in integrity}
    if set(eligibility) != {task.task_id for task in tasks}:
        raise ValueError("integrity records must cover every realistic task")
    if not all(item.semantic_score_eligible for item in integrity):
        raise ValueError("model matrix cannot run with an ineligible task")
    production = tuple(task.production_task for task in tasks)
    raw = RealWorldEvaluationRunner(
        model_profile,
        model,
        repository,
        mutation_representation=mutation_representation,
    ).run(production, snapshot)
    metadata = {task.task_id: task for task in tasks}
    results = tuple(
        summarize_realistic_result(
            metadata[result.task_id],
            result,
            repository_identity=snapshot.identity,
            model_profile=model_profile,
            model_artifact=model_artifact,
            context_capacity=model.context_capacity or 0,
            eligible=eligibility[result.task_id].semantic_score_eligible,
        )
        for result in raw.results
    )
    seeds = tuple(dict.fromkeys(result.seed for result in results))
    return RealisticSemanticRun(
        REALISTIC_SEMANTIC_V1,
        REALISTIC_SEMANTIC_SUITE_VERSION,
        REALISTIC_SEMANTIC_SCHEMA_VERSION,
        FORGE_MILESTONE,
        snapshot.identity,
        model_profile,
        model_artifact,
        model.context_capacity or 0,
        512,
        0.0,
        integrity,
        results,
        tuple(
            aggregate_realistic_results(model_profile, seed, results) for seed in seeds
        ),
        raw.canonical_unchanged,
    )


def summarize_realistic_result(
    task: RealisticSemanticTask,
    result: RealWorldTaskResult,
    *,
    repository_identity: str,
    model_profile: str,
    model_artifact: str,
    context_capacity: int,
    eligible: bool,
) -> RealisticSemanticResult:
    metrics = result.metrics
    mutation = metrics.mutations > 0
    semantic = eligible and mutation and result.oracle is EvaluationOutcome.PASS
    repair_used = metrics.repair_attempts > 0
    repair_success = (
        repair_used and semantic and metrics.reverification_result == "pass"
    )
    first_pass = semantic and not repair_used
    failure = classify_semantic_failure(task, result, eligible=eligible)
    verification = metrics.verification_plan_result
    if verification == "not_run":
        verification = metrics.verification_result
    discovery_status = (
        "PASS"
        if task.path_mode == "path_known" or metrics.expected_implementation_acquired
        else "FAIL"
    )
    if metrics.required_candidate_count:
        source_ready = (
            metrics.required_sources_ready == metrics.required_candidate_count
        )
    else:
        source_ready = metrics.expected_implementation_acquired
    source_status = "PASS" if source_ready else "FAIL"
    return RealisticSemanticResult(
        task.task_id,
        task.task_version,
        repository_identity,
        model_profile,
        model_artifact,
        result.seed,
        context_capacity,
        512,
        0.0,
        task.path_mode,
        task.mutation_kind,
        discovery_status,
        source_status,
        metrics.mutation_ready_reached,
        metrics.structured_mutation_valid > 0,
        metrics.preview_created > 0 or metrics.mutation_group_preview_created > 0,
        mutation,
        verification,
        metrics.verification_result,
        result.oracle.value,
        first_pass,
        semantic,
        repair_used,
        repair_success,
        failure.value,
        metrics.discovery_calls,
        metrics.source_reads,
        metrics.required_candidate_count,
        metrics.required_sources_ready,
        metrics.mutations,
        metrics.tool_executions,
        metrics.model_calls,
        result.usage.input_tokens,
        result.usage.output_tokens,
        metrics.context_peak_estimate,
        result.elapsed_seconds,
    )


def classify_semantic_failure(
    task: RealisticSemanticTask,
    result: RealWorldTaskResult,
    *,
    eligible: bool,
) -> SemanticFailureLayer:
    metrics = result.metrics
    if not eligible:
        return SemanticFailureLayer.SEMANTIC_SCORE_INELIGIBLE
    semantic_pass = metrics.mutations > 0 and result.oracle is EvaluationOutcome.PASS
    if (
        task.path_mode == "discovery_required"
        and not metrics.expected_implementation_acquired
    ):
        return SemanticFailureLayer.DISCOVERY_FAILED
    if metrics.required_sources_missing or metrics.source_acquisition_failures:
        return SemanticFailureLayer.SOURCE_ACQUISITION_FAILED
    if result.failure is RealWorldFailure.PROTOCOL:
        return SemanticFailureLayer.PROTOCOL_FAILED
    if not metrics.mutation_ready_reached:
        return SemanticFailureLayer.MUTATION_NOT_READY
    if not metrics.mutation_proposed or metrics.structured_mutation_valid == 0:
        return SemanticFailureLayer.EDIT_CONSTRUCTION_FAILED
    if metrics.preview_created == 0 and metrics.mutation_group_preview_created == 0:
        return SemanticFailureLayer.PREVIEW_FAILED
    if metrics.mutations == 0:
        return SemanticFailureLayer.TRANSACTION_FAILED
    verification = metrics.verification_plan_result
    if verification == "not_run":
        verification = metrics.verification_result
    if verification not in {"pass", "passed"}:
        return SemanticFailureLayer.VERIFICATION_FAILED
    if semantic_pass:
        return SemanticFailureLayer.PASS
    return SemanticFailureLayer.SEMANTIC_ORACLE_FAILED


def aggregate_realistic_results(
    profile: str,
    seed: int,
    results: tuple[RealisticSemanticResult, ...],
) -> RealisticSemanticAggregate:
    selected = tuple(
        item for item in results if item.model_profile == profile and item.seed == seed
    )
    return RealisticSemanticAggregate(
        profile,
        seed,
        len(selected),
        sum(item.first_pass_semantic for item in selected),
        sum(item.final_semantic for item in selected),
        sum(item.failure_layer == "PROTOCOL_FAILED" for item in selected),
        sum(
            item.failure_layer in {"DISCOVERY_FAILED", "SOURCE_ACQUISITION_FAILED"}
            for item in selected
        ),
        sum(item.failure_layer == "VERIFICATION_FAILED" for item in selected),
        sum(item.failure_layer == "SEMANTIC_ORACLE_FAILED" for item in selected),
        sum(item.repair_used for item in selected),
        sum(item.repair_success for item in selected),
        sum(item.tool_calls for item in selected),
        sum(item.model_calls for item in selected),
        sum(item.input_tokens or 0 for item in selected),
        sum(item.output_tokens or 0 for item in selected),
        all(
            item.input_tokens is not None and item.output_tokens is not None
            for item in selected
        ),
        sum(item.elapsed_seconds for item in selected),
    )


def realistic_semantic_to_dict(run: RealisticSemanticRun) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(asdict(run))  # type: ignore[return-value]


def write_realistic_semantic_json(run: RealisticSemanticRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(realistic_semantic_to_dict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_realistic_semantic(run: RealisticSemanticRun) -> str:
    lines = ["Model       Seed Task Ready Schema Tx Verify Oracle Final Failure"]
    for item in run.results:
        lines.append(
            f"{item.model_profile:<11} {item.seed:<4} {item.task_id:<4} "
            f"{'yes' if item.mutation_ready else 'no':<5} "
            f"{'yes' if item.schema_valid else 'no':<6} "
            f"{'yes' if item.transaction_executed else 'no':<2} "
            f"{item.verification_status:<8} {item.semantic_oracle:<6} "
            f"{'yes' if item.final_semantic else 'no':<5} {item.failure_layer}"
        )
    return "\n".join(lines)
