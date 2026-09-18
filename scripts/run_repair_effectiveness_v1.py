"""Run one model's paired A47 R0/R1 realistic repair cells."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
)
from forge.evaluation import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    RepairCondition,
    RepairOutputClass,
    RepairRegression,
    RepairTrialResult,
    build_repair_case_corpus,
    build_repair_effectiveness_run,
    hash_workspace,
    load_realistic_semantic_run,
    write_repair_effectiveness_json,
)
from forge.models import (
    LlamaCppConfig,
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    default_backend_registry,
    load_model_catalog,
)
from scripts.run_realistic_semantic_v1 import _artifact_identity, _snapshot


class _CaptureModel(Model):
    def __init__(self, delegate: Model, prefix: tuple[ModelResponse, ...] = ()) -> None:
        self.delegate = delegate
        self.prefix = prefix
        self.requests: list[ModelRequest] = []
        self.responses: list[ModelResponse] = []
        self.latencies: list[float] = []

    @property
    def identity(self) -> ModelIdentity:
        return self.delegate.identity

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.delegate.capabilities

    @property
    def context_capacity(self) -> int | None:
        return self.delegate.context_capacity

    def generate(self, request: ModelRequest) -> ModelResponse:
        index = len(self.requests)
        self.requests.append(request)
        started = time.perf_counter()
        response = (
            self.prefix[index]
            if index < len(self.prefix)
            else self.delegate.generate(request)
        )
        self.latencies.append(time.perf_counter() - started)
        self.responses.append(response)
        return response

    def close(self) -> None:
        # The outer evaluation owns the shared loaded model.
        return None


def _repair_request_index(model: _CaptureModel) -> int | None:
    for index, request in enumerate(model.requests):
        prompt = "\n".join(message.content for message in request.messages)
        if "final permitted repair mutation" in prompt:
            return index
    return None


def _trial(
    case_id: str,
    profile: str,
    task_id: str,
    condition: RepairCondition,
    result: object,
    model: _CaptureModel,
) -> RepairTrialResult:
    metrics = result.metrics
    repair_index = _repair_request_index(model)
    response = model.responses[repair_index] if repair_index is not None else None
    usage = response.usage if response is not None else None
    semantic = result.oracle is EvaluationOutcome.PASS
    verification = metrics.reverification_result == "passed"
    if semantic and metrics.repair_mutation_executed:
        output_class = RepairOutputClass.SEMANTIC_RECOVERY
    elif not metrics.repair_proposal_emitted:
        output_class = RepairOutputClass.SCHEMA_INVALID
    elif not metrics.repair_preview_created:
        output_class = RepairOutputClass.WRONG_PATH
    elif not metrics.repair_mutation_executed:
        output_class = RepairOutputClass.NO_OP
    elif not verification:
        output_class = RepairOutputClass.VERIFICATION_FAIL
    else:
        output_class = RepairOutputClass.SEMANTIC_FAIL
    return RepairTrialResult(
        case_id,
        profile,
        task_id,
        condition,
        output_class,
        RepairRegression.NO_CHANGE,
        metrics.repair_ready_reached,
        metrics.repair_proposal_emitted,
        metrics.repair_preview_created,
        metrics.repair_preview_created,
        metrics.repair_preview_created,
        metrics.repair_mutation_executed,
        metrics.reverification_executed,
        verification,
        semantic,
        usage.input_tokens if usage is not None else None,
        usage.output_tokens if usage is not None else None,
        model.latencies[repair_index] if repair_index is not None else 0.0,
        metrics.verification_plan_duration,
    )


def _compare(
    baseline: RepairTrialResult, enhanced: RepairTrialResult
) -> tuple[RepairTrialResult, RepairTrialResult]:
    if enhanced.semantic_passed and not baseline.semantic_passed:
        classification = RepairRegression.SEMANTIC_RECOVERED
    elif baseline.semantic_passed and not enhanced.semantic_passed:
        classification = RepairRegression.SEMANTIC_REGRESSED
    elif enhanced.verification_passed and not baseline.verification_passed:
        classification = RepairRegression.VERIFICATION_IMPROVED
    elif baseline.verification_passed and not enhanced.verification_passed:
        classification = RepairRegression.VERIFICATION_REGRESSED
    elif not enhanced.schema_valid and baseline.schema_valid:
        classification = RepairRegression.STRUCTURAL_FAILURE
    elif not enhanced.semantic_passed:
        classification = RepairRegression.SEMANTIC_STILL_FAIL
    else:
        classification = RepairRegression.NO_CHANGE
    return replace(baseline, regression=classification), replace(
        enhanced, regression=classification
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--corpus-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_run = load_realistic_semantic_run(args.source_result)
    corpus_runs = tuple(
        load_realistic_semantic_run(path) for path in args.corpus_result
    )
    cases = build_repair_case_corpus(corpus_runs)
    selected = tuple(
        case
        for case in cases
        if case.model_profile == args.model and case.eligible and case.seed == 42
    )
    if not selected:
        raise RuntimeError("no eligible recorded repair cases for selected model")
    task_ids = {case.task_id for case in selected}
    definitions = tuple(
        item
        for item in realistic_semantic_tasks((42,))
        if item.metadata.task_id in task_ids
    )
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.model)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A47 requires an existing llama.cpp profile")
    model = catalog.create(args.model)
    if model.context_capacity != 8192:
        model.close()
        raise RuntimeError("repair-effectiveness-v1 requires context 8192")
    canonical_before = hash_workspace(REPOSITORY)
    snapshot = _snapshot()
    trials: list[RepairTrialResult] = []
    try:
        for definition in definitions:
            case = next(
                case for case in selected if case.task_id == definition.metadata.task_id
            )
            baseline_model = _CaptureModel(model)
            baseline = (
                RealWorldEvaluationRunner(
                    args.model,
                    baseline_model,
                    REPOSITORY,
                    mutation_representation=profile.mutation_representation,
                    include_repair_mutation_history=False,
                )
                .run((definition.metadata.production_task,), snapshot)
                .results[0]
            )
            repair_index = _repair_request_index(baseline_model)
            prefix = (
                tuple(baseline_model.responses[:repair_index])
                if repair_index is not None
                else tuple(baseline_model.responses)
            )
            enhanced_model = _CaptureModel(model, prefix)
            enhanced = (
                RealWorldEvaluationRunner(
                    args.model,
                    enhanced_model,
                    REPOSITORY,
                    mutation_representation=profile.mutation_representation,
                    include_repair_mutation_history=True,
                )
                .run((definition.metadata.production_task,), snapshot)
                .results[0]
            )
            r0 = _trial(
                case.case_id,
                args.model,
                case.task_id,
                RepairCondition.CURRENT_REPAIR_BASELINE,
                baseline,
                baseline_model,
            )
            r1 = _trial(
                case.case_id,
                args.model,
                case.task_id,
                RepairCondition.CURRENT_SOURCE_WITH_FIRST_MUTATION,
                enhanced,
                enhanced_model,
            )
            trials.extend(_compare(r0, r1))
            print(
                f"{case.task_id}: R0={r0.output_class.value} "
                f"R1={r1.output_class.value}",
                flush=True,
            )
    finally:
        model.close()
    if canonical_before != hash_workspace(REPOSITORY):
        raise RuntimeError("canonical realistic semantic repository changed")
    run = build_repair_effectiveness_run(
        repository_identity=source_run.repository_identity,
        cases=cases,
        trials=tuple(trials),
    )
    write_repair_effectiveness_json(run, args.output)
    print(f"completed {len(selected)} paired repair cases", flush=True)
    print(
        "artifact=" + _artifact_identity(profile.backend_config.model_path), flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
