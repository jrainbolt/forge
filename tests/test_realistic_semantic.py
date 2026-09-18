from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
    validate_integrity,
)
from forge.evaluation import (
    REALISTIC_SEMANTIC_V1,
    EvaluationOutcome,
    RealWorldMetrics,
    RealWorldStatus,
    RealWorldTaskResult,
    SemanticFailureLayer,
    SemanticIntegrityResult,
    aggregate_realistic_results,
    classify_semantic_failure,
    realistic_semantic_to_dict,
    summarize_realistic_result,
)
from forge.evaluation.realistic_semantic import RealisticSemanticRun
from forge.models import ModelUsage


def _raw_result(task_id: str, seed: int = 42, **metric_changes):  # type: ignore[no-untyped-def]
    metrics = RealWorldMetrics(
        model_calls=2,
        tool_executions=4,
        expected_implementation_acquired=True,
        mutation_ready_reached=True,
        mutation_proposed=True,
        structured_mutation_valid=1,
        preview_created=1,
        mutations=1,
        verification_executed=1,
        verification_result="pass",
        verification_plan_result="pass",
        context_peak_estimate=6000,
    )
    metrics = replace(metrics, **metric_changes)
    return RealWorldTaskResult(
        task_id,
        "repair",
        "bounded_repair",
        seed,
        RealWorldStatus.PASS,
        EvaluationOutcome.PASS,
        EvaluationOutcome.PASS,
        EvaluationOutcome.PASS,
        None,
        None,
        "completed_verified",
        (),
        (),
        (),
        (),
        (),
        metrics,
        ModelUsage(input_tokens=500, output_tokens=100),
        3.0,
    )


def test_task_versions_counts_languages_and_mutation_shapes() -> None:
    tasks = tuple(item.metadata for item in realistic_semantic_tasks((42, 43)))
    assert [task.versioned_id for task in tasks] == [
        f"R{number:02d}-v1" for number in range(1, 9)
    ]
    assert sum(task.mutation_kind == "single_file" for task in tasks) == 4
    assert sum(task.mutation_kind == "multi_file" for task in tasks) == 4
    assert {task.language for task in tasks} == {"Python", "C17"}
    assert all(task.production_task.seeds == (42, 43) for task in tasks)


def test_path_authority_matches_discovery_design() -> None:
    tasks = tuple(item.metadata for item in realistic_semantic_tasks((42,)))
    for task in tasks[:4]:
        assert task.path_mode == "discovery_required"
        assert task.production_task.required_candidate_paths == ()
    for task in tasks[4:]:
        assert task.path_mode == "path_known"
        assert task.production_task.required_candidate_paths == (
            task.production_task.expected_changed_paths
        )


def test_all_oracles_are_discriminating_and_reject_wrong_mutations() -> None:
    integrity = validate_integrity(realistic_semantic_tasks((42,)))
    assert len(integrity) == 8
    assert all(not item.baseline_oracle_pass for item in integrity)
    assert all(item.reference_oracle_pass for item in integrity)
    assert all(not item.wrong_mutation_oracle_pass for item in integrity)
    assert all(item.semantic_score_eligible for item in integrity)


def test_repository_has_realistic_adjacent_file_count() -> None:
    files = tuple(path for path in REPOSITORY.rglob("*") if path.is_file())
    assert 20 <= len(files) <= 40
    assert len(tuple((REPOSITORY / "pyservice").glob("*.py"))) >= 8
    assert len(tuple((REPOSITORY / "cengine").glob("*.c"))) >= 6


def test_evaluator_reference_and_hidden_oracle_do_not_leak_to_prompts() -> None:
    definitions = realistic_semantic_tasks((42,))
    rendered = "\n".join(item.metadata.production_task.prompt for item in definitions)
    assert "known-good" not in rendered
    assert "hidden" not in rendered
    assert "oracle.py" not in rendered
    assert "SetupReplacement" not in rendered
    assert all(
        not hasattr(item.metadata.production_task, "reference") for item in definitions
    )


def test_success_result_separates_structure_verification_and_semantics() -> None:
    task = realistic_semantic_tasks((42,))[0].metadata
    result = summarize_realistic_result(
        task,
        _raw_result(task.task_id),
        repository_identity="repo-sha",
        model_profile="mock",
        model_artifact="mock.gguf:sha256:abc",
        context_capacity=8192,
        eligible=True,
    )
    assert result.seed == 42 and result.repository_identity == "repo-sha"
    assert result.mutation_ready and result.schema_valid
    assert result.preview_created and result.transaction_executed
    assert result.build_test_status == "pass"
    assert result.semantic_oracle == "PASS"
    assert result.first_pass_semantic and result.final_semantic
    assert result.failure_layer == SemanticFailureLayer.PASS.value


def test_repaired_success_is_not_counted_as_first_pass() -> None:
    task = realistic_semantic_tasks((42,))[0].metadata
    raw = _raw_result(
        task.task_id,
        mutations=2,
        repair_attempts=1,
        reverification_executed=True,
        reverification_result="pass",
    )
    result = summarize_realistic_result(
        task,
        raw,
        repository_identity="repo",
        model_profile="mock",
        model_artifact="artifact",
        context_capacity=8192,
        eligible=True,
    )
    assert result.repair_used and result.repair_success
    assert not result.first_pass_semantic and result.final_semantic


def test_failure_classification_distinguishes_layers() -> None:
    discovery = realistic_semantic_tasks((42,))[0].metadata
    raw = _raw_result(
        discovery.task_id,
        expected_implementation_acquired=False,
        mutation_ready_reached=False,
        mutation_proposed=False,
        structured_mutation_valid=0,
        preview_created=0,
        mutations=0,
        verification_executed=0,
        verification_result="not_run",
        verification_plan_result="not_run",
    )
    assert classify_semantic_failure(discovery, raw, eligible=True) is (
        SemanticFailureLayer.DISCOVERY_FAILED
    )
    assert classify_semantic_failure(discovery, raw, eligible=False) is (
        SemanticFailureLayer.SEMANTIC_SCORE_INELIGIBLE
    )


def test_semantic_oracle_failure_is_distinct_from_verification() -> None:
    task = realistic_semantic_tasks((42,))[0].metadata
    raw = replace(_raw_result(task.task_id), oracle=EvaluationOutcome.FAIL)
    assert classify_semantic_failure(task, raw, eligible=True) is (
        SemanticFailureLayer.SEMANTIC_ORACLE_FAILED
    )
    failed_verification = replace(
        raw,
        metrics=replace(
            raw.metrics,
            verification_result="fail",
            verification_plan_result="fail",
        ),
    )
    assert classify_semantic_failure(task, failed_verification, eligible=True) is (
        SemanticFailureLayer.VERIFICATION_FAILED
    )
    verification_only = replace(
        _raw_result(task.task_id),
        metrics=replace(
            raw.metrics,
            verification_result="fail",
            verification_plan_result="step_failed",
        ),
    )
    assert classify_semantic_failure(task, verification_only, eligible=True) is (
        SemanticFailureLayer.VERIFICATION_FAILED
    )


def test_verified_semantic_pass_outranks_incidental_source_failure() -> None:
    task = realistic_semantic_tasks((42,))[4].metadata
    raw = _raw_result(
        task.task_id,
        required_candidate_count=2,
        required_sources_ready=2,
        source_acquisition_failures=1,
    )

    assert classify_semantic_failure(task, raw, eligible=True) is (
        SemanticFailureLayer.PASS
    )


def test_aggregation_records_seed_repair_cost_and_failure_counts() -> None:
    task = realistic_semantic_tasks((42,))[0].metadata
    passed = summarize_realistic_result(
        task,
        _raw_result(task.task_id),
        repository_identity="repo",
        model_profile="mock",
        model_artifact="artifact",
        context_capacity=8192,
        eligible=True,
    )
    repaired = replace(
        passed,
        task_id="R02",
        first_pass_semantic=False,
        repair_used=True,
        repair_success=True,
    )
    aggregate = aggregate_realistic_results("mock", 42, (passed, repaired))
    assert aggregate.tasks == 2
    assert aggregate.first_pass_semantic_passes == 1
    assert aggregate.final_semantic_passes == 2
    assert aggregate.repairs_attempted == aggregate.repairs_successful == 1
    assert aggregate.model_calls == 4 and aggregate.seed == 42


def test_result_serialization_contains_no_reference_source() -> None:
    integrity = (SemanticIntegrityResult("R01", 1, False, True, False, True),)
    run = RealisticSemanticRun(
        REALISTIC_SEMANTIC_V1,
        1,
        1,
        "A45",
        "repo",
        "mock",
        "artifact",
        8192,
        512,
        0.0,
        integrity,
        (),
        (),
        True,
    )
    payload = realistic_semantic_to_dict(run)
    rendered = json.dumps(payload)
    assert payload["suite"] == REALISTIC_SEMANTIC_V1
    assert "reference_oracle_pass" in rendered
    assert "new_text" not in rendered and "old_text" not in rendered


def test_benchmark_assets_are_outside_installable_package() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    assert project["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert REPOSITORY.is_relative_to(root / "benchmarks")
    assert not REPOSITORY.is_relative_to(root / "src")
