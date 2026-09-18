"""A48 deterministic verification-signal alignment scenarios."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
)
from forge.evaluation import (
    VERIFICATION_ALIGNMENT_SCHEMA_VERSION,
    VERIFICATION_ALIGNMENT_SUITE_VERSION,
    VERIFICATION_ALIGNMENT_V1,
    RepairObservability,
    TargetedTestResult,
    VerificationAlignmentClass,
    VerificationConfusionCounts,
    VerificationCoverageObservation,
    VerificationSourceState,
    aggregate_confusion,
    classify_result_pattern,
    compare_targeted_classification,
    repair_observability,
    run_verification_alignment_v1,
    verification_alignment_to_dict,
)
from forge.orchestration import CodingTaskResult


@pytest.fixture(scope="module")
def alignment_run(tmp_path_factory: pytest.TempPathFactory):
    definitions = realistic_semantic_tasks((42,))
    selected = (definitions[0], definitions[4])
    return run_verification_alignment_v1(
        tmp_path_factory.mktemp("a48"), REPOSITORY, selected
    )


def _observation(
    semantic: bool,
    verification: str,
    state: VerificationSourceState = VerificationSourceState.BASELINE,
) -> VerificationCoverageObservation:
    return VerificationCoverageObservation(
        "synthetic",
        state,
        semantic,
        "pass",
        "pass",
        verification,
        verification,
        None if verification == "pass" else "test",
        "independent evaluator state",
        0.01,
        (),
    )


def test_a01_baseline_wrong_and_verification_fails(alignment_run) -> None:
    baseline = alignment_run.tasks[0].baseline
    assert baseline.source_state is VerificationSourceState.BASELINE
    assert not baseline.semantic_truth
    assert baseline.final_verification_result == "fail"


def test_a02_reference_correct_and_verification_passes(alignment_run) -> None:
    reference = alignment_run.tasks[0].reference
    assert reference.source_state is VerificationSourceState.REFERENCE
    assert reference.semantic_truth
    assert reference.final_verification_result == "pass"


def test_a03_wrong_mutation_fails_and_is_fully_discriminating(
    alignment_run,
) -> None:
    task = alignment_run.tasks[0]
    assert task.wrong.source_state is VerificationSourceState.WRONG
    assert not task.wrong.semantic_truth
    assert task.wrong.final_verification_result == "fail"
    assert task.classification is VerificationAlignmentClass.FULLY_DISCRIMINATING


def test_a04_baseline_acceptance_is_an_observability_gap(alignment_run) -> None:
    task = alignment_run.tasks[1]
    assert task.classification is VerificationAlignmentClass.BASELINE_ACCEPTING
    assert (
        task.repair_observability[0] is RepairObservability.SEMANTIC_FAILURE_INVISIBLE
    )


def test_a05_reference_rejection_is_potentially_harmful() -> None:
    assert (
        classify_result_pattern("fail", "fail", "fail")
        is VerificationAlignmentClass.REFERENCE_REJECTING
    )
    assert (
        repair_observability(_observation(True, "fail"))
        is RepairObservability.POTENTIALLY_HARMFUL_TRIGGER
    )


def test_a06_wrong_acceptance_is_explicit() -> None:
    assert (
        classify_result_pattern("fail", "pass", "pass")
        is VerificationAlignmentClass.REFERENCE_ACCEPTING_BUT_WRONG_ACCEPTING
    )


def test_a07_confusion_matrix_aggregates_exactly() -> None:
    observations = (
        _observation(True, "pass"),
        _observation(True, "fail"),
        _observation(False, "pass"),
        _observation(False, "fail"),
    )
    assert aggregate_confusion(observations) == VerificationConfusionCounts(1, 1, 1, 1)


def test_a08_repair_implications_cover_all_signal_cases() -> None:
    assert repair_observability(_observation(False, "fail")) is (
        RepairObservability.LEGITIMATELY_TRIGGERABLE
    )
    assert repair_observability(_observation(False, "pass")) is (
        RepairObservability.SEMANTIC_FAILURE_INVISIBLE
    )
    assert repair_observability(_observation(True, "fail")) is (
        RepairObservability.POTENTIALLY_HARMFUL_TRIGGER
    )


def test_a09_hidden_evaluator_authority_is_absent_from_production_result(
    alignment_run,
) -> None:
    production_fields = {item.name for item in fields(CodingTaskResult)}
    assert production_fields.isdisjoint(
        {"semantic_truth", "reference_mutation", "wrong_mutation", "hidden_oracle"}
    )
    serialized = str(verification_alignment_to_dict(alignment_run))
    assert "SetupReplacement" not in serialized
    assert "oracle_commands" not in serialized


def test_a10_preselected_targeted_test_can_improve_discrimination() -> None:
    assert (
        compare_targeted_classification(
            VerificationAlignmentClass.BASELINE_ACCEPTING,
            VerificationAlignmentClass.FULLY_DISCRIMINATING,
        )
        is TargetedTestResult.IMPROVES_DISCRIMINATION
    )


def test_a11_non_discriminating_targeted_test_is_not_useful(
    alignment_run,
) -> None:
    targeted = alignment_run.tasks[1].targeted_test
    assert targeted.classification is VerificationAlignmentClass.BASELINE_ACCEPTING
    assert targeted.comparison is TargetedTestResult.UNCHANGED


def test_a12_full_verification_plan_remains_unchanged(alignment_run) -> None:
    expected = (
        "project.configure",
        "project.build",
        "project.test",
    )
    assert all(
        definition.metadata.production_task.verification_plan.steps == expected
        for definition in realistic_semantic_tasks((42,))
    )
    assert all(len(task.reference.steps) == 3 for task in alignment_run.tasks)


def test_versioning_canonical_safety_and_package_exclusion(alignment_run) -> None:
    assert alignment_run.suite == VERIFICATION_ALIGNMENT_V1
    assert alignment_run.suite_version == VERIFICATION_ALIGNMENT_SUITE_VERSION == 1
    assert alignment_run.schema_version == VERIFICATION_ALIGNMENT_SCHEMA_VERSION == 1
    assert alignment_run.canonical_unchanged
    assert REPOSITORY.is_dir()
    project = Path(__file__).resolve().parents[1]
    assert not (project / "src/forge" / "benchmarks").exists()
    assert not (project / "src/forge" / "eval-results").exists()
