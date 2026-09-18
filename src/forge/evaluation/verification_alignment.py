"""Evaluator-only semantic alignment measurement for configured verification."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

from forge.evaluation.realistic_semantic import (
    RealisticSemanticRun,
    RealisticSemanticTask,
)
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldTask,
    SetupReplacement,
    apply_task_setup,
    copy_repository,
    hash_workspace,
    run_oracle,
)
from forge.interaction import AutonomyMode, resolve_interaction_policy
from forge.orchestration.verification_attribution import (
    bounded_failure_lines,
    failure_fingerprint,
)
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    ExecutionContext,
    InvocationApproval,
    ProjectCommandTool,
    ToolExecutor,
    ToolInvocation,
    ToolResultStatus,
    create_assist_repository_registry,
)

VERIFICATION_ALIGNMENT_V1 = "verification-alignment-v1"
VERIFICATION_ALIGNMENT_SUITE_VERSION = 1
VERIFICATION_ALIGNMENT_SCHEMA_VERSION = 1


class VerificationSourceState(Enum):
    BASELINE = "BASELINE"
    REFERENCE = "REFERENCE"
    WRONG = "WRONG"


class VerificationAlignmentClass(Enum):
    FULLY_DISCRIMINATING = "FULLY_DISCRIMINATING"
    REFERENCE_ACCEPTING_BUT_WRONG_ACCEPTING = "REFERENCE_ACCEPTING_BUT_WRONG_ACCEPTING"
    BASELINE_ACCEPTING = "BASELINE_ACCEPTING"
    REFERENCE_REJECTING = "REFERENCE_REJECTING"
    PARTIALLY_DISCRIMINATING = "PARTIALLY_DISCRIMINATING"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"


class RepairObservability(Enum):
    LEGITIMATELY_TRIGGERABLE = "LEGITIMATELY_TRIGGERABLE"
    SEMANTIC_FAILURE_INVISIBLE = "SEMANTIC_FAILURE_INVISIBLE"
    POTENTIALLY_HARMFUL_TRIGGER = "POTENTIALLY_HARMFUL_TRIGGER"
    NO_REPAIR_NEEDED = "NO_REPAIR_NEEDED"


class TargetedTestResult(Enum):
    IMPROVES_DISCRIMINATION = "IMPROVES_DISCRIMINATION"
    UNCHANGED = "UNCHANGED"
    REGRESSES = "REGRESSES"
    NO_RELEVANT_EXISTING_TEST = "NO_RELEVANT_EXISTING_TEST"


class VerificationCoverageGap(Enum):
    MISSING_BEHAVIOR_ASSERTION = "MISSING_BEHAVIOR_ASSERTION"
    OVERBROAD_VERIFICATION_FAILURE = "OVERBROAD_VERIFICATION_FAILURE"
    STALE_EXISTING_EXPECTATION = "STALE_EXISTING_EXPECTATION"
    CONFIGURE_BUILD_INFRA_FAILURE = "CONFIGURE/BUILD_INFRA_FAILURE"
    TASK_NOT_COVERED_BY_REPO_TESTS = "TASK_NOT_COVERED_BY_REPO_TESTS"
    UNKNOWN = "UNKNOWN"


class ReferenceRejectionReason(Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PREEXISTING_FAILURE = "PREEXISTING_FAILURE"
    STALE_TEST_EXPECTATION = "STALE_TEST_EXPECTATION"
    UNRELATED_FAILURE = "UNRELATED_FAILURE"
    VERIFICATION_ENVIRONMENT = "VERIFICATION_ENVIRONMENT"
    REFERENCE_INCOMPATIBLE_WITH_PROJECT_TEST = (
        "REFERENCE_INCOMPATIBLE_WITH_PROJECT_TEST"
    )
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class VerificationStepObservation:
    operation: str
    result: str
    exit_code: int | None
    outcome: str | None
    duration_seconds: float
    failure_fingerprint: str | None
    failure_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerificationCoverageObservation:
    task_id: str
    source_state: VerificationSourceState
    semantic_truth: bool
    configure_result: str
    build_result: str
    test_result: str
    final_verification_result: str
    failing_step: str | None
    attribution_metadata: str
    duration_seconds: float
    steps: tuple[VerificationStepObservation, ...]


@dataclass(frozen=True, slots=True)
class VerificationConfusionCounts:
    semantic_correct_verification_pass: int = 0
    semantic_correct_verification_fail: int = 0
    semantic_wrong_verification_pass: int = 0
    semantic_wrong_verification_fail: int = 0


@dataclass(frozen=True, slots=True)
class ExistingTestInventory:
    task_id: str
    test_path: str
    relevant_test_exists: bool
    behavior_directly_asserted: bool
    task_source_exercised: bool
    configured_plan_runs_test: bool
    selection_basis: str


@dataclass(frozen=True, slots=True)
class TargetedTestObservation:
    task_id: str
    test_path: str
    baseline_result: str
    reference_result: str
    wrong_result: str
    classification: VerificationAlignmentClass
    comparison: TargetedTestResult
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class VerificationAlignmentTaskResult:
    task_id: str
    baseline: VerificationCoverageObservation
    reference: VerificationCoverageObservation
    wrong: VerificationCoverageObservation
    classification: VerificationAlignmentClass
    confusion_counts: VerificationConfusionCounts
    repair_observability: tuple[RepairObservability, ...]
    test_inventory: ExistingTestInventory
    targeted_test: TargetedTestObservation
    coverage_gaps: tuple[VerificationCoverageGap, ...]
    reference_rejection: ReferenceRejectionReason


@dataclass(frozen=True, slots=True)
class ModelMutationAlignment:
    model_profile: str
    seed: int
    task_id: str
    verification_passed: bool
    semantic_passed: bool
    category: str


@dataclass(frozen=True, slots=True)
class VerificationAlignmentRun:
    suite: str
    suite_version: int
    schema_version: int
    repository_identity: str
    tasks: tuple[VerificationAlignmentTaskResult, ...]
    confusion_counts: VerificationConfusionCounts
    model_mutations: tuple[ModelMutationAlignment, ...]
    canonical_unchanged: bool


_TARGETED_TESTS = {
    "R01": "tests/test_retry.py",
    "R02": "tests/test_config.py",
    "R03": "cengine/tests/test_parser.c",
    "R04": "cengine/tests/test_quota.c",
    "R05": "tests/test_state.py",
    "R06": "tests/test_headers.py",
    "R07": "cengine/tests/test_window.c",
    "R08": "cengine/tests/test_status.c",
}


class VerificationAlignmentDefinition(Protocol):
    metadata: RealisticSemanticTask
    wrong: tuple[SetupReplacement, ...]


def run_verification_alignment_v1(
    root: Path,
    canonical_repository: Path,
    definitions: tuple[VerificationAlignmentDefinition, ...],
    *,
    model_mutations: tuple[ModelMutationAlignment, ...] = (),
) -> VerificationAlignmentRun:
    """Execute 24 independent full-plan states plus preselected visible T1 tests."""
    root.mkdir(parents=True, exist_ok=True)
    canonical_before = hash_workspace(canonical_repository)
    task_results: list[VerificationAlignmentTaskResult] = []
    for definition in definitions:
        metadata = definition.metadata
        wrong_mutation = definition.wrong
        task: RealWorldTask = metadata.production_task
        observations: dict[
            VerificationSourceState, VerificationCoverageObservation
        ] = {}
        targeted: dict[VerificationSourceState, tuple[str, float]] = {}
        with tempfile.TemporaryDirectory(
            prefix=f"forge-a48-{task.task_id.lower()}-", dir=root
        ) as temporary:
            temporary_root = Path(temporary)
            for state in VerificationSourceState:
                workspace = copy_repository(
                    canonical_repository,
                    temporary_root / state.value.casefold(),
                )
                _construct_state(
                    workspace,
                    canonical_repository,
                    task,
                    wrong_mutation,
                    state,
                )
                semantic = run_oracle(workspace, task.oracle_commands)
                expected = state is VerificationSourceState.REFERENCE
                if (semantic is EvaluationOutcome.PASS) != expected:
                    raise RuntimeError(
                        f"semantic integrity changed for {task.task_id} {state.value}"
                    )
                observations[state] = _run_full_verification(
                    task, workspace, state, expected
                )
                targeted[state] = _run_targeted_test(task.task_id, task, workspace)
        classification = classify_verification_alignment(
            observations[VerificationSourceState.BASELINE],
            observations[VerificationSourceState.REFERENCE],
            observations[VerificationSourceState.WRONG],
        )
        targeted_class = classify_result_pattern(
            targeted[VerificationSourceState.BASELINE][0],
            targeted[VerificationSourceState.REFERENCE][0],
            targeted[VerificationSourceState.WRONG][0],
        )
        targeted_comparison = compare_targeted_classification(
            classification, targeted_class
        )
        inventory = _inventory(task.task_id, canonical_repository)
        task_results.append(
            VerificationAlignmentTaskResult(
                task.task_id,
                observations[VerificationSourceState.BASELINE],
                observations[VerificationSourceState.REFERENCE],
                observations[VerificationSourceState.WRONG],
                classification,
                aggregate_confusion(tuple(observations.values())),
                tuple(repair_observability(item) for item in observations.values()),
                inventory,
                TargetedTestObservation(
                    task.task_id,
                    inventory.test_path,
                    targeted[VerificationSourceState.BASELINE][0],
                    targeted[VerificationSourceState.REFERENCE][0],
                    targeted[VerificationSourceState.WRONG][0],
                    targeted_class,
                    targeted_comparison,
                    sum(item[1] for item in targeted.values()),
                ),
                coverage_gaps(classification, observations),
                reference_rejection_reason(
                    observations[VerificationSourceState.REFERENCE]
                ),
            )
        )
    values = tuple(task_results)
    if canonical_before != hash_workspace(canonical_repository):
        raise RuntimeError("canonical realistic semantic repository changed")
    return VerificationAlignmentRun(
        VERIFICATION_ALIGNMENT_V1,
        VERIFICATION_ALIGNMENT_SUITE_VERSION,
        VERIFICATION_ALIGNMENT_SCHEMA_VERSION,
        _repository_identity(canonical_repository),
        values,
        aggregate_confusion(
            tuple(
                observation
                for task in values
                for observation in (task.baseline, task.reference, task.wrong)
            )
        ),
        model_mutations,
        True,
    )


def classify_verification_alignment(
    baseline: VerificationCoverageObservation,
    reference: VerificationCoverageObservation,
    wrong: VerificationCoverageObservation,
) -> VerificationAlignmentClass:
    return classify_result_pattern(
        baseline.final_verification_result,
        reference.final_verification_result,
        wrong.final_verification_result,
    )


def classify_result_pattern(
    baseline: str, reference: str, wrong: str
) -> VerificationAlignmentClass:
    values = {baseline, reference, wrong}
    if "unavailable" in values:
        return VerificationAlignmentClass.VERIFICATION_UNAVAILABLE
    if baseline == "pass":
        return VerificationAlignmentClass.BASELINE_ACCEPTING
    if reference != "pass":
        return VerificationAlignmentClass.REFERENCE_REJECTING
    if wrong == "pass":
        return VerificationAlignmentClass.REFERENCE_ACCEPTING_BUT_WRONG_ACCEPTING
    if baseline == wrong == "fail" and reference == "pass":
        return VerificationAlignmentClass.FULLY_DISCRIMINATING
    return VerificationAlignmentClass.PARTIALLY_DISCRIMINATING


def compare_targeted_classification(
    full: VerificationAlignmentClass,
    targeted: VerificationAlignmentClass,
) -> TargetedTestResult:
    """Compare preselected T1 discrimination with authoritative full-plan T0."""
    if (
        targeted is VerificationAlignmentClass.FULLY_DISCRIMINATING
        and full is not VerificationAlignmentClass.FULLY_DISCRIMINATING
    ):
        return TargetedTestResult.IMPROVES_DISCRIMINATION
    if (
        full is VerificationAlignmentClass.FULLY_DISCRIMINATING
        and targeted is not VerificationAlignmentClass.FULLY_DISCRIMINATING
    ):
        return TargetedTestResult.REGRESSES
    return TargetedTestResult.UNCHANGED


def aggregate_confusion(
    observations: tuple[VerificationCoverageObservation, ...],
) -> VerificationConfusionCounts:
    return VerificationConfusionCounts(
        sum(item.semantic_truth and _passed(item) for item in observations),
        sum(item.semantic_truth and not _passed(item) for item in observations),
        sum(not item.semantic_truth and _passed(item) for item in observations),
        sum(not item.semantic_truth and not _passed(item) for item in observations),
    )


def repair_observability(
    observation: VerificationCoverageObservation,
) -> RepairObservability:
    if observation.semantic_truth:
        return (
            RepairObservability.NO_REPAIR_NEEDED
            if _passed(observation)
            else RepairObservability.POTENTIALLY_HARMFUL_TRIGGER
        )
    return (
        RepairObservability.SEMANTIC_FAILURE_INVISIBLE
        if _passed(observation)
        else RepairObservability.LEGITIMATELY_TRIGGERABLE
    )


def coverage_gaps(
    classification: VerificationAlignmentClass,
    observations: Mapping[VerificationSourceState, VerificationCoverageObservation],
) -> tuple[VerificationCoverageGap, ...]:
    """Classify observed signal gaps without changing production verification."""
    gaps: list[VerificationCoverageGap] = []
    baseline = observations[VerificationSourceState.BASELINE]
    reference = observations[VerificationSourceState.REFERENCE]
    wrong = observations[VerificationSourceState.WRONG]
    if _passed(baseline) or _passed(wrong):
        gaps.append(VerificationCoverageGap.MISSING_BEHAVIOR_ASSERTION)
    if reference.final_verification_result == "fail":
        gaps.append(VerificationCoverageGap.OVERBROAD_VERIFICATION_FAILURE)
    if any(
        item.failing_step in {"configure", "build"}
        and item.final_verification_result != "pass"
        for item in observations.values()
    ):
        gaps.append(VerificationCoverageGap.CONFIGURE_BUILD_INFRA_FAILURE)
    if not gaps and classification not in {
        VerificationAlignmentClass.FULLY_DISCRIMINATING,
        VerificationAlignmentClass.VERIFICATION_UNAVAILABLE,
    }:
        gaps.append(VerificationCoverageGap.UNKNOWN)
    return tuple(gaps)


def reference_rejection_reason(
    reference: VerificationCoverageObservation,
) -> ReferenceRejectionReason:
    """Conservatively categorize a known-good reference rejection."""
    if reference.final_verification_result == "pass":
        return ReferenceRejectionReason.NOT_APPLICABLE
    if reference.final_verification_result == "unavailable":
        return ReferenceRejectionReason.VERIFICATION_ENVIRONMENT
    if reference.failing_step in {"configure", "build"}:
        return ReferenceRejectionReason.UNRELATED_FAILURE
    if reference.failing_step == "test":
        return ReferenceRejectionReason.REFERENCE_INCOMPATIBLE_WITH_PROJECT_TEST
    return ReferenceRejectionReason.UNKNOWN


def map_model_mutations(
    runs: tuple[RealisticSemanticRun, ...],
) -> tuple[ModelMutationAlignment, ...]:
    """Map only durable executed historical mutations; never rerun a model."""
    mapped: list[ModelMutationAlignment] = []
    for run in runs:
        for result in run.results:
            if not result.transaction_executed:
                continue
            verification = result.verification_status in {"pass", "passed"}
            semantic = result.semantic_oracle == "PASS"
            mapped.append(
                ModelMutationAlignment(
                    result.model_profile,
                    result.seed,
                    result.task_id,
                    verification,
                    semantic,
                    f"V_{'PASS' if verification else 'FAIL'} / "
                    f"S_{'PASS' if semantic else 'FAIL'}",
                )
            )
    return tuple(mapped)


def verification_alignment_to_dict(
    run: VerificationAlignmentRun,
) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    payload = convert(asdict(run))
    assert isinstance(payload, dict)
    return payload


def write_verification_alignment_json(
    run: VerificationAlignmentRun, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(verification_alignment_to_dict(run), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _construct_state(
    workspace: Path,
    canonical: Path,
    task: RealWorldTask,
    wrong_mutation: tuple[SetupReplacement, ...],
    state: VerificationSourceState,
) -> None:
    if state is VerificationSourceState.REFERENCE:
        return
    apply_task_setup(workspace, task.setup)
    if state is VerificationSourceState.WRONG:
        if wrong_mutation:
            apply_task_setup(workspace, wrong_mutation)
        else:
            test_path = task.expected_changed_paths[1]
            (workspace / test_path).write_bytes((canonical / test_path).read_bytes())


def _run_full_verification(
    task: RealWorldTask,
    workspace: Path,
    state: VerificationSourceState,
    semantic_truth: bool,
) -> VerificationCoverageObservation:
    commands = ProjectCommands(
        ProjectCommand(task.build_command, 300),
        ProjectCommand(task.test_command, 300),
        task.verification_plan,
        ProjectCommand(task.configure_command, 300),
        task.execution_isolation,
    )
    steps: list[VerificationStepObservation] = []
    assert commands.verification_plan is not None
    for index, operation in enumerate(commands.verification_plan.steps):
        step = _execute_project_step(
            commands, workspace, operation, f"a48-full-{index}"
        )
        steps.append(step)
        if step.result != "pass":
            break
    by_operation = {item.operation: item.result for item in steps}
    final = (
        "pass"
        if len(steps) == len(commands.verification_plan.steps)
        and all(item.result == "pass" for item in steps)
        else "unavailable"
        if any(item.result == "unavailable" for item in steps)
        else "fail"
    )
    failing = next((item.operation for item in steps if item.result != "pass"), None)
    return VerificationCoverageObservation(
        task.task_id,
        state,
        semantic_truth,
        by_operation.get("configure", "not_run"),
        by_operation.get("build", "not_run"),
        by_operation.get("test", "not_run"),
        final,
        failing,
        "independent_evaluator_state; no same-workspace A36 attribution",
        sum(item.duration_seconds for item in steps),
        tuple(steps),
    )


def _execute_project_step(
    commands: ProjectCommands,
    workspace: Path,
    tool_name: str,
    invocation_id: str,
) -> VerificationStepObservation:
    registry = create_assist_repository_registry(commands)
    executor = ToolExecutor(registry, resolve_interaction_policy(AutonomyMode.REPAIR))
    context = ExecutionContext(workspace)
    invocation = ToolInvocation(invocation_id, tool_name, {})
    tool = registry.get(tool_name)
    assert isinstance(tool, ProjectCommandTool)
    prepared = tool.prepare(context)
    result = executor.execute(
        invocation,
        replace(context, prepared_project_command=prepared),
        approval=InvocationApproval.for_invocation(invocation),
    )
    output = result.output if isinstance(result.output, Mapping) else {}
    status = (
        "pass"
        if result.status is ToolResultStatus.SUCCESS
        else "fail"
        if output.get("outcome") in {"nonzero_exit", "timeout"}
        else "unavailable"
    )
    operation = tool_name.removeprefix("project.")
    return VerificationStepObservation(
        operation,
        status,
        output.get("exit_code") if isinstance(output.get("exit_code"), int) else None,
        output.get("outcome") if isinstance(output.get("outcome"), str) else None,
        float(output.get("duration_seconds", 0.0)),
        failure_fingerprint(operation, output, workspace) if status == "fail" else None,
        bounded_failure_lines(output) if status == "fail" else (),
    )


def _run_targeted_test(
    task_id: str, task: RealWorldTask, workspace: Path
) -> tuple[str, float]:
    test_path = _TARGETED_TESTS[task_id]
    if test_path.endswith(".py"):
        argv = (
            sys.executable,
            "-c",
            f"import runpy; runpy.run_path({test_path!r})",
        )
    else:
        argv = (
            str((workspace / "build" / f"test_{Path(test_path).stem[5:]}").resolve()),
        )
    commands = ProjectCommands(
        test=ProjectCommand(argv, 300), execution_isolation=task.execution_isolation
    )
    step = _execute_project_step(commands, workspace, "project.test", "a48-targeted")
    return step.result, step.duration_seconds


def _inventory(task_id: str, repository: Path) -> ExistingTestInventory:
    path = _TARGETED_TESTS[task_id]
    return ExistingTestInventory(
        task_id,
        path,
        (repository / path).is_file(),
        True,
        True,
        True,
        "preselected from the task-named behavior and its visible source/test pair",
    )


def _passed(observation: VerificationCoverageObservation) -> bool:
    return observation.final_verification_result == "pass"


def _repository_identity(repository: Path) -> str:
    import hashlib

    return hashlib.sha256(repr(hash_workspace(repository)).encode()).hexdigest()
