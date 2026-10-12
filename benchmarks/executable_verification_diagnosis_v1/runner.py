"""Evaluator-only A90 candidate-check simulation over production workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import run_cell as run_a78
from benchmarks.realistic_coding_v2.suite import FrozenTask
from benchmarks.semantic_obligation_completeness_v1.checks import check
from forge.models import Model, MutationRepresentationPolicy

from .suite import OBLIGATIONS, RUN_ID, VerificationObligation


@dataclass(frozen=True, slots=True)
class CheckResult:
    obligation_id: str
    obligation_identity: str
    component: str
    frozen_coverage: str
    observed_coverage: str
    derivability: str
    source_class: str
    outcome: str


def _evaluate(
    obligations: tuple[VerificationObligation, ...], workspace: Path
) -> tuple[CheckResult, ...]:
    values = []
    for item in obligations:
        outcome = None if item.check_id is None else check(item.check_id, workspace)
        values.append(
            CheckResult(
                item.obligation_id,
                item.identity,
                item.component,
                item.coverage.value,
                item.coverage.value,
                item.derivability.value,
                item.source_class,
                "CHECK_UNAVAILABLE"
                if outcome is None
                else "CHECK_PASS"
                if outcome
                else "CHECK_FAIL",
            )
        )
    return tuple(values)


def _failure_category(
    checks: tuple[CheckResult, ...], verification_pass: bool, semantic_pass: bool
) -> str:
    if semantic_pass:
        return "PASS"
    failed = [item for item in checks if item.outcome == "CHECK_FAIL"]
    if verification_pass and failed:
        if any(
            item.frozen_coverage == "VERIFICATION_PARTIALLY_COVERS" for item in failed
        ):
            return "PARTIALLY_COVERED_BEHAVIOR"
        if any(item.frozen_coverage == "VERIFICATION_COVERS" for item in failed):
            return "VERIFICATION_ASSERTION_TOO_WEAK"
        return "UNCOVERED_BEHAVIOR"
    if verification_pass:
        return "VERIFICATION_PASSED_WRONG_BEHAVIOR"
    if failed:
        return "MODEL_FAILED_DESPITE_GOOD_VERIFICATION"
    return "OTHER"


def run_cell(
    definition: FrozenTask,
    backend: Model,
    *,
    selected_profile: str,
    artifact: str,
    artifact_size: int,
    model_config_identity: str,
    repository_identity: str,
    corpus_identity: str,
    representation: MutationRepresentationPolicy,
) -> dict[str, object]:
    obligations = OBLIGATIONS[definition.task_id]
    primary: list[tuple[CheckResult, ...]] = []
    repaired: list[tuple[CheckResult, ...]] = []

    def inspect_primary(workspace: Path) -> None:
        primary.append(_evaluate(obligations, workspace))

    def inspect_final(workspace: Path) -> None:
        repaired.append(_evaluate(obligations, workspace))

    inner = run_a78(
        definition,
        selected_profile,
        backend,
        artifact=artifact,
        artifact_size=artifact_size,
        model_config_identity=model_config_identity,
        repository_identity=repository_identity,
        corpus_identity=corpus_identity,
        representation=representation,
        run_identity=RUN_ID,
        primary_workspace_callback=inspect_primary,
        final_workspace_callback=inspect_final,
    )
    checks = (
        primary[0]
        if primary
        else tuple(
            CheckResult(
                item.obligation_id,
                item.identity,
                item.component,
                item.coverage.value,
                item.coverage.value,
                item.derivability.value,
                item.source_class,
                "CHECK_UNAVAILABLE",
            )
            for item in obligations
        )
    )
    verification_pass = bool(inner.proposal.get("verification_pass", False))
    aligned = tuple(
        CheckResult(
            **{
                **asdict(item),
                "observed_coverage": (
                    "VERIFICATION_DOES_NOT_COVER"
                    if verification_pass and item.outcome == "CHECK_FAIL"
                    else item.observed_coverage
                ),
            }
        )
        for item in checks
    )
    semantic_pass = bool(inner.primary_semantic_pass)
    return {
        "task_id": definition.task_id,
        "operation_class": definition.operation_class.value,
        "language": definition.language,
        "verification_inventory": {
            "steps": definition.production_task.verification_plan.steps
            if definition.production_task.verification_plan
            else (),
            "source_class": "TRUSTED_PROJECT_CONFIGURATION",
        },
        "checks": tuple(asdict(item) for item in aligned),
        "existing_verification_pass": verification_pass,
        "candidate_checks_pass": all(item.outcome == "CHECK_PASS" for item in aligned),
        "candidate_check_failures": sum(
            item.outcome == "CHECK_FAIL" for item in aligned
        ),
        "semantic_pass": semantic_pass,
        "failure_category": _failure_category(
            aligned, verification_pass, semantic_pass
        ),
        "repair_attempted": inner.repair is not None,
        "repair_checks": tuple(asdict(item) for item in repaired[-1])
        if repaired
        else (),
        "repair_semantic_pass": (
            bool(inner.repair.get("semantic_pass", False)) if inner.repair else None
        ),
        "workflow_terminal": inner.workflow_trace.get("final_status"),
        "generation_latency_seconds": inner.median_generation_latency_seconds,
    }
