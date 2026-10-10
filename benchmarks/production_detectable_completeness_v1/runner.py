"""A82 structural completeness execution over unchanged production workflows."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import run_cell as run_a78_cell
from benchmarks.realistic_coding_v2.suite import FrozenTask
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.models import Model, MutationRepresentationPolicy

from .checker import AMBIGUOUS, COMPLETE, INCOMPLETE, NOT_CHECKABLE, evaluate
from .suite import RUN_ID, SUITE, VERSION, CompletenessCheck, CorpusCase


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    check_identity: str
    kind: str
    obligation_identity: str
    authorized_scope_identity: str
    status: str


@dataclass(frozen=True, slots=True)
class StructuralCell:
    suite: str
    suite_version: int
    run_identity: str
    corpus_identity: str
    task_id: str
    corpus_role: str
    selected_profile: str
    primary_checks: tuple[CheckResult, ...]
    primary_structural_classification: str
    verification_pass: bool
    semantic_pass: bool
    repair_checks: tuple[CheckResult, ...] | None
    repair_structural_classification: str | None
    repair_verification_pass: bool | None
    repair_semantic_pass: bool | None
    workflow_attempt: dict[str, object]
    mutation_requests: tuple[dict[str, object], ...]
    proposal_request_lineage: dict[str, str]
    proposal_transaction_lineage: dict[str, tuple[str, ...]]


def evaluate_checks(
    checks: tuple[CompletenessCheck, ...], workspace: Path
) -> tuple[CheckResult, ...]:
    return tuple(
        CheckResult(
            check.check_id,
            check.identity,
            check.kind.value,
            hashlib.sha256(f"{check.check_id}:obligation".encode()).hexdigest(),
            hashlib.sha256(
                "\0".join(sorted(check.authorized_scope)).encode()
            ).hexdigest(),
            evaluate(check, workspace),
        )
        for check in checks
    )


def classify(results: tuple[CheckResult, ...]) -> str:
    if not results:
        return "NO_STRUCTURAL_CHECKS_APPLICABLE"
    if any(item.status == INCOMPLETE for item in results):
        return "STRUCTURAL_OBLIGATION_MISSING"
    if any(item.status in {NOT_CHECKABLE, AMBIGUOUS} for item in results):
        return "STRUCTURAL_CHECK_PARTIAL"
    if all(item.status == COMPLETE for item in results):
        return "STRUCTURALLY_COMPLETE"
    return "STRUCTURAL_CHECK_PARTIAL"


def run_cell(
    case: CorpusCase,
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
) -> StructuralCell:
    primary: list[tuple[CheckResult, ...]] = []
    final: list[tuple[CheckResult, ...]] = []

    def inspect_primary(workspace: Path) -> None:
        primary.append(evaluate_checks(case.checks, workspace))

    def inspect_final(workspace: Path) -> None:
        final.append(evaluate_checks(case.checks, workspace))

    inner = run_a78_cell(
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
    primary_results = (
        primary[0]
        if primary
        else tuple(
            CheckResult(
                check.check_id,
                check.identity,
                check.kind.value,
                hashlib.sha256(f"{check.check_id}:obligation".encode()).hexdigest(),
                hashlib.sha256(
                    "\0".join(sorted(check.authorized_scope)).encode()
                ).hexdigest(),
                NOT_CHECKABLE,
            )
            for check in case.checks
        )
    )
    final_results = final[-1] if final else primary_results
    repair = inner.repair
    return StructuralCell(
        SUITE,
        VERSION,
        RUN_ID,
        corpus_identity,
        definition.task_id,
        case.role,
        selected_profile,
        primary_results,
        classify(primary_results),
        bool(inner.proposal["verification_pass"]),
        inner.primary_semantic_pass,
        final_results if repair is not None else None,
        classify(final_results) if repair is not None else None,
        bool(repair["verification_pass"]) if repair is not None else None,
        bool(repair["semantic_pass"]) if repair is not None else None,
        inner.workflow_attempt,
        inner.mutation_requests,
        inner.proposal_request_lineage,
        inner.proposal_transaction_lineage,
    )


def commit_cell(path: Path, cell: StructuralCell) -> None:
    payload = asdict(cell)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A82 result is not source-free")
    atomic_checkpoint(path, payload)
