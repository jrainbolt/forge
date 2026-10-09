"""A81 obligation analysis over unchanged complete production workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import run_cell as run_a78_cell
from benchmarks.realistic_coding_v2.suite import FrozenTask, OperationClass
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.models import Model, MutationRepresentationPolicy

from .checks import check
from .suite import OBLIGATIONS, RUN_ID, SUITE, VERSION, Obligation

SATISFIED = "OBLIGATION_SATISFIED"
UNSATISFIED = "OBLIGATION_UNSATISFIED"
UNAVAILABLE = "OBLIGATION_CHECK_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ObligationResult:
    obligation_id: str
    obligation_identity: str
    origin: str
    component: str
    production_state: str
    satisfaction: str
    verification_coverage: str
    relationship: bool


@dataclass(frozen=True, slots=True)
class DiagnosticCell:
    suite: str
    suite_version: int
    run_identity: str
    corpus_identity: str
    task_id: str
    operation_class: str
    selected_profile: str
    primary_obligations: tuple[ObligationResult, ...]
    primary_completeness: str
    semantic_taxonomy: str
    relationship_classification: str
    verification_pass: bool
    full_semantic_pass: bool
    repair_obligations: tuple[ObligationResult, ...] | None
    repair_comparison: str
    repair_verification_pass: bool | None
    repair_semantic_pass: bool | None
    workflow_attempt: dict[str, object]
    mutation_requests: tuple[dict[str, object], ...]
    proposal_request_lineage: dict[str, str]
    proposal_transaction_lineage: dict[str, tuple[str, ...]]
    generation_calls: int
    median_generation_latency_seconds: float | None


def evaluate_obligations(
    obligations: tuple[Obligation, ...], workspace: Path
) -> tuple[ObligationResult, ...]:
    results = []
    for obligation in obligations:
        outcome = (
            None
            if obligation.check_id is None
            else check(obligation.check_id, workspace)
        )
        satisfaction = (
            UNAVAILABLE if outcome is None else SATISFIED if outcome else UNSATISFIED
        )
        results.append(
            ObligationResult(
                obligation.obligation_id,
                obligation.identity,
                obligation.origin.value,
                obligation.component,
                "OBLIGATION_PRESENT_IN_PRODUCTION_STATE",
                satisfaction,
                obligation.coverage.value,
                obligation.relationship,
            )
        )
    return tuple(results)


def classify_completeness(
    results: tuple[ObligationResult, ...], full_semantic_pass: bool
) -> str:
    missed = sum(item.satisfaction == UNSATISFIED for item in results)
    unavailable = any(item.satisfaction == UNAVAILABLE for item in results)
    if missed == 0 and full_semantic_pass and not unavailable:
        return "ALL_OBLIGATIONS_SATISFIED"
    if missed == 1:
        return "ONE_OBLIGATION_MISSED"
    if missed > 1:
        return "MULTIPLE_OBLIGATIONS_MISSED"
    if not full_semantic_pass and not unavailable:
        return "OBLIGATION_IMPLEMENTED_INCORRECTLY"
    return "UNKNOWN"


def align_verification_coverage(
    results: tuple[ObligationResult, ...], verification_pass: bool
) -> tuple[ObligationResult, ...]:
    """Classify demonstrated non-coverage from the verification actually run."""
    if not verification_pass:
        return results
    return tuple(
        replace(item, verification_coverage="VERIFICATION_DOES_NOT_COVER_OBLIGATION")
        if item.satisfaction == UNSATISFIED
        else item
        for item in results
    )


def classify_create(results: tuple[ObligationResult, ...]) -> str:
    missed = [item for item in results if item.satisfaction == UNSATISFIED]
    if not missed:
        return "PASS"
    ids = {item.obligation_id for item in missed}
    if any("interface" in item for item in ids):
        return "CREATE_INTERFACE_WRONG"
    if any(
        token in item
        for item in ids
        for token in ("negative", "nonpositive", "all-bytes")
    ):
        return "CREATE_EDGE_CASE_MISSED"
    if any(item.origin == "REQUIRED_INVARIANT" for item in missed):
        return "CREATE_INVARIANT_MISSED"
    return "CREATE_BEHAVIOR_WRONG"


def classify_mixed(results: tuple[ObligationResult, ...]) -> str:
    missed = [item for item in results if item.satisfaction == UNSATISFIED]
    if not missed:
        return "PASS"
    components = {item.component for item in missed}
    if "RELATIONSHIP" in components:
        return "EDIT_CREATE_RELATIONSHIP_MISSING"
    if components == {"EDIT"}:
        return "EDIT_OBLIGATION_WRONG"
    if components == {"CREATE"}:
        return "CREATE_OBLIGATION_WRONG"
    if any(item.origin == "REQUIRED_INVARIANT" for item in missed):
        return "SHARED_INVARIANT_MISSED"
    return "PARTIAL_BEHAVIOR"


def classify_relationship(results: tuple[ObligationResult, ...]) -> str:
    relationships = [item for item in results if item.relationship]
    if not relationships:
        return "NO_REQUIRED_RELATIONSHIP_IN_PRODUCTION_STATE"
    if any(item.satisfaction == UNAVAILABLE for item in relationships):
        return "RELATIONSHIP_CHECK_UNAVAILABLE"
    if any(item.satisfaction == UNSATISFIED for item in relationships):
        return "RELATIONSHIP_OBLIGATION_UNSATISFIED"
    return "RELATIONSHIP_OBLIGATION_SATISFIED"


def compare_repair(
    primary: tuple[ObligationResult, ...], final: tuple[ObligationResult, ...]
) -> str:
    before = {item.obligation_id: item.satisfaction for item in primary}
    after = {item.obligation_id: item.satisfaction for item in final}
    fixed = any(
        before[key] == UNSATISFIED and after[key] == SATISFIED for key in before
    )
    regressed = any(
        before[key] == SATISFIED and after[key] == UNSATISFIED for key in before
    )
    if fixed and regressed:
        return "REPAIR_FIXES_AND_REGRESSES_OBLIGATIONS"
    if fixed:
        return "REPAIR_FIXES_MISSED_OBLIGATION"
    if regressed:
        return "REPAIR_REGRESSES_SATISFIED_OBLIGATION"
    return "REPAIR_LEAVES_OBLIGATIONS_UNCHANGED"


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
) -> DiagnosticCell:
    obligations = OBLIGATIONS[definition.task_id]
    primary: list[tuple[ObligationResult, ...]] = []
    final: list[tuple[ObligationResult, ...]] = []

    def inspect_primary(workspace: Path) -> None:
        primary.append(evaluate_obligations(obligations, workspace))

    def inspect_final(workspace: Path) -> None:
        final.append(evaluate_obligations(obligations, workspace))

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
            ObligationResult(
                item.obligation_id,
                item.identity,
                item.origin.value,
                item.component,
                "OBLIGATION_PRESENT_IN_PRODUCTION_STATE",
                UNAVAILABLE,
                item.coverage.value,
                item.relationship,
            )
            for item in obligations
        )
    )
    verification_pass = bool(inner.proposal["verification_pass"])
    primary_results = align_verification_coverage(primary_results, verification_pass)
    final_results = final[-1] if final else primary_results
    repair_verification_pass = (
        bool(inner.repair["verification_pass"]) if inner.repair is not None else None
    )
    if repair_verification_pass is not None:
        final_results = align_verification_coverage(
            final_results, repair_verification_pass
        )
    taxonomy = (
        classify_create(primary_results)
        if definition.operation_class is OperationClass.CREATE
        else classify_mixed(primary_results)
        if definition.operation_class is OperationClass.MIXED_EDIT_CREATE
        else "CONTROL_PASS"
        if inner.primary_semantic_pass
        else "CONTROL_SEMANTIC_FAILURE"
    )
    return DiagnosticCell(
        SUITE,
        VERSION,
        RUN_ID,
        corpus_identity,
        definition.task_id,
        definition.operation_class.value,
        selected_profile,
        primary_results,
        classify_completeness(primary_results, inner.primary_semantic_pass),
        taxonomy,
        classify_relationship(primary_results),
        verification_pass,
        inner.primary_semantic_pass,
        final_results if inner.repair is not None else None,
        compare_repair(primary_results, final_results)
        if inner.repair is not None
        else "REPAIR_NOT_RUN",
        repair_verification_pass,
        bool(inner.repair["semantic_pass"]) if inner.repair is not None else None,
        inner.workflow_attempt,
        inner.mutation_requests,
        inner.proposal_request_lineage,
        inner.proposal_transaction_lineage,
        inner.generation_calls,
        inner.median_generation_latency_seconds,
    )


def commit_cell(path: Path, cell: DiagnosticCell) -> None:
    payload = asdict(cell)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A81 result is not source-free")
    atomic_checkpoint(path, payload)
