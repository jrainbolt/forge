"""A83 inert shadow observer over the normal post-mutation lifecycle."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import run_cell as run_a78_cell
from benchmarks.production_detectable_completeness_v1.checker import (
    AMBIGUOUS,
    COMPLETE,
    INCOMPLETE,
    NOT_CHECKABLE,
    evaluate,
)
from benchmarks.realistic_coding_v2.suite import FrozenTask
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.models import Model, MutationRepresentationPolicy

from .suite import RUN_ID, SUITE, VERSION, ShadowCase


@dataclass(frozen=True, slots=True)
class ShadowCheckResult:
    check_id: str
    check_identity: str
    check_kind: str
    obligation_identity: str
    scope_identity: str
    status: str
    latency_seconds: float
    compiler_subprocess: bool


@dataclass(frozen=True, slots=True)
class ShadowSnapshot:
    checks: tuple[ShadowCheckResult, ...]
    decision: str
    wall_time_seconds: float
    compiler_subprocesses: int
    deterministic_repeat: bool | None
    repeat_wall_time_seconds: float | None


@dataclass(frozen=True, slots=True)
class ShadowCell:
    suite: str
    suite_version: int
    run_identity: str
    corpus_identity: str
    task_id: str
    operation_class: str
    structural_truth: str
    selected_profile: str
    primary_shadow: ShadowSnapshot
    transaction_applied: bool
    verification_pass: bool
    semantic_pass: bool
    repair_shadow: ShadowSnapshot | None
    repair_change: str
    repair_verification_pass: bool | None
    repair_semantic_pass: bool | None
    workflow_attempt: dict[str, object]
    mutation_requests: tuple[dict[str, object], ...]
    proposal_request_lineage: dict[str, str]
    proposal_transaction_lineage: dict[str, tuple[str, ...]]


def decide(results: tuple[ShadowCheckResult, ...]) -> str:
    if not results:
        return "STRUCTURAL_SHADOW_NOT_APPLICABLE"
    if any(item.status == INCOMPLETE for item in results):
        return "STRUCTURAL_SHADOW_FAIL"
    if any(item.status in {NOT_CHECKABLE, AMBIGUOUS} for item in results):
        return "STRUCTURAL_SHADOW_PARTIAL"
    if all(item.status == COMPLETE for item in results):
        return "STRUCTURAL_SHADOW_PASS"
    return "STRUCTURAL_SHADOW_PARTIAL"


def _once(case: ShadowCase, workspace: Path) -> tuple[ShadowCheckResult, ...]:
    results = []
    for check in case.checks:
        started = time.perf_counter()
        compiler = (
            check.kind.value == "REQUIRED_COMPONENT_ROLE"
            and check.source_path.endswith(".c")
        )
        try:
            status = evaluate(check, workspace)
        except Exception:  # Shadow observation must never alter the workflow.
            status = NOT_CHECKABLE
        results.append(
            ShadowCheckResult(
                check.check_id,
                check.identity,
                check.kind.value,
                hashlib.sha256(f"{check.check_id}:obligation".encode()).hexdigest(),
                hashlib.sha256(
                    "\0".join(sorted(check.authorized_scope)).encode()
                ).hexdigest(),
                status,
                time.perf_counter() - started,
                compiler,
            )
        )
    return tuple(results)


def observe(case: ShadowCase, workspace: Path) -> ShadowSnapshot:
    started = time.perf_counter()
    try:
        results = _once(case, workspace)
        elapsed = time.perf_counter() - started
        repeated = None
        repeat_elapsed = None
        if case.determinism_control:
            repeat_started = time.perf_counter()
            repeat = _once(case, workspace)
            repeat_elapsed = time.perf_counter() - repeat_started
            repeated = tuple(
                (item.check_identity, item.status) for item in results
            ) == tuple((item.check_identity, item.status) for item in repeat)
        return ShadowSnapshot(
            results,
            decide(results),
            elapsed,
            sum(item.compiler_subprocess for item in results),
            repeated,
            repeat_elapsed,
        )
    except Exception:  # The outer boundary is deliberately non-interfering.
        return ShadowSnapshot(
            (),
            "STRUCTURAL_SHADOW_PARTIAL",
            time.perf_counter() - started,
            0,
            None,
            None,
        )


def compare_repair(primary: ShadowSnapshot, repair: ShadowSnapshot | None) -> str:
    if repair is None:
        return "REPAIR_NOT_RUN"
    rank = {
        "STRUCTURAL_SHADOW_FAIL": 0,
        "STRUCTURAL_SHADOW_PARTIAL": 1,
        "STRUCTURAL_SHADOW_NOT_APPLICABLE": 1,
        "STRUCTURAL_SHADOW_PASS": 2,
    }
    if rank[repair.decision] > rank[primary.decision]:
        return "STRUCTURAL_COMPLETENESS_IMPROVED"
    if rank[repair.decision] < rank[primary.decision]:
        return "STRUCTURAL_COMPLETENESS_REGRESSED"
    return "STRUCTURAL_COMPLETENESS_UNCHANGED"


def run_cell(
    case: ShadowCase,
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
) -> ShadowCell:
    primary: list[ShadowSnapshot] = []
    final: list[ShadowSnapshot] = []

    def inspect_primary(workspace: Path) -> None:
        primary.append(observe(case, workspace))

    def inspect_final(workspace: Path) -> None:
        final.append(observe(case, workspace))

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
    primary_snapshot = (
        primary[0]
        if primary
        else ShadowSnapshot((), "STRUCTURAL_SHADOW_NOT_APPLICABLE", 0.0, 0, None, None)
    )
    repair = inner.repair
    repair_snapshot = final[-1] if repair is not None and final else None
    return ShadowCell(
        SUITE,
        VERSION,
        RUN_ID,
        corpus_identity,
        definition.task_id,
        definition.operation_class.value,
        case.truth,
        selected_profile,
        primary_snapshot,
        bool(inner.proposal["transaction_applied"]),
        bool(inner.proposal["verification_pass"]),
        inner.primary_semantic_pass,
        repair_snapshot,
        compare_repair(primary_snapshot, repair_snapshot),
        bool(repair["verification_pass"]) if repair is not None else None,
        bool(repair["semantic_pass"]) if repair is not None else None,
        inner.workflow_attempt,
        inner.mutation_requests,
        inner.proposal_request_lineage,
        inner.proposal_transaction_lineage,
    )


def commit_cell(path: Path, cell: ShadowCell) -> None:
    payload = asdict(cell)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A83 result is not source-free")
    atomic_checkpoint(path, payload)
