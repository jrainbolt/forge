"""A84 non-blocking readiness execution and offline policy simulation."""

from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.production_detectable_completeness_v1.checker import (
    NOT_CHECKABLE,
    evaluate_detailed,
)
from benchmarks.production_detectable_completeness_v1.suite import (
    CheckKind,
    CompletenessCheck,
)
from benchmarks.realistic_coding_v2.suite import FrozenTask
from benchmarks.structural_completeness_shadow_v1.runner import (
    ShadowCell,
    ShadowCheckResult,
    ShadowSnapshot,
    decide,
)
from benchmarks.structural_completeness_shadow_v1.runner import (
    run_cell as run_shadow_cell,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.models import Model, MutationRepresentationPolicy

from .suite import RUN_ID, SUITE, VERSION, ReadinessCase


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    suite: str
    suite_version: int
    run_identity: str
    corpus_identity: str
    case_id: str
    task_id: str | None
    model_call: bool
    structural_truth: str
    primary_shadow: dict[str, object]
    offline_policy_action: str
    transaction_applied: bool | None
    verification_pass: bool | None
    semantic_pass: bool | None
    repair_shadow: dict[str, object] | None
    repair_change: str
    workflow_attempt: dict[str, object] | None
    mutation_requests: tuple[dict[str, object], ...]
    proposal_request_lineage: dict[str, str]
    proposal_transaction_lineage: dict[str, tuple[str, ...]]


def proposed_policy(decision: str, operational_failure: bool = False) -> str:
    if operational_failure:
        return "FAIL_OPEN_CONTINUE_WITH_DIAGNOSTIC"
    return {
        "STRUCTURAL_SHADOW_PASS": "ALLOW",
        "STRUCTURAL_SHADOW_FAIL": "BLOCK",
        "STRUCTURAL_SHADOW_PARTIAL": "CONTINUE_WITH_DIAGNOSTIC",
        "STRUCTURAL_SHADOW_NOT_APPLICABLE": "ALLOW_NOT_APPLICABLE",
    }[decision]


def _operational_failure(snapshot: dict[str, object]) -> bool:
    checks = snapshot.get("checks", ())
    return any(
        isinstance(item, dict) and item.get("operational_failure") is not None
        for item in checks  # type: ignore[union-attr]
    )


def from_model(
    case: ReadinessCase,
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
) -> ReadinessResult:
    from benchmarks.structural_completeness_shadow_v1.suite import ShadowCase

    inner: ShadowCell = run_shadow_cell(
        ShadowCase(
            case.task_id or case.case_id,
            case.truth,
            case.checks,
            case.determinism_control,
        ),
        definition,
        backend,
        selected_profile=selected_profile,
        artifact=artifact,
        artifact_size=artifact_size,
        model_config_identity=model_config_identity,
        repository_identity=repository_identity,
        corpus_identity=corpus_identity,
        representation=representation,
        run_identity=RUN_ID,
    )
    primary = asdict(inner.primary_shadow)
    return ReadinessResult(
        SUITE,
        VERSION,
        RUN_ID,
        corpus_identity,
        case.case_id,
        case.task_id,
        True,
        case.truth,
        primary,
        proposed_policy(inner.primary_shadow.decision, _operational_failure(primary)),
        inner.transaction_applied,
        inner.verification_pass,
        inner.semantic_pass,
        asdict(inner.repair_shadow) if inner.repair_shadow else None,
        inner.repair_change,
        inner.workflow_attempt,
        inner.mutation_requests,
        inner.proposal_request_lineage,
        inner.proposal_transaction_lineage,
    )


def _control_check(scenario: str) -> tuple[CompletenessCheck, str]:
    if scenario == "PARSER_FAILURE":
        return (
            CompletenessCheck(
                "OP01-parser",
                CheckKind.REQUIRED_SYMBOL_PRESENCE,
                "module.py",
                None,
                "serve",
                None,
                ("module.py",),
            ),
            "def broken(:\n",
        )
    return (
        CompletenessCheck(
            "OP02-language",
            CheckKind.REQUIRED_SYMBOL_PRESENCE,
            "module.rs",
            None,
            "serve",
            None,
            ("module.rs",),
        ),
        "fn serve() {}\n",
    )


def run_control(case: ReadinessCase, *, corpus_identity: str) -> ReadinessResult:
    scenario = case.control_scenario or "NOT_APPLICABLE"
    if scenario == "NOT_APPLICABLE":
        snapshot = ShadowSnapshot(
            (), "STRUCTURAL_SHADOW_NOT_APPLICABLE", 0.0, 0, True, 0.0
        )
    elif scenario == "CHECKER_INTERNAL_ERROR":
        snapshot = ShadowSnapshot(
            (
                ShadowCheckResult(
                    "OP03-internal",
                    "a" * 64,
                    "REQUIRED_SYMBOL_PRESENCE",
                    "b" * 64,
                    "c" * 64,
                    NOT_CHECKABLE,
                    0.0,
                    False,
                    "CHECKER_INTERNAL_ERROR",
                    "NOT_INVOKED",
                ),
            ),
            "STRUCTURAL_SHADOW_PARTIAL",
            0.0,
            0,
            True,
            0.0,
        )
    else:
        check, source = _control_check(scenario)
        with tempfile.TemporaryDirectory(prefix="forge-a84-control-") as name:
            workspace = Path(name)
            (workspace / check.source_path).write_text(source)
            started = time.perf_counter()
            detail = evaluate_detailed(check, workspace)
            elapsed = time.perf_counter() - started
            repeat = evaluate_detailed(check, workspace)
        result = ShadowCheckResult(
            check.check_id,
            check.identity,
            check.kind.value,
            "d" * 64,
            "e" * 64,
            detail.status,
            elapsed,
            False,
            detail.operational_failure,
            detail.subprocess_exit,
        )
        snapshot = ShadowSnapshot(
            (result,),
            decide((result,)),
            elapsed,
            0,
            (detail.status, detail.operational_failure)
            == (repeat.status, repeat.operational_failure),
            0.0,
        )
    payload = asdict(snapshot)
    return ReadinessResult(
        SUITE,
        VERSION,
        RUN_ID,
        corpus_identity,
        case.case_id,
        None,
        False,
        case.truth,
        payload,
        proposed_policy(snapshot.decision, _operational_failure(payload)),
        None,
        None,
        None,
        None,
        "REPAIR_NOT_APPLICABLE",
        None,
        (),
        {},
        {},
    )


def commit(path: Path, result: ReadinessResult) -> None:
    payload = asdict(result)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A84 result is not source-free")
    atomic_checkpoint(path, payload)
