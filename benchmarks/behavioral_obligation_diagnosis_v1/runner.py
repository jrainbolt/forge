"""A76 execution wrapper over the accepted proposal-local A75 runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.behavioral_obligation_diagnosis_v1.suite import (
    RUN_ID,
    SCHEMA_VERSION,
    SEED,
    SUITE,
    VERSION,
    Condition,
    DiagnosticCase,
    Obligation,
)
from benchmarks.grounded_mutation_contract_v1.runner import run_cell as run_a75_cell
from benchmarks.grounded_mutation_contract_v1.suite import Condition as A75Condition
from benchmarks.grounded_mutation_planning_v1.suite import PlanningCase
from benchmarks.realistic_coding_v2.suite import FrozenTask
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.models import Model, MutationRepresentationPolicy


@dataclass(frozen=True, slots=True)
class CellResult:
    suite: str
    run_identity: str
    suite_version: int
    schema_version: int
    cell_id: str
    case_id: str
    condition: str
    profile: str
    task_id: str
    operation_class: str
    seed: int
    corpus_identity: str
    repository_identity: str
    model_artifact: str
    model_config_identity: str
    goal_id: str | None
    goal_identity: str | None
    oracle_identity: str
    full_task_oracle_identity: str
    authority_paths: tuple[str, ...]
    paired_input_identity: dict[str, object] | None
    mutation_requests: tuple[dict[str, object], ...]
    proposal_request_lineage: dict[str, str]
    proposal_transaction_lineage: dict[str, tuple[str, ...]]
    proposal: dict[str, object]
    obligation_oracle_pass: bool | None
    full_task_semantic_pass: bool
    repair: dict[str, object] | None
    failure: str
    workflow_trace: dict[str, object]


def run_cell(
    case: DiagnosticCase,
    condition: Condition,
    definition: FrozenTask,
    backend: Model,
    *,
    obligation: Obligation | None,
    artifact: str,
    model_config_identity: str,
    repository_identity: str,
    corpus_identity: str,
    representation: MutationRepresentationPolicy,
) -> CellResult:
    if condition is Condition.F1 and (obligation is None or not obligation.isolatable):
        raise ValueError("F1 requires one isolatable production obligation")
    production = definition.production_task
    task = (
        production
        if condition is Condition.F0
        else replace(production, prompt=obligation.description)  # type: ignore[union-attr]
    )
    diagnostic_definition = replace(definition, production_task=task)
    planning_case = PlanningCase(
        case.case_id, case.profile, case.task_id, case.rationale
    )
    inner = run_a75_cell(
        planning_case,
        A75Condition.M0,
        diagnostic_definition,
        backend,
        artifact=artifact,
        model_config_identity=model_config_identity,
        repository_identity=repository_identity,
        corpus_identity=corpus_identity,
        representation=representation,
    )
    full_oracle = _full_oracle(definition)
    goal_identity = (
        __import__("hashlib").sha256(obligation.description.encode()).hexdigest()
        if obligation
        else None
    )
    obligation_pass = inner.primary.semantic_pass if obligation else None
    return CellResult(
        SUITE,
        RUN_ID,
        VERSION,
        SCHEMA_VERSION,
        f"{case.case_id}-{condition.name}"
        + (f"-{obligation.goal_id}" if obligation else ""),
        case.case_id,
        condition.value,
        case.profile,
        case.task_id,
        case.operation_class,
        SEED,
        corpus_identity,
        repository_identity,
        artifact,
        model_config_identity,
        obligation.goal_id if obligation else None,
        goal_identity,
        obligation.oracle_identity if obligation else full_oracle,
        full_oracle,
        task.allowed_paths,
        inner.paired_input_identity,
        inner.mutation_requests,
        inner.proposal_request_lineage,
        inner.proposal_transaction_lineage,
        asdict(inner.primary),
        obligation_pass,
        inner.primary.semantic_pass,
        asdict(inner.repair) if inner.repair else None,
        inner.failure,
        inner.workflow_trace,
    )


def _full_oracle(definition: FrozenTask) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(definition.production_task.oracle_commands).encode()
    ).hexdigest()


def checkpoint_path(root: Path, cell_id: str) -> Path:
    return root / "cells" / f"{cell_id}.json"


def commit_cell(path: Path, result: CellResult) -> None:
    payload = asdict(result)
    if not standard_result_is_source_free(payload):
        raise RuntimeError("A76 result is not source-free")
    atomic_checkpoint(path, payload)


def read_cell(path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    payload = resume_checkpoint(path)
    if payload is not None and any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"A76 checkpoint identity mismatch: {path.name}")
    return payload
