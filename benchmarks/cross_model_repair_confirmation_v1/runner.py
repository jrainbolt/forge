"""A59 predeclaration, exactly-once primary capture, and repair comparison."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.cross_model_repair_confirmation_v1.suite import (
    HISTORICAL_EXCLUSION,
    HYPOTHESIS,
    PRIMARY_PROFILES,
    SUITE,
    THRESHOLD,
    VERSION,
    definition_identity,
    integrity,
    tasks,
)
from benchmarks.realistic_coding_v2.runner import _atomic_json, snapshot
from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.repair_framing_v1.runner import (
    CaptureModel,
    PrimaryRecord,
    _restore_response,
    _step_improved,
    _usage,
    load_primary,
    tree_hash,
)
from benchmarks.repair_framing_v1.suite import Case
from forge.evaluation.realworld import EvaluationOutcome, RealWorldEvaluationRunner
from forge.models import Model, MutationRepresentationPolicy


@dataclass(frozen=True, slots=True)
class RepairCell:
    suite: str
    schema_version: int
    case_id: str
    task_id: str
    primary_profile: str
    repair_profile: str
    condition: str
    seed: int
    model_artifact: str
    representation: str
    manifest_hash: str
    start_hash: str
    evidence_hash: str
    authorized_paths: tuple[str, ...]
    authority_sufficient: str
    schema_valid: bool
    authorized_path_set: bool
    material_mutation: bool
    transaction_executed: bool
    verification_improved: bool | None
    verification_executed: bool
    verification_passed: bool
    semantic_passed: bool
    semantic_recovered: bool
    mutation_hash: str | None
    failure_taxonomy: str
    input_tokens: int | None
    output_tokens: int | None
    model_calls: int
    generation_seconds: float
    verification_seconds: float
    total_seconds: float


def candidate_cases() -> tuple[Case, ...]:
    values = []
    index = 1
    for profile in PRIMARY_PROFILES:
        for definition in tasks():
            values.append(Case(f"QK{index:02d}", profile, definition))
            index += 1
    return tuple(values)


def identity(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def freeze_plan(
    root: Path,
    artifacts: dict[str, str],
    representations: dict[str, MutationRepresentationPolicy],
) -> dict[str, object]:
    checks = integrity()
    if not all(item["eligible"] for item in checks):
        raise ValueError("A59 frozen task integrity failed")
    payload: dict[str, object] = {
        "suite": SUITE,
        "schema_version": VERSION,
        "definition_identity": definition_identity(),
        "hypothesis": HYPOTHESIS,
        "routing_threshold": THRESHOLD,
        "historical_exclusion": HISTORICAL_EXCLUSION,
        "eligibility": (
            "primary transaction; verification fail; semantic fail; "
            "not preexisting/unrelated; reference within repair authority"
        ),
        "candidates": [
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "task_version": case.definition.version,
                "language": case.definition.language,
                "operation_class": case.definition.operation_class.value,
                "primary_profile": case.profile,
                "seed": case.seed,
                "representation": representations[case.profile].value,
                "primary_model_artifact": artifacts[case.profile],
            }
            for case in candidate_cases()
        ],
        "integrity": checks,
        "repair_profiles": ("same-model", "qwen-large"),
        "model_artifacts": artifacts,
    }
    path = root / "plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("A59 predeclared plan changed")
    else:
        _atomic_json(path, payload)
    return payload


def freeze_cases(root: Path, plan: dict[str, object]) -> dict[str, object]:
    records = [load_primary(root, case)[0] for case in candidate_cases()]
    eligible = [record for record in records if record.eligible]
    payload: dict[str, object] = {
        "suite": SUITE,
        "schema_version": VERSION,
        "plan_hash": identity(plan),
        "eligible_count": len(eligible),
        "records": [asdict(record) for record in records],
        "scored_case_ids": [record.case_id for record in eligible],
    }
    path = root / "results" / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("A59 primary-result manifest changed")
    else:
        _atomic_json(path, payload)
    if len(eligible) < 8:
        raise ValueError(f"A59_BLOCKED_ELIGIBLE_CASES_{len(eligible)}")
    return payload


def cell_path(root: Path, case: Case, condition: str) -> Path:
    return root / "results" / f"{case.case_id}-{condition}.json"


def load_cell(
    root: Path, case: Case, condition: str, manifest_hash: str
) -> RepairCell | None:
    path = cell_path(root, case, condition)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    value["authorized_paths"] = tuple(value["authorized_paths"])
    cell = RepairCell(**value)
    if (
        cell.case_id != case.case_id
        or cell.condition != condition
        or cell.manifest_hash != manifest_hash
    ):
        raise ValueError("A59 committed repair identity mismatch")
    return cell


def _taxonomy(
    schema: bool,
    authorized: bool,
    transaction: bool,
    verification_executed: bool,
    verification_passed: bool,
    semantic: bool,
) -> str:
    if not schema:
        return "NO_VALID_PROPOSAL"
    if not authorized:
        return "UNAUTHORIZED_PATH"
    if not transaction:
        return "NO_CHANGE"
    if not verification_executed:
        return "STRUCTURAL_FAILURE"
    if verification_passed and semantic:
        return "SEMANTIC_RECOVERED"
    if verification_passed:
        return "VERIFICATION_PASS_SEMANTIC_FAIL"
    return "VERIFICATION_STILL_FAIL"


def run_repair(
    root: Path,
    case: Case,
    record: PrimaryRecord,
    replay: dict[str, object],
    backend: Model,
    repair_profile: str,
    artifact: str,
    representation: MutationRepresentationPolicy,
    condition: str,
    manifest_hash: str,
) -> RepairCell:
    existing = load_cell(root, case, condition, manifest_hash)
    if existing is not None:
        return existing
    started = time.perf_counter()
    prefix = tuple(_restore_response(item, backend) for item in replay["responses"])
    recorder = CaptureModel(backend, prefix=prefix)
    starts: list[str] = []

    def on_primary(_task, workspace: Path) -> None:  # type: ignore[no-untyped-def]
        value = tree_hash(workspace)
        starts.append(value)
        if value != record.post_primary_hash:
            raise ValueError("A59 repair start differs from frozen primary")

    raw = (
        RealWorldEvaluationRunner(
            record.profile,
            recorder,
            REPOSITORY,
            mutation_representation=representation,
            primary_mutation_callback=on_primary,
        )
        .run((case.definition.production_task,), snapshot(record.post_primary_hash))
        .results[0]
    )
    metrics = raw.metrics
    schema = bool(metrics.repair_preview_created or metrics.repair_mutation_executed)
    transaction = bool(metrics.repair_mutation_executed)
    verification_executed = bool(metrics.reverification_executed)
    verification_passed = metrics.reverification_result in {"pass", "passed"}
    semantic = raw.oracle is EvaluationOutcome.PASS
    authorized = bool(starts) and not raw.unexpected_paths
    repair_text = tuple(item.text for item in recorder.responses[len(prefix) :])
    mutation_hash = identity(repair_text) if schema else None
    input_tokens, output_tokens = _usage(recorder, len(prefix))
    total = (
        time.perf_counter() - recorder.repair_started
        if recorder.repair_started is not None
        else time.perf_counter() - started
    )
    cell = RepairCell(
        SUITE,
        VERSION,
        case.case_id,
        case.task_id,
        record.profile,
        repair_profile,
        condition,
        case.seed,
        artifact,
        representation.value,
        manifest_hash,
        record.post_primary_hash,
        record.evidence_hash,
        record.authorized_paths,
        record.authority_sufficient,
        schema,
        authorized,
        transaction,
        transaction,
        _step_improved(record.failure_step, metrics.verification_plan_failed_step),
        verification_executed,
        verification_passed,
        semantic,
        transaction and verification_executed and verification_passed and semantic,
        mutation_hash,
        _taxonomy(
            schema,
            authorized,
            transaction,
            verification_executed,
            verification_passed,
            semantic,
        ),
        input_tokens,
        output_tokens,
        recorder.backend_calls,
        round(recorder.generation_seconds, 3),
        round(metrics.verification_plan_duration, 3),
        round(total, 3),
    )
    _atomic_json(cell_path(root, case, condition), asdict(cell))
    return cell
