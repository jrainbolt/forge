"""A58 manifest, production repair handoff, and source-free checkpoints."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.cross_model_repair_v1.suite import SUITE, VERSION, Pair, classify_case
from benchmarks.realistic_coding_v2.runner import _atomic_json, snapshot
from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.repair_framing_v1.runner import (
    CaptureModel,
    PrimaryRecord,
    _restore_response,
    _sha,
    _step_improved,
    _usage,
    load_primary,
    load_trial,
    tree_hash,
    workspace_path,
)
from forge.evaluation.realworld import EvaluationOutcome, RealWorldEvaluationRunner
from forge.models import Model, MutationRepresentationPolicy


@dataclass(frozen=True, slots=True)
class Cell:
    suite: str
    schema_version: int
    case_id: str
    task_id: str
    primary_profile: str
    repair_profile: str
    same_model: bool
    reused_a57: bool
    repair_model_artifact: str
    representation: str
    manifest_hash: str
    start_hash: str
    primary_mutation_hash: str
    evidence_hash: str
    authorized_paths: tuple[str, ...]
    authority_sufficient: str
    original_task_present: bool
    model_identity_injected: bool
    evidence_identical: bool
    schema_valid: bool
    authorized_path_set: bool
    material_mutation: bool
    transaction_executed: bool
    full_verification_executed: bool
    verification_improved: bool | None
    verification_passed: bool
    semantic_passed: bool
    semantic_recovered: bool
    failing_step: str | None
    failure_taxonomy: str
    mutation_hash: str | None
    mutation_shape: str
    input_tokens: int | None
    output_tokens: int | None
    model_calls: int
    generation_seconds: float
    verification_seconds: float
    total_seconds: float | None


def manifest_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def freeze_manifest(
    root: Path,
    a57_root: Path,
    pair_values: tuple[Pair, ...],
    artifacts: dict[str, str],
    representations: dict[str, MutationRepresentationPolicy],
) -> dict[str, object]:
    """Bind all case and repair-profile identities before any A58 repair."""
    a57_payload = json.loads(
        (a57_root / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    a57_hash = manifest_hash(a57_payload)
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for pair in pair_values:
        if pair.case.case_id in seen:
            continue
        seen.add(pair.case.case_id)
        record, replay = load_primary(a57_root, pair.case)
        if not record.eligible or record.authority_sufficient != "YES":
            raise ValueError("A58 requires eligible authority-sufficient A57 cases")
        if record.model_artifact != artifacts[record.profile]:
            raise ValueError("A57 primary model artifact differs from A58 catalog")
        if tree_hash(workspace_path(a57_root, pair.case)) != record.post_primary_hash:
            raise ValueError("A57 post-primary bytes changed")
        evidence = replay.get("verification_evidence")
        if not isinstance(evidence, str) or _sha(evidence) != record.evidence_hash:
            raise ValueError("A57 bounded verification evidence changed")
        records.append(
            {
                "case_id": record.case_id,
                "task_id": record.task_id,
                "task_version": record.task_version,
                "primary_profile": record.profile,
                "primary_mutation_hash": record.primary_mutation_hash,
                "post_primary_hash": record.post_primary_hash,
                "evidence_hash": record.evidence_hash,
                "authorized_paths": record.authorized_paths,
                "representation": representations[record.profile].value,
                "authority_sufficient": record.authority_sufficient,
            }
        )
    payload: dict[str, object] = {
        "suite": SUITE,
        "schema_version": VERSION,
        "a57_manifest_hash": a57_hash,
        "cases": records,
        "repair_model_artifacts": artifacts,
    }
    path = root / "results" / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen A58 manifest mismatch")
    else:
        _atomic_json(path, payload)
    return payload


def cell_path(root: Path, pair: Pair) -> Path:
    return root / "results" / f"{pair.key}.json"


def load_cell(root: Path, pair: Pair, expected_manifest: str) -> Cell | None:
    path = cell_path(root, pair)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    value["authorized_paths"] = tuple(value["authorized_paths"])
    if (
        value["total_seconds"] is not None
        and value["total_seconds"] < value["generation_seconds"]
    ):
        value["total_seconds"] = None
    cell = Cell(**value)
    if (
        cell.case_id != pair.case.case_id
        or cell.repair_profile != pair.repair_profile
        or cell.manifest_hash != expected_manifest
    ):
        raise ValueError("A58 committed cell identity mismatch")
    return cell


def _taxonomy(
    *,
    schema: bool,
    authorized: bool,
    material: bool,
    transaction: bool,
    verification_executed: bool,
    verification_passed: bool,
    semantic: bool,
) -> str:
    if not schema:
        return "NO_VALID_PROPOSAL"
    if not authorized:
        return "UNAUTHORIZED_PATH"
    if not material:
        return "NO_CHANGE"
    if not transaction:
        return "TRANSACTION_FAILURE"
    if not verification_executed:
        return "STRUCTURAL_FAILURE"
    if verification_passed and semantic:
        return "SEMANTIC_RECOVERED"
    if verification_passed:
        return "VERIFICATION_PASS_SEMANTIC_FAIL"
    return "VERIFICATION_STILL_FAIL"


def _shape(schema: bool, transaction: bool, semantic: bool) -> str:
    if not schema:
        return "NO_SCHEMA"
    if not transaction:
        return "PROPOSAL_ONLY"
    return "TRANSACTION_SEMANTIC_PASS" if semantic else "TRANSACTION_SEMANTIC_FAIL"


def reuse_diagonal(
    root: Path,
    a57_root: Path,
    pair: Pair,
    record: PrimaryRecord,
    artifact: str,
    representation: MutationRepresentationPolicy,
    expected_manifest: str,
) -> Cell:
    if not pair.same_model:
        raise ValueError("only the same-model diagonal can reuse A57")
    existing = load_cell(root, pair, expected_manifest)
    if existing is not None:
        return existing
    a57_manifest = manifest_hash(
        json.loads((a57_root / "results" / "manifest.json").read_text(encoding="utf-8"))
    )
    trial = load_trial(a57_root, pair.case, "R0", a57_manifest)
    if trial is None or trial.start_hash != record.post_primary_hash:
        raise ValueError("A57 diagonal result identity is unavailable")
    if artifact != record.model_artifact:
        raise ValueError("same-model artifact identity differs from A57")
    taxonomy = _taxonomy(
        schema=trial.schema_valid,
        authorized=trial.authorized_path_set,
        material=trial.material_mutation,
        transaction=trial.transaction_executed,
        verification_executed=trial.full_verification_executed,
        verification_passed=trial.verification_passed,
        semantic=trial.semantic_passed,
    )
    cell = Cell(
        SUITE,
        VERSION,
        record.case_id,
        record.task_id,
        record.profile,
        pair.repair_profile,
        True,
        True,
        artifact,
        representation.value,
        expected_manifest,
        record.post_primary_hash,
        record.primary_mutation_hash,
        record.evidence_hash,
        record.authorized_paths,
        record.authority_sufficient,
        trial.original_task_present,
        False,
        trial.evidence_present,
        trial.schema_valid,
        trial.authorized_path_set,
        trial.material_mutation,
        trial.transaction_executed,
        trial.full_verification_executed,
        trial.fingerprint_changed,
        trial.verification_passed,
        trial.semantic_passed,
        trial.semantic_recovered,
        trial.failing_step,
        taxonomy,
        None,
        _shape(trial.schema_valid, trial.transaction_executed, trial.semantic_passed),
        trial.input_tokens,
        trial.output_tokens,
        trial.model_calls,
        trial.generation_seconds,
        trial.verification_seconds,
        trial.total_seconds,
    )
    _atomic_json(cell_path(root, pair), asdict(cell))
    return cell


def run_cross_cell(
    root: Path,
    a57_root: Path,
    pair: Pair,
    record: PrimaryRecord,
    replay: dict[str, object],
    backend: Model,
    artifact: str,
    representation: MutationRepresentationPolicy,
    expected_manifest: str,
) -> Cell:
    """Run one off-diagonal backend against the exact production R0 request."""
    if pair.same_model:
        raise ValueError("same-model cells must reuse A57")
    existing = load_cell(root, pair, expected_manifest)
    if existing is not None:
        return existing
    started = time.perf_counter()
    prefix = tuple(_restore_response(item, backend) for item in replay["responses"])
    recorder = CaptureModel(backend, prefix=prefix)
    start_hashes: list[str] = []

    def on_primary(_task, workspace: Path) -> None:  # type: ignore[no-untyped-def]
        current = tree_hash(workspace)
        start_hashes.append(current)
        if current != record.post_primary_hash:
            raise ValueError("cross-model post-primary state differs from A57")

    raw = (
        RealWorldEvaluationRunner(
            record.profile,
            recorder,
            REPOSITORY,
            mutation_representation=representation,
            primary_mutation_callback=on_primary,
        )
        .run(
            (pair.case.definition.production_task,), snapshot(record.post_primary_hash)
        )
        .results[0]
    )
    metrics = raw.metrics
    schema = bool(metrics.repair_preview_created or metrics.repair_mutation_executed)
    transaction = bool(metrics.repair_mutation_executed)
    verification_executed = bool(metrics.reverification_executed)
    verification_passed = metrics.reverification_result in {"pass", "passed"}
    semantic = raw.oracle is EvaluationOutcome.PASS
    authorized = bool(start_hashes) and not raw.unexpected_paths
    input_tokens, output_tokens = _usage(recorder, len(prefix))
    repair_payload = tuple(
        response.text for response in recorder.responses[len(prefix) :]
    )
    mutation_hash = (
        hashlib.sha256(json.dumps(repair_payload).encode()).hexdigest()
        if schema
        else None
    )
    failing_step = metrics.verification_plan_failed_step
    improvement = _step_improved(record.failure_step, failing_step)
    total = (
        time.perf_counter() - recorder.repair_started
        if recorder.repair_started is not None
        else time.perf_counter() - started
    )
    cell = Cell(
        SUITE,
        VERSION,
        record.case_id,
        record.task_id,
        record.profile,
        pair.repair_profile,
        False,
        False,
        artifact,
        representation.value,
        expected_manifest,
        record.post_primary_hash,
        record.primary_mutation_hash,
        record.evidence_hash,
        record.authorized_paths,
        record.authority_sufficient,
        True,
        False,
        True,
        schema,
        authorized,
        transaction,
        transaction,
        verification_executed,
        improvement,
        verification_passed,
        semantic,
        transaction and verification_executed and verification_passed and semantic,
        failing_step,
        _taxonomy(
            schema=schema,
            authorized=authorized,
            material=transaction,
            transaction=transaction,
            verification_executed=verification_executed,
            verification_passed=verification_passed,
            semantic=semantic,
        ),
        mutation_hash,
        _shape(schema, transaction, semantic),
        input_tokens,
        output_tokens,
        recorder.backend_calls,
        round(recorder.generation_seconds, 3),
        round(metrics.verification_plan_duration, 3),
        round(total, 3),
    )
    _atomic_json(cell_path(root, pair), asdict(cell))
    return cell


def summarize_case(cells: tuple[Cell, ...]) -> dict[str, object]:
    if len(cells) != 3 or len({cell.repair_profile for cell in cells}) != 3:
        raise ValueError("A58 case summary requires three repair profiles")
    recovered = frozenset(
        cell.repair_profile for cell in cells if cell.semantic_recovered
    )
    primary = cells[0].primary_profile
    wrong_transactions = [
        cell for cell in cells if cell.transaction_executed and not cell.semantic_passed
    ]
    wrong_hashes = {cell.mutation_hash for cell in wrong_transactions}
    if len(wrong_transactions) < 2 or None in wrong_hashes:
        diversity = "INSUFFICIENT_COMPARABLE_MUTATIONS"
    elif len(wrong_hashes) == 1:
        diversity = "CONVERGENT_WRONG_FIX"
    else:
        diversity = "DIVERGENT_WRONG_FIX"
    return {
        "case_id": cells[0].case_id,
        "primary_profile": primary,
        "classification": classify_case(recovered, primary),
        "corrective_diversity": diversity,
        "repair_profiles": [cell.repair_profile for cell in cells],
        "semantic_recoveries": sorted(recovered),
    }
