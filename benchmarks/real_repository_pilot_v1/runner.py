"""Source-free exactly-once production runner for the A60 Foundation pilot."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.real_repository_pilot_v1.suite import (
    SUITE,
    VERSION,
    AuthorityMode,
    Integrity,
    PilotTask,
    definition_identity,
    repository_identity,
    snapshot_description,
    source_manifest,
    task_relevant_hashes,
)
from benchmarks.realistic_coding_v2.runner import MUTATION_TYPES, RecordingModel
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    run_oracle,
)
from forge.models import Model, MutationRepresentationPolicy


class Failure:
    DISCOVERY_FAILED = "DISCOVERY_FAILED"
    SOURCE_ACQUISITION_FAILED = "SOURCE_ACQUISITION_FAILED"
    CONTEXT_INSUFFICIENT = "CONTEXT_INSUFFICIENT"
    AUTHORITY_NOT_READY = "AUTHORITY_NOT_READY"
    PROTOCOL_FAILED = "PROTOCOL_FAILED"
    CONSTRUCTION_FAILED = "CONSTRUCTION_FAILED"
    PREVIEW_FAILED = "PREVIEW_FAILED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    REPAIR_FAILED = "REPAIR_FAILED"
    SEMANTIC_FAILED = "SEMANTIC_FAILED"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class PilotCell:
    suite: str
    schema_version: int
    task_id: str
    task_version: int
    operation_class: str
    authority_mode: str
    model_profile: str
    model_artifact: str
    seed: int
    repository_identity: str
    manifest_identity: str
    discovery_success: bool
    discovery_calls: int
    source_reads: int
    irrelevant_files_read: int
    symbol_index_used: bool
    source_acquired: bool
    context_quality: str
    context_files: tuple[str, ...]
    authority_readiness: str
    schema_valid: bool
    path_valid: bool
    operation_role_valid: bool
    proposal_complete: bool
    preview_created: bool
    transaction_success: bool
    verification_outcome: str
    repair_eligible: bool
    repair_attempted: bool
    repair_proposal_valid: bool
    verification_changed: bool
    semantic_recovered: bool
    semantic_pass: bool
    first_failure_layer: str
    final_status: str
    changed_paths: tuple[str, ...]
    input_tokens: int | None
    output_tokens: int | None
    model_calls: int
    tool_calls: int
    context_peak: int
    generation_seconds: float
    verification_seconds: float
    total_seconds: float
    index_builds: int
    index_refreshes: int
    index_seconds: float


def context_quality(
    definition: PilotTask, inspected: tuple[str, ...], ready: int
) -> str:
    required = set(definition.edit_paths)
    available = required.intersection(inspected)
    if ready >= len(required) and available == required:
        return "SUFFICIENT"
    if available or ready:
        return "PARTIAL"
    return "MISDIRECTED"


def first_failure(
    definition: PilotTask,
    *,
    discovery: bool,
    acquired: bool,
    context: str,
    ready: bool,
    schema: bool,
    path_valid: bool,
    role_valid: bool,
    complete: bool,
    preview: bool,
    transaction: bool,
    verification: bool,
    repair_attempted: bool,
    semantic: bool,
) -> str:
    if definition.authority_mode is AuthorityMode.DISCOVERY_REQUIRED and not discovery:
        return Failure.DISCOVERY_FAILED
    if not acquired:
        return Failure.SOURCE_ACQUISITION_FAILED
    if context != "SUFFICIENT":
        return Failure.CONTEXT_INSUFFICIENT
    if not ready:
        return Failure.AUTHORITY_NOT_READY
    if not schema:
        return Failure.PROTOCOL_FAILED
    if not path_valid or not role_valid or not complete:
        return Failure.CONSTRUCTION_FAILED
    if not preview:
        return Failure.PREVIEW_FAILED
    if not transaction:
        return Failure.TRANSACTION_FAILED
    if not verification:
        return (
            Failure.REPAIR_FAILED if repair_attempted else Failure.VERIFICATION_FAILED
        )
    if not semantic:
        return Failure.SEMANTIC_FAILED
    return Failure.PASS


def readiness(definition: PilotTask, ready: bool) -> str:
    if not ready:
        return "NOT_READY"
    return (
        "EDIT_READY"
        if definition.operation_class.value == "EDIT_SINGLE"
        else "EDIT_READY"
    )


def proposal_shape(
    definition: PilotTask, envelopes: tuple[dict[str, object], ...]
) -> tuple[bool, int, int, int]:
    first = next(
        (item for item in envelopes if item.get("type") in MUTATION_TYPES), None
    )
    if first is None:
        return False, 0, len(definition.edit_paths), 0
    children = first.get("children")
    if not isinstance(children, list):
        return False, 0, len(definition.edit_paths), 0
    paths = [item.get("path") for item in children if isinstance(item, dict)]
    duplicate = len(paths) - len(set(paths))
    missing = len(set(definition.edit_paths) - set(paths))
    expected_type = (
        {"line_range_edit", "structured_edit"}
        if len(definition.edit_paths) == 1
        else {"multi_file_line_range_edit", "multi_file_structured_edit"}
    )
    return (
        first.get("type") in expected_type and set(paths) == set(definition.edit_paths),
        duplicate,
        missing,
        0,
    )


def run_cell(
    definition: PilotTask,
    backend: Model,
    snapshot: Path,
    *,
    profile: str,
    artifact: str,
    manifest: dict[str, object],
    representation: MutationRepresentationPolicy,
) -> PilotCell:
    recorder = RecordingModel(backend, frozenset(definition.edit_paths))
    first_semantic: bool | None = None

    def after_primary(task, workspace: Path) -> None:  # type: ignore[no-untyped-def]
        nonlocal first_semantic
        first_semantic = (
            run_oracle(workspace, task.oracle_commands) is EvaluationOutcome.PASS
        )

    raw = (
        RealWorldEvaluationRunner(
            profile,
            recorder,
            snapshot,
            mutation_representation=representation,
            primary_mutation_callback=after_primary,
        )
        .run(
            (replace(definition.production_task, seeds=(42,)),),
            snapshot_description(snapshot),
        )
        .results[0]
    )
    metrics = raw.metrics
    envelopes = tuple(recorder.envelopes)
    role_valid, duplicate, missing, unexpected_create = proposal_shape(
        definition, envelopes
    )
    inspected = tuple(metrics.files_inspected)
    quality = context_quality(definition, inspected, metrics.required_sources_ready)
    acquired = metrics.required_sources_ready >= len(definition.edit_paths)
    discovery = metrics.expected_implementation_acquired and acquired
    schema = bool(
        metrics.structured_mutation_valid
        or metrics.preview_created
        or metrics.mutation_group_preview_created
        or metrics.mutations
    )
    preview = bool(
        metrics.preview_created
        or metrics.mutation_group_preview_created
        or metrics.mutations
    )
    transaction = bool(metrics.mutations) and not raw.unexpected_paths
    verification = (
        metrics.verification_plan_result == "pass"
        or metrics.reverification_result == "pass"
    )
    semantic = raw.oracle is EvaluationOutcome.PASS
    repair_attempted = bool(metrics.repair_attempts)
    changed = tuple(
        path if path in definition.edit_paths else "<unauthorized>"
        for path in raw.changed_paths
    )
    path_valid = not raw.unexpected_paths and unexpected_create == 0
    complete = missing == 0 and duplicate == 0
    failure = first_failure(
        definition,
        discovery=discovery,
        acquired=acquired,
        context=quality,
        ready=metrics.mutation_ready_reached,
        schema=schema,
        path_valid=path_valid,
        role_valid=role_valid,
        complete=complete,
        preview=preview,
        transaction=transaction,
        verification=verification,
        repair_attempted=repair_attempted,
        semantic=semantic,
    )
    verification_seconds = metrics.verification_plan_duration or (
        metrics.configure_duration
        + metrics.verification_build_duration
        + metrics.verification_test_duration
    )
    return PilotCell(
        SUITE,
        VERSION,
        definition.task_id,
        definition.version,
        definition.operation_class.value,
        definition.authority_mode.value,
        profile,
        artifact,
        42,
        str(manifest["repository_identity"]),
        str(manifest["manifest_identity"]),
        discovery,
        metrics.discovery_calls,
        metrics.source_reads,
        len(set(inspected) - set(definition.edit_paths)),
        bool(metrics.lexical_index_builds),
        acquired,
        quality,
        inspected,
        readiness(definition, metrics.mutation_ready_reached),
        schema,
        path_valid,
        role_valid,
        complete,
        preview,
        transaction,
        metrics.verification_plan_result,
        metrics.repair_diagnosis_entered,
        repair_attempted,
        bool(metrics.repair_preview_created or metrics.repair_mutation_executed),
        metrics.reverification_result == "pass",
        repair_attempted and first_semantic is False and semantic,
        semantic,
        failure,
        raw.final_status,
        changed,
        raw.usage.input_tokens,
        raw.usage.output_tokens,
        recorder.calls,
        metrics.tool_executions,
        metrics.context_peak_estimate,
        round(recorder.generation_seconds, 3),
        round(verification_seconds, 3),
        round(raw.elapsed_seconds, 3),
        metrics.lexical_index_builds,
        metrics.lexical_index_refreshes,
        round(metrics.lexical_index_duration_seconds, 3),
    )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_manifest(
    directory: Path,
    canonical: Path,
    snapshot: Path,
    definitions: tuple[PilotTask, ...],
    integrity: tuple[Integrity, ...],
) -> dict[str, object]:
    if len(definitions) != 8 or not all(item.eligible for item in integrity):
        raise ValueError("A60 task corpus is not semantically eligible")
    payload: dict[str, object] = {
        "suite": SUITE,
        "schema_version": VERSION,
        "canonical_identity": repository_identity(canonical),
        "repository_identity": repository_identity(snapshot),
        "source_file_manifest": source_manifest(snapshot),
        "task_relevant_hashes": task_relevant_hashes(snapshot, definitions),
        "build_configuration_identity": hashlib.sha256(
            (snapshot / "CMakeLists.txt").read_bytes()
        ).hexdigest(),
        "manifest_identity": definition_identity(definitions, snapshot),
        "task_ids": [item.versioned_id for item in definitions],
        "integrity": [asdict(item) for item in integrity],
        "ephemeral_acceptance": False,
        "routing": False,
    }
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise ValueError("A60 frozen manifest changed")
    else:
        _atomic_json(path, payload)
    return payload


def checkpoint_path(directory: Path, profile: str, task_id: str) -> Path:
    return directory / "results" / f"{profile}-seed42-{task_id}.json"


def read_checkpoint(
    path: Path, expected: dict[str, object]
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"A60 checkpoint identity mismatch: {path.name}")
    return payload


def commit_cell(path: Path, cell: PilotCell) -> None:
    _atomic_json(path, asdict(cell))


def standard_result_is_source_free(payload: dict[str, object]) -> bool:
    forbidden = {"replacement", "source", "old_text", "new_text", "content"}
    return not any(key in forbidden for key in payload)


def summarize(cells: tuple[dict[str, object], ...]) -> dict[str, object]:
    by_profile: dict[str, dict[str, object]] = {}
    for profile in sorted({str(cell["model_profile"]) for cell in cells}):
        selected = [cell for cell in cells if cell["model_profile"] == profile]
        by_profile[profile] = {
            "cells": len(selected),
            "discovery": sum(bool(cell["discovery_success"]) for cell in selected),
            "context_sufficient": sum(
                cell["context_quality"] == "SUFFICIENT" for cell in selected
            ),
            "valid_proposals": sum(
                bool(cell["schema_valid"]) and bool(cell["operation_role_valid"])
                for cell in selected
            ),
            "transactions": sum(bool(cell["transaction_success"]) for cell in selected),
            "verification_pass": sum(
                cell["verification_outcome"] == "pass" for cell in selected
            ),
            "semantic_pass": sum(bool(cell["semantic_pass"]) for cell in selected),
            "repair_recoveries": sum(
                bool(cell["semantic_recovered"]) for cell in selected
            ),
            "total_seconds": round(
                sum(float(cell["total_seconds"]) for cell in selected), 3
            ),
        }
    return {"suite": SUITE, "cells": len(cells), "profiles": by_profile}
