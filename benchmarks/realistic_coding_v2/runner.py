"""Production-path A56 cell execution with source-free durable checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.realistic_coding_v2.suite import (
    BUILD,
    REPOSITORY,
    SUITE,
    TEST,
    VERSION,
    AuthorityMode,
    FrozenTask,
    Integrity,
    OperationClass,
    manifest_identity,
)
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    RealWorldTaskResult,
    RepositorySnapshot,
    hash_workspace,
    run_oracle,
)
from forge.models import (
    FinishReason,
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    MutationRepresentationPolicy,
)

MUTATION_TYPES = frozenset(
    {
        "structured_edit",
        "line_range_edit",
        "multi_file_structured_edit",
        "multi_file_line_range_edit",
        "create_file",
        "multi_file_create",
        "multi_file_change",
    }
)


class Failure:
    DISCOVERY_FAILED = "DISCOVERY_FAILED"
    SOURCE_ACQUISITION_FAILED = "SOURCE_ACQUISITION_FAILED"
    AUTHORITY_NOT_READY = "AUTHORITY_NOT_READY"
    PROTOCOL_FAILED = "PROTOCOL_FAILED"
    EDIT_CONSTRUCTION_FAILED = "EDIT_CONSTRUCTION_FAILED"
    CREATE_CONSTRUCTION_FAILED = "CREATE_CONSTRUCTION_FAILED"
    MIXED_CONSTRUCTION_FAILED = "MIXED_CONSTRUCTION_FAILED"
    PREVIEW_FAILED = "PREVIEW_FAILED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    REPAIR_FAILED = "REPAIR_FAILED"
    SEMANTIC_ORACLE_FAILED = "SEMANTIC_ORACLE_FAILED"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class CellResult:
    suite: str
    schema_version: int
    task_id: str
    task_version: int
    operation_class: str
    language: str
    authority_mode: str
    model_profile: str
    model_artifact: str
    seed: int
    repository_identity: str
    manifest_identity: str
    required_modify: int
    required_create: int
    proposed_create: int
    authorized_create: int
    unexpected_create_attempts: int
    successful_create: int
    proposal_role_correct: bool
    duplicate_operations: int
    missing_operations: int
    proposal_envelopes: tuple[dict[str, object], ...]
    discovery_calls: int
    source_reads: int
    expected_source_acquired: bool
    required_sources_ready: int
    mutation_ready: bool
    schema_valid: bool
    preview_created: bool
    transaction_executed: bool
    verification_result: str
    repair_eligible: bool
    repair_attempted: bool
    repair_structurally_valid: bool
    verification_improved: bool
    first_pass_semantic: bool | None
    final_semantic: bool
    repair_recovered: bool
    failure_layer: str
    final_status: str
    changed_paths: tuple[str, ...]
    unexpected_paths: tuple[str, ...]
    model_calls: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    truncated_generations: int
    generation_seconds: float
    execution_seconds: float
    verification_seconds: float
    total_seconds: float


class RecordingModel(Model):
    """Record timings and envelope shape; never retain generated source text."""

    def __init__(self, backend: Model, authorized_paths: frozenset[str]) -> None:
        self.backend = backend
        self.authorized_paths = authorized_paths
        self.envelopes: list[dict[str, object]] = []
        self.generation_seconds = 0.0
        self.truncated = 0
        self.calls = 0

    @property
    def identity(self) -> ModelIdentity:
        return self.backend.identity

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.backend.capabilities

    @property
    def context_capacity(self) -> int | None:
        return self.backend.context_capacity

    def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        try:
            response = self.backend.generate(request)
        finally:
            self.generation_seconds += time.perf_counter() - started
            self.calls += 1
        if response.finish_reason is FinishReason.MAX_TOKENS:
            self.truncated += 1
        self.envelopes.append(envelope_shape(response.text, self.authorized_paths))
        return response

    def close(self) -> None:
        self.backend.close()


def envelope_shape(
    text: str, authorized_paths: frozenset[str] = frozenset()
) -> dict[str, object]:
    """Erase all implementation content while preserving operation roles/paths."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return {"type": "invalid_json", "children": []}
    if not isinstance(payload, dict):
        return {"type": "non_object", "children": []}
    raw_kind = payload.get("type")
    kind = (
        raw_kind
        if isinstance(raw_kind, str)
        and raw_kind in MUTATION_TYPES | {"tool_call", "final"}
        else "other"
    )

    def child(kind_value: object, path_value: object) -> dict[str, object]:
        safe_kind = (
            kind_value
            if isinstance(kind_value, str) and kind_value in MUTATION_TYPES | {"edit"}
            else "other"
        )
        safe_path = (
            path_value
            if isinstance(path_value, str) and path_value in authorized_paths
            else "<unauthorized>"
        )
        return {"type": safe_kind, "path": safe_path}

    if kind == "multi_file_change":
        values = payload.get("operations")
        children = (
            [
                child(item.get("type"), item.get("path"))
                for item in values
                if isinstance(item, dict)
            ]
            if isinstance(values, list)
            else []
        )
    elif kind in {"multi_file_line_range_edit", "multi_file_structured_edit"}:
        values = payload.get("edits")
        children = (
            [
                child("edit", item.get("path"))
                for item in values
                if isinstance(item, dict)
            ]
            if isinstance(values, list)
            else []
        )
    elif kind == "multi_file_create":
        values = payload.get("creates")
        children = (
            [
                child("create_file", item.get("path"))
                for item in values
                if isinstance(item, dict)
            ]
            if isinstance(values, list)
            else []
        )
    elif kind in {"line_range_edit", "structured_edit", "create_file"}:
        children = [child(kind, payload.get("path"))]
    else:
        children = []
    return {"type": kind, "children": children}


def _proposal_shape(
    definition: FrozenTask, envelopes: tuple[dict[str, object], ...]
) -> tuple[bool, int, int, int, int]:
    first = next(
        (item for item in envelopes if item.get("type") in MUTATION_TYPES), None
    )
    if first is None:
        return False, 0, len(definition.edit_paths) + len(definition.create_paths), 0, 0
    children = first.get("children")
    if not isinstance(children, list):
        return False, 0, len(definition.edit_paths) + len(definition.create_paths), 0, 0
    roles = []
    for child in children:
        if not isinstance(child, dict):
            continue
        kind = child.get("type")
        role = "CREATE" if kind == "create_file" else "MODIFY"
        roles.append((role, child.get("path")))
    duplicate = len(roles) - len({path for _, path in roles})
    expected = {
        *(("MODIFY", path) for path in definition.edit_paths),
        *(("CREATE", path) for path in definition.create_paths),
    }
    actual = set(roles)
    missing = len(expected - actual)
    proposed_create = sum(role == "CREATE" for role, _ in roles)
    unexpected_create = sum(
        role == "CREATE" and path not in definition.create_paths for role, path in roles
    )
    envelope_type = first.get("type")
    expected_type = {
        OperationClass.EDIT_SINGLE: {"line_range_edit", "structured_edit"},
        OperationClass.EDIT_MULTI: {
            "multi_file_line_range_edit",
            "multi_file_structured_edit",
        },
        OperationClass.CREATE: {"create_file", "multi_file_create"},
        OperationClass.MIXED_EDIT_CREATE: {"multi_file_change"},
    }[definition.operation_class]
    return (
        envelope_type in expected_type
        and not duplicate
        and not missing
        and actual == expected,
        duplicate,
        missing,
        proposed_create,
        unexpected_create,
    )


def classify_failure(
    definition: FrozenTask,
    raw: RealWorldTaskResult,
    *,
    proposal_role_correct: bool,
    first_pass: bool | None,
    mutation_envelope_seen: bool = False,
) -> str:
    """Name the first integrated failure layer; keep semantics separately."""
    metrics = raw.metrics
    if (
        definition.authority_mode is AuthorityMode.DISCOVERY_REQUIRED
        and not metrics.expected_implementation_acquired
    ):
        return Failure.DISCOVERY_FAILED
    if not metrics.mutation_ready_reached and (
        metrics.required_sources_missing or metrics.source_acquisition_failures
    ):
        return Failure.SOURCE_ACQUISITION_FAILED
    if not metrics.mutation_ready_reached:
        return Failure.AUTHORITY_NOT_READY
    construction_failure = {
        OperationClass.EDIT_SINGLE: Failure.EDIT_CONSTRUCTION_FAILED,
        OperationClass.EDIT_MULTI: Failure.EDIT_CONSTRUCTION_FAILED,
        OperationClass.CREATE: Failure.CREATE_CONSTRUCTION_FAILED,
        OperationClass.MIXED_EDIT_CREATE: Failure.MIXED_CONSTRUCTION_FAILED,
    }[definition.operation_class]
    if not metrics.mutation_proposed:
        return (
            construction_failure
            if mutation_envelope_seen or proposal_role_correct
            else Failure.PROTOCOL_FAILED
        )
    structurally_valid = bool(
        metrics.structured_mutation_valid
        or metrics.preview_created
        or metrics.mutation_group_preview_created
        or metrics.mutations
    )
    if not proposal_role_correct or not structurally_valid:
        return construction_failure
    if not metrics.preview_created and not metrics.mutation_group_preview_created:
        return Failure.PREVIEW_FAILED
    if not metrics.mutations:
        return Failure.TRANSACTION_FAILED
    if raw.unexpected_paths:
        return Failure.TRANSACTION_FAILED
    if metrics.repair_attempts and raw.oracle is not EvaluationOutcome.PASS:
        return Failure.REPAIR_FAILED
    if not (
        metrics.verification_result in {"pass", "passed"}
        or metrics.reverification_result in {"pass", "passed"}
    ):
        return Failure.VERIFICATION_FAILED
    if raw.oracle is not EvaluationOutcome.PASS:
        return Failure.SEMANTIC_ORACLE_FAILED
    return Failure.PASS


def repair_recovered(
    first_pass_semantic: bool | None, final_semantic: bool, repair_attempted: bool
) -> bool:
    return repair_attempted and first_pass_semantic is False and final_semantic


def aggregate_cells(cells: tuple[CellResult, ...]) -> dict[str, dict[str, int]]:
    """Keep operation classes separate from profile-level totals."""
    result = {
        item.value: {
            "cells": 0,
            "role_correct": 0,
            "transactions": 0,
            "semantic": 0,
        }
        for item in OperationClass
    }
    for cell in cells:
        group = result[cell.operation_class]
        group["cells"] += 1
        group["role_correct"] += int(cell.proposal_role_correct)
        group["transactions"] += int(cell.transaction_executed)
        group["semantic"] += int(cell.final_semantic)
    return result


def snapshot(repository_identity: str) -> RepositorySnapshot:
    paths = tuple(path for path in REPOSITORY.rglob("*") if path.is_file())
    sources = tuple(
        path
        for path in paths
        if path.suffix in {".py", ".c", ".h"} and "tests" not in path.parts
    )
    tests = tuple(
        path
        for path in paths
        if path.suffix in {".py", ".c", ".h"} and "tests" in path.parts
    )
    return RepositorySnapshot(
        "service-engine-v2",
        repository_identity,
        "Python+C17",
        len(sources),
        len(tests),
        sum(len(path.read_text().splitlines()) for path in (*sources, *tests)),
        BUILD,
        TEST,
        EvaluationOutcome.PASS,
        0.0,
    )


def run_cell(
    definition: FrozenTask,
    backend: Model,
    *,
    profile: str,
    artifact: str,
    seed: int,
    repository_identity: str,
    manifest: str,
    representation: MutationRepresentationPolicy,
) -> CellResult:
    """Run one production session and retain only source-free measurements."""
    recorder = RecordingModel(
        backend, frozenset((*definition.edit_paths, *definition.create_paths))
    )
    first_semantic: bool | None = None

    def primary_callback(task, workspace: Path) -> None:  # type: ignore[no-untyped-def]
        nonlocal first_semantic
        try:
            first_semantic = (
                run_oracle(workspace, task.oracle_commands) is EvaluationOutcome.PASS
            )
        except Exception:
            first_semantic = False

    task = replace(definition.production_task, seeds=(seed,))
    raw_run = RealWorldEvaluationRunner(
        profile,
        recorder,
        REPOSITORY,
        mutation_representation=representation,
        primary_mutation_callback=primary_callback,
    ).run((task,), snapshot(repository_identity))
    raw = raw_run.results[0]
    metrics = raw.metrics
    envelopes = tuple(recorder.envelopes)
    role_correct, duplicate, missing, proposed_create, unexpected_create = (
        _proposal_shape(definition, envelopes)
    )
    final_semantic = raw.oracle is EvaluationOutcome.PASS
    verification_seconds = metrics.verification_plan_duration or (
        metrics.configure_duration
        + metrics.verification_build_duration
        + metrics.verification_test_duration
    )
    execution_seconds = max(
        0.0, raw.elapsed_seconds - recorder.generation_seconds - verification_seconds
    )
    failure = classify_failure(
        definition,
        raw,
        proposal_role_correct=role_correct,
        first_pass=first_semantic,
        mutation_envelope_seen=any(
            item.get("type") in MUTATION_TYPES for item in envelopes
        ),
    )
    after = dict(raw.after_hashes)
    before = dict(raw.before_hashes)
    successful_create = sum(
        path in after and path not in before for path in definition.create_paths
    )
    return CellResult(
        SUITE,
        VERSION,
        definition.task_id,
        definition.version,
        definition.operation_class.value,
        definition.language,
        definition.authority_mode.value,
        profile,
        artifact,
        seed,
        repository_identity,
        manifest,
        len(definition.edit_paths),
        len(definition.create_paths),
        proposed_create,
        len(definition.create_paths),
        unexpected_create,
        successful_create,
        role_correct,
        duplicate,
        missing,
        envelopes,
        metrics.discovery_calls,
        metrics.source_reads,
        metrics.expected_implementation_acquired,
        metrics.required_sources_ready,
        metrics.mutation_ready_reached,
        bool(
            metrics.structured_mutation_valid
            or metrics.preview_created
            or metrics.mutation_group_preview_created
            or metrics.mutations
        ),
        bool(
            metrics.preview_created
            or metrics.mutation_group_preview_created
            or metrics.mutations
        ),
        bool(metrics.mutations),
        metrics.verification_plan_result,
        metrics.repair_diagnosis_entered,
        bool(metrics.repair_attempts),
        bool(metrics.repair_preview_created or metrics.repair_mutation_executed),
        metrics.reverification_result == "pass",
        first_semantic,
        final_semantic,
        repair_recovered(first_semantic, final_semantic, bool(metrics.repair_attempts)),
        failure,
        raw.final_status,
        tuple(
            path if path in recorder.authorized_paths else "<unauthorized>"
            for path in raw.changed_paths
        ),
        tuple("<unauthorized>" for _ in raw.unexpected_paths),
        recorder.calls,
        metrics.tool_executions,
        raw.usage.input_tokens,
        raw.usage.output_tokens,
        recorder.truncated,
        round(recorder.generation_seconds, 3),
        round(execution_seconds, 3),
        round(verification_seconds, 3),
        round(raw.elapsed_seconds, 3),
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Commit one cell once; never overwrite an existing checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
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
    checkpoint_dir: Path,
    definitions: tuple[FrozenTask, ...],
    integrity: tuple[Integrity, ...],
) -> dict[str, object]:
    """Write or verify frozen evaluator identity before any model loads."""
    if not all(item.eligible for item in integrity):
        raise ValueError("semantic-score-ineligible A56 task")
    repository_identity = hashlib.sha256(
        repr(hash_workspace(REPOSITORY)).encode()
    ).hexdigest()
    payload: dict[str, object] = {
        "suite": SUITE,
        "schema_version": VERSION,
        "repository_identity": repository_identity,
        "manifest_identity": manifest_identity(definitions),
        "task_ids": [item.versioned_id for item in definitions],
        "integrity": [asdict(item) for item in integrity],
    }
    destination = checkpoint_dir / "manifest.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("frozen A56 manifest differs from current definitions")
    else:
        _atomic_json(destination, payload)
    return payload


def checkpoint_path(directory: Path, profile: str, seed: int, task_id: str) -> Path:
    return directory / f"{profile}-seed{seed}-{task_id}.json"


def read_checkpoint(
    path: Path,
    *,
    profile: str,
    seed: int,
    definition: FrozenTask,
    artifact: str,
    manifest: dict[str, object],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "suite": SUITE,
        "schema_version": VERSION,
        "task_id": definition.task_id,
        "task_version": definition.version,
        "operation_class": definition.operation_class.value,
        "model_profile": profile,
        "model_artifact": artifact,
        "seed": seed,
        "repository_identity": manifest["repository_identity"],
        "manifest_identity": manifest["manifest_identity"],
    }
    if not isinstance(value, dict) or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise ValueError(f"checkpoint identity mismatch: {path.name}")
    return value


def commit_cell(path: Path, cell: CellResult) -> None:
    _atomic_json(path, asdict(cell))
