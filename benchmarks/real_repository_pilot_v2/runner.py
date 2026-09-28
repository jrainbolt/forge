"""Exactly-once A62 runner and independent discovery scoring."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.real_repository_discovery_boundary_v1.suite import DiscoveryScore
from benchmarks.real_repository_pilot_v1.runner import proposal_shape
from benchmarks.real_repository_pilot_v1.suite import (
    AuthorityMode,
    snapshot_description,
)
from benchmarks.real_repository_pilot_v2.suite import (
    SEED,
    SUITE,
    VERSION,
    PilotV2Task,
    validate_boundary,
)
from benchmarks.realistic_coding_v2.runner import RecordingModel
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
)
from forge.models import Model, MutationRepresentationPolicy


class Failure:
    DISCOVERY_FAILED = "DISCOVERY_FAILED"
    SOURCE_ACQUISITION_FAILED = "SOURCE_ACQUISITION_FAILED"
    CONTEXT_INSUFFICIENT = "CONTEXT_INSUFFICIENT"
    AUTHORITY_NOT_READY = "AUTHORITY_NOT_READY"
    PROTOCOL_FAILED = "PROTOCOL_FAILED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    REPAIR_FAILED = "REPAIR_FAILED"
    SEMANTIC_FAILED = "SEMANTIC_FAILED"
    PASS = "PASS"


@dataclass(frozen=True, slots=True)
class PilotV2Cell:
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
    discovery_outcome: str
    discovery_calls: int
    search_calls: int
    symbol_calls: int
    index_builds: int
    index_queries: int
    index_refreshes: int
    index_seconds: float
    source_reads: int
    source_acquired: bool
    context_quality: str
    context_files: tuple[str, ...]
    mutation_ready: bool
    schema_valid: bool
    transaction_success: bool
    verification_result: str
    repair_attempted: bool
    repair_recovered: bool
    semantic_pass: bool
    first_failure_layer: str
    final_status: str
    changed_paths: tuple[str, ...]
    input_tokens: int | None
    output_tokens: int | None
    model_calls: int
    tool_calls: int
    generation_seconds: float
    verification_seconds: float
    total_seconds: float


def classify_context(
    definition: PilotV2Task,
    acquired: tuple[str, ...],
    *,
    alternative_valid: bool = False,
) -> tuple[str, str]:
    expected = set(definition.edit_paths)
    found = expected.intersection(acquired)
    if found == expected:
        return DiscoveryScore.EXPECTED_PATH_FOUND.value, "SUFFICIENT"
    if found:
        return DiscoveryScore.INSUFFICIENT_SOURCE.value, "PARTIAL"
    if acquired and alternative_valid:
        return DiscoveryScore.ALTERNATIVE_VALID_SOURCE_FOUND.value, "SUFFICIENT"
    if acquired:
        return DiscoveryScore.MISDIRECTED_SOURCE.value, "MISDIRECTED"
    return DiscoveryScore.INSUFFICIENT_SOURCE.value, "MISDIRECTED"


def first_failure(
    definition: PilotV2Task,
    *,
    discovery_calls: int,
    source_acquired: bool,
    context_quality: str,
    ready: bool,
    schema: bool,
    transaction: bool,
    verification: bool,
    repair_attempted: bool,
    semantic: bool,
) -> str:
    if (
        definition.authority_mode is AuthorityMode.DISCOVERY_REQUIRED
        and discovery_calls == 0
    ):
        return Failure.DISCOVERY_FAILED
    if not source_acquired:
        return Failure.SOURCE_ACQUISITION_FAILED
    if context_quality != "SUFFICIENT":
        return Failure.CONTEXT_INSUFFICIENT
    if not ready:
        return Failure.AUTHORITY_NOT_READY
    if not schema:
        return Failure.PROTOCOL_FAILED
    if not transaction:
        return Failure.TRANSACTION_FAILED
    if not verification:
        return (
            Failure.REPAIR_FAILED if repair_attempted else Failure.VERIFICATION_FAILED
        )
    if not semantic:
        return Failure.SEMANTIC_FAILED
    return Failure.PASS


def run_cell(
    definition: PilotV2Task,
    backend: Model,
    snapshot: Path,
    *,
    profile: str,
    artifact: str,
    repository_identity: str,
    manifest_identity: str,
    representation: MutationRepresentationPolicy,
) -> PilotV2Cell:
    validate_boundary(definition)
    recorder = RecordingModel(backend, frozenset(definition.edit_paths))
    raw = (
        RealWorldEvaluationRunner(
            profile,
            recorder,
            snapshot,
            mutation_representation=representation,
            authorize_discovered_sources=True,
        )
        .run((definition.production_task,), snapshot_description(snapshot))
        .results[0]
    )
    metrics = raw.metrics
    inspected = tuple(metrics.files_inspected)
    outcome, quality = classify_context(definition, inspected)
    source_acquired = bool(inspected)
    schema, _duplicate, _missing, _unexpected = proposal_shape(
        definition, tuple(recorder.envelopes)
    )
    transaction = bool(metrics.mutations) and not raw.unexpected_paths
    verification = (
        metrics.verification_plan_result == "pass"
        or metrics.reverification_result == "pass"
    )
    repair_attempted = bool(metrics.repair_attempts)
    semantic = raw.oracle is EvaluationOutcome.PASS
    failure = first_failure(
        definition,
        discovery_calls=metrics.discovery_calls,
        source_acquired=source_acquired,
        context_quality=quality,
        ready=metrics.mutation_ready_reached,
        schema=schema,
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
    return PilotV2Cell(
        SUITE,
        VERSION,
        definition.task_id,
        definition.version,
        definition.operation_class.value,
        definition.authority_mode.value,
        profile,
        artifact,
        SEED,
        repository_identity,
        manifest_identity,
        outcome,
        metrics.discovery_calls,
        metrics.bootstrap_executions,
        0,
        metrics.lexical_index_builds,
        0,
        metrics.lexical_index_refreshes,
        round(metrics.lexical_index_duration_seconds, 3),
        metrics.source_reads,
        source_acquired,
        quality,
        inspected,
        metrics.mutation_ready_reached,
        schema,
        transaction,
        metrics.verification_plan_result,
        repair_attempted,
        repair_attempted and verification and semantic,
        semantic,
        failure,
        raw.final_status,
        raw.changed_paths,
        raw.usage.input_tokens,
        raw.usage.output_tokens,
        recorder.calls,
        metrics.tool_executions,
        round(recorder.generation_seconds, 3),
        round(verification_seconds, 3),
        round(raw.elapsed_seconds, 3),
    )


def checkpoint_path(root: Path, profile: str, task_id: str) -> Path:
    return root / "results" / f"{profile}-seed42-{task_id}.json"


def atomic_json(path: Path, payload: object) -> None:
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


def commit_cell(path: Path, cell: PilotV2Cell) -> None:
    atomic_json(path, asdict(cell))


def read_checkpoint(
    path: Path, expected: dict[str, object]
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"A62 checkpoint identity mismatch: {path.name}")
    return payload


def standard_result_is_source_free(payload: dict[str, object]) -> bool:
    forbidden = {"replacement", "source", "old_text", "new_text", "content"}
    return not any(key in forbidden for key in payload)


def summarize(cells: tuple[dict[str, object], ...]) -> dict[str, object]:
    profiles: dict[str, object] = {}
    for profile in sorted({str(cell["model_profile"]) for cell in cells}):
        selected = [cell for cell in cells if cell["model_profile"] == profile]
        profiles[profile] = {
            "cells": len(selected),
            "context_sufficient": sum(
                item["context_quality"] == "SUFFICIENT" for item in selected
            ),
            "transactions": sum(bool(item["transaction_success"]) for item in selected),
            "verification_pass": sum(
                item["verification_result"] == "pass" for item in selected
            ),
            "semantic_pass": sum(bool(item["semantic_pass"]) for item in selected),
            "first_failures": {
                layer: sum(item["first_failure_layer"] == layer for item in selected)
                for layer in sorted(
                    {str(item["first_failure_layer"]) for item in selected}
                )
            },
        }
    return {"suite": SUITE, "cells": len(cells), "profiles": profiles}
