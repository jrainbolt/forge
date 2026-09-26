"""A57 primary capture, isolated conditions, and source-free checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.realistic_coding_v2.runner import _atomic_json, snapshot
from benchmarks.realistic_coding_v2.suite import REPOSITORY, manifest_identity, tasks
from benchmarks.repair_framing_v1.suite import (
    SUITE,
    VERSION,
    Case,
    authority_sufficient,
    corrective_task,
)
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    hash_workspace,
)
from forge.models import (
    FinishReason,
    Model,
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    MutationRepresentationPolicy,
)
from forge.orchestration.verification_attribution import failure_fingerprint

LOCAL_WARNING = "LOCAL EVALUATOR DATA — MAY CONTAIN MODEL-GENERATED SOURCE."
MUTATION_TOOLS = frozenset(
    {
        "repository.apply_patch",
        "repository.apply_multi_patch",
        "repository.apply_file_operations",
        "repository.create_text_files",
        "repository.write_file",
    }
)


class StopAtRepair(RuntimeError):
    """Planned capture boundary, not a model or production failure."""


@dataclass(frozen=True, slots=True)
class PrimaryRecord:
    case_id: str
    profile: str
    task_id: str
    task_version: int
    seed: int
    operation_class: str
    language: str
    a56_manifest: str
    model_artifact: str
    primary_mutation_hash: str
    post_primary_hash: str
    source_hashes: tuple[tuple[str, str], ...]
    evidence_hash: str
    failure_step: str
    failure_fingerprint: str | None
    authorized_paths: tuple[str, ...]
    authority_sufficient: str
    eligible: bool
    exclusion: str | None


@dataclass(frozen=True, slots=True)
class Trial:
    suite: str
    schema_version: int
    case_id: str
    condition: str
    profile: str
    task_id: str
    seed: int
    start_hash: str
    manifest_hash: str
    authorized_paths: tuple[str, ...]
    original_task_present: bool
    evidence_present: bool
    repair_framing: bool
    create_authority: bool
    fresh_source_acquired: bool
    ready: bool
    schema_valid: bool
    authorized_path_set: bool
    material_mutation: bool
    transaction_executed: bool
    full_verification_executed: bool
    verification_passed: bool
    failing_step: str | None
    failure_fingerprint: str | None
    fingerprint_changed: bool | None
    semantic_passed: bool
    semantic_recovered: bool
    classification: str
    input_tokens: int | None
    output_tokens: int | None
    model_calls: int
    generation_seconds: float
    verification_seconds: float
    total_seconds: float


def tree_hash(root: Path) -> str:
    """Bind all file bytes, paths, and modes in a post-primary snapshot."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"file\0")
            digest.update(str(path.stat().st_mode & 0o777).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
        elif path.is_dir():
            digest.update(b"dir\0")
    return digest.hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _diagnostic(request: ModelRequest) -> str | None:
    """Select the exact bounded failing project result already in R0 context."""
    for message in request.messages:
        try:
            payload = json.loads(message.content)
        except ValueError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("type") == "tool_result"
            and payload.get("status") == "failure"
            and payload.get("tool")
            in {"project.configure", "project.build", "project.test"}
        ):
            return message.content
    return None


def _evidence_failure(evidence: str | None, workspace: Path) -> tuple[str, str | None]:
    if evidence is None:
        return "unavailable", None
    value = json.loads(evidence)
    tool = value.get("tool", "unavailable")
    output = value.get("output")
    fingerprint = (
        failure_fingerprint(tool, output, workspace)
        if isinstance(tool, str) and isinstance(output, Mapping)
        else None
    )
    return str(tool), fingerprint


class CaptureModel(Model):
    def __init__(
        self,
        backend: Model,
        *,
        prefix: tuple[ModelResponse, ...] = (),
        stop_at_repair: bool = False,
    ) -> None:
        self.backend = backend
        self.prefix = prefix
        self.stop_at_repair = stop_at_repair
        self.responses: list[ModelResponse] = []
        self.requests: list[ModelRequest] = []
        self.repair_request: ModelRequest | None = None
        self.repair_started: float | None = None
        self.generation_seconds = 0.0
        self.backend_calls = 0

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
        if "final permitted repair mutation" in "\n".join(
            message.content for message in request.messages
        ):
            self.repair_request = request
            if self.repair_started is None:
                self.repair_started = time.perf_counter()
            if self.stop_at_repair:
                raise StopAtRepair("planned A57 capture before repair generation")
        index = len(self.responses)
        self.requests.append(request)
        if index < len(self.prefix):
            response = self.prefix[index]
        else:
            started = time.perf_counter()
            response = self.backend.generate(request)
            self.generation_seconds += time.perf_counter() - started
            self.backend_calls += 1
        self.responses.append(response)
        return response

    def close(self) -> None:
        return None


def _response_payload(response: ModelResponse) -> dict[str, object]:
    return {
        "text": response.text,
        "finish_reason": response.finish_reason.value,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def _restore_response(payload: dict[str, object], model: Model) -> ModelResponse:
    return ModelResponse(
        str(payload["text"]),
        FinishReason(str(payload["finish_reason"])),
        model.identity,
        ModelUsage(
            payload["input_tokens"],  # type: ignore[arg-type]
            payload["output_tokens"],  # type: ignore[arg-type]
        ),
    )


def _failure_activity(item: object, workspace: Path) -> tuple[str, str | None] | None:
    tool = getattr(item, "tool_name", None)
    status = getattr(item, "status", None)
    output = getattr(item, "output", None)
    if (
        tool not in {"project.configure", "project.build", "project.test"}
        or getattr(status, "value", status) != "failure"
        or not isinstance(output, Mapping)
    ):
        return None
    return tool, failure_fingerprint(tool, output, workspace)


def _workspace_copy(root: Path, source: Path, name: str) -> Path:
    destination = root / name
    if destination.exists():
        raise FileExistsError(destination)
    return Path(shutil.copytree(source, destination))


def primary_path(root: Path, case: Case) -> Path:
    return root / "replay" / f"{case.case_id}-primary.json"


def workspace_path(root: Path, case: Case) -> Path:
    return root / "replay" / f"{case.case_id}-workspace"


def capture_primary(
    root: Path,
    case: Case,
    backend: Model,
    *,
    artifact: str,
    a56_manifest: str,
    representation: MutationRepresentationPolicy,
) -> PrimaryRecord:
    """Generate one planned primary, stop at R0 repair, and commit local replay."""
    destination = primary_path(root, case)
    if destination.exists():
        return load_primary(root, case)[0]
    root.joinpath("replay").mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{case.case_id}-", dir=root / "replay"))
    snap: Path | None = None

    def on_primary(_task, workspace: Path) -> None:  # type: ignore[no-untyped-def]
        nonlocal snap
        snap = _workspace_copy(temporary, workspace, "workspace")

    recorder = CaptureModel(backend, stop_at_repair=True)
    production = case.definition.production_task
    raw = (
        RealWorldEvaluationRunner(
            case.profile,
            recorder,
            REPOSITORY,
            mutation_representation=representation,
            primary_mutation_callback=on_primary,
        )
        .run((production,), snapshot(_sha(repr(hash_workspace(REPOSITORY)))))
        .results[0]
    )
    request = recorder.repair_request
    evidence = _diagnostic(request) if request is not None else None
    failure_step, fingerprint = _evidence_failure(evidence, snap or REPOSITORY)
    eligible = bool(
        snap is not None
        and request is not None
        and evidence is not None
        and raw.metrics.mutations == 1
        and raw.oracle is EvaluationOutcome.FAIL
        and raw.metrics.verification_plan_result == "step_failed"
        and raw.metrics.repair_ready_reached
        and not raw.unexpected_paths
        and raw.metrics.attribution_result != "preexisting_or_unrelated"
    )
    exclusion = None if eligible else "PLANNED_PRIMARY_NOT_REPAIR_ELIGIBLE"
    changed = tuple(raw.changed_paths)
    sufficient = (
        authority_sufficient(case, snap, changed)
        if eligible and snap is not None
        else "UNKNOWN"
    )
    if eligible and sufficient != "YES":
        eligible = False
        exclusion = f"REPAIR_AUTHORITY_{sufficient}"
    if snap is not None:
        frozen_workspace = workspace_path(root, case)
        if frozen_workspace.exists():
            if tree_hash(frozen_workspace) != tree_hash(snap):
                raise ValueError("existing A57 snapshot identity mismatch")
        else:
            os.rename(snap, frozen_workspace)
        post_hash = tree_hash(frozen_workspace)
        hashes = hash_workspace(frozen_workspace)
    else:
        post_hash = "unavailable"
        hashes = ()
    temporary.rmdir()
    prefix = tuple(_response_payload(item) for item in recorder.responses)
    prefix_hash = hashlib.sha256(
        json.dumps(prefix, sort_keys=True).encode()
    ).hexdigest()
    record = PrimaryRecord(
        case.case_id,
        case.profile,
        case.task_id,
        case.definition.version,
        case.seed,
        case.definition.operation_class.value,
        case.definition.language,
        a56_manifest,
        artifact,
        prefix_hash,
        post_hash,
        hashes,
        _sha(evidence) if evidence is not None else "unavailable",
        failure_step,
        fingerprint,
        changed,
        sufficient,
        eligible,
        exclusion,
    )
    _atomic_json(
        destination,
        {
            "warning": LOCAL_WARNING,
            "record": asdict(record),
            "responses": prefix,
            "verification_evidence": evidence,
        },
    )
    return record


def load_primary(root: Path, case: Case) -> tuple[PrimaryRecord, dict[str, object]]:
    payload = json.loads(primary_path(root, case).read_text(encoding="utf-8"))
    if payload.get("warning") != LOCAL_WARNING:
        raise ValueError("A57 replay artifact warning missing")
    value = payload["record"]
    value["source_hashes"] = tuple(tuple(pair) for pair in value["source_hashes"])
    value["authorized_paths"] = tuple(value["authorized_paths"])
    record = PrimaryRecord(**value)
    if record.case_id != case.case_id or record.profile != case.profile:
        raise ValueError("A57 primary identity mismatch")
    if (
        record.post_primary_hash != "unavailable"
        and tree_hash(workspace_path(root, case)) != record.post_primary_hash
    ):
        raise ValueError("A57 replay workspace identity mismatch")
    prefix_hash = hashlib.sha256(
        json.dumps(payload["responses"], sort_keys=True).encode()
    ).hexdigest()
    if prefix_hash != record.primary_mutation_hash:
        raise ValueError("A57 primary response replay tampered")
    evidence = payload.get("verification_evidence")
    evidence_hash = _sha(evidence) if isinstance(evidence, str) else "unavailable"
    if evidence_hash != record.evidence_hash:
        raise ValueError("A57 verification evidence tampered")
    return record, payload


def freeze_manifest(
    root: Path, records: tuple[PrimaryRecord, ...]
) -> dict[str, object]:
    """Freeze source-free identities before any R0/R1/R2 condition."""
    if sum(record.eligible for record in records) < 6:
        raise ValueError("fewer than six A57 eligible post-primary states")
    payload: dict[str, object] = {
        "suite": SUITE,
        "schema_version": VERSION,
        "a56_definition_identity": manifest_identity(tasks()),
        "canonical_repository_hash": _sha(repr(hash_workspace(REPOSITORY))),
        "cases": [asdict(record) for record in records],
    }
    path = root / "results" / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("frozen A57 case manifest mismatch")
    else:
        _atomic_json(path, payload)
    return payload


def trial_path(root: Path, case: Case, condition: str) -> Path:
    return root / "results" / f"{case.case_id}-{condition}.json"


def load_trial(
    root: Path, case: Case, condition: str, manifest_hash: str
) -> Trial | None:
    path = trial_path(root, case, condition)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    value["authorized_paths"] = tuple(value["authorized_paths"])
    trial = Trial(**value)
    if (
        trial.case_id != case.case_id
        or trial.condition != condition
        or trial.manifest_hash != manifest_hash
    ):
        raise ValueError("A57 committed condition identity mismatch")
    return trial


def _usage(model: CaptureModel, offset: int) -> tuple[int | None, int | None]:
    values = [item.usage for item in model.responses[offset:]]
    input_tokens = (
        sum(item.input_tokens for item in values if item.input_tokens is not None)
        if values and all(item.input_tokens is not None for item in values)
        else None
    )
    output_tokens = (
        sum(item.output_tokens for item in values if item.output_tokens is not None)
        if values and all(item.output_tokens is not None for item in values)
        else None
    )
    return input_tokens, output_tokens


def _classification(
    *,
    schema: bool,
    changed: bool,
    transaction: bool,
    verification_executed: bool,
    verified: bool,
    semantic: bool,
    verification_improved: bool | None,
) -> str:
    if not schema:
        return "NO_VALID_PROPOSAL"
    if not changed:
        return "NO_CHANGE"
    if not transaction:
        return "TRANSACTION_FAILURE"
    if not verification_executed:
        return "STRUCTURAL_FAILURE"
    if verified and semantic:
        return "SEMANTIC_RECOVERED"
    if verified:
        return "SEMANTIC_STILL_FAIL"
    if semantic:
        return "SEMANTIC_REGRESSED"
    if verification_improved is True:
        return "VERIFICATION_IMPROVED"
    if verification_improved is False:
        return "VERIFICATION_REGRESSED"
    return "VERIFICATION_UNCHANGED"


def _step_improved(initial: str, current: str | None) -> bool | None:
    if current is None:
        return None
    order = {
        "project.configure": 0,
        "configure": 0,
        "project.build": 1,
        "build": 1,
        "project.test": 2,
        "test": 2,
    }
    if initial not in order or current not in order or initial == current:
        return None
    return order[current] > order[initial]


def run_condition(
    root: Path,
    case: Case,
    record: PrimaryRecord,
    replay: dict[str, object],
    backend: Model,
    representation: MutationRepresentationPolicy,
    condition: str,
    manifest_hash: str,
) -> Trial:
    """Run one condition once on its own disposable copy; commit before next."""
    if condition not in {"R0", "R1", "R2"}:
        raise ValueError("unknown A57 condition")
    existing = load_trial(root, case, condition, manifest_hash)
    if existing is not None:
        return existing
    if not record.eligible or record.authority_sufficient != "YES":
        raise ValueError("A57 condition cannot run on an ineligible primary")
    started = time.perf_counter()
    start_hashes: list[str] = []

    if condition == "R0":
        prefix = tuple(_restore_response(item, backend) for item in replay["responses"])
        recorder = CaptureModel(backend, prefix=prefix)

        def on_primary(_task, workspace: Path) -> None:  # type: ignore[no-untyped-def]
            current = tree_hash(workspace)
            start_hashes.append(current)
            if current != record.post_primary_hash:
                raise ValueError("R0 post-primary state does not match frozen replay")

        task = case.definition.production_task
        raw = (
            RealWorldEvaluationRunner(
                case.profile,
                recorder,
                REPOSITORY,
                mutation_representation=representation,
                primary_mutation_callback=on_primary,
            )
            .run((task,), snapshot(record.post_primary_hash))
            .results[0]
        )
        metrics = raw.metrics
        schema = bool(
            metrics.repair_preview_created or metrics.repair_mutation_executed
        )
        transaction = bool(metrics.repair_mutation_executed)
        verification = metrics.reverification_result in {"pass", "passed"}
        verification_executed = bool(metrics.reverification_executed)
        source_ready = bool(metrics.repair_source_refreshed)
        ready = bool(metrics.repair_ready_reached)
        offset = len(prefix)
        request = recorder.repair_request
        evidence_present = request is not None and _diagnostic(request) is not None
        repair_framing = True
        create_authority = False
        authorized = bool(start_hashes) and not raw.unexpected_paths
    else:
        if tree_hash(workspace_path(root, case)) != record.post_primary_hash:
            raise ValueError("A57 post-primary replay changed")
        evidence = replay["verification_evidence"] if condition == "R2" else None
        assert evidence is None or isinstance(evidence, str)
        task = corrective_task(case, record.authorized_paths, evidence)
        recorder = CaptureModel(backend)
        raw = (
            RealWorldEvaluationRunner(
                case.profile,
                recorder,
                workspace_path(root, case),
                mutation_representation=representation,
            )
            .run((task,), snapshot(record.post_primary_hash))
            .results[0]
        )
        metrics = raw.metrics
        schema = bool(
            metrics.structured_mutation_valid
            or metrics.preview_created
            or metrics.mutation_group_preview_created
            or metrics.mutations
        )
        transaction = bool(metrics.mutations)
        verification = metrics.verification_plan_result == "pass"
        verification_executed = bool(metrics.verification_plan_steps_started)
        source_ready = bool(metrics.expected_implementation_acquired)
        ready = bool(metrics.mutation_ready_reached)
        offset = 0
        evidence_present = condition == "R2"
        repair_framing = False
        create_authority = bool(task.create_candidate_paths)
        authorized = not raw.unexpected_paths
        start_hashes.append(record.post_primary_hash)
    semantic = raw.oracle is EvaluationOutcome.PASS
    failing_step = metrics.verification_plan_failed_step
    fingerprint = None
    fingerprint_changed = None
    verification_improved = _step_improved(record.failure_step, failing_step)
    input_tokens, output_tokens = _usage(recorder, offset)
    material = transaction
    total_seconds = (
        time.perf_counter() - recorder.repair_started
        if condition == "R0" and recorder.repair_started is not None
        else time.perf_counter() - started
    )
    trial = Trial(
        SUITE,
        VERSION,
        case.case_id,
        condition,
        case.profile,
        case.task_id,
        case.seed,
        record.post_primary_hash,
        manifest_hash,
        record.authorized_paths,
        case.definition.production_task.prompt in task.prompt,
        evidence_present,
        repair_framing,
        create_authority,
        source_ready,
        ready,
        schema,
        authorized,
        material,
        transaction,
        verification_executed,
        verification,
        failing_step,
        fingerprint,
        fingerprint_changed,
        semantic,
        transaction and verification_executed and verification and semantic,
        _classification(
            schema=schema,
            changed=material,
            transaction=transaction,
            verification_executed=verification_executed,
            verified=verification,
            semantic=semantic,
            verification_improved=verification_improved,
        ),
        input_tokens,
        output_tokens,
        recorder.backend_calls,
        round(recorder.generation_seconds, 3),
        round(metrics.verification_plan_duration, 3),
        round(total_seconds, 3),
    )
    _atomic_json(trial_path(root, case, condition), asdict(trial))
    return trial


def diagnose(trials: tuple[Trial, Trial, Trial]) -> str:
    by_condition = {item.condition: item for item in trials}
    r0, r1, r2 = (by_condition[key] for key in ("R0", "R1", "R2"))
    if r2.semantic_recovered and not r1.semantic_recovered:
        return "VERIFICATION_EVIDENCE_HELPFUL"
    if r1.semantic_recovered and not r2.semantic_recovered:
        return "VERIFICATION_EVIDENCE_HARMFUL"
    if not r0.semantic_recovered and (r1.semantic_recovered or r2.semantic_recovered):
        return "REPAIR_FRAMING_LIMITATION"
    if not any(item.semantic_recovered for item in trials):
        return "MODEL_LIMITED"
    if r0.semantic_recovered:
        return "CURRENT_REPAIR_COMPETITIVE"
    return "MIXED_INCONCLUSIVE"
