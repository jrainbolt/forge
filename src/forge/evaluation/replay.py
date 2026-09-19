"""Local-only evaluator replay bundles with fail-closed reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from forge.evaluation.acceptance_test_synthesis import (
    GeneratedAcceptanceTest,
    execute_candidate,
    validate_generated_acceptance_test,
)
from forge.evaluation.realworld import (
    SetupReplacement,
    apply_task_setup,
    copy_repository,
)

EVALUATION_REPLAY_V1 = "evaluation-replay-v1"
REPLAY_FORMAT_VERSION = 1
MAX_REPLAY_PAYLOAD_BYTES = 64 * 1024
LOCAL_REPLAY_WARNING = "LOCAL EVALUATOR DATA — MAY CONTAIN MODEL-GENERATED SOURCE."


class ReplayArtifactType(Enum):
    MUTATION_PROPOSAL = "MUTATION_PROPOSAL"
    GENERATED_ACCEPTANCE_TEST = "GENERATED_ACCEPTANCE_TEST"


class ReplayFailure(Enum):
    ARTIFACT_INVALID = "ARTIFACT_INVALID"
    ARTIFACT_TAMPERED = "ARTIFACT_TAMPERED"
    REPOSITORY_MISMATCH = "REPOSITORY_MISMATCH"
    SOURCE_STATE_MISMATCH = "SOURCE_STATE_MISMATCH"
    REPLAY_UNSAFE = "REPLAY_UNSAFE"
    PATCH_UNAVAILABLE = "PATCH_UNAVAILABLE"
    TEST_UNAVAILABLE = "TEST_UNAVAILABLE"
    REPLAY_PASS = "REPLAY_PASS"
    REPLAY_FAIL = "REPLAY_FAIL"


class ReplayValidationError(ValueError):
    def __init__(self, failure: ReplayFailure, message: str) -> None:
        super().__init__(message)
        self.failure = failure


@dataclass(frozen=True, slots=True)
class ReplayBundle:
    format_version: int
    benchmark_identity: str
    repository_identity: str
    task_id: str
    task_version: int
    source_state_identity: str
    model_identity: str
    seed: int
    artifact_type: ReplayArtifactType
    payload: dict[str, object]
    payload_sha256: str
    created_by_evaluator: bool = True
    warning: str = LOCAL_REPLAY_WARNING

    @property
    def artifact_id(self) -> str:
        material = (
            self.benchmark_identity,
            self.task_id,
            self.task_version,
            self.model_identity,
            self.seed,
            self.artifact_type.value,
        )
        suffix = hashlib.sha256(json.dumps(material).encode()).hexdigest()[:16]
        safe_model = "".join(
            character if character.isalnum() else "-"
            for character in self.model_identity.casefold()
        ).strip("-")
        return (
            f"{self.benchmark_identity.split(':')[0]}-{self.task_id.casefold()}-"
            f"{safe_model}-seed{self.seed}-{self.artifact_type.value.casefold()}-{suffix}"
        )


@dataclass(frozen=True, slots=True)
class ReplayManifestEntry:
    artifact_id: str
    artifact_type: str
    task_id: str
    model_identity: str
    seed: int
    payload_sha256: str
    benchmark_identity: str
    filename: str


@dataclass(frozen=True, slots=True)
class MutationReplayResult:
    workspace: Path
    file_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CrossModelObservation:
    task_id: str
    test_model: str
    patch_model: str
    patch_semantic_truth: str
    test_outcome: str
    agreement: str
    test_artifact_id: str
    test_payload_sha256: str
    patch_artifact_id: str
    patch_payload_sha256: str
    benchmark_identity: str
    replay_seconds: float


@dataclass(frozen=True, slots=True)
class EvaluationReplayRun:
    suite: str
    suite_version: int
    schema_version: int
    benchmark_identity: str
    repository_identity: str
    planned_patch_cells: tuple[tuple[str, str], ...]
    planned_test_cells: tuple[tuple[str, str], ...]
    unavailable_patch_cells: tuple[tuple[str, str], ...]
    unavailable_test_cells: tuple[tuple[str, str], ...]
    observations: tuple[CrossModelObservation, ...]
    canonical_unchanged: bool


@dataclass(frozen=True, slots=True)
class ReplayCell:
    task_id: str
    model_identity: str
    artifact_type: str
    status: str
    artifact_id: str | None
    payload_sha256: str | None
    elapsed_seconds: float
    semantic_truth: str | None = None


@dataclass(frozen=True, slots=True)
class TestQualificationObservation:
    task_id: str
    test_model: str
    baseline: str
    reference: str
    reference_repeat: str
    wrong: str
    qualification: str
    artifact_id: str
    payload_sha256: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class AlternativeObservation:
    task_id: str
    test_model: str
    outcome: str
    artifact_id: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationReplayReport:
    suite: str
    suite_version: int
    schema_version: int
    benchmark_identity: str
    repository_identity: str
    seed: int
    patch_cells: tuple[ReplayCell, ...]
    test_cells: tuple[ReplayCell, ...]
    qualifications: tuple[TestQualificationObservation, ...]
    cross_model_observations: tuple[CrossModelObservation, ...]
    alternatives: tuple[AlternativeObservation, ...]
    model_coupled_signal: bool
    canonical_unchanged: bool


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_sha256(payload: dict[str, object]) -> str:
    encoded = canonical_json(payload)
    if len(encoded) > MAX_REPLAY_PAYLOAD_BYTES:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "replay payload exceeds size limit"
        )
    return hashlib.sha256(encoded).hexdigest()


def source_state_identity(root: Path) -> str:
    resolved = root.resolve(strict=True)
    values = []
    for path in sorted(resolved.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(resolved).as_posix()
            if any(
                part in {"build", "__pycache__", ".pytest_cache"} for part in path.parts
            ):
                continue
            values.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    return hashlib.sha256(canonical_json(values)).hexdigest()


def repository_identity(root: Path) -> str:
    return source_state_identity(root)


def build_mutation_payload(
    baseline: Path, mutated: Path, paths: tuple[str, ...]
) -> dict[str, object]:
    """Capture bounded accepted line replacements, never an arbitrary tree."""
    edits: list[dict[str, object]] = []
    for relative in paths:
        before_path = _confined_file(baseline, relative)
        after_path = _confined_file(mutated, relative)
        before = before_path.read_text(encoding="utf-8")
        after = after_path.read_text(encoding="utf-8")
        if before == after:
            continue
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        prefix = 0
        while (
            prefix < len(before_lines)
            and prefix < len(after_lines)
            and before_lines[prefix] == after_lines[prefix]
        ):
            prefix += 1
        suffix = 0
        while (
            suffix < len(before_lines) - prefix
            and suffix < len(after_lines) - prefix
            and before_lines[-suffix - 1] == after_lines[-suffix - 1]
        ):
            suffix += 1
        before_end = len(before_lines) - suffix
        after_end = len(after_lines) - suffix
        if prefix == before_end:
            # Represent an insertion as replacement of one unchanged anchor line.
            # Replay still enforces ordinary LINE_RANGE old-text currentness.
            if prefix < len(before_lines):
                before_end += 1
                after_end += 1
            elif prefix > 0:
                prefix -= 1
            else:
                raise ReplayValidationError(
                    ReplayFailure.ARTIFACT_INVALID,
                    "cannot anchor insertion in an empty source file",
                )
        old_text = "".join(before_lines[prefix:before_end])
        new_text = "".join(after_lines[prefix:after_end])
        edits.append(
            {
                "path": relative,
                "start_line": prefix + 1,
                "end_line": before_end,
                "old_text": old_text,
                "new_text": new_text,
                "pre_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "post_sha256": hashlib.sha256(after.encode()).hexdigest(),
            }
        )
    if not edits:
        raise ReplayValidationError(
            ReplayFailure.PATCH_UNAVAILABLE, "accepted mutation changed no allowed path"
        )
    return {"edits": edits, "grouped": len(edits) > 1}


def create_replay_bundle(
    *,
    benchmark_identity: str,
    repository_identity_value: str,
    task_id: str,
    task_version: int,
    source_state_identity_value: str,
    model_identity: str,
    seed: int,
    artifact_type: ReplayArtifactType,
    payload: dict[str, object],
) -> ReplayBundle:
    _validate_payload_shape(artifact_type, payload)
    return ReplayBundle(
        REPLAY_FORMAT_VERSION,
        benchmark_identity,
        repository_identity_value,
        task_id,
        task_version,
        source_state_identity_value,
        model_identity,
        seed,
        artifact_type,
        payload,
        payload_sha256(payload),
    )


def replay_bundle_to_dict(bundle: ReplayBundle) -> dict[str, object]:
    value = asdict(bundle)
    value["artifact_type"] = bundle.artifact_type.value
    return value


def replay_bundle_from_dict(value: object) -> ReplayBundle:
    if not isinstance(value, dict):
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "replay bundle must be an object"
        )
    try:
        artifact_type = ReplayArtifactType(value["artifact_type"])
        payload = value["payload"]
        if not isinstance(payload, dict):
            raise TypeError
        bundle = ReplayBundle(
            int(value["format_version"]),
            str(value["benchmark_identity"]),
            str(value["repository_identity"]),
            str(value["task_id"]),
            int(value["task_version"]),
            str(value["source_state_identity"]),
            str(value["model_identity"]),
            int(value["seed"]),
            artifact_type,
            payload,
            str(value["payload_sha256"]),
            value.get("created_by_evaluator") is True,
            str(value.get("warning", "")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "replay bundle fields are malformed"
        ) from error
    validate_replay_bundle(bundle)
    return bundle


def validate_replay_bundle(bundle: ReplayBundle) -> None:
    if bundle.format_version != REPLAY_FORMAT_VERSION:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "unsupported replay format version"
        )
    if not bundle.created_by_evaluator or bundle.warning != LOCAL_REPLAY_WARNING:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "replay provenance is invalid"
        )
    for label, value in (
        ("benchmark", bundle.benchmark_identity),
        ("task", bundle.task_id),
        ("model", bundle.model_identity),
    ):
        if not value or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) is None:
            raise ReplayValidationError(
                ReplayFailure.ARTIFACT_INVALID, f"{label} identity is invalid"
            )
    _validate_payload_shape(bundle.artifact_type, bundle.payload)
    if payload_sha256(bundle.payload) != bundle.payload_sha256:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_TAMPERED, "replay payload hash mismatch"
        )


def write_replay_bundle_atomic(
    bundle: ReplayBundle, replay_root: Path
) -> ReplayManifestEntry:
    validate_replay_bundle(bundle)
    replay_root.mkdir(parents=True, exist_ok=True)
    existing = completed_artifact(replay_root, bundle.artifact_id)
    if existing is not None:
        if existing != bundle:
            raise ReplayValidationError(
                ReplayFailure.ARTIFACT_TAMPERED,
                "committed replay cell cannot be overwritten",
            )
        return next(
            item
            for item in load_replay_manifest(replay_root)
            if item.artifact_id == bundle.artifact_id
        )
    filename = f"{bundle.artifact_id}.json"
    _atomic_json_write(replay_root / filename, replay_bundle_to_dict(bundle))
    entry = ReplayManifestEntry(
        bundle.artifact_id,
        bundle.artifact_type.value,
        bundle.task_id,
        bundle.model_identity,
        bundle.seed,
        bundle.payload_sha256,
        bundle.benchmark_identity,
        filename,
    )
    entries = {item.artifact_id: item for item in load_replay_manifest(replay_root)}
    entries[entry.artifact_id] = entry
    _atomic_json_write(
        replay_root / "manifest.json",
        {
            "format_version": REPLAY_FORMAT_VERSION,
            "warning": LOCAL_REPLAY_WARNING,
            "artifacts": [asdict(entries[key]) for key in sorted(entries)],
        },
    )
    return entry


def load_replay_bundle(path: Path) -> ReplayBundle:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "cannot read replay bundle"
        ) from error
    return replay_bundle_from_dict(value)


def load_replay_manifest(replay_root: Path) -> tuple[ReplayManifestEntry, ...]:
    path = replay_root / "manifest.json"
    if not path.is_file():
        return ()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format_version") != REPLAY_FORMAT_VERSION:
            raise ValueError
        entries = tuple(ReplayManifestEntry(**item) for item in value["artifacts"])
        if len({item.artifact_id for item in entries}) != len(entries):
            raise ValueError
        for item in entries:
            if (
                not item.artifact_id
                or Path(item.filename).name != item.filename
                or item.filename != f"{item.artifact_id}.json"
                or not item.filename.endswith(".json")
            ):
                raise ValueError
        return entries
    except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "replay manifest is malformed"
        ) from error


def completed_artifact(
    replay_root: Path,
    artifact_id: str,
) -> ReplayBundle | None:
    entries = {item.artifact_id: item for item in load_replay_manifest(replay_root)}
    entry = entries.get(artifact_id)
    if entry is None:
        return None
    bundle = load_replay_bundle(replay_root / entry.filename)
    if bundle.payload_sha256 != entry.payload_sha256:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_TAMPERED, "manifest payload hash mismatch"
        )
    if (
        bundle.artifact_id != entry.artifact_id
        or bundle.artifact_type.value != entry.artifact_type
        or bundle.task_id != entry.task_id
        or bundle.model_identity != entry.model_identity
        or bundle.seed != entry.seed
        or bundle.benchmark_identity != entry.benchmark_identity
    ):
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_TAMPERED, "manifest identity mismatch"
        )
    return bundle


def replay_mutation_bundle(
    bundle: ReplayBundle,
    canonical_repository: Path,
    destination: Path,
    setup: tuple[SetupReplacement, ...],
    *,
    expected_benchmark_identity: str,
    expected_task_id: str | None = None,
) -> MutationReplayResult:
    validate_replay_bundle(bundle)
    if bundle.artifact_type is not ReplayArtifactType.MUTATION_PROPOSAL:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "bundle is not a mutation proposal"
        )
    _validate_identities(bundle, canonical_repository, expected_benchmark_identity)
    if expected_task_id is not None and bundle.task_id != expected_task_id:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "task identity mismatch"
        )
    workspace = copy_repository(canonical_repository, destination)
    apply_task_setup(workspace, setup)
    if source_state_identity(workspace) != bundle.source_state_identity:
        raise ReplayValidationError(
            ReplayFailure.SOURCE_STATE_MISMATCH, "source-state identity mismatch"
        )
    edits = bundle.payload["edits"]
    assert isinstance(edits, list)
    prepared: list[tuple[Path, bytes, bytes, str]] = []
    for item in edits:
        assert isinstance(item, dict)
        path = _confined_file(workspace, str(item["path"]))
        original = path.read_bytes()
        if hashlib.sha256(original).hexdigest() != item["pre_sha256"]:
            raise ReplayValidationError(
                ReplayFailure.SOURCE_STATE_MISMATCH, "mutation pre-hash mismatch"
            )
        text = original.decode("utf-8")
        lines = text.splitlines(keepends=True)
        start = int(item["start_line"])
        end = int(item["end_line"])
        old_text = str(item["old_text"])
        if not 1 <= start <= end <= len(lines):
            raise ReplayValidationError(
                ReplayFailure.SOURCE_STATE_MISMATCH, "mutation range is stale"
            )
        if "".join(lines[start - 1 : end]) != old_text:
            raise ReplayValidationError(
                ReplayFailure.SOURCE_STATE_MISMATCH, "mutation old text mismatch"
            )
        replacement = str(item["new_text"])
        updated = "".join((*lines[: start - 1], replacement, *lines[end:]))
        if hashlib.sha256(updated.encode()).hexdigest() != item["post_sha256"]:
            raise ReplayValidationError(
                ReplayFailure.ARTIFACT_INVALID, "mutation post-hash mismatch"
            )
        prepared.append((path, original, updated.encode(), str(item["path"])))
    touched: list[tuple[Path, bytes]] = []
    try:
        for path, original, content, _relative in prepared:
            touched.append((path, original))
            path.write_bytes(content)
    except OSError as error:
        try:
            for path, original in reversed(touched):
                path.write_bytes(original)
        except OSError as rollback_error:
            raise ReplayValidationError(
                ReplayFailure.REPLAY_UNSAFE,
                "grouped replay write failed and rollback was incomplete",
            ) from rollback_error
        raise ReplayValidationError(
            ReplayFailure.REPLAY_UNSAFE, "grouped replay write failed; rolled back"
        ) from error
    return MutationReplayResult(
        workspace,
        tuple(
            (relative, hashlib.sha256(content).hexdigest())
            for _path, _original, content, relative in prepared
        ),
    )


def replay_generated_test_bundle(
    bundle: ReplayBundle,
    workspace: Path,
    *,
    expected_benchmark_identity: str,
    canonical_repository: Path,
    expected_task_id: str | None = None,
    source_setup: tuple[SetupReplacement, ...] = (),
) -> tuple[ReplayFailure, float]:
    started = time.perf_counter()
    validate_replay_bundle(bundle)
    if bundle.artifact_type is not ReplayArtifactType.GENERATED_ACCEPTANCE_TEST:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "bundle is not a generated test"
        )
    _validate_identities(bundle, canonical_repository, expected_benchmark_identity)
    if expected_task_id is not None and bundle.task_id != expected_task_id:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "task identity mismatch"
        )
    with tempfile.TemporaryDirectory(prefix="forge-a50-test-source-") as name:
        original_source = copy_repository(canonical_repository, Path(name) / "source")
        apply_task_setup(original_source, source_setup)
        if source_state_identity(original_source) != bundle.source_state_identity:
            raise ReplayValidationError(
                ReplayFailure.SOURCE_STATE_MISMATCH,
                "generated-test source-state identity mismatch",
            )
    candidate = GeneratedAcceptanceTest(
        str(bundle.payload["test_name"]), str(bundle.payload["test_source"])
    )
    language = str(bundle.payload["language"])
    validation = validate_generated_acceptance_test(candidate, language)
    if not validation.safe:
        return ReplayFailure.REPLAY_UNSAFE, time.perf_counter() - started
    if not validation.structurally_valid:
        return ReplayFailure.TEST_UNAVAILABLE, time.perf_counter() - started
    result = execute_candidate(candidate, language, workspace)
    failure = (
        ReplayFailure.REPLAY_PASS
        if result == "pass"
        else ReplayFailure.REPLAY_FAIL
        if result == "fail"
        else ReplayFailure.TEST_UNAVAILABLE
    )
    return failure, time.perf_counter() - started


def evaluation_replay_to_dict(run: EvaluationReplayRun) -> dict[str, object]:
    value = asdict(run)
    return value


def evaluation_replay_report_to_dict(
    report: EvaluationReplayReport,
) -> dict[str, object]:
    return asdict(report)


def write_evaluation_replay_report_json(
    report: EvaluationReplayReport, path: Path
) -> None:
    payload = evaluation_replay_report_to_dict(report)
    _validate_standard_result_is_source_free(payload)
    _atomic_json_write(path, payload)


def record_unavailable_cell_atomic(
    checkpoint_root: Path,
    *,
    task_id: str,
    model_identity: str,
    artifact_type: ReplayArtifactType,
    reason: ReplayFailure,
    elapsed_seconds: float,
) -> Path:
    """Commit a no-artifact result so resume cannot success-seek by rerunning it."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(
        character if character.isalnum() else "-"
        for character in model_identity.casefold()
    ).strip("-")
    path = checkpoint_root / (
        f"{task_id.casefold()}-{safe_model}-{artifact_type.value.casefold()}.json"
    )
    _atomic_json_write(
        path,
        {
            "format_version": REPLAY_FORMAT_VERSION,
            "task_id": task_id,
            "model_identity": model_identity,
            "seed": 42,
            "artifact_type": artifact_type.value,
            "status": reason.value,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    return path


def record_cell_result_atomic(
    checkpoint_root: Path,
    cell: ReplayCell,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    tool_calls: int | None = None,
    model_calls: int | None = None,
    verification_result: str | None = None,
) -> Path:
    """Write bounded source-free generation metadata after the raw artifact commit."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(
        character if character.isalnum() else "-"
        for character in cell.model_identity.casefold()
    ).strip("-")
    path = checkpoint_root / (
        f"{cell.task_id.casefold()}-{safe_model}-{cell.artifact_type.casefold()}.json"
    )
    _atomic_json_write(
        path,
        {
            **asdict(cell),
            "format_version": REPLAY_FORMAT_VERSION,
            "seed": 42,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tool_calls": tool_calls,
            "model_calls": model_calls,
            "verification_result": verification_result,
        },
    )
    return path


def load_committed_cell(
    replay_root: Path,
    checkpoint_root: Path,
    *,
    task_id: str,
    model_identity: str,
    artifact_type: ReplayArtifactType,
) -> ReplayBundle | ReplayFailure | None:
    for entry in load_replay_manifest(replay_root):
        if (
            entry.task_id == task_id
            and entry.model_identity == model_identity
            and entry.seed == 42
            and entry.artifact_type == artifact_type.value
        ):
            return completed_artifact(replay_root, entry.artifact_id)
    safe_model = "".join(
        character if character.isalnum() else "-"
        for character in model_identity.casefold()
    ).strip("-")
    path = checkpoint_root / (
        f"{task_id.casefold()}-{safe_model}-{artifact_type.value.casefold()}.json"
    )
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value["format_version"] != REPLAY_FORMAT_VERSION
            or value["task_id"] != task_id
            or value["model_identity"] != model_identity
            or value["seed"] != 42
            or value["artifact_type"] != artifact_type.value
        ):
            raise ValueError
        return ReplayFailure(value["status"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "cell checkpoint is malformed"
        ) from error


def qualify_replayed_test(
    bundle: ReplayBundle,
    root: Path,
    canonical_repository: Path,
    task: object,
    wrong_mutation: tuple[SetupReplacement, ...],
) -> TestQualificationObservation:
    """Rebuild independent B/R/W states and execute the currently revalidated test."""
    from forge.evaluation.acceptance_test_synthesis import classify_qualification

    metadata = task
    started = time.perf_counter()
    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="forge-a50-qualify-", dir=root) as name:
        temporary = Path(name)
        workspaces: dict[str, Path] = {}
        for state in ("baseline", "reference", "wrong"):
            workspace = copy_repository(canonical_repository, temporary / state)
            if state != "reference":
                apply_task_setup(workspace, metadata.production_task.setup)
            if state == "wrong":
                if wrong_mutation:
                    apply_task_setup(workspace, wrong_mutation)
                else:
                    test_path = metadata.production_task.expected_changed_paths[1]
                    shutil.copyfile(
                        canonical_repository / test_path, workspace / test_path
                    )
            workspaces[state] = workspace
            outcome, _ = replay_generated_test_bundle(
                bundle,
                workspace,
                expected_benchmark_identity=bundle.benchmark_identity,
                canonical_repository=canonical_repository,
                expected_task_id=metadata.task_id,
                source_setup=metadata.production_task.setup,
            )
            results[state] = _test_outcome(outcome)
        repeated, _ = replay_generated_test_bundle(
            bundle,
            workspaces["reference"],
            expected_benchmark_identity=bundle.benchmark_identity,
            canonical_repository=canonical_repository,
            expected_task_id=metadata.task_id,
            source_setup=metadata.production_task.setup,
        )
        results["reference_repeat"] = _test_outcome(repeated)
    qualification = classify_qualification(
        results["baseline"],
        results["reference"],
        results["wrong"],
        reference_repeat=results["reference_repeat"],
    )
    return TestQualificationObservation(
        metadata.task_id,
        bundle.model_identity,
        results["baseline"],
        results["reference"],
        results["reference_repeat"],
        results["wrong"],
        qualification.value,
        bundle.artifact_id,
        bundle.payload_sha256,
        time.perf_counter() - started,
    )


def write_evaluation_replay_json(run: EvaluationReplayRun, path: Path) -> None:
    _atomic_json_write(path, evaluation_replay_to_dict(run))


def _validate_identities(
    bundle: ReplayBundle,
    canonical_repository: Path,
    expected_benchmark_identity: str,
) -> None:
    if bundle.benchmark_identity != expected_benchmark_identity:
        raise ReplayValidationError(
            ReplayFailure.REPOSITORY_MISMATCH, "benchmark identity mismatch"
        )
    if repository_identity(canonical_repository) != bundle.repository_identity:
        raise ReplayValidationError(
            ReplayFailure.REPOSITORY_MISMATCH, "repository identity mismatch"
        )


def _validate_payload_shape(
    artifact_type: ReplayArtifactType, payload: dict[str, object]
) -> None:
    if any(
        marker in canonical_json(payload).decode("utf-8").casefold()
        for marker in ("hidden oracle", "reference mutation", "oracle.py", "api_key")
    ):
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "replay contains forbidden evaluator data"
        )
    if artifact_type is ReplayArtifactType.GENERATED_ACCEPTANCE_TEST:
        if set(payload) != {
            "language",
            "test_name",
            "test_source",
            "validation_version",
        } or not all(isinstance(payload[key], str) for key in payload):
            raise ReplayValidationError(
                ReplayFailure.ARTIFACT_INVALID, "generated-test payload is malformed"
            )
        return
    if (
        set(payload) != {"edits", "grouped"}
        or not isinstance(payload["edits"], list)
        or not isinstance(payload["grouped"], bool)
    ):
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "mutation payload is malformed"
        )
    edits = payload["edits"]
    if not edits or len(edits) > 4:
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "mutation edit count is invalid"
        )
    required = {
        "path",
        "start_line",
        "end_line",
        "old_text",
        "new_text",
        "pre_sha256",
        "post_sha256",
    }
    for item in edits:
        if not isinstance(item, dict) or set(item) != required:
            raise ReplayValidationError(
                ReplayFailure.ARTIFACT_INVALID, "mutation edit is malformed"
            )
        _validate_relative_path(str(item["path"]))
        if not isinstance(item["path"], str):
            raise ReplayValidationError(
                ReplayFailure.ARTIFACT_INVALID, "mutation path is malformed"
            )
        if (
            not isinstance(item["start_line"], int)
            or isinstance(item["start_line"], bool)
            or not isinstance(item["end_line"], int)
            or isinstance(item["end_line"], bool)
            or item["start_line"] < 1
            or item["end_line"] < item["start_line"]
        ):
            raise ReplayValidationError(
                ReplayFailure.ARTIFACT_INVALID, "mutation range is malformed"
            )
        if not all(
            isinstance(item[key], str)
            for key in ("old_text", "new_text", "pre_sha256", "post_sha256")
        ):
            raise ReplayValidationError(
                ReplayFailure.ARTIFACT_INVALID, "mutation text or hashes malformed"
            )
        if any(
            len(str(item[key])) != 64
            or any(character not in "0123456789abcdef" for character in str(item[key]))
            for key in ("pre_sha256", "post_sha256")
        ):
            raise ReplayValidationError(
                ReplayFailure.ARTIFACT_INVALID, "mutation hashes are malformed"
            )
    if payload["grouped"] is not (len(edits) > 1):
        raise ReplayValidationError(
            ReplayFailure.ARTIFACT_INVALID, "grouped marker does not match edits"
        )


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ReplayValidationError(
            ReplayFailure.REPLAY_UNSAFE, "replay path escapes workspace"
        )


def _confined_file(root: Path, value: str) -> Path:
    _validate_relative_path(value)
    path = root.joinpath(*PurePosixPath(value).parts)
    resolved = path.resolve(strict=True)
    if root.resolve() not in resolved.parents or not resolved.is_file():
        raise ReplayValidationError(
            ReplayFailure.REPLAY_UNSAFE, "replay path is not a confined file"
        )
    return resolved


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _test_outcome(outcome: ReplayFailure) -> str:
    if outcome is ReplayFailure.REPLAY_PASS:
        return "pass"
    if outcome is ReplayFailure.REPLAY_FAIL:
        return "fail"
    return "unavailable"


def _validate_standard_result_is_source_free(payload: dict[str, object]) -> None:
    forbidden_keys = {"payload", "test_source", "old_text", "new_text", "source"}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if forbidden_keys.intersection(value):
                raise ReplayValidationError(
                    ReplayFailure.ARTIFACT_INVALID,
                    "standard result contains replay source content",
                )
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
