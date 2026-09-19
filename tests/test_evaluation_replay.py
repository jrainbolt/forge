"""evaluation-replay-v1 deterministic D01-D16 acceptance scenarios."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from forge.evaluation.replay import (
    EVALUATION_REPLAY_V1,
    REPLAY_FORMAT_VERSION,
    EvaluationReplayReport,
    ReplayArtifactType,
    ReplayBundle,
    ReplayCell,
    ReplayFailure,
    ReplayValidationError,
    build_mutation_payload,
    completed_artifact,
    create_replay_bundle,
    load_committed_cell,
    load_replay_bundle,
    load_replay_manifest,
    payload_sha256,
    record_unavailable_cell_atomic,
    replay_bundle_from_dict,
    replay_bundle_to_dict,
    replay_generated_test_bundle,
    replay_mutation_bundle,
    repository_identity,
    source_state_identity,
    validate_replay_bundle,
    write_evaluation_replay_report_json,
    write_replay_bundle_atomic,
)


def _repository(root: Path) -> Path:
    repository = root / "canonical"
    repository.mkdir()
    (repository / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "beta.py").write_text("OTHER = 2\n", encoding="utf-8")
    return repository


def _mutation_bundle(
    repository: Path,
    *,
    paths: tuple[str, ...] = ("alpha.py",),
) -> ReplayBundle:
    baseline = repository.parent / "baseline"
    mutated = repository.parent / "mutated"
    baseline.mkdir(exist_ok=True)
    mutated.mkdir(exist_ok=True)
    for name in ("alpha.py", "beta.py"):
        content = (repository / name).read_bytes()
        (baseline / name).write_bytes(content)
        (mutated / name).write_bytes(content)
    (mutated / "alpha.py").write_text("VALUE = 3\n", encoding="utf-8")
    if "beta.py" in paths:
        (mutated / "beta.py").write_text("OTHER = 4\n", encoding="utf-8")
    return create_replay_bundle(
        benchmark_identity=EVALUATION_REPLAY_V1,
        repository_identity_value=repository_identity(repository),
        task_id="D01",
        task_version=1,
        source_state_identity_value=source_state_identity(baseline),
        model_identity="mock-model",
        seed=42,
        artifact_type=ReplayArtifactType.MUTATION_PROPOSAL,
        payload=build_mutation_payload(baseline, mutated, paths),
    )


def _test_bundle(repository: Path, source: str = "assert 1 + 1 == 2\n") -> ReplayBundle:
    return create_replay_bundle(
        benchmark_identity=EVALUATION_REPLAY_V1,
        repository_identity_value=repository_identity(repository),
        task_id="D02",
        task_version=1,
        source_state_identity_value=source_state_identity(repository),
        model_identity="mock-model",
        seed=42,
        artifact_type=ReplayArtifactType.GENERATED_ACCEPTANCE_TEST,
        payload={
            "language": "Python",
            "test_name": "addition",
            "test_source": source,
            "validation_version": "A49-v1",
        },
    )


def test_d01_mutation_bundle_serializes_exact_payload(tmp_path: Path) -> None:
    bundle = _mutation_bundle(_repository(tmp_path))
    restored = replay_bundle_from_dict(replay_bundle_to_dict(bundle))
    assert restored == bundle
    assert restored.payload["edits"][0]["new_text"] == "VALUE = 3\n"


def test_d02_generated_test_bundle_serializes_exact_payload(tmp_path: Path) -> None:
    bundle = _test_bundle(_repository(tmp_path))
    assert replay_bundle_from_dict(replay_bundle_to_dict(bundle)) == bundle
    assert bundle.payload["test_source"] == "assert 1 + 1 == 2\n"


def test_d03_payload_hash_is_verified(tmp_path: Path) -> None:
    bundle = _mutation_bundle(_repository(tmp_path))
    assert len(bundle.payload_sha256) == 64
    validate_replay_bundle(bundle)


def test_d04_tampered_payload_is_rejected(tmp_path: Path) -> None:
    bundle = _test_bundle(_repository(tmp_path))
    tampered = replace(bundle, payload={**bundle.payload, "test_name": "changed"})
    with pytest.raises(ReplayValidationError) as captured:
        validate_replay_bundle(tampered)
    assert captured.value.failure is ReplayFailure.ARTIFACT_TAMPERED


def test_d05_repository_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _mutation_bundle(repository)
    (repository / "unrelated.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ReplayValidationError) as captured:
        replay_mutation_bundle(
            bundle,
            repository,
            tmp_path / "out",
            (),
            expected_benchmark_identity=EVALUATION_REPLAY_V1,
        )
    assert captured.value.failure is ReplayFailure.REPOSITORY_MISMATCH


def test_d06_source_state_mismatch_is_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = replace(_mutation_bundle(repository), source_state_identity="0" * 64)
    with pytest.raises(ReplayValidationError) as captured:
        replay_mutation_bundle(
            bundle,
            repository,
            tmp_path / "out",
            (),
            expected_benchmark_identity=EVALUATION_REPLAY_V1,
        )
    assert captured.value.failure is ReplayFailure.SOURCE_STATE_MISMATCH


def test_d07_path_escape_is_rejected(tmp_path: Path) -> None:
    bundle = _mutation_bundle(_repository(tmp_path))
    edit = {**bundle.payload["edits"][0], "path": "../escape.py"}
    with pytest.raises(ReplayValidationError) as captured:
        create_replay_bundle(
            benchmark_identity=EVALUATION_REPLAY_V1,
            repository_identity_value=bundle.repository_identity,
            task_id="D07",
            task_version=1,
            source_state_identity_value=bundle.source_state_identity,
            model_identity="mock",
            seed=42,
            artifact_type=ReplayArtifactType.MUTATION_PROPOSAL,
            payload={"edits": [edit], "grouped": False},
        )
    assert captured.value.failure is ReplayFailure.REPLAY_UNSAFE


def test_d08_single_mutation_replay_is_deterministic(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _mutation_bundle(repository)
    first = replay_mutation_bundle(
        bundle,
        repository,
        tmp_path / "first",
        (),
        expected_benchmark_identity=EVALUATION_REPLAY_V1,
    )
    second = replay_mutation_bundle(
        bundle,
        repository,
        tmp_path / "second",
        (),
        expected_benchmark_identity=EVALUATION_REPLAY_V1,
    )
    assert first.file_hashes == second.file_hashes


def test_d09_grouped_mutation_replay_is_deterministic(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _mutation_bundle(repository, paths=("alpha.py", "beta.py"))
    first = replay_mutation_bundle(
        bundle,
        repository,
        tmp_path / "first",
        (),
        expected_benchmark_identity=EVALUATION_REPLAY_V1,
    )
    second = replay_mutation_bundle(
        bundle,
        repository,
        tmp_path / "second",
        (),
        expected_benchmark_identity=EVALUATION_REPLAY_V1,
    )
    assert len(first.file_hashes) == 2
    assert first.file_hashes == second.file_hashes


def test_d10_replayed_test_is_revalidated_and_deterministic(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _test_bundle(repository)
    outcomes = []
    for index in range(2):
        workspace = tmp_path / f"workspace-{index}"
        shutil.copytree(repository, workspace)
        outcomes.append(
            replay_generated_test_bundle(
                bundle,
                workspace,
                expected_benchmark_identity=EVALUATION_REPLAY_V1,
                canonical_repository=repository,
            )[0]
        )
    assert outcomes == [ReplayFailure.REPLAY_PASS, ReplayFailure.REPLAY_PASS]


def test_d11_unsafe_replayed_test_is_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = _test_bundle(repository, "import socket\n")
    outcome, _ = replay_generated_test_bundle(
        bundle,
        repository,
        expected_benchmark_identity=EVALUATION_REPLAY_V1,
        canonical_repository=repository,
    )
    assert outcome is ReplayFailure.REPLAY_UNSAFE


def test_d12_evaluator_truth_is_excluded_from_replay(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    with pytest.raises(ReplayValidationError):
        _test_bundle(repository, "# hidden oracle\nassert True\n")


def test_d13_standard_result_is_source_free(tmp_path: Path) -> None:
    cell = ReplayCell(
        "R05", "mock", "MUTATION_PROPOSAL", "available", "id", "a" * 64, 0.1
    )
    report = EvaluationReplayReport(
        EVALUATION_REPLAY_V1,
        1,
        1,
        "benchmark",
        "repository",
        42,
        (cell,),
        (),
        (),
        (),
        (),
        False,
        True,
    )
    output = tmp_path / "result.json"
    write_evaluation_replay_report_json(report, output)
    text = output.read_text(encoding="utf-8")
    assert "test_source" not in text and "new_text" not in text


def test_d14_atomic_artifact_and_manifest_commit(tmp_path: Path) -> None:
    bundle = _test_bundle(_repository(tmp_path))
    root = tmp_path / "replay"
    entry = write_replay_bundle_atomic(bundle, root)
    assert load_replay_bundle(root / entry.filename) == bundle
    assert load_replay_manifest(root) == (entry,)
    assert not tuple(root.glob("*.tmp"))


def test_d15_resume_skips_committed_available_and_unavailable_cells(
    tmp_path: Path,
) -> None:
    bundle = _test_bundle(_repository(tmp_path))
    replay_root = tmp_path / "replay"
    checkpoints = tmp_path / "checkpoints"
    write_replay_bundle_atomic(bundle, replay_root)
    assert completed_artifact(replay_root, bundle.artifact_id) == bundle
    assert (
        load_committed_cell(
            replay_root,
            checkpoints,
            task_id="D02",
            model_identity="mock-model",
            artifact_type=ReplayArtifactType.GENERATED_ACCEPTANCE_TEST,
        )
        == bundle
    )
    record_unavailable_cell_atomic(
        checkpoints,
        task_id="R05",
        model_identity="mock-model",
        artifact_type=ReplayArtifactType.MUTATION_PROPOSAL,
        reason=ReplayFailure.PATCH_UNAVAILABLE,
        elapsed_seconds=0.2,
    )
    assert (
        load_committed_cell(
            replay_root,
            checkpoints,
            task_id="R05",
            model_identity="mock-model",
            artifact_type=ReplayArtifactType.MUTATION_PROPOSAL,
        )
        is ReplayFailure.PATCH_UNAVAILABLE
    )


def test_d16_replay_payload_location_is_excluded_from_package() -> None:
    project = Path(__file__).resolve().parents[1]
    configuration = (project / "pyproject.toml").read_text(encoding="utf-8")
    ignored = (project / ".gitignore").read_text(encoding="utf-8")
    assert 'package-dir = { "" = "src" }' in configuration
    assert "eval-results/" in ignored
    assert not (project / "src" / "eval-results").exists()


def test_replay_format_rejects_unsupported_version(tmp_path: Path) -> None:
    bundle = _test_bundle(_repository(tmp_path))
    with pytest.raises(ReplayValidationError):
        validate_replay_bundle(
            replace(bundle, format_version=REPLAY_FORMAT_VERSION + 1)
        )


def test_manifest_contains_hash_not_payload(tmp_path: Path) -> None:
    bundle = _test_bundle(_repository(tmp_path))
    root = tmp_path / "replay"
    write_replay_bundle_atomic(bundle, root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["payload_sha256"] == bundle.payload_sha256
    assert "payload" not in manifest["artifacts"][0]
    assert (
        hashlib.sha256(
            json.dumps(bundle.payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == bundle.payload_sha256
    )


def test_insertion_only_capture_uses_adjacent_line_anchor(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    mutated = tmp_path / "mutated"
    baseline.mkdir()
    mutated.mkdir()
    (baseline / "test.py").write_text("first\nlast\n", encoding="utf-8")
    (mutated / "test.py").write_text("first\ninserted\nlast\n", encoding="utf-8")
    payload = build_mutation_payload(baseline, mutated, ("test.py",))
    edit = payload["edits"][0]
    assert edit["old_text"] == "last\n"
    assert edit["new_text"] == "inserted\nlast\n"


def test_manifest_cannot_redirect_outside_replay_root(tmp_path: Path) -> None:
    bundle = _test_bundle(_repository(tmp_path))
    root = tmp_path / "replay"
    write_replay_bundle_atomic(bundle, root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["filename"] = "../other.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ReplayValidationError) as captured:
        completed_artifact(root, bundle.artifact_id)
    assert captured.value.failure is ReplayFailure.ARTIFACT_INVALID


def test_committed_cell_cannot_be_overwritten(tmp_path: Path) -> None:
    bundle = _test_bundle(_repository(tmp_path))
    root = tmp_path / "replay"
    first = write_replay_bundle_atomic(bundle, root)
    assert write_replay_bundle_atomic(bundle, root) == first
    payload = {**bundle.payload, "test_name": "changed"}
    altered = replace(bundle, payload=payload, payload_sha256=payload_sha256(payload))
    with pytest.raises(ReplayValidationError) as captured:
        write_replay_bundle_atomic(altered, root)
    assert captured.value.failure is ReplayFailure.ARTIFACT_TAMPERED


def test_identity_cannot_escape_artifact_storage(tmp_path: Path) -> None:
    bundle = _test_bundle(_repository(tmp_path))
    with pytest.raises(ReplayValidationError) as captured:
        validate_replay_bundle(replace(bundle, task_id="../escape"))
    assert captured.value.failure is ReplayFailure.ARTIFACT_INVALID


def test_generated_test_source_state_mismatch_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bundle = replace(_test_bundle(repository), source_state_identity="0" * 64)
    with pytest.raises(ReplayValidationError) as captured:
        replay_generated_test_bundle(
            bundle,
            repository,
            expected_benchmark_identity=EVALUATION_REPLAY_V1,
            canonical_repository=repository,
        )
    assert captured.value.failure is ReplayFailure.SOURCE_STATE_MISMATCH


def test_grouped_replay_write_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    bundle = _mutation_bundle(repository, paths=("alpha.py", "beta.py"))
    destination = tmp_path / "failed-replay"
    original_write = Path.write_bytes
    failures = 0

    def fail_second_write(path: Path, content: bytes) -> int:
        nonlocal failures
        if path.name == "beta.py" and failures == 0:
            failures += 1
            raise OSError("injected write failure")
        return original_write(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_second_write)
    with pytest.raises(ReplayValidationError) as captured:
        replay_mutation_bundle(
            bundle,
            repository,
            destination,
            (),
            expected_benchmark_identity=EVALUATION_REPLAY_V1,
        )
    assert captured.value.failure is ReplayFailure.REPLAY_UNSAFE
    assert (destination / "alpha.py").read_bytes() == (
        repository / "alpha.py"
    ).read_bytes()
    assert (destination / "beta.py").read_bytes() == (
        repository / "beta.py"
    ).read_bytes()
