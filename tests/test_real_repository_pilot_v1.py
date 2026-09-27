"""A60 model-free snapshot, corpus, classification, and durability controls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.real_repository_pilot_v1.runner import (
    Failure,
    _atomic_json,
    checkpoint_path,
    context_quality,
    first_failure,
    proposal_shape,
    read_checkpoint,
    standard_result_is_source_free,
    summarize,
)
from benchmarks.real_repository_pilot_v1.suite import (
    AuthorityMode,
    Integrity,
    OperationClass,
    create_snapshot,
    definition_identity,
    repository_identity,
    source_manifest,
    tasks,
)


@pytest.fixture(scope="module")
def definitions():  # type: ignore[no-untyped-def]
    return tasks()


def _tiny_repository(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.c").write_text("int a(void) { return 1; }\n")
    (root / "CMakeLists.txt").write_text("project(tiny C)\n")
    (root / "build").mkdir()
    (root / "build" / "generated.c").write_text("secret\n")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ignored\n")
    return root


def test_r01_snapshot_manifest_deterministic(tmp_path: Path) -> None:
    root = _tiny_repository(tmp_path / "repo")
    assert source_manifest(root) == source_manifest(root)
    assert repository_identity(root) == repository_identity(root)
    assert [path for path, _ in source_manifest(root)] == ["CMakeLists.txt", "src/a.c"]


def test_r02_canonical_root_is_never_writable_target(tmp_path: Path) -> None:
    root = _tiny_repository(tmp_path / "repo")
    with pytest.raises(ValueError):
        create_snapshot(root, root)


def test_r03_disposable_workspace_isolation(tmp_path: Path) -> None:
    canonical = _tiny_repository(tmp_path / "repo")
    before = repository_identity(canonical)
    snapshot = create_snapshot(canonical, tmp_path / "snapshot")
    (snapshot / "src/a.c").write_text("changed\n")
    assert repository_identity(canonical) == before


def test_r04_task_version_and_freeze(definitions, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    root = _tiny_repository(tmp_path / "repo")
    assert [item.versioned_id for item in definitions] == [
        f"F{i:02d}-v1" for i in range(1, 9)
    ]
    assert definition_identity(definitions, root) == definition_identity(
        definitions, root
    )


def test_r05_baseline_reference_oracle_integrity() -> None:
    eligible = Integrity(
        "F01", False, True, False, False, True, False, "FULLY_DISCRIMINATING", True
    )
    assert (
        not eligible.baseline_oracle
        and eligible.reference_oracle
        and not eligible.wrong_oracle
        and eligible.eligible
    )


def test_r06_hidden_reference_excluded(definitions) -> None:  # type: ignore[no-untyped-def]
    for item in definitions:
        prompt = item.production_task.prompt.lower()
        assert "reference" not in prompt and "oracle" not in prompt
        assert item.production_task.oracle_commands[0][1].endswith("oracle.py")


def test_r07_discovery_required_metadata(definitions) -> None:  # type: ignore[no-untyped-def]
    assert (
        sum(
            item.authority_mode is AuthorityMode.DISCOVERY_REQUIRED
            for item in definitions
        )
        == 6
    )
    assert (
        sum(item.authority_mode is AuthorityMode.PATH_KNOWN for item in definitions)
        == 2
    )


def test_r08_source_authority_requires_trusted_read(definitions) -> None:  # type: ignore[no-untyped-def]
    assert all(
        item.production_task.required_candidate_paths == item.edit_paths
        for item in definitions
    )


def test_r09_context_classification(definitions) -> None:  # type: ignore[no-untyped-def]
    item = definitions[4]
    assert context_quality(item, item.edit_paths, 2) == "SUFFICIENT"
    assert context_quality(item, item.edit_paths[:1], 1) == "PARTIAL"
    assert context_quality(item, ("README.md",), 0) == "MISDIRECTED"


def _failure(definition, **changes):  # type: ignore[no-untyped-def]
    values = dict(
        discovery=True,
        acquired=True,
        context="SUFFICIENT",
        ready=True,
        schema=True,
        path_valid=True,
        role_valid=True,
        complete=True,
        preview=True,
        transaction=True,
        verification=True,
        repair_attempted=False,
        semantic=True,
    )
    values.update(changes)
    return first_failure(definition, **values)


def test_r10_first_failure_classification(definitions) -> None:  # type: ignore[no-untyped-def]
    assert _failure(definitions[0], discovery=False) == Failure.DISCOVERY_FAILED
    assert _failure(definitions[0], context="PARTIAL") == Failure.CONTEXT_INSUFFICIENT
    assert _failure(definitions[0], semantic=False) == Failure.SEMANTIC_FAILED


def test_r11_transaction_separate_from_semantics(definitions) -> None:  # type: ignore[no-untyped-def]
    assert (
        _failure(definitions[0], transaction=False, semantic=False)
        == Failure.TRANSACTION_FAILED
    )


def test_r12_verification_separate_from_oracle(definitions) -> None:  # type: ignore[no-untyped-def]
    assert (
        _failure(definitions[0], verification=False, semantic=True)
        == Failure.VERIFICATION_FAILED
    )
    assert (
        _failure(definitions[0], verification=True, semantic=False)
        == Failure.SEMANTIC_FAILED
    )


def test_r13_repair_accounting(definitions) -> None:  # type: ignore[no-untyped-def]
    assert (
        _failure(definitions[0], verification=False, repair_attempted=True)
        == Failure.REPAIR_FAILED
    )


def test_r14_per_cell_checkpoint(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "qwen-small", "F01")
    payload = {"suite": "real-repository-pilot-v1", "task_id": "F01"}
    _atomic_json(path, payload)
    assert json.loads(path.read_text()) == payload


def test_r15_resume_skips_committed(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "qwen-small", "F01")
    payload = {"suite": "real-repository-pilot-v1", "task_id": "F01"}
    _atomic_json(path, payload)
    assert read_checkpoint(path, payload) == payload
    with pytest.raises(FileExistsError):
        _atomic_json(path, payload)


def test_r16_standard_result_source_free() -> None:
    assert standard_result_is_source_free({"task_id": "F01", "status": "PASS"})
    assert not standard_result_is_source_free({"task_id": "F01", "source": "body"})


def test_r17_package_exclusion() -> None:
    config = Path("pyproject.toml").read_text()
    assert 'package-dir = { "" = "src" }' in config
    assert "real_repository_pilot_v1" not in config


def test_r18_canonical_identity_post_run(tmp_path: Path) -> None:
    canonical = _tiny_repository(tmp_path / "repo")
    before = repository_identity(canonical)
    snapshot = create_snapshot(canonical, tmp_path / "snapshot")
    (snapshot / "src/a.c").write_text("mutation\n")
    assert repository_identity(canonical) == before


def test_r19_no_benchmark_specific_production_branch() -> None:
    production = "\n".join(
        path.read_text(errors="ignore") for path in Path("src/forge").rglob("*.py")
    )
    assert "real-repository-pilot-v1" not in production
    assert "factory_simulation_clock_can_advance" not in production


def test_r20_ephemeral_acceptance_disabled(definitions) -> None:  # type: ignore[no-untyped-def]
    assert all(
        not hasattr(item.production_task, "ephemeral_acceptance")
        for item in definitions
    )


def test_r21_no_model_routing_or_default_mutation(definitions) -> None:  # type: ignore[no-untyped-def]
    assert all(
        "qwen" not in item.production_task.prompt.lower()
        and "codestral" not in item.production_task.prompt.lower()
        for item in definitions
    )
    assert {item.operation_class for item in definitions} == {
        OperationClass.EDIT_SINGLE,
        OperationClass.EDIT_MULTI,
    }


def test_proposal_completeness(definitions) -> None:  # type: ignore[no-untyped-def]
    item = definitions[4]
    valid, duplicate, missing, unexpected = proposal_shape(
        item,
        (
            {
                "type": "multi_file_line_range_edit",
                "children": [
                    {"type": "edit", "path": path} for path in item.edit_paths
                ],
            },
        ),
    )
    assert valid and duplicate == missing == unexpected == 0


def test_summary_is_not_opaque_score() -> None:
    cell = {
        "model_profile": "qwen-small",
        "discovery_success": True,
        "context_quality": "SUFFICIENT",
        "schema_valid": True,
        "operation_role_valid": True,
        "transaction_success": True,
        "verification_outcome": "pass",
        "semantic_pass": True,
        "semantic_recovered": False,
        "total_seconds": 1.0,
    }
    summary = summarize((cell,))
    assert summary["profiles"]["qwen-small"]["discovery"] == 1
    assert "score" not in summary["profiles"]["qwen-small"]
