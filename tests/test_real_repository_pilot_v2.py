"""A62 deterministic evaluator, durability, and boundary controls."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.real_repository_pilot_v1.suite import (
    AuthorityMode,
    create_snapshot,
    repository_identity,
)
from benchmarks.real_repository_pilot_v2.runner import (
    Failure,
    atomic_json,
    checkpoint_path,
    classify_context,
    first_failure,
    read_checkpoint,
    standard_result_is_source_free,
)
from benchmarks.real_repository_pilot_v2.suite import (
    SUITE,
    VERSION,
    definition_identity,
    tasks,
    validate_boundary,
)
from forge.evaluation.realworld import DiscoveredSourceApproval
from forge.project_config import ProjectCommands


@pytest.fixture(scope="module")
def definitions():  # type: ignore[no-untyped-def]
    return tasks()


def _discovery(definitions):  # type: ignore[no-untyped-def]
    return tuple(
        item
        for item in definitions
        if item.authority_mode is AuthorityMode.DISCOVERY_REQUIRED
    )


def test_new_evaluator_and_version_identity(definitions, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "a.c").write_text("int a;\n")
    assert SUITE == "real-repository-pilot-v2" and VERSION == 2
    assert len(definitions) == 8
    assert definition_identity(definitions, tmp_path)


def test_hidden_expected_paths_excluded(definitions) -> None:  # type: ignore[no-untyped-def]
    for item in _discovery(definitions):
        validate_boundary(item)
        production = item.production_task
        assert not production.required_candidate_paths
        assert not production.expected_changed_paths
        assert not production.allowed_paths


def test_discovery_required_starts_without_path_authority(definitions) -> None:  # type: ignore[no-untyped-def]
    assert len(_discovery(definitions)) == 6


def test_boundary_leak_fails_closed(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    leaked = replace(
        item,
        production_task=replace(
            item.production_task, required_candidate_paths=item.edit_paths
        ),
    )
    with pytest.raises(ValueError, match="DISCOVERY_EXPECTATION_LEAK"):
        validate_boundary(leaked)


def test_path_known_controls_work(definitions) -> None:  # type: ignore[no-untyped-def]
    known = [
        item for item in definitions if item.authority_mode is AuthorityMode.PATH_KNOWN
    ]
    assert len(known) == 2
    for item in known:
        assert set(item.production_task.required_candidate_paths) == set(
            item.edit_paths
        )
        assert all(path in item.production_task.prompt for path in item.edit_paths)


def test_expected_alternate_misdirected_and_insufficient(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    assert classify_context(item, item.edit_paths) == (
        "EXPECTED_PATH_FOUND",
        "SUFFICIENT",
    )
    assert classify_context(item, ("src/valid.c",), alternative_valid=True) == (
        "ALTERNATIVE_VALID_SOURCE_FOUND",
        "SUFFICIENT",
    )
    assert classify_context(item, ("src/wrong.c",)) == (
        "MISDIRECTED_SOURCE",
        "MISDIRECTED",
    )
    assert classify_context(item, ()) == ("INSUFFICIENT_SOURCE", "MISDIRECTED")


def test_context_quality_independent_from_readiness(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    _outcome, quality = classify_context(item, ("src/wrong.c",))
    assert quality == "MISDIRECTED"


def test_trusted_read_required_for_dynamic_approval(
    definitions, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    approval = DiscoveredSourceApproval(
        item.production_task, tmp_path, ProjectCommands()
    )
    approval.observe(
        SimpleNamespace(
            status="success", tool_name="repository.lexical_search", path="src/a.c"
        )
    )
    assert "src/a.c" not in approval.authorized_paths
    approval.observe(
        SimpleNamespace(
            status="success", tool_name="repository.read_file", path="src/a.c"
        )
    )
    assert "src/a.c" in approval.authorized_paths


def test_checkpoint_resume_exactly_once(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "qwen-large", "F01")
    payload = {"suite": SUITE, "task_id": "F01", "model_profile": "qwen-large"}
    atomic_json(path, payload)
    assert read_checkpoint(path, payload) == payload
    with pytest.raises(FileExistsError):
        atomic_json(path, payload)


def _failure(definition, **changes):  # type: ignore[no-untyped-def]
    values = dict(
        discovery_calls=1,
        source_acquired=True,
        context_quality="SUFFICIENT",
        ready=True,
        schema=True,
        transaction=True,
        verification=True,
        repair_attempted=False,
        semantic=True,
    )
    values.update(changes)
    return first_failure(definition, **values)


def test_first_failure_classification(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    assert _failure(item, discovery_calls=0) == Failure.DISCOVERY_FAILED
    assert _failure(item, context_quality="MISDIRECTED") == Failure.CONTEXT_INSUFFICIENT
    assert _failure(item, transaction=False) == Failure.TRANSACTION_FAILED
    assert _failure(item, semantic=False) == Failure.SEMANTIC_FAILED


def test_verification_and_oracle_are_separate(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    assert (
        _failure(item, verification=False, semantic=True) == Failure.VERIFICATION_FAILED
    )
    assert _failure(item, verification=True, semantic=False) == Failure.SEMANTIC_FAILED


def test_canonical_foundation_safety(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "a.c").write_text("int a;\n")
    before = repository_identity(canonical)
    snapshot = create_snapshot(canonical, tmp_path / "snapshot")
    (snapshot / "a.c").write_text("changed\n")
    assert repository_identity(canonical) == before


def test_standard_results_are_source_free() -> None:
    assert standard_result_is_source_free({"suite": SUITE, "status": "PASS"})
    assert not standard_result_is_source_free({"suite": SUITE, "source": "body"})


def test_package_exclusion() -> None:
    config = Path("pyproject.toml").read_text()
    assert "real_repository_pilot_v2" not in config


def test_historical_blocked_statuses_preserved() -> None:
    roadmap = Path("docs/ROADMAP.md").read_text()
    assert "A59 remains blocked" in roadmap
    assert "A60 remains blocked" in roadmap
