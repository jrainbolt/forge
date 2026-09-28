"""A61 deterministic real-repository discovery boundary suite (D01-D20)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.real_repository_discovery_boundary_v1.harness import (
    DiscoveryEvent,
    DiscoveryTrace,
    EventKind,
    deterministic_trace,
    validate_trace,
)
from benchmarks.real_repository_discovery_boundary_v1.suite import (
    SUITE,
    AuthorityMode,
    DiscoveryExpectationLeak,
    DiscoveryScore,
    ExpectedImplementationPath,
    RequiredCandidatePath,
    production_inputs,
    score_acquired_source,
    tasks,
    validate_production_boundary,
)
from benchmarks.real_repository_pilot_v1.runner import standard_result_is_source_free
from benchmarks.real_repository_pilot_v1.suite import tasks as a60_tasks


@pytest.fixture(scope="module")
def definitions():  # type: ignore[no-untyped-def]
    return tasks()


def _discovery(definitions):  # type: ignore[no-untyped-def]
    return tuple(
        item
        for item in definitions
        if item.authority_mode is AuthorityMode.DISCOVERY_REQUIRED
    )


def test_d01_hidden_expected_path_cannot_become_required_candidate(definitions) -> None:  # type: ignore[no-untyped-def]
    assert all(
        not item.production_task.required_candidate_paths
        for item in _discovery(definitions)
    )


def test_d02_evaluator_expectation_remains_available_for_scoring(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    assert item.expected_implementation_paths
    assert (
        score_acquired_source(item, item.expected_values, sufficient=True)
        is DiscoveryScore.EXPECTED_PATH_FOUND
    )


def test_d03_discovery_task_starts_with_zero_path_authority(definitions) -> None:  # type: ignore[no-untyped-def]
    for item in _discovery(definitions):
        values = production_inputs(item)
        assert not any(values[key] for key in values if key != "task_text")


def test_d04_path_known_user_task_may_carry_explicit_path(definitions) -> None:  # type: ignore[no-untyped-def]
    known = [
        item for item in definitions if item.authority_mode is AuthorityMode.PATH_KNOWN
    ]
    assert known
    for item in known:
        assert item.production_task.required_candidate_paths
        assert all(
            path in item.production_task.prompt
            for path in item.production_task.required_candidate_paths
        )


def test_d05_search_hit_is_not_trusted_source() -> None:
    trace = DiscoveryTrace(
        (
            DiscoveryEvent(EventKind.SEARCH),
            DiscoveryEvent(EventKind.SOURCE_HINT, "src/a.c"),
            DiscoveryEvent(EventKind.MUTATION_CANDIDATE, "src/a.c", True),
        )
    )
    with pytest.raises(ValueError, match="UNTRUSTED_MUTATION_CANDIDATE"):
        validate_trace(trace, discovery_required=True)


def test_d06_trusted_read_establishes_source_authority() -> None:
    trace = deterministic_trace("src/a.c")
    validate_trace(trace, discovery_required=True)
    assert trace.mutation_candidates == ("src/a.c",)


def test_d07_hidden_path_excluded_from_context_planner_input(definitions) -> None:  # type: ignore[no-untyped-def]
    for item in _discovery(definitions):
        inputs = production_inputs(item)
        assert not set(item.expected_values).intersection(
            inputs["expected_changed_paths"]
        )


def test_d08_hidden_path_excluded_from_retrieval_seed(definitions) -> None:  # type: ignore[no-untyped-def]
    assert all(
        not item.production_task.required_candidate_paths
        for item in _discovery(definitions)
    )


def test_d09_hidden_path_excluded_from_mutation_readiness(definitions) -> None:  # type: ignore[no-untyped-def]
    for item in _discovery(definitions):
        assert not item.production_task.allowed_paths
        assert not item.production_task.expected_changed_paths


def test_d10_expectation_leak_fails_closed(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    leaked = replace(
        item,
        production_task=replace(
            item.production_task, required_candidate_paths=item.expected_values
        ),
    )
    with pytest.raises(DiscoveryExpectationLeak, match="DISCOVERY_EXPECTATION_LEAK"):
        validate_production_boundary(leaked)


def test_d11_real_discovery_events_recorded() -> None:
    trace = deterministic_trace("src/a.c")
    assert trace.discovery_calls == trace.search_calls == trace.source_reads == 1
    assert trace.context_files == ("src/a.c",)


def test_d12_zero_discovery_flagged_for_discovery_task() -> None:
    with pytest.raises(ValueError, match="ZERO_DISCOVERY_ACTIVITY"):
        validate_trace(DiscoveryTrace(()), discovery_required=True)


def test_d13_alternate_valid_source_may_succeed(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    assert (
        score_acquired_source(
            item,
            ("src/alternate.c",),
            sufficient=True,
            alternative_valid=True,
        )
        is DiscoveryScore.ALTERNATIVE_VALID_SOURCE_FOUND
    )


def test_unvalidated_alternate_is_misdirected(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    assert (
        score_acquired_source(item, ("src/nearby.c",), sufficient=True)
        is DiscoveryScore.MISDIRECTED_SOURCE
    )


def test_d14_context_classification_uses_acquired_source_only(definitions) -> None:  # type: ignore[no-untyped-def]
    item = _discovery(definitions)[0]
    assert (
        score_acquired_source(item, (), sufficient=True)
        is DiscoveryScore.INSUFFICIENT_SOURCE
    )


def test_d15_path_known_control_still_works(definitions) -> None:  # type: ignore[no-untyped-def]
    item = next(
        item for item in definitions if item.authority_mode is AuthorityMode.PATH_KNOWN
    )
    assert set(item.production_task.required_candidate_paths) == set(
        item.expected_values
    )


def test_d16_standard_result_source_free() -> None:
    assert standard_result_is_source_free({"suite": SUITE, "status": "PASS"})
    assert not standard_result_is_source_free({"suite": SUITE, "source": "hidden"})


def test_d17_canonical_foundation_target_cannot_be_mutated(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    before = canonical.stat().st_mtime_ns
    trace = deterministic_trace("src/a.c")
    validate_trace(trace, discovery_required=True)
    assert canonical.stat().st_mtime_ns == before


def test_d18_packaging_exclusion() -> None:
    config = Path("pyproject.toml").read_text()
    assert "benchmarks" not in config.split("[tool.setuptools.package-data]", 1)[-1]
    assert "real_repository_discovery_boundary_v1" not in config


def test_d19_a60_historical_result_remains_blocked() -> None:
    evaluation = Path("docs/EVALUATION.md").read_text()
    assert "A60 is blocked" in evaluation
    assert all(
        item.production_task.required_candidate_paths == item.edit_paths
        for item in a60_tasks()
    )


def test_d20_future_pilot_requires_new_version() -> None:
    assert SUITE != "real-repository-pilot-v1"
    assert SUITE.endswith("-v1")


def test_distinct_path_types_prevent_accidental_equality() -> None:
    expected = ExpectedImplementationPath("src/a.c")
    required = RequiredCandidatePath("src/a.c")
    assert expected != required
