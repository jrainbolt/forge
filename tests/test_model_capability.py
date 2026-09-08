from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from forge.evaluation import (
    MODEL_CAPABILITY_V1,
    EvaluationOutcome,
    ModelAvailability,
    ModelCapabilityRunner,
    RealWorldLevel,
    RealWorldTask,
    RepositorySnapshot,
    build_model_matrix,
    merge_model_capability_runs,
    model_capability_to_dict,
    render_model_capability_report,
    summarize_model,
)
from forge.interaction import AutonomyMode
from forge.models import (
    BackendDefinition,
    BackendRegistry,
    MockModel,
    ModelCatalog,
    ModelError,
    ModelProfile,
)


def _call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": identifier, "tool": tool, "arguments": arguments}
    )


def _catalog() -> ModelCatalog:
    responses = (
        "smoke",
        _call("read", "repository.read_file", {"path": "main.c"}),
        json.dumps({"type": "final", "answer": "main.c contains target_symbol"}),
        _call("read-2", "repository.read_file", {"path": "main.c"}),
        json.dumps({"type": "final", "answer": "main.c contains target_symbol"}),
    )

    def build(config: object) -> MockModel:
        if config == "missing":
            raise ModelError("configured artifact is unavailable")
        return MockModel(responses)

    registry = BackendRegistry(
        {"fake": BackendDefinition(lambda _name, raw: raw, build)}
    )
    return ModelCatalog(
        (
            ModelProfile("available", "fake", "mock-code", "available"),
            ModelProfile("missing", "fake", "mock-missing", "missing"),
        ),
        registry,
    )


def _task() -> RealWorldTask:
    return RealWorldTask(
        "T01",
        RealWorldLevel.REPOSITORY_REASONING,
        AutonomyMode.READ,
        "Locate target_symbol.",
        ("main.c",),
        seeds=(7, 42),
    )


def test_matrix_expansion_reuses_tasks_and_pairs_fixed_seeds() -> None:
    task = _task()
    matrix = build_model_matrix(("small", "large"), (task,), {"T01": (7, 42)})

    assert [(cell.profile, cell.task_id, cell.seed) for cell in matrix] == [
        ("small", "T01", 7),
        ("small", "T01", 42),
        ("large", "T01", 7),
        ("large", "T01", 42),
    ]
    assert task.seeds == (7, 42)


def test_runner_records_available_and_unavailable_profiles_and_keeps_going(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.c").write_text("void target_symbol(void) {}\n")
    snapshot = RepositorySnapshot(
        "fixture",
        "fixture-identity",
        "C",
        1,
        0,
        1,
        (),
        (),
        EvaluationOutcome.PASS,
        0.0,
    )

    run = ModelCapabilityRunner(_catalog(), tmp_path, snapshot).run(
        ("missing", "available"), (_task(),), {"T01": (7, 42)}
    )

    missing, available = run.profiles
    assert run.suite == MODEL_CAPABILITY_V1
    assert run.canonical_unchanged
    assert missing.availability.status is ModelAvailability.UNAVAILABLE
    assert missing.availability.stage == "load"
    assert missing.realworld is None
    assert available.availability.status is ModelAvailability.AVAILABLE
    assert available.summary.runs == 2
    assert available.summary.grounded_read_only == 2
    assert available.summary.read_only_runs == 2

    payload = model_capability_to_dict(run)
    assert payload["profiles"][0]["availability"]["status"] == "UNAVAILABLE"
    assert payload["profiles"][1]["rates"]["valid_delta"] == 0.0
    rendered = render_model_capability_report(run)
    assert "UNAVAILABLE" in rendered
    assert "2/2" in rendered

    merged = merge_model_capability_runs(
        (
            replace(run, profiles=(missing,)),
            replace(run, profiles=(available,)),
        )
    )
    assert tuple(profile.configuration.profile for profile in merged.profiles) == (
        "missing",
        "available",
    )

    read_result = available.realworld.results[0]
    coding_result = replace(
        read_result,
        level="single_change",
        mode="assist",
        oracle=EvaluationOutcome.PASS,
        metrics=replace(
            read_result.metrics,
            actual_delta_proposed=True,
            preview_created=1,
            mutations=1,
            verification_result="passed",
        ),
    )
    coding_summary = summarize_model((coding_result,))
    assert coding_summary.valid_delta_rate == 1.0
    assert coding_summary.preview_rate == 1.0
    assert coding_summary.mutation_rate == 1.0
    assert coding_summary.verification_pass_rate == 1.0
    assert coding_summary.oracle_pass_rate == 1.0


def test_seed_plan_must_match_selected_unchanged_tasks() -> None:
    try:
        build_model_matrix(("small",), (_task(),), {})
    except ValueError as error:
        assert "every selected task" in str(error)
    else:
        raise AssertionError("missing seed pairing must fail")
