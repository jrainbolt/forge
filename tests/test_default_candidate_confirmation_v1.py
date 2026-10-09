from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.default_candidate_confirmation_v1.runner import (
    checkpoint_path,
    read_cell,
)
from benchmarks.default_candidate_confirmation_v1.suite import (
    PRIMARY_PROFILES,
    PROFILES,
    REPRESENTATION,
    RUN_ID,
    corpus_identity,
    tasks,
    validate_corpus,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY, OperationClass
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.models import MutationRepresentationPolicy


def test_held_out_corpus_has_three_tasks_per_class() -> None:
    validate_corpus()
    assert len(tasks()) == 12
    assert {
        kind: sum(item.operation_class is kind for item in tasks())
        for kind in OperationClass
    } == {kind: 3 for kind in OperationClass}


def test_primary_and_control_profiles_are_frozen() -> None:
    assert PRIMARY_PROFILES == ("qwen-small", "qwen-large")
    assert PROFILES == ("qwen-small", "qwen-large", "codestral-22b")
    assert REPRESENTATION is MutationRepresentationPolicy.LINE_RANGE


def test_frozen_inputs_and_identity_are_deterministic() -> None:
    first = corpus_identity()
    assert first == corpus_identity()
    assert all(item.production_task.seeds == (42,) for item in tasks())
    assert all(item.production_task.max_mutations == 2 for item in tasks())


def test_tasks_are_complete_workflow_not_a77_atomic_prompts() -> None:
    prompts = {item.production_task.prompt for item in tasks()}
    assert len(prompts) == 12
    assert all(item.task_id.startswith("H") for item in tasks())
    assert all(item.production_task.verification_plan is not None for item in tasks())


def test_exactly_once_checkpoint_and_source_free_result(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "H01", "qwen-small")
    payload = {"run_identity": RUN_ID, "cell_id": "A78-H01-qwen-small"}
    atomic_checkpoint(path, payload)
    assert read_cell(path, {"run_identity": RUN_ID}) == payload
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, {"regenerated": True})
    assert standard_result_is_source_free(payload)


def test_no_model_specific_prompt_routing_or_default_change() -> None:
    suite = (
        Path(__file__).parents[1]
        / "benchmarks/default_candidate_confirmation_v1/suite.py"
    )
    runner = (
        Path(__file__).parents[1] / "scripts/run_default_candidate_confirmation_v1.py"
    )
    text = suite.read_text() + runner.read_text()
    assert "if profile" not in text
    assert "catalog.create(args.profile)" in text
    assert "default_model" not in text
    assert "route" not in text.casefold()


def test_canonical_source_identity_remains_stable() -> None:
    before = source_state_identity(REPOSITORY)
    tasks()
    assert source_state_identity(REPOSITORY) == before


def test_results_and_external_artifacts_are_not_packaged() -> None:
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "default_candidate_confirmation_v1" not in configuration
    payload = {"artifact": "model.gguf:sha256:" + "a" * 64}
    assert "/Users/" not in json.dumps(payload)
