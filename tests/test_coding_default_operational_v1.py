from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.coding_default_operational_v1.runner import (
    classify_create,
    classify_mixed,
)
from benchmarks.coding_default_operational_v1.suite import (
    REPRESENTATIVE_TASK_IDS,
    SelectionCondition,
    corpus_identity,
    tasks,
    validate_corpus,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.models import ModelDefaults, ModelRole


def _cell(**overrides):  # type: ignore[no-untyped-def]
    proposal = {
        "model_output": True,
        "schema_valid": True,
        "mechanically_materializable": True,
        "production_validatable": True,
        "transaction_applied": True,
        "verification_pass": True,
        "semantic_failure": None,
    }
    proposal.update(overrides)
    return SimpleNamespace(proposal=proposal, primary_semantic_pass=True)


def test_frozen_corpus_shape_and_identity() -> None:
    validate_corpus()
    assert len(tasks()) == 10
    assert REPRESENTATIVE_TASK_IDS == ("H01", "H04", "H07", "H10")
    assert len(corpus_identity()) == 64


@pytest.mark.parametrize(
    ("overrides", "semantic", "expected"),
    [
        ({"model_output": False}, False, "NO_PROPOSAL"),
        ({"schema_valid": False}, False, "MODEL_PROTOCOL_FAILURE"),
        (
            {"mechanically_materializable": False},
            False,
            "CREATE_MATERIALIZATION_FAILURE",
        ),
        ({"production_validatable": False}, False, "CREATE_AUTHORITY_FAILURE"),
        ({"transaction_applied": False}, False, "INTEGRATION_FAILURE"),
        ({"semantic_failure": "BUILD_PREVENTS_ORACLE"}, False, "BUILD_FAILURE"),
        ({"verification_pass": False}, False, "VERIFICATION_FAILURE"),
        ({}, False, "WRONG_BEHAVIOR"),
        ({}, True, "PASS"),
    ],
)
def test_create_failure_taxonomy(overrides, semantic: bool, expected: str) -> None:  # type: ignore[no-untyped-def]
    cell = _cell(**overrides)
    cell.primary_semantic_pass = semantic
    assert classify_create(cell) == expected


def test_mixed_failure_taxonomy_distinguishes_operation_sets() -> None:
    definition = next(item for item in tasks() if item.task_id == "H10")
    edit = definition.edit_paths[0]
    create = definition.create_paths[0]
    base = _cell()
    assert classify_mixed(definition, base, None) == "MISSING_EDIT"
    assert (
        classify_mixed(definition, base, {"children": [{"type": "edit", "path": edit}]})
        == "MISSING_CREATE"
    )
    assert (
        classify_mixed(
            definition,
            base,
            {"children": [{"type": "create_file", "path": edit}]},
        )
        == "WRONG_OPERATION_ROLE"
    )
    assert (
        classify_mixed(
            definition,
            base,
            {"children": [{"type": "edit", "path": edit}] * 2},
        )
        == "DUPLICATE_OPERATION"
    )
    complete = {
        "children": [
            {"type": "edit", "path": edit},
            {"type": "create_file", "path": create},
        ]
    }
    base.primary_semantic_pass = False
    base.proposal["semantic_failure"] = "PARTIAL_MULTI_FILE_CHANGE"
    assert classify_mixed(definition, base, complete) == "PARTIAL_IMPLEMENTATION"


def test_default_and_explicit_conditions_select_same_profile_name() -> None:
    defaults = ModelDefaults()
    assert defaults.profile_for(ModelRole.CODING) == "qwen-large"
    assert SelectionCondition.DEFAULT.value != SelectionCondition.EXPLICIT_LARGE.value


def test_source_free_checkpoint_resume_and_package_exclusion(tmp_path: Path) -> None:
    payload = {
        "classification": "MISSING_CREATE",
        "workflow_attempt_id": "w1",
        "mutation_request_ids": ["r1"],
        "repair": None,
    }
    assert standard_result_is_source_free(payload)
    path = tmp_path / "cell.json"
    atomic_checkpoint(path, payload)
    assert resume_checkpoint(path) == json.loads(json.dumps(payload))
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, payload)
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "coding_default_operational_v1" not in configuration
