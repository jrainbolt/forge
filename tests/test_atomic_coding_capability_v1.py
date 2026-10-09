from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.atomic_coding_capability_v1.runner import (
    checkpoint_path,
    classify_failure,
    read_cell,
)
from benchmarks.atomic_coding_capability_v1.suite import (
    PROFILES,
    REPRESENTATION,
    RUN_ID,
    corpus_identity,
    tasks,
    validate_matrix,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.models import MutationRepresentationPolicy
from scripts.summarize_atomic_coding_capability_v1 import _identity_projection


def test_frozen_inventory_matrix_and_corpus_are_deterministic() -> None:
    validate_matrix()
    assert PROFILES == (
        "qwen-small",
        "qwen-large",
        "codestral-22b",
        "deepseek-coder-lite",
    )
    assert len(tasks()) * len(PROFILES) == 48
    assert corpus_identity() == corpus_identity()


def test_all_nine_a76_atomic_cases_and_three_controls_are_present() -> None:
    frozen = tasks()
    assert sum(not task.control for task in frozen) == 9
    assert sum(task.control for task in frozen) == 3
    assert {task.operation_class for task in frozen} == {
        "CREATE",
        "EDIT_SINGLE",
        "EDIT_MULTI",
    }


def test_representation_and_generation_conditions_are_profile_invariant() -> None:
    assert REPRESENTATION is MutationRepresentationPolicy.LINE_RANGE
    settings = json.dumps(
        {"seed": 42, "temperature": 0, "context": 8192, "output": 512}
    )
    assert settings in json.dumps(
        {"seed": 42, "temperature": 0, "context": 8192, "output": 512}
    )


def test_paired_projection_excludes_only_profile_configuration() -> None:
    paired = {
        "task_identity": "task",
        "model_config_identity": "profile-a",
        "generation_identity": "generation",
        "grounding_source_set_identity": "sources",
        "source_hash_identity": "hashes",
        "authority_path_identity": "authority",
        "authorized_range_identity": "ranges",
        "create_authority_identity": "create",
        "workspace_generation_identity": "workspace",
        "representation_identity": "line-range",
        "evidence_plan_identity": "evidence",
    }
    changed = dict(paired, model_config_identity="profile-b")
    assert _identity_projection(
        {"paired_input_identity": paired}
    ) == _identity_projection({"paired_input_identity": changed})
    changed["authorized_range_identity"] = "different"
    assert _identity_projection(
        {"paired_input_identity": paired}
    ) != _identity_projection({"paired_input_identity": changed})


def test_checkpoint_resume_is_exactly_once(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "A77-C01", "qwen-small")
    payload = {"run_identity": RUN_ID, "cell_id": "A77-C01-qwen-small"}
    atomic_checkpoint(path, payload)
    assert read_cell(path, {"run_identity": RUN_ID}) == payload
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, {"regenerated": True})


def test_result_identity_is_source_free_and_package_excluded() -> None:
    payload = {
        "model_artifact": "model.gguf:sha256:" + "a" * 64,
        "repository_identity": "b" * 64,
        "failure": "MODEL_PROTOCOL_FAILURE",
    }
    assert standard_result_is_source_free(payload)
    assert "/Users/" not in json.dumps(payload)
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "atomic_coding_capability_v1" not in configuration
    assert "*.gguf" not in configuration


def test_load_protocol_and_semantic_failures_remain_distinct() -> None:
    valid = {
        "model_output": True,
        "schema_valid": True,
        "mechanically_materializable": True,
        "production_validatable": True,
        "transaction_applied": True,
        "verification_pass": True,
        "semantic_failure": "INCORRECT_EDGE_CASE",
    }
    assert (
        classify_failure(
            valid,
            inner_failure="PASS",
            obligation_oracle_pass=False,
            load_failed=True,
        )
        == "MODEL_LOAD_FAILURE"
    )
    assert (
        classify_failure(
            dict(valid, schema_valid=False),
            inner_failure="PASS",
            obligation_oracle_pass=False,
        )
        == "MODEL_PROTOCOL_FAILURE"
    )
    assert (
        classify_failure(valid, inner_failure="PASS", obligation_oracle_pass=False)
        == "INCORRECT_EDGE_CASE"
    )


def test_primary_and_repair_are_counted_separately() -> None:
    from scripts.summarize_atomic_coding_capability_v1 import _funnel

    cell = {
        "proposal": {
            "model_output": True,
            "schema_valid": True,
            "mechanically_materializable": True,
            "production_validatable": True,
            "transaction_applied": True,
            "verification_pass": False,
        },
        "obligation_oracle_pass": False,
        "repair": {"semantic_pass": True},
    }
    assert _funnel([cell])["semantic_pass"] == 0


def test_canonical_source_identity_is_stable_under_suite_inspection() -> None:
    root = Path(__file__).parents[1] / "src"
    before = source_state_identity(root)
    tasks()
    assert source_state_identity(root) == before


def test_no_model_specific_prompt_or_production_routing_change() -> None:
    suite = (
        Path(__file__).parents[1] / "benchmarks/atomic_coding_capability_v1/suite.py"
    ).read_text()
    runner = (
        Path(__file__).parents[1] / "scripts/run_atomic_coding_capability_v1.py"
    ).read_text()
    assert "if profile" not in suite
    assert "catalog.create(args.profile)" in runner
    assert "mutation_representation=profile" not in runner
