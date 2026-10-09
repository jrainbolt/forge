from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.default_candidate_confirmation_v1.qualification import (
    WORKFLOW_REACHABLE,
    WORKFLOW_UNREACHABLE,
    WorkflowReachability,
    qualify_workflow_reachability,
)
from benchmarks.default_candidate_confirmation_v1.suite import (
    REPRESENTATION,
    VERSION,
    tasks,
)
from benchmarks.realistic_coding_v2.suite import (
    BUILD,
    CONFIGURE,
    REPOSITORY,
    TEST,
    baseline_workspace,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.realworld import EvaluationOutcome, run_oracle


@pytest.fixture(scope="module")
def reachability() -> dict[str, WorkflowReachability]:
    return {
        definition.task_id: qualify_workflow_reachability(
            definition, representation=REPRESENTATION
        )
        for definition in tasks()
    }


def test_all_twelve_baselines_reach_model_boundary(
    reachability: dict[str, WorkflowReachability],
) -> None:
    assert len(reachability) == 12
    assert all(
        item.classification == WORKFLOW_REACHABLE for item in reachability.values()
    )
    assert all(item.setup_succeeded for item in reachability.values())
    assert all(
        item.session_reached and item.model_calls >= 1 for item in reachability.values()
    )


def test_setup_failure_is_detected_before_model_invocation() -> None:
    h09 = next(item for item in tasks() if item.task_id == "H09")
    unreachable = replace(
        h09,
        production_task=replace(h09.production_task, setup_commands=(CONFIGURE,)),
    )
    result = qualify_workflow_reachability(unreachable, representation=REPRESENTATION)
    assert result.classification == WORKFLOW_UNREACHABLE
    assert not result.setup_succeeded
    assert not result.session_reached
    assert result.model_calls == 0
    assert result.failure == "INFRASTRUCTURE"


def test_h09_semantic_failure_is_workflow_reachable_and_reference_is_unchanged(
    tmp_path: Path,
    reachability: dict[str, WorkflowReachability],
) -> None:
    h09 = next(item for item in tasks() if item.task_id == "H09")
    baseline = baseline_workspace(h09, tmp_path / "baseline")
    assert (
        run_oracle(baseline, h09.production_task.oracle_commands)
        is EvaluationOutcome.FAIL
    )
    assert reachability["H09"].classification == WORKFLOW_REACHABLE

    reference = tmp_path / "reference"
    shutil.copytree(baseline, reference)
    target = reference / "pyservice/config.py"
    target.write_bytes((REPOSITORY / "pyservice/config.py").read_bytes())
    assert (
        run_oracle(reference, h09.production_task.oracle_commands)
        is EvaluationOutcome.PASS
    )
    assert run_oracle(reference, (CONFIGURE, BUILD, TEST)) is EvaluationOutcome.PASS
    assert h09.production_task.prompt.startswith("Create missing pyservice/config.py")
    assert h09.create_paths == ("pyservice/config.py",)


def test_reachability_payload_is_source_free_and_package_excluded(
    reachability: dict[str, WorkflowReachability],
) -> None:
    payload = {key: value.payload() for key, value in reachability.items()}
    encoded = json.dumps(payload)
    assert VERSION == 2
    assert "/Users/" not in encoded
    assert "qualification probe" not in encoded
    assert standard_result_is_source_free(payload)
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "default_candidate_confirmation_v1" not in configuration
    assert "eval-results" not in configuration
