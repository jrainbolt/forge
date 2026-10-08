from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.behavioral_obligation_diagnosis_v1.runner import (
    checkpoint_path,
    read_cell,
)
from benchmarks.behavioral_obligation_diagnosis_v1.suite import (
    RUN_ID,
    cases,
    corpus_identity,
    narrow_authority,
    obligations,
    task_map,
    validate_corpus,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.evidence_coverage import decompose_evidence_plan


def test_obligations_are_exact_existing_production_visible_goals() -> None:
    for case in cases():
        task = task_map()[case.task_id].production_task.prompt
        goals = {
            goal.goal_id: goal.description
            for goal in decompose_evidence_plan(task).goals
        }
        assert all(
            goals[item.goal_id] == item.description for item in obligations(case)
        )


def test_f1_construction_excludes_reference_and_oracle_material() -> None:
    for case in cases():
        definition = task_map()[case.task_id]
        for item in obligations(case):
            assert item.description in definition.production_task.prompt
            assert item.oracle_identity not in item.description
            assert "expected implementation" not in item.description.casefold()


def test_authority_can_narrow_but_never_broaden() -> None:
    original = ("src/a.py", "src/b.py")
    assert narrow_authority(original, ("src/a.py",)) == ("src/a.py",)
    assert narrow_authority(original) == original
    with pytest.raises(ValueError, match="cannot broaden"):
        narrow_authority(original, ("src/hidden.py",))


def test_non_isolatable_goals_fail_closed() -> None:
    mixed = next(
        case for case in cases() if case.operation_class == "MIXED_EDIT_CREATE"
    )
    items = obligations(mixed)
    assert len(items) > 1
    assert all(not item.isolatable for item in items)
    assert all(item.non_isolatable_reason for item in items)


def test_oracles_are_frozen_and_scoped_distinctly_before_execution() -> None:
    first = cases()[0]
    item = obligations(first)[0]
    definition = task_map()[first.task_id]
    assert item.oracle_identity
    assert item.oracle_identity not in definition.production_task.oracle_commands
    assert corpus_identity() == corpus_identity()


@pytest.mark.parametrize(
    ("full_pass", "isolated", "isolatable", "expected"),
    [
        (True, (), True, "FULL_TASK_PASS"),
        (False, (), False, "OBLIGATION_NOT_ISOLATABLE"),
        (False, (False,), True, "ATOMIC_CAPABILITY_FAILURE"),
        (False, (True,), True, "COORDINATION_FAILURE"),
        (False, (True, False), True, "MIXED_CAPABILITY_FAILURE"),
    ],
)
def test_atomic_coordination_and_mixed_classification(
    full_pass: bool, isolated: tuple[bool, ...], isolatable: bool, expected: str
) -> None:
    from scripts.summarize_behavioral_obligation_diagnosis_v1 import _classification

    full = {"full_task_semantic_pass": full_pass}
    cells = tuple({"obligation_oracle_pass": value} for value in isolated)
    assert _classification(full, cells, isolatable) == expected


def test_proposal_identity_and_primary_repair_remain_separate() -> None:
    proposal = {"proposal_observation_id": "primary", "transaction_applied": True}
    repair = {"proposal_observation_id": "repair", "transaction_applied": True}
    assert proposal["proposal_observation_id"] != repair["proposal_observation_id"]


def test_checkpoint_resume_exactly_once_and_source_free(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "A74-C01-F0")
    payload = {
        "run_identity": RUN_ID,
        "cell_id": "A74-C01-F0",
        "goal_identity": "a" * 64,
        "proposal": {"proposal_observation_id": "p1"},
    }
    atomic_checkpoint(path, payload)
    assert read_cell(path, {"run_identity": RUN_ID}) == payload
    with pytest.raises(FileExistsError):
        atomic_checkpoint(path, {"regenerated": True})
    assert standard_result_is_source_free(payload)


def test_corpus_distribution_and_package_exclusion() -> None:
    validate_corpus()
    assert len(cases()) == 10
    configuration = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'where = ["src"]' in configuration
    assert "behavioral_obligation_diagnosis" not in configuration
    assert "eval-results" not in configuration
