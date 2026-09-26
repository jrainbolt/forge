"""A57 evaluator invariants; no model or network is needed."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.realistic_coding_v2.runner import _atomic_json
from benchmarks.repair_framing_v1.runner import (
    PrimaryRecord,
    Trial,
    _diagnostic,
    _failure_activity,
    diagnose,
    load_trial,
    tree_hash,
    trial_path,
)
from benchmarks.repair_framing_v1.suite import (
    a56_eligibility,
    cases,
    corrective_task,
)


@pytest.fixture
def case():
    return cases()[1]  # CREATE: the created source is now an existing edit target.


@pytest.fixture
def primary(case):
    return PrimaryRecord(
        case.case_id,
        case.profile,
        case.task_id,
        case.definition.version,
        case.seed,
        case.definition.operation_class.value,
        case.definition.language,
        "a56",
        "model:sha256:abc",
        "primary",
        "workspace",
        (),
        "evidence",
        "project.test",
        "failure",
        tuple(case.definition.production_task.expected_changed_paths),
        "YES",
        True,
        None,
    )


def _trial(case, primary, condition: str, *, recovered: bool = False) -> Trial:
    return Trial(
        "repair-framing-v1",
        1,
        case.case_id,
        condition,
        case.profile,
        case.task_id,
        case.seed,
        primary.post_primary_hash,
        "manifest",
        primary.authorized_paths,
        True,
        condition != "R1",
        condition == "R0",
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        recovered,
        None,
        None,
        None,
        recovered,
        recovered,
        "SEMANTIC_RECOVERED" if recovered else "SEMANTIC_STILL_FAIL",
        100,
        20,
        1,
        1.0,
        1.0,
        2.0,
    )


def test_f01_independent_copies_have_identical_bytes(tmp_path: Path) -> None:
    root = tmp_path / "base"
    root.mkdir()
    (root / "source.py").write_bytes(b"mutated\n")
    import shutil

    copies = [tmp_path / key for key in ("R0", "R1", "R2")]
    for copy in copies:
        shutil.copytree(root, copy)
    assert len({tree_hash(copy) for copy in copies}) == 1
    (copies[0] / "source.py").write_bytes(b"new\n")
    assert tree_hash(copies[0]) != tree_hash(copies[1])
    assert tree_hash(copies[1]) == tree_hash(copies[2])


def test_f02_r0_is_a_production_replay_path() -> None:
    from inspect import getsource

    from benchmarks.repair_framing_v1.runner import run_condition

    source = getsource(run_condition)
    assert "RealWorldEvaluationRunner(" in source
    assert "case.definition.production_task" in source
    assert "prefix=prefix" in source


def test_f03_r1_excludes_repair_framing(case, primary) -> None:
    prompt = corrective_task(case, primary.authorized_paths, None).prompt.lower()
    assert "repair" not in prompt
    assert "previous patch" not in prompt


def test_f04_r1_excludes_verification_evidence(case, primary) -> None:
    prompt = corrective_task(case, primary.authorized_paths, None).prompt
    assert "VERIFICATION_SENTINEL" not in prompt


def test_f05_r2_includes_exact_bounded_evidence(case, primary) -> None:
    evidence = '{"type":"tool_result","output":"VERIFICATION_SENTINEL"}'
    prompt = corrective_task(case, primary.authorized_paths, evidence).prompt
    assert evidence in prompt
    assert prompt.count(evidence) == 1


def test_f06_no_hidden_oracle_in_corrective_prompts(case, primary) -> None:
    for evidence in (None, '{"output":"configured test failed"}'):
        prompt = corrective_task(case, primary.authorized_paths, evidence).prompt
        assert "oracle_commands" not in prompt
        assert "reference" not in prompt.lower()


def test_f07_original_task_preserved(case, primary) -> None:
    original = case.definition.production_task.prompt
    assert original in corrective_task(case, primary.authorized_paths, None).prompt
    assert original in corrective_task(case, primary.authorized_paths, "failure").prompt


def test_f08_exact_path_authority(case, primary) -> None:
    task = corrective_task(case, primary.authorized_paths, None)
    assert task.allowed_paths == primary.authorized_paths
    assert task.required_candidate_paths == primary.authorized_paths
    with pytest.raises(ValueError, match="widen"):
        corrective_task(case, (*primary.authorized_paths, "extra.py"), None)


def test_f09_no_create_authority(case, primary) -> None:
    task = corrective_task(case, primary.authorized_paths, None)
    assert task.create_candidate_paths == ()
    assert task.setup_absent_paths == ()
    assert not task.mixed_file_operations


def test_f10_fresh_current_source_required(case, primary) -> None:
    task = corrective_task(case, primary.authorized_paths, None)
    assert task.required_candidate_paths == primary.authorized_paths
    assert "current workspace state" in task.prompt


def test_f11_full_verification_plan_retained(case, primary) -> None:
    original = case.definition.production_task
    task = corrective_task(case, primary.authorized_paths, "failure")
    assert task.configure_command == original.configure_command
    assert task.build_command == original.build_command
    assert task.test_command == original.test_command


def test_f12_verification_and_semantic_separate(case, primary) -> None:
    trial = replace(
        _trial(case, primary, "R2"),
        verification_passed=True,
        semantic_passed=False,
        semantic_recovered=False,
    )
    assert trial.verification_passed and not trial.semantic_recovered


def test_f13_diagnosis(case, primary) -> None:
    failed = tuple(_trial(case, primary, key) for key in ("R0", "R1", "R2"))
    assert diagnose(failed) == "MODEL_LIMITED"
    helpful = (failed[0], failed[1], replace(failed[2], semantic_recovered=True))
    assert diagnose(helpful) == "VERIFICATION_EVIDENCE_HELPFUL"


def test_f14_standard_result_is_source_free(case, primary) -> None:
    text = json.dumps(asdict(_trial(case, primary, "R1")))
    assert "source.py contents" not in text
    assert "verification_evidence" not in text
    assert "responses" not in text


def test_f15_checkpoint_resume(tmp_path: Path, case, primary) -> None:
    trial = _trial(case, primary, "R1")
    _atomic_json(trial_path(tmp_path, case, "R1"), asdict(trial))
    assert load_trial(tmp_path, case, "R1", "manifest") == trial
    with pytest.raises(ValueError, match="identity"):
        load_trial(tmp_path, case, "R1", "wrong")


def test_f16_verification_pass_oracle_fail_excluded() -> None:
    eligible, reason = a56_eligibility(
        {
            "transaction_executed": True,
            "final_semantic": False,
            "verification_result": "pass",
            "repair_eligible": True,
        }
    )
    assert not eligible and reason == "UNOBSERVABLE_TO_PRODUCTION_REPAIR"


def test_f17_created_file_is_existing_edit_target(case, primary) -> None:
    assert case.definition.operation_class.value == "CREATE"
    task = corrective_task(case, primary.authorized_paths, None)
    assert task.create_candidate_paths == ()
    assert task.required_candidate_paths


def test_diagnostic_selects_only_failing_configured_result() -> None:
    evidence = json.dumps(
        {"type": "tool_result", "tool": "project.test", "status": "failure"}
    )
    request = SimpleNamespace(
        messages=(
            SimpleNamespace(content='{"type":"tool_result","status":"success"}'),
            SimpleNamespace(content=evidence),
        )
    )
    assert _diagnostic(request) == evidence


def test_unrelated_tool_failure_not_used_as_verification_evidence(
    tmp_path: Path,
) -> None:
    item = SimpleNamespace(
        tool_name="repository.read_file", status="failure", output={}
    )
    assert _failure_activity(item, tmp_path) is None


def test_tree_hash_detects_file_mode_and_content(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"one")
    original = tree_hash(tmp_path)
    path.write_bytes(b"two")
    assert tree_hash(tmp_path) != original
