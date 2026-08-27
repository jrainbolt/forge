from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.evaluation import (
    EvaluationOutcome,
    ExpectedApproval,
    RealWorldEvaluationRunner,
    RealWorldFailure,
    RealWorldLevel,
    RealWorldMetrics,
    RealWorldStatus,
    RealWorldTask,
    RealWorldTaskResult,
    RepositorySnapshot,
    SetupReplacement,
    apply_task_setup,
    changed_paths,
    copy_repository,
    foundation_realworld_tasks,
    hash_workspace,
    realworld_run_to_dict,
    run_oracle,
    summarize_results,
)
from forge.evaluation.realworld import TaskSetupError, score_task_result
from forge.interaction import AutonomyMode
from forge.models import MockModel, ModelUsage
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import MutationPreview, PreparedProjectCommand


def task(**values: object) -> RealWorldTask:
    defaults: dict[str, object] = {
        "task_id": "E99",
        "level": RealWorldLevel.REPOSITORY_REASONING,
        "mode": AutonomyMode.READ,
        "prompt": "Find the implementation.",
        "expected_files": ("src/main.py",),
    }
    defaults.update(values)
    return RealWorldTask(**defaults)  # type: ignore[arg-type]


def repository(tmp_path: Path) -> Path:
    root = tmp_path / "canonical"
    (root / "src").mkdir(parents=True)
    (root / "src/main.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def test_suite_is_versioned_bounded_and_records_seed_policy() -> None:
    tasks = foundation_realworld_tasks()
    assert [value.task_id for value in tasks] == [f"E0{i}" for i in range(1, 9)]
    assert all(value.seeds == (7, 42) for value in tasks[:3])
    assert all(value.seeds == (42,) for value in tasks[3:])
    assert tasks[-1].unsupported_reason is not None


def test_copy_isolated_from_canonical_and_omits_git(tmp_path: Path) -> None:
    source = repository(tmp_path)
    (source / ".git").mkdir()
    (source / ".git/config").write_text("canonical", encoding="utf-8")
    before = hash_workspace(source)
    copied = copy_repository(source, tmp_path / "copy")
    (copied / "src/main.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert hash_workspace(source) == before
    assert not (copied / ".git").exists()


def test_setup_requires_one_exact_match_and_is_confined(tmp_path: Path) -> None:
    root = repository(tmp_path)
    apply_task_setup(root, (SetupReplacement("src/main.py", "VALUE = 1", "VALUE = 2"),))
    assert "VALUE = 2" in (root / "src/main.py").read_text()
    with pytest.raises(TaskSetupError, match="matched 0 times"):
        apply_task_setup(root, (SetupReplacement("src/main.py", "missing", "new"),))
    with pytest.raises(TaskSetupError, match="unavailable"):
        apply_task_setup(root, (SetupReplacement("../escape", "old", "new"),))


def test_hashes_detect_modified_created_and_deleted_files(tmp_path: Path) -> None:
    root = repository(tmp_path)
    before = hash_workspace(root)
    (root / "src/main.py").write_text("VALUE = 2\n")
    (root / "new.py").write_text("new\n")
    assert changed_paths(before, hash_workspace(root)) == ("new.py", "src/main.py")


def test_expected_approval_restricts_paths_and_exact_commands(tmp_path: Path) -> None:
    root = repository(tmp_path)
    current_task = task(
        mode=AutonomyMode.ASSIST,
        allowed_paths=("src/main.py",),
        max_mutations=1,
    )
    commands = ProjectCommands(test=ProjectCommand(("pytest",), 30))
    approval = ExpectedApproval(current_task, root, commands)
    allowed = MutationPreview("src/main.py", "patch", "diff", "a", "b")
    denied = MutationPreview("other.py", "patch", "diff", "a", "b")
    assert approval(SimpleNamespace(), allowed)  # type: ignore[arg-type]
    assert not approval(SimpleNamespace(), denied)  # type: ignore[arg-type]
    assert approval(
        SimpleNamespace(),  # type: ignore[arg-type]
        PreparedProjectCommand("test", ("pytest",), root.resolve(), 30.0),
    )
    assert not approval(
        SimpleNamespace(),  # type: ignore[arg-type]
        PreparedProjectCommand("test", ("pytest", "-q"), root.resolve(), 30.0),
    )
    assert (approval.approved, approval.rejected) == (2, 2)


def test_independent_oracle_uses_real_argument_array_subprocess(tmp_path: Path) -> None:
    root = repository(tmp_path)
    assert run_oracle(root, (("true",),)) is EvaluationOutcome.PASS
    assert run_oracle(root, (("false",),)) is EvaluationOutcome.FAIL
    assert run_oracle(root, ()) is EvaluationOutcome.NOT_RUN


def test_scoring_detects_unexpected_mutation_and_taxonomizes_it(tmp_path: Path) -> None:
    root = repository(tmp_path)
    before = hash_workspace(root)
    (root / "wrong.py").write_text("wrong\n")
    after = hash_workspace(root)
    current_task = task(
        mode=AutonomyMode.ASSIST,
        allowed_paths=("src/main.py",),
        expected_changed_paths=("src/main.py",),
        max_mutations=1,
    )
    approval = ExpectedApproval(current_task, root, ProjectCommands())
    response = SimpleNamespace(
        tool_activity=(),
        coding_task=SimpleNamespace(mutation_count=1, status=None),
        agent_task=None,
        usage=ModelUsage(),
        orchestration_steps=1,
    )
    result = score_task_result(
        current_task,
        42,
        response,
        None,
        None,
        before,
        after,
        changed_paths(before, after),
        ("wrong.py",),
        EvaluationOutcome.FAIL,
        approval,
        1.0,
    )
    assert result.status is RealWorldStatus.FAIL
    assert result.failure is RealWorldFailure.MUTATION
    assert result.unexpected_paths == ("wrong.py",)


def test_failure_scoring_retains_activity_recorded_before_exception(
    tmp_path: Path,
) -> None:
    root = repository(tmp_path)
    current_task = task()
    hashes = hash_workspace(root)
    activity = SimpleNamespace(
        path="src/main.py",
        status="success",
        tool_name="repository.read_file",
    )
    scored = score_task_result(
        current_task,
        42,
        None,
        RealWorldFailure.TOOL_LIMIT,
        "limit",
        hashes,
        hashes,
        (),
        (),
        EvaluationOutcome.NOT_RUN,
        ExpectedApproval(current_task, root, ProjectCommands()),
        1.0,
        (activity,),
    )
    assert scored.metrics.tool_executions == 1
    assert scored.metrics.model_calls == 1
    assert scored.metrics.source_reads == 1
    assert scored.expected_files_found == ("src/main.py",)


def test_unsupported_is_infrastructure_success_not_model_failure(
    tmp_path: Path,
) -> None:
    root = repository(tmp_path)
    current_task = task(
        level=RealWorldLevel.BOUNDED_REPAIR,
        mode=AutonomyMode.AGENT,
        expected_files=("src/main.py", "tests/test_main.py"),
        unsupported_reason="multi-file ceiling",
    )
    snapshot = RepositorySnapshot(
        "fixture",
        "abc",
        "Python",
        1,
        0,
        1,
        ("true",),
        ("true",),
        EvaluationOutcome.PASS,
        0.1,
    )
    run = RealWorldEvaluationRunner("mock", MockModel(("unused",)), root).run(
        (current_task,), snapshot
    )
    result = run.results[0]
    assert result.status is RealWorldStatus.UNSUPPORTED
    assert result.infrastructure is EvaluationOutcome.PASS
    assert result.model is EvaluationOutcome.NOT_RUN
    assert result.failure is RealWorldFailure.CAPABILITY_UNSUPPORTED
    assert run.canonical_unchanged


def result(status: RealWorldStatus, *, level: RealWorldLevel) -> RealWorldTaskResult:
    return RealWorldTaskResult(
        "E",
        "read",
        level.value,
        42,
        status,
        EvaluationOutcome.PASS,
        EvaluationOutcome.PASS,
        EvaluationOutcome.PASS,
        None,
        None,
        "completed",
        ("src/main.py",),
        (),
        (),
        (),
        (),
        RealWorldMetrics(tool_executions=2, source_reads=1),
        ModelUsage(),
        2.0,
    )


def test_aggregate_metrics_and_serialization_are_deterministic(tmp_path: Path) -> None:
    values = (
        result(RealWorldStatus.PASS, level=RealWorldLevel.REPOSITORY_REASONING),
        result(RealWorldStatus.PARTIAL, level=RealWorldLevel.SINGLE_CHANGE),
    )
    summary = summarize_results(values)
    assert (summary.runs, summary.passed, summary.partial) == (2, 1, 1)
    assert summary.mean_tool_executions == 2.0
    root = repository(tmp_path)
    snapshot = RepositorySnapshot(
        "fixture",
        "abc",
        "Python",
        1,
        0,
        1,
        ("true",),
        ("true",),
        EvaluationOutcome.PASS,
        0.1,
    )
    run = RealWorldEvaluationRunner("mock", MockModel(("unused",)), root).run(
        (
            task(
                level=RealWorldLevel.BOUNDED_REPAIR,
                mode=AutonomyMode.AGENT,
                unsupported_reason="ceiling",
            ),
        ),
        snapshot,
    )
    document = realworld_run_to_dict(run)
    assert document["suite"] == "realworld-v1"
    assert document["results"][0]["seed"] == 42  # type: ignore[index]
