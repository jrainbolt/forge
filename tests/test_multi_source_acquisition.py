from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from forge.evaluation.multi_source_acquisition import run_multi_source_acquisition_v1
from forge.interaction import AutonomyMode
from forge.models import MockModel
from forge.orchestration import (
    CodingTaskState,
    DuplicateToolCallIdError,
    RepositoryChatSession,
)
from forge.project_config import ProjectCommand, ProjectCommands, VerificationPlan
from forge.tools import (
    PermissionDecision,
    RuleBasedPolicy,
    create_assist_repository_policy,
    create_assist_repository_registry,
    create_readonly_repository_policy,
    create_readonly_repository_registry,
)


def _read(identifier: str, path: str) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "id": identifier,
            "tool": "repository.read_file",
            "arguments": {"path": path},
        }
    )


def _group(old: int, new: int) -> str:
    return json.dumps(
        {
            "type": "multi_file_structured_edit",
            "edits": [
                {"path": "a.py", "old_text": f"A = {old}", "new_text": f"A = {new}"},
                {"path": "b.py", "old_text": f"B = {old}", "new_text": f"B = {new}"},
            ],
        }
    )


def _workspace(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 1\n")


def _session(
    tmp_path: Path,
    model: MockModel,
    *,
    policy=None,  # type: ignore[no-untyped-def]
    commands: ProjectCommands | None = None,
    max_tools: int | None = None,
    repair: bool = False,
) -> RepositoryChatSession:
    configured = commands or ProjectCommands()
    return RepositoryChatSession(
        "acquisition",
        model,
        tmp_path,
        mode=AutonomyMode.REPAIR if repair else AutonomyMode.AGENT,
        registry=create_assist_repository_registry(configured),
        policy=policy or create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        minimum_source_files=2,
        required_candidate_paths=("b.py", "a.py"),
        require_relevant_source=False,
        skip_verification=commands is None,
        verification_plan=configured.verification_plan,
        max_tool_executions=max_tools,
    )


def test_q01_two_required_sources_are_acquired_before_first_model_call(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    model = MockModel((_group(1, 2), json.dumps({"type": "final", "answer": "done"})))
    response = _session(tmp_path, model).ask("Change both files")

    assert [item.path for item in response.tool_activity[:2]] == ["a.py", "b.py"]
    assert all(
        item.acquisition_origin == "orchestrator_required"
        for item in response.tool_activity[:2]
    )
    assert response.coding_task is not None
    metrics = response.coding_task.source_acquisition_metrics
    assert metrics.required_sources_ready == 2
    assert metrics.deterministic_source_reads == 2
    assert response.coding_task.transition_metrics.entries == 1


def test_q02_already_trusted_source_is_not_missing_or_reacquired() -> None:
    state = CodingTaskState(0)
    state.establish_required_candidates(("b.py", "a.py"))
    state.note_source_acquired(
        "a.py", "a" * 64, start_line=1, end_line=1, deterministic=False
    )
    assert tuple(item.path for item in state.missing_required_candidates) == ("b.py",)
    assert state.source_acquisition_metrics.model_source_reads == 1


def test_q03_required_candidate_order_is_deterministic() -> None:
    state = CodingTaskState(0)
    state.establish_required_candidates(("z.py", "a.py", "m.py"))
    assert state.required_candidate_paths == ("a.py", "m.py", "z.py")


def test_q04_read_deny_blocks_without_model_call(tmp_path: Path) -> None:
    _workspace(tmp_path)
    model = MockModel((json.dumps({"type": "final", "answer": "unused"}),))
    policy = RuleBasedPolicy(
        {
            "repository.read_file": PermissionDecision.DENY,
            "repository.apply_patch": PermissionDecision.ASK,
            "repository.apply_multi_patch": PermissionDecision.ASK,
        }
    )
    session = _session(tmp_path, model, policy=policy)
    with pytest.raises(Exception, match="required source acquisition failed"):
        session.ask("Change both files")
    assert len(model.requests) == 0
    assert session.last_coding_task is not None
    assert (
        session.last_coding_task.source_acquisition_metrics.source_acquisition_failures
        == 1
    )


def test_q05_missing_required_file_fails_once_without_model_loop(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    model = MockModel((json.dumps({"type": "final", "answer": "unused"}),))
    session = _session(tmp_path, model)
    with pytest.raises(Exception, match="required source acquisition failed"):
        session.ask("Change both files")
    assert len(model.requests) == 0
    assert session.last_coding_task is not None
    attempts = {
        item.path: item.acquisition_attempts
        for item in session.last_coding_task.required_candidates
    }
    assert attempts == {"a.py": 1, "b.py": 1}


def test_q06_budget_reserves_mutation_and_three_verification_steps(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    command = ProjectCommand((sys.executable, "-c", "pass"), 5)
    plan = VerificationPlan(
        "three-step", ("project.configure", "project.build", "project.test")
    )
    commands = ProjectCommands(command, command, plan, command)
    model = MockModel((_group(1, 2),))
    session = _session(tmp_path, model, commands=commands, max_tools=5)
    with pytest.raises(Exception, match="reserved tool budget"):
        session.ask("Change both files")
    assert len(model.requests) == 0
    assert session.last_activity == ()


def test_q07_generation_change_makes_required_sources_missing_again() -> None:
    state = CodingTaskState(0)
    state.establish_required_candidates(("a.py", "b.py"))
    for path in state.required_candidate_paths:
        state.note_source_acquired(
            path, path[0] * 64, start_line=1, end_line=1, deterministic=True
        )
    assert state.required_sources_ready
    state.invalidate_mutation_ready(1)
    assert not state.required_sources_ready
    assert state.source_acquisition_metrics.required_sources_missing == 2


def test_q08_grouped_repair_reacquires_both_original_paths(tmp_path: Path) -> None:
    _workspace(tmp_path)
    model = MockModel(
        (
            _group(1, 2),
            _group(2, 3),
            json.dumps({"type": "final", "answer": "repaired"}),
        )
    )
    test = ProjectCommand(
        (
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "sys.exit(0 if 'A = 3' in Path('a.py').read_text() and "
            "'B = 3' in Path('b.py').read_text() else 1)",
        ),
        5,
    )
    response = _session(
        tmp_path,
        model,
        commands=ProjectCommands(test=test),
        repair=True,
    ).ask("Change both files and repair if verification fails")
    assert response.coding_task is not None
    assert response.coding_task.mutation_count == 2
    assert (
        response.coding_task.source_acquisition_metrics.deterministic_source_reads == 4
    )
    acquired = [
        item.path
        for item in response.tool_activity
        if item.acquisition_origin == "orchestrator_required"
    ]
    assert acquired == ["a.py", "b.py", "a.py", "b.py"]


def test_q09_duplicate_id_gets_one_correction_then_new_id_executes(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    model = MockModel(
        (
            _read("same", "a.py"),
            _read("same", "b.py"),
            _read("new", "b.py"),
            _group(1, 2),
            json.dumps({"type": "final", "answer": "done"}),
        )
    )
    session = RepositoryChatSession(
        "ids",
        model,
        tmp_path,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        minimum_source_files=2,
        require_relevant_source=False,
        skip_verification=True,
    )
    response = session.ask("Change both files")
    assert [item.invocation_id for item in response.tool_activity[:2]] == [
        "same",
        "new",
    ]
    assert response.protocol_corrections == 1
    assert response.coding_task is not None
    assert response.coding_task.source_acquisition_metrics.duplicate_tool_call_ids == 1


def test_q10_second_duplicate_is_terminal_without_tool_execution(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    model = MockModel(
        (_read("same", "a.py"), _read("same", "b.py"), _read("same", "b.py"))
    )
    session = RepositoryChatSession(
        "ids",
        model,
        tmp_path,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        minimum_source_files=2,
        require_relevant_source=False,
        skip_verification=True,
    )
    with pytest.raises(DuplicateToolCallIdError) as caught:
        session.ask("Change both files")
    assert caught.value.classification == "DUPLICATE_TOOL_CALL_ID"
    assert [item.invocation_id for item in session.last_activity] == ["same"]
    assert len(model.requests) == 3
    assert session.last_coding_task is not None
    assert (
        session.last_coding_task.source_acquisition_metrics.duplicate_tool_call_ids == 2
    )


def test_tool_call_id_scope_is_session_local(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        (root / "a.py").write_text("A = 1\n")
        model = MockModel(
            (_read("shared", "a.py"), json.dumps({"type": "final", "answer": "ok"}))
        )
        response = RepositoryChatSession(
            "scope",
            model,
            root,
            registry=create_readonly_repository_registry(),
            policy=create_readonly_repository_policy(),
            require_relevant_source=False,
        ).ask("Inspect a.py")
        assert response.tool_activity[0].invocation_id == "shared"


def test_tool_call_ids_remain_reserved_across_turn_compaction(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    model = MockModel(
        (
            _read("shared", "a.py"),
            json.dumps({"type": "final", "answer": "first"}),
            _read("shared", "a.py"),
            _read("fresh", "a.py"),
            json.dumps({"type": "final", "answer": "second"}),
        )
    )
    session = RepositoryChatSession(
        "scope",
        model,
        tmp_path,
        registry=create_readonly_repository_registry(),
        policy=create_readonly_repository_policy(),
        require_relevant_source=False,
    )
    session.ask("Inspect a.py once")
    second = session.ask("Inspect a.py again")
    assert second.protocol_corrections == 1
    assert tuple(item.invocation_id for item in second.tool_activity) == ("fresh",)


def test_same_tool_call_with_different_ids_is_legal(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")

    def listing(identifier: str) -> str:
        return json.dumps(
            {
                "type": "tool_call",
                "id": identifier,
                "tool": "repository.list_directory",
                "arguments": {"path": "."},
            }
        )

    model = MockModel(
        (
            listing("first"),
            listing("second"),
            _read("source", "a.py"),
            json.dumps({"type": "final", "answer": "done"}),
        )
    )
    response = RepositoryChatSession(
        "scope",
        model,
        tmp_path,
        registry=create_readonly_repository_registry(),
        policy=create_readonly_repository_policy(),
        require_relevant_source=False,
    ).ask("Inspect a.py")
    assert tuple(item.invocation_id for item in response.tool_activity) == (
        "first",
        "second",
        "source",
    )


def test_multi_source_acquisition_v1_ten_scenarios_pass(tmp_path: Path) -> None:
    run = run_multi_source_acquisition_v1(tmp_path / "suite")
    assert tuple(case.case_id for case in run.cases) == tuple(
        f"Q{number:02d}" for number in range(1, 11)
    )
    assert run.tasks_passed == run.tasks_total == 10
