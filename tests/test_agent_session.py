from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from forge.models import MockModel
from forge.orchestration import (
    AgentCancelled,
    AgentStopReason,
    CodingTaskStatus,
    RepositoryChatSession,
    RepositoryOrchestrationError,
)
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)


def call(call_id: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": call_id, "tool": tool, "arguments": arguments}
    )


def final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src/value.py").write_text("VALUE = 1\n")
    (root / "src/other.py").write_text("OTHER = 1\n")
    return root


def patch_arguments(workspace: Path) -> dict[str, object]:
    data = (workspace / "src/value.py").read_bytes()
    return {
        "path": "src/value.py",
        "expected_sha256": hashlib.sha256(data).hexdigest(),
        "edits": [{"old": "VALUE = 1", "new": "VALUE = 2"}],
    }


def agent_session(
    workspace: Path,
    model: MockModel,
    *,
    approve=lambda *_args: True,
    test_code: str | None = None,
    **limits: object,
) -> RepositoryChatSession:
    commands = ProjectCommands(
        test=(
            ProjectCommand((sys.executable, "-c", test_code), 5)
            if test_code is not None
            else None
        )
    )
    return RepositoryChatSession(
        "fixture",
        model,
        workspace,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        approval_callback=approve,
        require_relevant_source=False,
        agent_mode=True,
        **limits,
    )


def test_agent_multi_step_read_only_progress_and_metrics(workspace: Path) -> None:
    model = MockModel(
        (
            call("search-a", "repository.search_files", {"query": "VALUE"}),
            call("read-a", "repository.read_file", {"path": "src/value.py"}),
            call("search-b", "repository.search_files", {"query": "OTHER"}),
            call("read-b", "repository.read_file", {"path": "src/other.py"}),
            final("The two values are separate."),
        )
    )
    response = agent_session(workspace, model).run_agent_task("Trace both values")
    result = response.agent_task
    assert result.stop_reason is AgentStopReason.COMPLETED
    assert result.status == CodingTaskStatus.COMPLETED_READ_ONLY.value
    assert result.iterations == result.model_calls == 5
    assert result.tool_calls == 4
    assert result.unique_files_read == ("src/other.py", "src/value.py")
    assert result.mutation_count == 0


def test_agent_no_progress_stops_after_three_empty_searches(workspace: Path) -> None:
    model = MockModel(
        tuple(
            call(f"search-{index}", "repository.search_files", {"query": query})
            for index, query in enumerate(("absent-one", "absent-two", "absent-three"))
        )
    )
    session = agent_session(workspace, model)
    with pytest.raises(RepositoryOrchestrationError, match="no-progress"):
        session.run_agent_task("Investigate absent symbols")
    assert session.last_agent_task.stop_reason is AgentStopReason.NO_PROGRESS
    assert session.last_agent_task.tool_calls == 3


def test_agent_repeated_call_stops_before_third_execution(workspace: Path) -> None:
    repeated = call("first", "repository.search_files", {"query": "VALUE"})
    model = MockModel(
        (
            repeated,
            call("second", "repository.search_files", {"query": "VALUE"}),
            call("third", "repository.search_files", {"query": "VALUE"}),
        )
    )
    session = agent_session(workspace, model)
    with pytest.raises(RepositoryOrchestrationError, match="repeated"):
        session.run_agent_task("Find VALUE repeatedly")
    assert session.last_agent_task.stop_reason is AgentStopReason.REPEATED_CALL
    assert session.last_agent_task.tool_calls == 2


def test_agent_tool_model_and_iteration_limits_are_independent(workspace: Path) -> None:
    tool_session = agent_session(
        workspace,
        MockModel(
            (
                call("one", "repository.read_file", {"path": "src/value.py"}),
                call("two", "repository.read_file", {"path": "src/other.py"}),
            )
        ),
        max_tool_executions=1,
    )
    with pytest.raises(RepositoryOrchestrationError, match="tool execution"):
        tool_session.run_agent_task("Read files")
    assert tool_session.last_agent_task.stop_reason is AgentStopReason.TOOL_LIMIT

    model_session = agent_session(
        workspace,
        MockModel(
            (
                call("one", "repository.read_file", {"path": "src/value.py"}),
                call("two", "repository.read_file", {"path": "src/other.py"}),
            )
        ),
        max_model_calls=2,
        max_steps=5,
    )
    with pytest.raises(RepositoryOrchestrationError, match="model-call"):
        model_session.run_agent_task("Keep reading")
    assert model_session.last_agent_task.stop_reason is AgentStopReason.MODEL_CALL_LIMIT

    iteration_session = agent_session(
        workspace,
        MockModel(
            (
                call("one", "repository.read_file", {"path": "src/value.py"}),
                call("two", "repository.read_file", {"path": "src/other.py"}),
            )
        ),
        max_model_calls=5,
        max_steps=2,
    )
    with pytest.raises(RepositoryOrchestrationError, match="step limit"):
        iteration_session.run_agent_task("Keep reading")
    assert (
        iteration_session.last_agent_task.stop_reason is AgentStopReason.ITERATION_LIMIT
    )


def test_agent_one_mutation_reread_and_verified_completion(workspace: Path) -> None:
    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments(workspace)),
            call("reread", "repository.read_file", {"path": "src/value.py"}),
            call("test", "project.test", {}),
            final("Change and test complete."),
        )
    )
    response = agent_session(
        workspace,
        model,
        test_code=(
            "from pathlib import Path; "
            "assert 'VALUE = 2' in Path('src/value.py').read_text()"
        ),
    ).run_agent_task("Change and verify VALUE")
    result = response.agent_task
    assert result.stop_reason is AgentStopReason.COMPLETED
    assert result.status == CodingTaskStatus.COMPLETED_VERIFIED.value
    assert result.mutation_count == 1
    assert result.approval_requests == result.approvals_granted == 2
    assert result.approvals_rejected == 0


def test_agent_second_mutation_is_blocked(workspace: Path) -> None:
    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments(workspace)),
            call("second", "repository.apply_patch", patch_arguments(workspace)),
            final("Second change blocked."),
        )
    )
    response = agent_session(workspace, model).run_agent_task("Change twice")
    assert response.agent_task.stop_reason is AgentStopReason.SECOND_MUTATION_BLOCKED
    assert response.agent_task.mutation_count == 1
    assert (workspace / "src/value.py").read_text() == "VALUE = 2\n"


def test_agent_failed_verification_can_read_but_not_repair(workspace: Path) -> None:
    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments(workspace)),
            final("Verification failed; the change remains."),
        )
    )
    response = agent_session(
        workspace, model, test_code="import sys; sys.exit(3)"
    ).run_agent_task("Change, test, and explain failure")
    assert response.agent_task.stop_reason is AgentStopReason.VERIFICATION_FAILED
    assert response.agent_task.test.exit_code == 3
    assert response.agent_task.mutation_count == 1
    assert response.tool_activity[-1].status == "failure"


def test_agent_mutation_rejection_and_cancellation(workspace: Path) -> None:
    rejected_model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments(workspace)),
            final("Rejected."),
        )
    )
    rejected = agent_session(
        workspace, rejected_model, approve=lambda *_args: False
    ).run_agent_task("Change VALUE")
    assert rejected.agent_task.stop_reason is AgentStopReason.USER_REJECTED
    assert rejected.agent_task.approvals_rejected == 1
    assert (workspace / "src/value.py").read_text() == "VALUE = 1\n"

    def cancel(*_args: object) -> bool:
        raise AgentCancelled("cancelled")

    cancelled_model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments(workspace)),
        )
    )
    cancelled_session = agent_session(workspace, cancelled_model, approve=cancel)
    with pytest.raises(AgentCancelled):
        cancelled_session.run_agent_task("Change VALUE")
    assert cancelled_session.last_agent_task.stop_reason is AgentStopReason.CANCELLED
    assert cancelled_session.last_agent_task.approval_requests == 1
    assert (workspace / "src/value.py").read_text() == "VALUE = 1\n"


def test_agent_unknown_tool_and_task_state_reset(workspace: Path) -> None:
    first = agent_session(
        workspace,
        MockModel(
            (
                call("read", "repository.read_file", {"path": "src/value.py"}),
                call("shell", "shell.exec", {"command": "touch bad"}),
                final("Unavailable."),
            )
        ),
    )
    response = first.run_agent_task("Try a forbidden shell")
    assert response.agent_task.stop_reason is AgentStopReason.TOOL_ERROR
    assert not (workspace / "bad").exists()

    second = agent_session(
        workspace,
        MockModel(
            (
                call("read", "repository.read_file", {"path": "src/value.py"}),
                final("Fresh task."),
            )
        ),
    ).run_agent_task("Read safely")
    assert second.agent_task.iterations == 2
    assert second.agent_task.tool_calls == 1
