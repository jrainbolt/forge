from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from forge.models import MockModel
from forge.orchestration import (
    CodingTaskStatus,
    RepositoryChatSession,
    RepositoryOrchestrationError,
)
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    MutationPreview,
    ToolInvocation,
    create_assist_repository_policy,
    create_assist_repository_registry,
)


def call(call_id: str, tool: str, arguments: Mapping[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": call_id, "tool": tool, "arguments": arguments}
    )


def final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "src").mkdir(parents=True)
    (root / "src/value.py").write_bytes(b"VALUE = 1\n")
    return root


def assist_session(
    model: MockModel,
    workspace: Path,
    *,
    approve=None,  # type: ignore[no-untyped-def]
    activity_callback=None,  # type: ignore[no-untyped-def]
) -> RepositoryChatSession:
    return RepositoryChatSession(
        "fixture",
        model,
        workspace,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        approval_callback=approve,
        activity_callback=activity_callback,
        minimum_source_files=1,
        require_relevant_source=False,
    )


def patch_arguments(path: str = "src/value.py") -> dict[str, object]:
    return {
        "path": path,
        "expected_sha256": digest(b"VALUE = 1\n"),
        "edits": [{"old": "VALUE = 1", "new": "VALUE = 2"}],
    }


def test_assist_read_patch_preview_approval_and_evidence_invalidation(
    workspace: Path,
) -> None:
    previews: list[MutationPreview] = []

    def approve(invocation: ToolInvocation, preview: MutationPreview) -> bool:
        assert invocation.tool_name == "repository.apply_patch"
        previews.append(preview)
        assert (workspace / "src/value.py").read_bytes() == b"VALUE = 1\n"
        return True

    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments()),
            final("The file mutation succeeded; code correctness was not tested."),
        )
    )
    response = assist_session(model, workspace, approve=approve).ask("Change VALUE")
    assert (workspace / "src/value.py").read_bytes() == b"VALUE = 2\n"
    assert len(previews) == 1
    assert [activity.status for activity in response.tool_activity] == [
        "success",
        "success",
    ]
    assert response.tool_activity[0].current_source is False
    assert response.tool_activity[1].evidence == "patch_success"
    assert "repository.apply_patch" not in str(model.requests[2].output.schema)


def test_assist_read_then_matching_replace_is_approved(workspace: Path) -> None:
    arguments = {
        "path": "src/value.py",
        "content": "VALUE = 5\n",
        "mode": "replace",
        "expected_sha256": digest(b"VALUE = 1\n"),
    }
    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("replace", "repository.write_file", arguments),
            final("Replacement succeeded; tests were not run."),
        )
    )
    response = assist_session(model, workspace, approve=lambda *_args: True).ask(
        "Replace VALUE"
    )
    assert (workspace / "src/value.py").read_bytes() == b"VALUE = 5\n"
    assert response.tool_activity[-1].evidence == "write_success"


def test_rejection_after_preview_performs_zero_mutation(workspace: Path) -> None:
    previews = 0

    def reject(_invocation: ToolInvocation, _preview: MutationPreview) -> bool:
        nonlocal previews
        previews += 1
        return False

    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments()),
            final("The proposal was not approved and no mutation occurred."),
        )
    )
    response = assist_session(model, workspace, approve=reject).ask("Change VALUE")
    assert previews == 1
    assert (workspace / "src/value.py").read_bytes() == b"VALUE = 1\n"
    assert response.tool_activity[-1].status == "approval_required"
    assert "not executed" in model.requests[2].messages[-1].content


def test_existing_file_mutation_without_read_fails_before_preview(
    workspace: Path,
) -> None:
    callback_calls = 0

    def approve(_invocation: ToolInvocation, _preview: MutationPreview) -> bool:
        nonlocal callback_calls
        callback_calls += 1
        return True

    model = MockModel(
        (
            call("blind", "repository.apply_patch", patch_arguments()),
            call("read", "repository.read_file", {"path": "src/value.py"}),
            final("The blind mutation was rejected."),
        )
    )
    response = assist_session(model, workspace, approve=approve).ask("Change VALUE")
    assert callback_calls == 0
    assert response.tool_activity[0].status == "failure"
    assert "current-turn read" in model.requests[1].messages[-1].content
    assert (workspace / "src/value.py").read_bytes() == b"VALUE = 1\n"


def test_reading_different_file_does_not_authorize_mutation(workspace: Path) -> None:
    (workspace / "src/other.py").write_text("OTHER = 1\n")
    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/other.py"}),
            call("patch", "repository.apply_patch", patch_arguments()),
            final("The mutation was rejected."),
        )
    )
    response = assist_session(model, workspace, approve=lambda *_args: True).ask(
        "Inspect OTHER and change VALUE"
    )
    assert response.tool_activity[-1].status == "failure"
    assert (workspace / "src/value.py").read_bytes() == b"VALUE = 1\n"


def test_external_change_after_preview_causes_precondition_failure(
    workspace: Path,
) -> None:
    def alter_then_approve(
        _invocation: ToolInvocation, _preview: MutationPreview
    ) -> bool:
        (workspace / "src/value.py").write_bytes(b"HUMAN = 3\n")
        return True

    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments()),
            final("The stale mutation failed."),
        )
    )
    response = assist_session(model, workspace, approve=alter_then_approve).ask(
        "Change VALUE"
    )
    assert response.tool_activity[-1].status == "failure"
    assert "precondition" in model.requests[2].messages[-1].content
    assert (workspace / "src/value.py").read_bytes() == b"HUMAN = 3\n"


def test_new_file_requires_inspected_parent_then_can_be_approved(
    workspace: Path,
) -> None:
    arguments = {"path": "src/new.py", "content": "NEW = 1\n", "mode": "create"}
    model = MockModel(
        (
            call("list", "repository.list_directory", {"path": "src"}),
            call("create", "repository.write_file", arguments),
            final("The new file was created; it was not tested."),
        )
    )
    response = assist_session(model, workspace, approve=lambda *_args: True).ask(
        "Create a source file"
    )
    assert (workspace / "src/new.py").read_text() == "NEW = 1\n"
    assert response.tool_activity[-1].evidence == "write_success"


def test_new_file_without_context_is_rejected(workspace: Path) -> None:
    model = MockModel(
        (
            call(
                "create",
                "repository.write_file",
                {"path": "src/new.py", "content": "NEW = 1\n", "mode": "create"},
            ),
            call("read", "repository.read_file", {"path": "src/value.py"}),
            final("Creation was rejected."),
        )
    )
    response = assist_session(model, workspace, approve=lambda *_args: True).ask(
        "Create a source file"
    )
    assert response.tool_activity[0].status == "failure"
    assert not (workspace / "src/new.py").exists()


def test_successful_mutation_remains_real_if_later_generation_fails(
    workspace: Path,
) -> None:
    observed = []
    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments()),
            "not json",
            "still not json",
        )
    )
    session = assist_session(
        model,
        workspace,
        approve=lambda *_args: True,
        activity_callback=observed.append,
    )
    with pytest.raises(RepositoryOrchestrationError):
        session.ask("Change VALUE")
    assert (workspace / "src/value.py").read_bytes() == b"VALUE = 2\n"
    assert observed[-1].evidence == "patch_success"
    assert session.conversation.turns == ()
    assert session.last_coding_task.status is CodingTaskStatus.MUTATED_TASK_FAILED


def test_project_test_rejection_starts_no_process(workspace: Path) -> None:
    marker = workspace / "ran"
    model = MockModel(
        (
            call("test", "project.test", {}),
            final("The test was rejected."),
        )
    )
    commands = ProjectCommands(
        test=ProjectCommand(
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
            ),
            5,
        )
    )
    session = RepositoryChatSession(
        "fixture",
        model,
        workspace,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        require_relevant_source=False,
        approval_callback=lambda *_args: False,
    )
    response = session.ask("Run tests")
    assert response.tool_activity[0].status == "approval_required"
    assert response.tool_activity[0].evidence == "test_result"
    assert not marker.exists()


def test_project_approval_uses_configured_snapshot(workspace: Path) -> None:
    first = workspace / "first"
    second = workspace / "second"
    commands = ProjectCommands(
        test=ProjectCommand(
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(first)!r}).write_text('A')",
            ),
            5,
        )
    )
    previews = []

    def approve(_invocation: object, preview: object) -> bool:
        previews.append(preview)
        commands = ProjectCommands(  # noqa: F841 - proves source rebinding is inert
            test=ProjectCommand(
                (
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(second)!r}).write_text('B')",
                ),
                5,
            )
        )
        return True

    model = MockModel((call("test", "project.test", {}), final("Tests pass.")))
    response = RepositoryChatSession(
        "fixture",
        model,
        workspace,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        require_relevant_source=False,
        approval_callback=approve,
    ).ask("Run tests")
    assert response.tool_activity[0].status == "success"
    assert previews[0].argv == commands.test.argv
    assert first.read_text() == "A"
    assert not second.exists()


def test_mutation_invalidates_verification_generation_then_retest_refreshes(
    workspace: Path,
) -> None:
    test_command = ProjectCommand(
        (
            sys.executable,
            "-c",
            "from pathlib import Path; assert Path('src/value.py').is_file()",
        ),
        5,
    )
    model = MockModel(
        (
            call("test-before", "project.test", {}),
            call("read", "repository.read_file", {"path": "src/value.py"}),
            call("patch", "repository.apply_patch", patch_arguments()),
            call("test-after", "project.test", {}),
            final("The current tests pass."),
        )
    )
    response = RepositoryChatSession(
        "fixture",
        model,
        workspace,
        registry=create_assist_repository_registry(ProjectCommands(test=test_command)),
        policy=create_assist_repository_policy(),
        require_relevant_source=False,
        approval_callback=lambda *_args: True,
    ).ask("Change VALUE and test")
    before, _read, patch, after = response.tool_activity
    assert before.status == "success"
    assert before.current_verification is False
    assert before.generation == 0
    assert patch.generation == 1
    assert after.status == "success"
    assert after.generation == 1
    assert after.current_verification is True
