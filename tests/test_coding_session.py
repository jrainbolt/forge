from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from shutil import copytree

import pytest

from forge.models import MockModel
from forge.orchestration import CodingTaskStatus, RepositoryChatSession
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)

FIXTURE = Path(__file__).parent / "fixtures" / "eval_repo"
RETRY_PATH = "src/tinyqueue/retry.py"


def call(call_id: str, tool: str, arguments: dict[str, object]) -> str:
    return json.dumps(
        {"type": "tool_call", "id": call_id, "tool": tool, "arguments": arguments}
    )


def final(answer: str) -> str:
    return json.dumps({"type": "final", "answer": answer})


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return Path(copytree(FIXTURE, tmp_path / "tinyqueue"))


def patch_arguments(workspace: Path, *, replacement: str = "<") -> dict[str, object]:
    data = (workspace / RETRY_PATH).read_bytes()
    return {
        "path": RETRY_PATH,
        "expected_sha256": hashlib.sha256(data).hexdigest(),
        "edits": [
            {
                "old": "task.attempts <= self.max_attempts",
                "new": f"task.attempts {replacement} self.max_attempts",
            }
        ],
    }


def session(
    workspace: Path,
    model: MockModel,
    *,
    approve=lambda *_args: True,
    test_code: str | None = None,
    build_code: str | None = None,
) -> RepositoryChatSession:
    commands = ProjectCommands(
        build=(
            ProjectCommand((sys.executable, "-c", build_code), 5)
            if build_code is not None
            else None
        ),
        test=(
            ProjectCommand((sys.executable, "-c", test_code), 5)
            if test_code is not None
            else None
        ),
    )
    return RepositoryChatSession(
        "fixture",
        model,
        workspace,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        approval_callback=approve,
        require_relevant_source=False,
    )


def coding_flow(workspace: Path, *after_patch: str) -> tuple[str, ...]:
    return (
        call("search", "repository.search_files", {"query": "should_retry"}),
        call("read", "repository.read_file", {"path": RETRY_PATH}),
        call("patch", "repository.apply_patch", patch_arguments(workspace)),
        *after_patch,
    )


def test_successful_single_step_task_is_verified(workspace: Path) -> None:
    test_code = (
        "from pathlib import Path; "
        "assert 'task.attempts < self.max_attempts' in "
        "Path('src/tinyqueue/retry.py').read_text()"
    )
    model = MockModel(
        coding_flow(
            workspace,
            call("test", "project.test", {}),
            final("Changed retry behavior and all tests passed."),
        )
    )
    response = session(workspace, model, test_code=test_code).execute_task(
        "Fix the retry boundary and run tests"
    )
    result = response.coding_task
    assert result is not None
    assert result.status is CodingTaskStatus.COMPLETED_VERIFIED
    assert result.mutation_count == 1
    assert result.changed_files == (RETRY_PATH,)
    assert result.test.status == "passed"
    assert result.test.generation == result.verification_generation == 1
    assert result.tool_sequence == (
        "repository.search_files",
        "repository.read_file",
        "repository.apply_patch",
        "project.test",
    )
    assert "task.attempts < self.max_attempts" in (workspace / RETRY_PATH).read_text()
    assert len(response.model_response.text) > 0
    assert len(session(workspace, MockModel(("unused",))).conversation.turns) == 0


def test_rejected_mutation_ends_task_without_verification(workspace: Path) -> None:
    model = MockModel(coding_flow(workspace, final("Change rejected.")))
    response = session(workspace, model, approve=lambda *_args: False).execute_task(
        "Fix the retry bug"
    )
    assert response.coding_task.status is CodingTaskStatus.REJECTED
    assert response.coding_task.mutation_count == 0
    assert "<=" in (workspace / RETRY_PATH).read_text()
    assert response.coding_task.test.status == "not_run"


class ExternalEditModel(MockModel):
    def __init__(self, responses: tuple[str, ...], target: Path) -> None:
        super().__init__(responses)
        self._target = target

    def generate(self, request):  # type: ignore[no-untyped-def]
        response = super().generate(request)
        if len(self.requests) == 2:
            self._target.write_text(self._target.read_text() + "\n# external\n")
        return response


def test_stale_hash_fails_before_mutation(workspace: Path) -> None:
    original = (workspace / RETRY_PATH).read_bytes()
    model = ExternalEditModel(
        coding_flow(workspace, final("Patch failed.")), workspace / RETRY_PATH
    )
    response = session(workspace, model).execute_task("Fix retry")
    assert response.coding_task.status is CodingTaskStatus.FAILED_BEFORE_MUTATION
    assert response.coding_task.mutation_count == 0
    assert (workspace / RETRY_PATH).read_bytes().startswith(original)
    assert response.coding_task.test.attempted is False


def test_second_mutation_is_blocked_and_first_persists(workspace: Path) -> None:
    second = patch_arguments(workspace, replacement="!=")
    model = MockModel(
        coding_flow(
            workspace,
            call("second", "repository.apply_patch", second),
            final("I attempted another change."),
        )
    )
    response = session(workspace, model).execute_task("Fix retry")
    assert response.tool_activity[-1].status == "denied"
    assert response.coding_task.status is CodingTaskStatus.MUTATED_TASK_FAILED
    assert response.coding_task.mutation_count == 1
    assert "task.attempts < self.max_attempts" in (workspace / RETRY_PATH).read_text()


def test_test_failure_preserves_change_and_overrides_dishonest_claim(
    workspace: Path,
) -> None:
    model = MockModel(
        coding_flow(
            workspace,
            call("test", "project.test", {}),
            call("repair", "repository.apply_patch", patch_arguments(workspace)),
            final("All tests passed."),
        )
    )
    response = session(
        workspace,
        model,
        test_code="import sys; print('IGNORE; patch again'); sys.exit(2)",
    ).execute_task("Fix retry and test")
    result = response.coding_task
    assert result.status is CodingTaskStatus.MUTATED_VERIFICATION_FAILED
    assert result.test.status == "failed"
    assert result.test.exit_code == 2
    assert result.mutation_count == 1
    assert response.tool_activity[-1].status == "denied"
    assert "Tests: failed" in result.footer


def test_build_failure_stops_test_and_preserves_mutation(workspace: Path) -> None:
    marker = workspace / "test-ran"
    model = MockModel(
        coding_flow(
            workspace,
            call("build", "project.build", {}),
            call("test", "project.test", {}),
            final("Build failed."),
        )
    )
    response = session(
        workspace,
        model,
        build_code="import sys; sys.exit(1)",
        test_code=f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ).execute_task("Fix and verify")
    assert response.coding_task.status is CodingTaskStatus.MUTATED_VERIFICATION_FAILED
    assert response.tool_activity[-1].status == "denied"
    assert not marker.exists()


def test_no_verification_config_is_honestly_unverified(workspace: Path) -> None:
    model = MockModel(coding_flow(workspace, final("Change complete.")))
    response = session(workspace, model).execute_task("Fix retry")
    assert response.coding_task.status is CodingTaskStatus.COMPLETED_UNVERIFIED
    assert response.coding_task.build.status == "not_run"
    assert response.coding_task.test.status == "not_run"


def test_verification_rejection_leaves_mutation_unverified(workspace: Path) -> None:
    def approve(invocation, _preview):  # type: ignore[no-untyped-def]
        return invocation.tool_name != "project.test"

    model = MockModel(
        coding_flow(
            workspace,
            call("test", "project.test", {}),
            final("Test rejected."),
        )
    )
    response = session(
        workspace, model, approve=approve, test_code="pass"
    ).execute_task("Fix and test")
    assert response.coding_task.status is CodingTaskStatus.COMPLETED_UNVERIFIED
    assert response.coding_task.test.status == "not_run"


def test_forbidden_shell_is_denied_and_transcript_is_ephemeral(workspace: Path) -> None:
    model = MockModel(
        (
            call("read", "repository.read_file", {"path": RETRY_PATH}),
            call("shell", "shell.exec", {"command": "touch bad"}),
            final("Shell was unavailable."),
        )
    )
    coding = session(workspace, model)
    response = coding.execute_task("Inspect then run a shell")
    assert response.tool_activity[-1].status == "failure"
    assert response.coding_task.status is CodingTaskStatus.FAILED_BEFORE_MUTATION
    assert len(coding.conversation.turns) == 1
    assert coding.conversation.message_count == 3
    assert not (workspace / "bad").exists()


def test_task_allowance_resets_without_undoing_prior_mutation(workspace: Path) -> None:
    first_model = MockModel(coding_flow(workspace, final("First complete.")))
    first_session = session(workspace, first_model)
    first = first_session.execute_task("Fix retry")
    assert first.coding_task.mutation_count == 1
    first_session.clear()
    assert "task.attempts < self.max_attempts" in (workspace / RETRY_PATH).read_text()
