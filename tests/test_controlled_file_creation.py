"""controlled-file-creation-v1: exact candidate authority and rollback contracts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from forge.interaction import AutonomyMode, resolve_interaction_policy
from forge.models import MockModel
from forge.orchestration import (
    MutationCandidate,
    RepositoryChatSession,
    StructuredEditProposal,
    validate_structured_edit,
)
from forge.orchestration.repository_session import RepositoryOrchestrationError
from forge.project_config import ProjectCommand, ProjectCommands
from forge.repository_index import RepositoryIndex
from forge.tools import (
    ExecutionContext,
    InvocationApproval,
    ToolExecutor,
    ToolInvocation,
    ToolResultStatus,
    create_assist_repository_policy,
    create_assist_repository_registry,
)
from forge.tools.controlled_creation import (
    CREATE_MODE,
    MAX_CREATE_BYTES,
    CreateTextFilesTool,
    authorize_create_candidate,
    create_group_id,
    preview_create_text_files,
)
from forge.tools.registry import ToolRegistry
from forge.tools.repository import ReadFileTool
from forge.tools.tool import ToolError


def _context(root: Path, *paths: str) -> ExecutionContext:
    candidates = tuple(
        authorize_create_candidate(root, path, 0, "explicit_task_path")
        for path in sorted(paths)
    )
    return ExecutionContext(root, create_candidates=candidates)


def _arguments(context: ExecutionContext, files: dict[str, str]) -> dict[str, object]:
    creates = tuple({"path": path, "content": files[path]} for path in sorted(files))
    selected = tuple(
        candidate for candidate in context.create_candidates if candidate.path in files
    )
    try:
        group_id = create_group_id(creates, selected, 0)
    except ToolError:
        group_id = "0" * 64
    return {
        "group_id": group_id,
        "workspace_generation": 0,
        "creates": creates,
    }


def _executor(tool: CreateTextFilesTool | None = None) -> ToolExecutor:
    return ToolExecutor(
        ToolRegistry((tool or CreateTextFilesTool(),)),
        create_assist_repository_policy(),
    )


def _run(
    context: ExecutionContext,
    arguments: dict[str, object],
    tool: CreateTextFilesTool | None = None,
):
    invocation = ToolInvocation("create", "repository.create_text_files", arguments)
    executor = _executor(tool)
    assert (
        executor.execute(invocation, context).status
        is ToolResultStatus.APPROVAL_REQUIRED
    )
    return executor.execute(
        invocation, context, approval=InvocationApproval.for_invocation(invocation)
    )


def test_c01_single_creation_preview_and_exact_bytes(tmp_path: Path) -> None:
    context = _context(tmp_path, "hello.py")
    arguments = _arguments(context, {"hello.py": "VALUE = 'α'\n"})
    preview = preview_create_text_files(arguments, context)
    assert preview.files[0].operation == "create"
    assert "CREATE hello.py" in preview.diff and "/dev/null" in preview.diff
    result = _run(context, arguments)
    assert result.status is ToolResultStatus.SUCCESS
    assert (tmp_path / "hello.py").read_bytes() == "VALUE = 'α'\n".encode()
    assert os.stat(tmp_path / "hello.py").st_mode & 0o777 == CREATE_MODE
    assert result.output["created_file_count"] == 1


def test_c02_two_file_creation_one_group(tmp_path: Path) -> None:
    context = _context(tmp_path, "a.py", "b.py")
    result = _run(context, _arguments(context, {"a.py": "a = 1\n", "b.py": "b = 2\n"}))
    assert result.status is ToolResultStatus.SUCCESS
    assert result.output["created_file_count"] == 2
    assert (tmp_path / "a.py").read_text() == "a = 1\n"
    assert (tmp_path / "b.py").read_text() == "b = 2\n"


def test_c03_mixed_edit_create_is_explicitly_not_in_v1(tmp_path: Path) -> None:
    (tmp_path / "old.py").write_text("VALUE = 1\n")
    context = _context(tmp_path, "new.py")
    with pytest.raises(ToolError, match="authorized path set"):
        preview_create_text_files(
            _arguments(context, {"old.py": "VALUE = 2\n", "new.py": "x = 1\n"}),
            context,
        )
    assert (tmp_path / "old.py").read_text() == "VALUE = 1\n"


def test_c04_surprise_create_rejects_complete_group(tmp_path: Path) -> None:
    context = _context(tmp_path, "a.py")
    result = _run(context, _arguments(context, {"a.py": "a", "surprise.py": "s"}))
    assert result.status is ToolResultStatus.FAILURE
    assert not (tmp_path / "a.py").exists()


def test_c05_existing_target_even_empty_is_not_create(tmp_path: Path) -> None:
    (tmp_path / "empty.py").touch()
    with pytest.raises(ToolError, match="exists"):
        _context(tmp_path, "empty.py")


def test_c06_missing_parent_never_created(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="parent"):
        _context(tmp_path, "missing/new.py")
    assert not (tmp_path / "missing").exists()


def test_c07_symlink_parent_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ToolError, match="real directory"):
        _context(tmp_path, "link/new.py")


def test_parent_identity_change_rejects_before_write(tmp_path: Path) -> None:
    parent = tmp_path / "pkg"
    parent.mkdir()
    context = _context(tmp_path, "pkg/new.py")
    arguments = _arguments(context, {"pkg/new.py": "value = 1\n"})
    parent.rename(tmp_path / "pkg-old")
    parent.mkdir()
    result = _run(context, arguments)
    assert result.status is ToolResultStatus.FAILURE
    assert not (parent / "new.py").exists()
    assert not (tmp_path / "pkg-old" / "new.py").exists()


@pytest.mark.parametrize(
    "content", ["x" * (MAX_CREATE_BYTES + 1), "a\x00b", "a\x01b", "\ud800"]
)
def test_c08_content_policy_rejects_invalid_text(tmp_path: Path, content: str) -> None:
    context = _context(tmp_path, "new.txt")
    result = _run(context, _arguments(context, {"new.txt": content}))
    assert result.status is ToolResultStatus.FAILURE
    assert not (tmp_path / "new.txt").exists()


def test_c09_approval_is_exact_invocation(tmp_path: Path) -> None:
    context = _context(tmp_path, "new.py")
    original = ToolInvocation(
        "create", "repository.create_text_files", _arguments(context, {"new.py": "A"})
    )
    changed = ToolInvocation(
        "create", "repository.create_text_files", _arguments(context, {"new.py": "B"})
    )
    result = _executor().execute(
        changed, context, approval=InvocationApproval.for_invocation(original)
    )
    assert result.status is ToolResultStatus.APPROVAL_REQUIRED
    assert not (tmp_path / "new.py").exists()


def test_c10_target_appears_after_preview_zero_writes(tmp_path: Path) -> None:
    (tmp_path / "old.py").write_text("unchanged")
    context = _context(tmp_path, "a.py", "b.py")
    arguments = _arguments(context, {"a.py": "a", "b.py": "b"})
    preview_create_text_files(arguments, context)
    (tmp_path / "b.py").write_text("other actor")
    result = _run(context, arguments)
    assert result.status is ToolResultStatus.FAILURE
    assert not (tmp_path / "a.py").exists()
    assert (tmp_path / "old.py").read_text() == "unchanged"
    assert (tmp_path / "b.py").read_text() == "other actor"


def test_c11_later_failure_rolls_back_created_path(tmp_path: Path) -> None:
    context = _context(tmp_path, "a.py", "b.py")
    normal = CreateTextFilesTool()._create_operation

    def fail_second(path: Path, data: bytes) -> tuple[int, int]:
        if path.name == "b.py":
            raise OSError("injected failure")
        return normal(path, data)

    result = _run(
        context,
        _arguments(context, {"a.py": "a", "b.py": "b"}),
        CreateTextFilesTool(create_operation=fail_second),
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["mutation_group_rollback_result"] == "restored"
    assert not (tmp_path / "a.py").exists() and not (tmp_path / "b.py").exists()


def test_c12_rollback_failure_is_integrity_failure(tmp_path: Path) -> None:
    context = _context(tmp_path, "a.py", "b.py")
    normal = CreateTextFilesTool()._create_operation

    def fail_second(path: Path, data: bytes) -> tuple[int, int]:
        if path.name == "b.py":
            raise OSError("injected failure")
        return normal(path, data)

    def fail_remove(path: Path) -> None:
        raise OSError("injected rollback failure")

    result = _run(
        context,
        _arguments(context, {"a.py": "a", "b.py": "b"}),
        CreateTextFilesTool(create_operation=fail_second, remove_operation=fail_remove),
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["workspace_integrity_failure"] is True


def test_c12_session_integrity_failure_is_terminal_before_verification(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("VALUE = 1\n")
    normal = CreateTextFilesTool()._create_operation

    def fail_second(path: Path, data: bytes) -> tuple[int, int]:
        if path.name == "b.py":
            raise OSError("injected later-child failure")
        return normal(path, data)

    def fail_remove(_path: Path) -> None:
        raise OSError("injected rollback failure")

    base = create_assist_repository_registry(include_creation=True)
    registry = ToolRegistry(
        CreateTextFilesTool(create_operation=fail_second, remove_operation=fail_remove)
        if metadata.name == "repository.create_text_files"
        else base.get(metadata.name)
        for metadata in base.metadata
    )

    def call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
        return json.dumps(
            {
                "type": "tool_call",
                "id": identifier,
                "tool": tool,
                "arguments": arguments,
            }
        )

    model = MockModel(
        (
            call("read", "repository.read_file", {"path": "main.py"}),
            json.dumps(
                {
                    "type": "multi_file_create",
                    "creates": [
                        {"path": "a.py", "content": "A = 1\n"},
                        {"path": "b.py", "content": "B = 2\n"},
                    ],
                }
            ),
        )
    )
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=registry,
        policy=create_assist_repository_policy(),
        create_candidate_paths=("a.py", "b.py"),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    )
    with pytest.raises(RepositoryOrchestrationError, match="fatal grouped"):
        session.execute_task("Create a.py and b.py")
    assert session.last_coding_task is not None
    assert session.last_coding_task.status.value == "workspace_integrity_failed"
    assert session.last_coding_task.test.status == "not_run"


def test_c13_session_creation_increments_generation_once(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("from helper import value\n")
    index = RepositoryIndex(tmp_path, cache_root=tmp_path.parent / "index-cache")
    index.build()

    def call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
        return json.dumps(
            {
                "type": "tool_call",
                "id": identifier,
                "tool": tool,
                "arguments": arguments,
            }
        )

    model = MockModel(
        (
            call("search", "repository.search_files", {"query": "helper"}),
            call("read", "repository.read_file", {"path": "main.py"}),
            json.dumps(
                {
                    "type": "create_file",
                    "path": "helper.py",
                    "content": "def value():\n    return 2\n",
                }
            ),
            json.dumps({"type": "final", "answer": "Created helper.py"}),
        )
    )
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(include_creation=True),
        policy=create_assist_repository_policy(),
        create_candidate_paths=("helper.py",),
        repository_index=index,
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    )
    response = session.execute_task("Create helper.py so main.py can import value")
    assert (tmp_path / "helper.py").exists(), [
        (activity.tool_name, activity.status) for activity in response.tool_activity
    ]
    assert (tmp_path / "helper.py").read_text() == "def value():\n    return 2\n"
    assert session._mutation_generation == 1
    assert session._create_paths == ()
    assert session._context.create_candidates == ()
    assert any(row["name"] == "value" for row in index.file_symbols("helper.py"))
    assert response.coding_task is not None and response.coding_task.mutation_count == 1


def test_c14_created_file_is_fresh_source_only_after_creation(tmp_path: Path) -> None:
    context = _context(tmp_path, "new.py")
    _run(context, _arguments(context, {"new.py": "value = 1\n"}))
    observed = ReadFileTool().execute({"path": "new.py"}, context)
    candidate = MutationCandidate("new.py", str(observed["sha256"]), 1, "fresh-read")
    edit = validate_structured_edit(
        StructuredEditProposal("new.py", "value = 1", "value = 2"),
        (candidate,),
        tmp_path,
        1,
    )
    assert edit.valid
    with pytest.raises(ToolError, match="exists"):
        authorize_create_candidate(tmp_path, "new.py", 1, "repair")


def test_c14_repair_reacquires_primary_created_file_and_cannot_create_again(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("from helper import value\n")

    def call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
        return json.dumps(
            {
                "type": "tool_call",
                "id": identifier,
                "tool": tool,
                "arguments": arguments,
            }
        )

    model = MockModel(
        (
            call("read-main", "repository.read_file", {"path": "main.py"}),
            json.dumps(
                {
                    "type": "create_file",
                    "path": "helper.py",
                    "content": "def value():\n    return 1\n",
                }
            ),
            call("test-1", "project.test", {}),
            call("read-helper", "repository.read_file", {"path": "helper.py"}),
            json.dumps(
                {
                    "type": "structured_edit",
                    "path": "helper.py",
                    "old_text": "return 1",
                    "new_text": "return 2",
                }
            ),
            call("test-2", "project.test", {}),
            json.dumps({"type": "final", "answer": "Created and repaired helper.py"}),
        )
    )
    commands = ProjectCommands(
        test=ProjectCommand(
            (sys.executable, "-c", "from helper import value; assert value() == 2"),
            5,
        )
    )
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(commands, include_creation=True),
        policy=create_assist_repository_policy(),
        create_candidate_paths=("helper.py",),
        agent_mode=True,
        repair_enabled=True,
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    )
    response = session.run_agent_task("Create helper.py so main.py imports value 2")
    assert (tmp_path / "helper.py").read_text() == "def value():\n    return 2\n"
    assert session._mutation_generation == 2
    assert response.coding_task is not None
    assert response.coding_task.mutation_count == 2
    assert session._context.create_candidates == ()


def test_repair_cannot_use_legacy_create_for_surprise_path(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("from helper import value\n")

    def call(identifier: str, tool: str, arguments: dict[str, object]) -> str:
        return json.dumps(
            {
                "type": "tool_call",
                "id": identifier,
                "tool": tool,
                "arguments": arguments,
            }
        )

    model = MockModel(
        (
            call("read-main", "repository.read_file", {"path": "main.py"}),
            json.dumps(
                {"type": "create_file", "path": "helper.py", "content": "VALUE = 1\n"}
            ),
            call("test-1", "project.test", {}),
            call(
                "surprise",
                "repository.write_file",
                {"path": "surprise.py", "content": "VALUE = 2\n", "mode": "create"},
            ),
        )
    )
    commands = ProjectCommands(
        test=ProjectCommand((sys.executable, "-c", "assert False"), 5)
    )
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(commands, include_creation=True),
        policy=create_assist_repository_policy(),
        create_candidate_paths=("helper.py",),
        agent_mode=True,
        repair_enabled=True,
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    )
    with pytest.raises(RepositoryOrchestrationError, match="CREATE_READY"):
        session.run_agent_task("Create helper.py")
    assert (tmp_path / "helper.py").exists()
    assert not (tmp_path / "surprise.py").exists()


@pytest.mark.parametrize(
    "path",
    (".git/config", ".forge-exec/test.py", "eval-results/x.py", "build/x.py"),
)
def test_c15_protected_roots_rejected(tmp_path: Path, path: str) -> None:
    (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ToolError, match="eligible"):
        _context(tmp_path, path)


def test_c16_ephemeral_candidate_is_not_create_authority(tmp_path: Path) -> None:
    assert ExecutionContext(tmp_path).create_candidates == ()
    context = ExecutionContext(tmp_path)
    invocation = ToolInvocation(
        "candidate",
        "repository.create_text_files",
        _arguments(context, {"test_generated.py": "assert True\n"}),
    )
    result = _executor().execute(
        invocation, context, approval=InvocationApproval.for_invocation(invocation)
    )
    assert result.status is ToolResultStatus.FAILURE
    assert not (tmp_path / "test_generated.py").exists()


def test_coding_task_cannot_use_legacy_parent_observation_to_create(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("VALUE = 1\n")
    model = MockModel(
        (
            json.dumps(
                {
                    "type": "tool_call",
                    "id": "read",
                    "tool": "repository.read_file",
                    "arguments": {"path": "main.py"},
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "id": "create",
                    "tool": "repository.write_file",
                    "arguments": {
                        "path": "surprise.py",
                        "content": "VALUE = 2\n",
                        "mode": "create",
                    },
                }
            ),
        )
    )
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(include_creation=True),
        policy=create_assist_repository_policy(),
        create_candidate_paths=("authorized.py",),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    )
    with pytest.raises(RepositoryOrchestrationError, match="CREATE_READY"):
        session.execute_task("Create surprise.py")
    assert not (tmp_path / "surprise.py").exists()


def test_empty_utf8_and_unicode_name_are_valid(tmp_path: Path) -> None:
    context = _context(tmp_path, "café.py")
    result = _run(context, _arguments(context, {"café.py": ""}))
    assert result.status is ToolResultStatus.SUCCESS
    assert (tmp_path / "café.py").read_bytes() == b""


def test_case_collision_obeys_actual_filesystem(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("original")
    case_aliases = os.path.lexists(tmp_path / "FOO.py")
    if case_aliases:
        with pytest.raises(ToolError, match="exists"):
            _context(tmp_path, "FOO.py")
    else:
        context = _context(tmp_path, "FOO.py")
        assert (
            _run(context, _arguments(context, {"FOO.py": "new"})).status
            is ToolResultStatus.SUCCESS
        )
        assert (tmp_path / "foo.py").read_text() == "original"


@pytest.mark.parametrize(
    "profile,expected",
    (
        ("safe", ToolResultStatus.DENIED),
        ("confirm", ToolResultStatus.APPROVAL_REQUIRED),
        ("trusted-exec", ToolResultStatus.APPROVAL_REQUIRED),
    ),
)
def test_creation_permission_profiles(
    tmp_path: Path, profile: str, expected: ToolResultStatus
) -> None:
    context = _context(tmp_path, "new.py")
    invocation = ToolInvocation(
        "create", "repository.create_text_files", _arguments(context, {"new.py": "x"})
    )
    policy = resolve_interaction_policy(AutonomyMode.ASSIST, profile)
    result = ToolExecutor(ToolRegistry((CreateTextFilesTool(),)), policy).execute(
        invocation, context
    )
    assert result.status is expected
    assert not (tmp_path / "new.py").exists()
