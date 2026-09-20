"""mixed-file-transaction-v1: bounded typed edit/create invariants."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from forge.interaction import AutonomyMode, resolve_interaction_policy
from forge.models import MockModel, MutationRepresentationPolicy
from forge.orchestration.protocol import (
    ToolCallOutcome,
    build_mutation_ready_output,
    parse_model_output,
)
from forge.orchestration.repository_session import (
    RepositoryChatSession,
    RepositoryOrchestrationError,
)
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    ExecutionContext,
    InvocationApproval,
    PermissionDecision,
    ToolExecutor,
    ToolInvocation,
    ToolResultStatus,
    create_assist_repository_policy,
    create_assist_repository_registry,
    create_repository_registry,
)
from forge.tools.controlled_creation import authorize_create_candidate
from forge.tools.mixed_transaction import (
    MixedFileTransactionTool,
    mixed_group_id,
    preview_mixed_file_transaction,
)
from forge.tools.registry import ToolRegistry
from forge.tools.tool import ToolError


def _fixture(
    root: Path,
    *,
    edits: tuple[str, ...] = ("a.py",),
    creates: tuple[str, ...] = ("b.py",),
) -> tuple[ExecutionContext, dict[str, object]]:
    for path in edits:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("VALUE = 1\n")
    for path in creates:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
    candidates = tuple(
        authorize_create_candidate(root, path, 0, "explicit_task_path")
        for path in creates
    )
    context = ExecutionContext(
        root, create_candidates=candidates, edit_candidates=edits
    )
    operations: list[dict[str, object]] = []
    for path in edits:
        import hashlib

        operations.append(
            {
                "type": "edit",
                "path": path,
                "expected_sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
                "edits": [{"old": "VALUE = 1\n", "new": "VALUE = 2\n"}],
            }
        )
    for path in creates:
        operations.append({"type": "create", "path": path, "content": "NEW = 1\n"})
    operations.sort(key=lambda item: str(item["path"]))
    arguments = {
        "workspace_generation": 0,
        "operations": tuple(operations),
        "group_id": mixed_group_id(tuple(operations), candidates, 0, context.workspace),
    }
    return context, arguments


def _run(
    context: ExecutionContext,
    arguments: dict[str, object],
    tool: MixedFileTransactionTool | None = None,
):
    invocation = ToolInvocation("mixed", "repository.apply_file_operations", arguments)
    executor = ToolExecutor(
        ToolRegistry((tool or MixedFileTransactionTool(),)),
        create_assist_repository_policy(),
    )
    assert (
        executor.execute(invocation, context).status
        is ToolResultStatus.APPROVAL_REQUIRED
    )
    return executor.execute(
        invocation, context, approval=InvocationApproval.for_invocation(invocation)
    )


def _rebind(
    context: ExecutionContext,
    arguments: dict[str, object],
    operations: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "workspace_generation": 0,
        "operations": tuple(operations),
        "group_id": mixed_group_id(
            tuple(operations), context.create_candidates, 0, context.workspace
        ),
    }


def test_m01_edit_create_success(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path)
    preview = preview_mixed_file_transaction(arguments, context)
    assert preview.paths == ("a.py", "b.py")
    assert "MODIFY a.py" in preview.diff and "CREATE b.py" in preview.diff
    result = _run(context, arguments)
    assert result.status is ToolResultStatus.SUCCESS
    assert (
        result.output["modified_file_count"] == result.output["created_file_count"] == 1
    )
    assert (tmp_path / "a.py").read_text() == "VALUE = 2\n"
    assert (tmp_path / "b.py").read_text() == "NEW = 1\n"


def test_m02_two_edits_two_creates(tmp_path: Path) -> None:
    context, arguments = _fixture(
        tmp_path, edits=("a.py", "c.py"), creates=("b.py", "d.py")
    )
    result = _run(context, arguments)
    assert result.status is ToolResultStatus.SUCCESS
    assert result.output["modified_file_count"] == 2
    assert result.output["created_file_count"] == 2


def test_m03_canonical_order_independent_of_model_order(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path)
    reversed_ops = list(reversed(arguments["operations"]))
    with pytest.raises(ToolError, match="canonical"):
        preview_mixed_file_transaction(
            _rebind(context, arguments, reversed_ops), context
        )


def test_m04_unauthorized_create_rejects_all(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path)
    operations = [dict(item) for item in arguments["operations"]]
    operations[1]["path"] = "surprise.py"
    with pytest.raises(ToolError, match="candidate authority"):
        _rebind(context, arguments, operations)
    assert (tmp_path / "a.py").read_text() == "VALUE = 1\n"


def test_m05_unauthorized_edit_rejects_all(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path)
    (tmp_path / "z.py").write_text("VALUE = 1\n")
    operations = [dict(item) for item in arguments["operations"]]
    operations[0]["path"] = "z.py"
    invalid = dict(arguments)
    invalid["operations"] = tuple(operations)
    result = _run(context, invalid)
    assert result.status is ToolResultStatus.FAILURE
    assert not (tmp_path / "b.py").exists()


def test_m06_duplicate_path_rejects(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path)
    operations = [dict(item) for item in arguments["operations"]]
    operations[1]["path"] = "a.py"
    invalid = dict(arguments)
    invalid["operations"] = tuple(operations)
    result = _run(context, invalid)
    assert result.status is ToolResultStatus.FAILURE
    assert (tmp_path / "a.py").read_text() == "VALUE = 1\n"


def test_m07_create_target_appears_before_apply(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path)
    preview_mixed_file_transaction(arguments, context)
    (tmp_path / "b.py").write_text("OTHER\n")
    result = _run(context, arguments)
    assert result.status is ToolResultStatus.FAILURE
    assert (tmp_path / "a.py").read_text() == "VALUE = 1\n"
    assert (tmp_path / "b.py").read_text() == "OTHER\n"


def test_m08_edit_changes_before_apply(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path)
    preview_mixed_file_transaction(arguments, context)
    (tmp_path / "a.py").write_text("VALUE = 3\n")
    result = _run(context, arguments)
    assert result.status is ToolResultStatus.FAILURE
    assert not (tmp_path / "b.py").exists()


def test_m09_missing_create_parent_rejects(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path, creates=("sub/b.py",))
    (tmp_path / "sub").rmdir()
    result = _run(context, arguments)
    assert result.status is ToolResultStatus.FAILURE
    assert (tmp_path / "a.py").read_text() == "VALUE = 1\n"


def test_m10_protected_create_path_rejects(tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        _fixture(tmp_path, creates=(".forge-exec/b.py",))


def test_m11_later_edit_failure_reverses_create_and_edit(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path, edits=("a.py", "c.py"), creates=("b.py",))

    def fail_last(src: Path, dst: Path) -> None:
        if dst.name == "c.py":
            raise OSError("injected")
        os.replace(src, dst)

    result = _run(
        context, arguments, MixedFileTransactionTool(replace_operation=fail_last)
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["mutation_group_rollback_result"] == "restored"
    assert (tmp_path / "a.py").read_text() == "VALUE = 1\n"
    assert (tmp_path / "c.py").read_text() == "VALUE = 1\n"
    assert not (tmp_path / "b.py").exists()


def test_m12_later_create_failure_restores_edits(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path, edits=("a.py", "b.py"), creates=("z.py",))

    def fail_create(_path: Path, _data: bytes) -> tuple[int, int]:
        raise OSError("injected")

    result = _run(
        context, arguments, MixedFileTransactionTool(create_operation=fail_create)
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["mutation_group_rollback_result"] == "restored"
    assert all(
        (tmp_path / path).read_text() == "VALUE = 1\n" for path in ("a.py", "b.py")
    )


def test_m13_created_identity_mismatch_is_integrity_failure(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path, edits=("a.py", "c.py"), creates=("b.py",))

    def replace_unknown(src: Path, dst: Path) -> None:
        if dst.name == "c.py":
            (tmp_path / "b.py").unlink()
            (tmp_path / "b.py").write_text("UNKNOWN\n")
            raise OSError("injected")
        os.replace(src, dst)

    result = _run(
        context, arguments, MixedFileTransactionTool(replace_operation=replace_unknown)
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["workspace_integrity_failure"] is True
    assert (tmp_path / "b.py").read_text() == "UNKNOWN\n"


def test_m14_edit_restore_failure_is_integrity_failure(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path, edits=("a.py", "c.py"), creates=("b.py",))

    def fail_last(src: Path, dst: Path) -> None:
        if dst.name == "c.py":
            raise OSError("injected")
        os.replace(src, dst)

    def fail_restore(_src: Path, _dst: Path) -> None:
        raise OSError("injected restore")

    result = _run(
        context,
        arguments,
        MixedFileTransactionTool(
            replace_operation=fail_last, rollback_operation=fail_restore
        ),
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["workspace_integrity_failure"] is True


def test_m15_approval_identity_binds_create_bytes(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path)
    original = ToolInvocation("mixed", "repository.apply_file_operations", arguments)
    operations = [dict(item) for item in arguments["operations"]]
    operations[1]["content"] = "CHANGED\n"
    changed = ToolInvocation(
        "mixed",
        "repository.apply_file_operations",
        _rebind(context, arguments, operations),
    )
    assert not InvocationApproval.for_invocation(original).matches(changed)


def test_m16_repair_schema_excludes_create(tmp_path: Path) -> None:
    schema = build_mutation_ready_output(
        ("a.py", "b.py"),
        representation=MutationRepresentationPolicy.LINE_RANGE,
        allow_single_file_subset=True,
    ).schema
    assert "multi_file_change" not in str(schema)
    assert "create_file" not in str(schema)


def test_m17_legacy_creation_rejected_in_production(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("VALUE = 1\n")
    model = MockModel(
        (
            json.dumps(
                {
                    "type": "tool_call",
                    "id": "create",
                    "tool": "repository.write_file",
                    "arguments": {"path": "b.py", "content": "NEW\n", "mode": "create"},
                }
            ),
        ),
        context_capacity=8192,
    )
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        require_relevant_source=False,
    )
    with pytest.raises(RepositoryOrchestrationError, match="CREATE_READY"):
        session.execute_task("Create b.py")
    assert not (tmp_path / "b.py").exists()


def test_m18_ephemeral_path_not_create_candidate(tmp_path: Path) -> None:
    context, arguments = _fixture(tmp_path)
    (tmp_path / "tests").mkdir()
    operations = [dict(item) for item in arguments["operations"]]
    operations[1]["path"] = "tests/test_generated.py"
    with pytest.raises(ToolError, match="candidate authority"):
        _rebind(context, arguments, operations)


def test_mixed_protocol_parses_and_schema_is_representation_aware() -> None:
    payload = {
        "type": "multi_file_change",
        "operations": [
            {
                "type": "line_range_edit",
                "path": "a.py",
                "start_line": 1,
                "end_line": 1,
                "new_text": "x",
            },
            {"type": "create_file", "path": "b.py", "content": "y"},
        ],
    }
    assert (
        parse_model_output(json.dumps(payload)).outcome
        is ToolCallOutcome.MULTI_FILE_CHANGE
    )
    schema = build_mutation_ready_output(
        ("a.py",),
        create_paths=("b.py",),
        representation=MutationRepresentationPolicy.LINE_RANGE,
    ).schema
    assert "line_range_edit" in str(schema)
    assert "structured_edit" not in str(schema)


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("safe", PermissionDecision.DENY),
        ("confirm", PermissionDecision.ASK),
        ("trusted-exec", PermissionDecision.ASK),
    ),
)
def test_mixed_creation_permission_profiles(
    profile: str, expected: PermissionDecision
) -> None:
    interaction = resolve_interaction_policy(AutonomyMode.AGENT, profile)
    registry = create_repository_registry(interaction, include_creation=True)
    if profile == "safe":
        assert "repository.apply_file_operations" not in {
            item.name for item in registry.metadata
        }
    else:
        tool = registry.get("repository.apply_file_operations")
        assert tool.metadata.capability.value == "write"
        assert interaction.decision_for(tool.metadata.capability) is expected


def test_m15_production_session_one_generation_one_mutation(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("VALUE = 1\n")
    proposal = json.dumps(
        {
            "type": "multi_file_change",
            "operations": [
                {"type": "create_file", "path": "b.py", "content": "NEW = 1\n"},
                {
                    "type": "line_range_edit",
                    "path": "a.py",
                    "start_line": 1,
                    "end_line": 1,
                    "new_text": "VALUE = 2",
                },
            ],
        }
    )
    model = MockModel((proposal, json.dumps({"type": "final", "answer": "Done"})))
    previews = []
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(include_creation=True),
        policy=create_assist_repository_policy(),
        required_candidate_paths=("a.py",),
        create_candidate_paths=("b.py",),
        mixed_file_operations=True,
        mutation_representation=MutationRepresentationPolicy.LINE_RANGE,
        approval_callback=lambda _invocation, preview: previews.append(preview) or True,
        require_relevant_source=False,
    )
    response = session.execute_task("Update a.py and create b.py")
    assert (tmp_path / "a.py").read_text() == "VALUE = 2\n"
    assert (tmp_path / "b.py").read_text() == "NEW = 1\n"
    assert session._mutation_generation == 1
    assert response.coding_task is not None and response.coding_task.mutation_count == 1
    assert len(previews) == 1
    assert previews[0].paths == ("a.py", "b.py")


def test_m16_repair_freshly_edits_primary_created_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("VALUE = 1\n")

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
            json.dumps(
                {
                    "type": "multi_file_change",
                    "operations": [
                        {
                            "type": "structured_edit",
                            "path": "a.py",
                            "old_text": "VALUE = 1",
                            "new_text": "from b import value\nVALUE = value()",
                        },
                        {
                            "type": "create_file",
                            "path": "b.py",
                            "content": "def value():\n    return 1\n",
                        },
                    ],
                }
            ),
            call("test-1", "project.test", {}),
            call("read-b", "repository.read_file", {"path": "b.py"}),
            json.dumps(
                {
                    "type": "structured_edit",
                    "path": "b.py",
                    "old_text": "return 1",
                    "new_text": "return 2",
                }
            ),
            call("test-2", "project.test", {}),
            json.dumps({"type": "final", "answer": "Done"}),
        ),
        context_capacity=8192,
    )
    commands = ProjectCommands(
        test=ProjectCommand(
            (sys.executable, "-c", "from a import VALUE; assert VALUE == 2"), 5
        )
    )
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(commands, include_creation=True),
        policy=create_assist_repository_policy(),
        required_candidate_paths=("a.py",),
        create_candidate_paths=("b.py",),
        mixed_file_operations=True,
        agent_mode=True,
        repair_enabled=True,
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
    )
    response = session.run_agent_task("Change a.py and add b.py")
    assert (tmp_path / "b.py").read_text() == "def value():\n    return 2\n"
    assert session._mutation_generation == 2
    assert response.coding_task is not None and response.coding_task.mutation_count == 2
    assert session._context.create_candidates == ()


def test_partial_mixed_group_gets_one_generic_correction(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("VALUE = 1\n")
    partial = json.dumps(
        {
            "type": "multi_file_change",
            "operations": [
                {"type": "create_file", "path": "b.py", "content": "NEW = 1\n"},
                {"type": "create_file", "path": "b.py", "content": "NEW = 2\n"},
            ],
        }
    )
    model = MockModel((partial, partial))
    session = RepositoryChatSession(
        "test",
        model,
        tmp_path,
        registry=create_assist_repository_registry(include_creation=True),
        policy=create_assist_repository_policy(),
        required_candidate_paths=("a.py",),
        create_candidate_paths=("b.py",),
        mixed_file_operations=True,
        require_relevant_source=False,
    )
    with pytest.raises(
        RepositoryOrchestrationError, match="invalid mixed group repeated"
    ):
        session.execute_task("Update a.py and create b.py")
    assert (tmp_path / "a.py").read_text() == "VALUE = 1\n"
    assert not (tmp_path / "b.py").exists()
    assert len(model.requests) == 2
    assert any(
        "exactly 2 operations" in message.content
        and "MODIFY paths: a.py" in message.content
        and "CREATE paths: b.py" in message.content
        for message in model.requests[1].messages
    )
