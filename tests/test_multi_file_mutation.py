from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from forge.evaluation.multi_file_mutation import run_multi_file_mutation_v1
from forge.interaction import AutonomyMode
from forge.models import MockModel, MutationRepresentationPolicy
from forge.orchestration import (
    CodingTaskStatus,
    LineRangeEditProposal,
    MultiFileLineRangeEditProposal,
    MultiFileStructuredEditProposal,
    MutationCandidate,
    RepositoryOrchestrationError,
    StructuredEditFailure,
    StructuredEditProposal,
    validate_multi_file_line_range_edit,
    validate_multi_file_structured_edit,
)
from forge.orchestration.protocol import (
    ToolCallOutcome,
    build_mutation_ready_output,
    parse_model_output,
)
from forge.orchestration.repository_session import RepositoryChatSession
from forge.project_config import ProjectCommand, ProjectCommands
from forge.tools import (
    AllowAllPolicy,
    ExecutionContext,
    InvocationApproval,
    MultiFileMutationPreview,
    MultiFilePatchTool,
    PermissionDecision,
    RuleBasedPolicy,
    ToolExecutor,
    ToolInvocation,
    ToolRegistry,
    ToolResultStatus,
    create_assist_repository_policy,
    create_assist_repository_registry,
    preview_multi_file_mutation,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(
    path: Path, workspace: Path, *, generation: int = 0
) -> MutationCandidate:
    return MutationCandidate(
        path.relative_to(workspace).as_posix(),
        _hash(path),
        generation,
        f"read-{path.name}",
        1,
        len(path.read_text().splitlines()),
    )


def _exact_group(workspace: Path):  # type: ignore[no-untyped-def]
    candidates = tuple(
        _candidate(workspace / name, workspace) for name in ("a.txt", "b.txt")
    )
    proposal = MultiFileStructuredEditProposal(
        (
            StructuredEditProposal("b.txt", "B = 1", "B = 2"),
            StructuredEditProposal("a.txt", "A = 1", "A = 2"),
        )
    )
    return validate_multi_file_structured_edit(proposal, candidates, workspace, 0)


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("A = 1\n", newline="")
    (tmp_path / "b.txt").write_text("B = 1\r\n", newline="")
    return tmp_path


def _execute(
    workspace: Path,
    arguments: dict[str, object],
    tool: MultiFilePatchTool | None = None,
):
    selected = tool or MultiFilePatchTool()
    invocation = ToolInvocation("group", "repository.apply_multi_patch", arguments)
    return ToolExecutor(ToolRegistry((selected,)), AllowAllPolicy()).execute(
        invocation, ExecutionContext(workspace)
    )


def test_multi_exact_is_canonical_one_preview_and_one_transaction(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    validation = _exact_group(workspace)
    assert validation.valid
    assert validation.arguments is not None
    assert [item["path"] for item in validation.arguments["patches"]] == [
        "a.txt",
        "b.txt",
    ]

    preview = preview_multi_file_mutation(
        validation.arguments, ExecutionContext(workspace)
    )
    result = _execute(workspace, validation.arguments)

    assert isinstance(preview, MultiFileMutationPreview)
    assert preview.paths == ("a.txt", "b.txt")
    assert "File 1: a.txt" in preview.diff and "File 2: b.txt" in preview.diff
    assert result.status is ToolResultStatus.SUCCESS
    assert result.output["mutation_file_count"] == 2
    assert (workspace / "a.txt").read_bytes() == b"A = 2\n"
    assert (workspace / "b.txt").read_bytes() == b"B = 2\r\n"
    assert not (workspace / ".forge-exec").exists()


def test_success_preserves_modes_and_unterminated_final_line(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    target = workspace / "c.txt"
    target.write_bytes(b"C = 1")
    os.chmod(target, 0o751)
    candidates = (
        _candidate(workspace / "a.txt", workspace),
        _candidate(target, workspace),
    )
    validation = validate_multi_file_structured_edit(
        MultiFileStructuredEditProposal(
            (
                StructuredEditProposal("a.txt", "A = 1", "A = 4"),
                StructuredEditProposal("c.txt", "C = 1", "C = 4"),
            )
        ),
        candidates,
        workspace,
        0,
    )
    assert validation.arguments is not None
    result = _execute(workspace, validation.arguments)
    assert result.status is ToolResultStatus.SUCCESS
    assert target.read_bytes() == b"C = 4"
    assert stat.S_IMODE(target.stat().st_mode) == 0o751


def test_group_bounds_and_canonical_identity_are_enforced(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    proposals = []
    candidates = []
    for index in range(5):
        path = workspace / f"{index}.txt"
        path.write_text(f"value = {index}\n")
        candidates.append(_candidate(path, workspace))
        proposals.append(
            StructuredEditProposal(
                path.name, f"value = {index}", f"value = {index + 1}"
            )
        )
    oversized = validate_multi_file_structured_edit(
        MultiFileStructuredEditProposal(tuple(proposals)),
        tuple(candidates),
        workspace,
        0,
    )
    assert oversized.failure is StructuredEditFailure.GROUP_BOUNDS

    valid = _exact_group(workspace)
    assert valid.arguments is not None
    tampered = {**valid.arguments, "group_id": "0" * 64}
    result = _execute(workspace, tampered)
    assert result.status is ToolResultStatus.FAILURE
    assert (workspace / "a.txt").read_text() == "A = 1\n"


def test_multi_line_range_reuses_per_file_range_semantics(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidates = tuple(
        _candidate(workspace / name, workspace) for name in ("a.txt", "b.txt")
    )
    validation = validate_multi_file_line_range_edit(
        MultiFileLineRangeEditProposal(
            (
                LineRangeEditProposal("b.txt", 1, 1, "B = 3"),
                LineRangeEditProposal("a.txt", 1, 1, "A = 3"),
            )
        ),
        candidates,
        workspace,
        0,
    )
    assert validation.valid and validation.arguments is not None
    result = _execute(workspace, validation.arguments)
    assert result.status is ToolResultStatus.SUCCESS
    assert (workspace / "a.txt").read_bytes() == b"A = 3\n"
    assert (workspace / "b.txt").read_bytes() == b"B = 3\r\n"


def test_invalid_or_noop_child_rejects_whole_group_before_preview(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    candidates = tuple(
        _candidate(workspace / name, workspace) for name in ("a.txt", "b.txt")
    )
    for second, failure in (
        (
            StructuredEditProposal("b.txt", "missing", "B = 2"),
            StructuredEditFailure.OLD_TEXT_NOT_FOUND,
        ),
        (
            StructuredEditProposal("b.txt", "B = 1", "B = 1"),
            StructuredEditFailure.NO_OP_EDIT,
        ),
    ):
        validation = validate_multi_file_structured_edit(
            MultiFileStructuredEditProposal(
                (StructuredEditProposal("a.txt", "A = 1", "A = 2"), second)
            ),
            candidates,
            workspace,
            0,
        )
        assert validation.failure is failure
        assert validation.arguments is None
        assert (workspace / "a.txt").read_text() == "A = 1\n"


def test_duplicate_unauthorized_and_symlink_paths_reject_group(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidates = tuple(
        _candidate(workspace / name, workspace) for name in ("a.txt", "b.txt")
    )
    duplicate = validate_multi_file_structured_edit(
        MultiFileStructuredEditProposal(
            (
                StructuredEditProposal("a.txt", "A = 1", "A = 2"),
                StructuredEditProposal("a.txt", "A = 1", "A = 3"),
            )
        ),
        candidates,
        workspace,
        0,
    )
    unauthorized = validate_multi_file_structured_edit(
        MultiFileStructuredEditProposal(
            (
                StructuredEditProposal("a.txt", "A = 1", "A = 2"),
                StructuredEditProposal("c.txt", "C = 1", "C = 2"),
            )
        ),
        candidates,
        workspace,
        0,
    )
    (workspace / "link.txt").symlink_to(workspace / "b.txt")
    symlink = validate_multi_file_structured_edit(
        MultiFileStructuredEditProposal(
            (
                StructuredEditProposal("a.txt", "A = 1", "A = 2"),
                StructuredEditProposal("link.txt", "B = 1", "B = 2"),
            )
        ),
        (
            *candidates,
            MutationCandidate("link.txt", _hash(workspace / "b.txt"), 0, "read-link"),
        ),
        workspace,
        0,
    )
    assert duplicate.failure is StructuredEditFailure.DUPLICATE_PATH
    assert unauthorized.failure is StructuredEditFailure.PATH_NOT_ELIGIBLE
    assert symlink.failure is StructuredEditFailure.PATH_NOT_ELIGIBLE


def test_stale_second_file_after_preview_causes_zero_writes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    validation = _exact_group(workspace)
    assert validation.arguments is not None
    preview_multi_file_mutation(validation.arguments, ExecutionContext(workspace))
    (workspace / "b.txt").write_text("B = external\n")

    result = _execute(workspace, validation.arguments)

    assert result.status is ToolResultStatus.FAILURE
    assert (workspace / "a.txt").read_text() == "A = 1\n"
    assert (workspace / "b.txt").read_text() == "B = external\n"


def test_approval_binds_exact_complete_group(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    validation = _exact_group(workspace)
    assert validation.arguments is not None
    original = ToolInvocation(
        "group", "repository.apply_multi_patch", validation.arguments
    )
    changed_arguments = dict(validation.arguments)
    changed_patches = [dict(item) for item in changed_arguments["patches"]]
    changed_patches[1] = {
        **changed_patches[1],
        "edits": ({"old": "B = 1", "new": "B = 9"},),
    }
    changed_arguments["patches"] = changed_patches
    changed = ToolInvocation("group", "repository.apply_multi_patch", changed_arguments)
    executor = ToolExecutor(
        ToolRegistry((MultiFilePatchTool(),)),
        RuleBasedPolicy({"repository.apply_multi_patch": PermissionDecision.ASK}),
    )
    result = executor.execute(
        changed,
        ExecutionContext(workspace),
        approval=InvocationApproval.for_invocation(original),
    )
    assert result.status is ToolResultStatus.APPROVAL_REQUIRED
    assert (workspace / "a.txt").read_text() == "A = 1\n"


def test_second_replace_failure_rolls_back_exact_bytes_and_modes(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    os.chmod(workspace / "a.txt", 0o754)
    before = {name: (workspace / name).read_bytes() for name in ("a.txt", "b.txt")}
    validation = _exact_group(workspace)
    assert validation.arguments is not None
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second replacement failure")
        os.replace(source, target)

    result = _execute(
        workspace,
        validation.arguments,
        MultiFilePatchTool(replace_operation=fail_second),
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["mutation_group_rollback_result"] == "restored"
    assert all((workspace / name).read_bytes() == data for name, data in before.items())
    assert stat.S_IMODE((workspace / "a.txt").stat().st_mode) == 0o754
    assert not (workspace / ".forge-exec").exists()


def test_rollback_failure_reports_fatal_integrity_state(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    validation = _exact_group(workspace)
    assert validation.arguments is not None
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected apply failure")
        os.replace(source, target)

    def fail_rollback(_source: Path, _target: Path) -> None:
        raise OSError("injected rollback failure")

    result = _execute(
        workspace,
        validation.arguments,
        MultiFilePatchTool(
            replace_operation=fail_second, rollback_operation=fail_rollback
        ),
    )
    assert result.status is ToolResultStatus.FAILURE
    assert result.output["workspace_integrity_failure"] is True
    assert result.output["mutation_group_rollback_result"] == "failed"


def test_integrity_failure_terminates_session_before_model_retry_or_verification(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 1\n")

    def read(identifier: str, path: str) -> str:
        return json.dumps(
            {
                "type": "tool_call",
                "id": identifier,
                "tool": "repository.read_file",
                "arguments": {"path": path},
            }
        )

    model = MockModel(
        (
            read("read-a", "a.py"),
            read("read-b", "b.py"),
            json.dumps(
                {
                    "type": "multi_file_structured_edit",
                    "edits": [
                        {"path": "a.py", "old_text": "A = 1", "new_text": "A = 2"},
                        {"path": "b.py", "old_text": "B = 1", "new_text": "B = 2"},
                    ],
                }
            ),
            json.dumps({"type": "final", "answer": "must not be requested"}),
        )
    )
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected apply failure")
        os.replace(source, target)

    def fail_rollback(_source: Path, _target: Path) -> None:
        raise OSError("injected rollback failure")

    registry = create_assist_repository_registry()
    vars(registry)["_tools"]["repository.apply_multi_patch"] = MultiFilePatchTool(
        replace_operation=fail_second, rollback_operation=fail_rollback
    )
    session = RepositoryChatSession(
        "fatal-multi",
        model,
        tmp_path,
        registry=registry,
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
        minimum_source_files=2,
    )

    with pytest.raises(RepositoryOrchestrationError, match="integrity failure"):
        session.execute_task("Change both files")

    assert len(model.requests) == 3
    assert session.last_coding_task is not None
    assert (
        session.last_coding_task.status is CodingTaskStatus.WORKSPACE_INTEGRITY_FAILED
    )
    assert session.last_coding_task.verification_gate_metrics.executed == 0


def test_protocol_and_schema_are_bounded_and_representation_specific() -> None:
    exact = build_mutation_ready_output(("b.c", "a.c"))
    line = build_mutation_ready_output(
        ("a.c", "b.c"), representation=MutationRepresentationPolicy.LINE_RANGE
    )
    parsed = parse_model_output(
        json.dumps(
            {
                "type": "multi_file_structured_edit",
                "edits": [
                    {"path": "a.c", "old_text": "a", "new_text": "A"},
                    {"path": "b.c", "old_text": "b", "new_text": "B"},
                ],
            }
        )
    )
    assert parsed.outcome is ToolCallOutcome.MULTI_FILE_STRUCTURED_EDIT
    assert "multi_file_structured_edit" in str(exact.schema)
    assert "multi_file_line_range_edit" not in str(exact.schema)
    assert "multi_file_line_range_edit" in str(line.schema)
    assert "'maxItems': 4" in str(line.schema)


def test_grouped_mutation_runs_normal_verification_once(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 1\n")

    def call(identifier: str, path: str) -> str:
        return json.dumps(
            {
                "type": "tool_call",
                "id": identifier,
                "tool": "repository.read_file",
                "arguments": {"path": path},
            }
        )

    model = MockModel(
        (
            call("read-a", "a.py"),
            call("read-b", "b.py"),
            json.dumps(
                {
                    "type": "multi_file_structured_edit",
                    "edits": [
                        {"path": "b.py", "old_text": "B = 1", "new_text": "B = 2"},
                        {"path": "a.py", "old_text": "A = 1", "new_text": "A = 2"},
                    ],
                }
            ),
            json.dumps({"type": "final", "answer": "Done."}),
        )
    )
    test = ProjectCommand(
        (
            sys.executable,
            "-c",
            "from pathlib import Path; assert 'A = 2' in Path('a.py').read_text(); "
            "assert 'B = 2' in Path('b.py').read_text()",
        ),
        5,
    )
    commands = ProjectCommands(test=test)
    previews: list[MultiFileMutationPreview] = []

    def approve(_invocation, preview):  # type: ignore[no-untyped-def]
        if isinstance(preview, MultiFileMutationPreview):
            previews.append(preview)
        return True

    response = RepositoryChatSession(
        "multi",
        model,
        tmp_path,
        registry=create_assist_repository_registry(commands),
        policy=create_assist_repository_policy(),
        approval_callback=approve,
        require_relevant_source=False,
        minimum_source_files=2,
    ).execute_task("Change a.py and b.py together")

    assert response.coding_task is not None
    assert response.coding_task.status.value == "completed_verified"
    assert response.coding_task.mutation_count == 1
    assert response.coding_task.changed_files == ("a.py", "b.py")
    assert response.coding_task.verification_gate_metrics.executed == 1
    assert len(previews) == 1
    assert "multi_file_structured_edit" in str(model.requests[2].output.schema)


def test_grouped_primary_and_grouped_repair_count_as_two_mutations(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 1\n")

    def read(identifier: str, path: str) -> str:
        return json.dumps(
            {
                "type": "tool_call",
                "id": identifier,
                "tool": "repository.read_file",
                "arguments": {"path": path},
            }
        )

    def group(old: int, new: int) -> str:
        return json.dumps(
            {
                "type": "multi_file_structured_edit",
                "edits": [
                    {
                        "path": "a.py",
                        "old_text": f"A = {old}",
                        "new_text": f"A = {new}",
                    },
                    {
                        "path": "b.py",
                        "old_text": f"B = {old}",
                        "new_text": f"B = {new}",
                    },
                ],
            }
        )

    model = MockModel(
        (
            read("read-a", "a.py"),
            read("read-b", "b.py"),
            group(1, 2),
            read("repair-a", "a.py"),
            read("repair-b", "b.py"),
            group(2, 3),
            json.dumps({"type": "final", "answer": "Repaired."}),
        )
    )
    test = ProjectCommand(
        (
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "ok='A = 3' in Path('a.py').read_text() and "
            "'B = 3' in Path('b.py').read_text(); sys.exit(0 if ok else 1)",
        ),
        5,
    )
    response = RepositoryChatSession(
        "multi-repair",
        model,
        tmp_path,
        mode=AutonomyMode.REPAIR,
        registry=create_assist_repository_registry(ProjectCommands(test=test)),
        policy=create_assist_repository_policy(),
        approval_callback=lambda *_args: True,
        require_relevant_source=False,
        minimum_source_files=2,
    ).execute_task("Change a.py and b.py together and repair if verification fails")

    assert response.coding_task is not None
    assert response.coding_task.status.value == "completed_repaired_verified"
    assert response.coding_task.mutation_count == 2
    assert len(response.coding_task.mutations) == 2
    assert all(len(item.files) == 2 for item in response.coding_task.mutations)
    assert (tmp_path / "a.py").read_text() == "A = 3\n"
    assert (tmp_path / "b.py").read_text() == "B = 3\n"


def test_multi_file_mutation_v1_ten_scenarios_pass(tmp_path: Path) -> None:
    run = run_multi_file_mutation_v1(tmp_path / "suite")
    assert tuple(case.case_id for case in run.cases) == tuple(
        f"M{number:02d}" for number in range(1, 11)
    )
    assert run.tasks_passed == run.tasks_total == 10
