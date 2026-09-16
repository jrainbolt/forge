"""Deterministic A41 atomic multi-file mutation evaluation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from forge.orchestration.coding_task import (
    CodingTaskPhase,
    CodingTaskState,
    MutationCandidate,
)
from forge.orchestration.structured_edit import (
    LineRangeEditProposal,
    MultiFileLineRangeEditProposal,
    MultiFileStructuredEditProposal,
    StructuredEditFailure,
    StructuredEditProposal,
    validate_multi_file_line_range_edit,
    validate_multi_file_structured_edit,
)
from forge.tools import (
    AllowAllPolicy,
    ExecutionContext,
    InvocationApproval,
    MultiFilePatchTool,
    ToolExecutor,
    ToolInvocation,
    ToolRegistry,
    ToolResultStatus,
    preview_multi_file_mutation,
)

MULTI_FILE_MUTATION_V1 = "multi-file-mutation-v1"


@dataclass(frozen=True, slots=True)
class MultiFileMutationCase:
    case_id: str
    passed: bool
    outcome: str


@dataclass(frozen=True, slots=True)
class MultiFileMutationRun:
    cases: tuple[MultiFileMutationCase, ...]

    @property
    def tasks_passed(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def tasks_total(self) -> int:
        return len(self.cases)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path, name: str) -> Path:
    workspace = root / name
    workspace.mkdir(parents=True)
    (workspace / "a.txt").write_text("A = 1\n")
    (workspace / "b.txt").write_text("B = 1\n")
    return workspace


def _candidates(workspace: Path, generation: int = 0) -> tuple[MutationCandidate, ...]:
    return tuple(
        MutationCandidate(
            name, _hash(workspace / name), generation, f"read-{name}", 1, 1
        )
        for name in ("a.txt", "b.txt")
    )


def _validation(workspace: Path):  # type: ignore[no-untyped-def]
    return validate_multi_file_structured_edit(
        MultiFileStructuredEditProposal(
            (
                StructuredEditProposal("a.txt", "A = 1", "A = 2"),
                StructuredEditProposal("b.txt", "B = 1", "B = 2"),
            )
        ),
        _candidates(workspace),
        workspace,
        0,
    )


def _execute(
    workspace: Path,
    arguments: dict[str, object],
    tool: MultiFilePatchTool | None = None,
):
    invocation = ToolInvocation("group", "repository.apply_multi_patch", arguments)
    return ToolExecutor(
        ToolRegistry((tool or MultiFilePatchTool(),)), AllowAllPolicy()
    ).execute(invocation, ExecutionContext(workspace))


def run_multi_file_mutation_v1(root: Path) -> MultiFileMutationRun:
    root.mkdir(parents=True, exist_ok=True)
    cases: list[MultiFileMutationCase] = []

    m01 = _fixture(root, "M01")
    exact = _validation(m01)
    preview = (
        preview_multi_file_mutation(exact.arguments, ExecutionContext(m01))
        if exact.arguments is not None
        else None
    )
    result = _execute(m01, exact.arguments) if exact.arguments is not None else None
    cases.append(
        MultiFileMutationCase(
            "M01",
            preview is not None
            and preview.paths == ("a.txt", "b.txt")
            and result is not None
            and result.status is ToolResultStatus.SUCCESS,
            "two_file_exact_text",
        )
    )

    m02 = _fixture(root, "M02")
    line = validate_multi_file_line_range_edit(
        MultiFileLineRangeEditProposal(
            (
                LineRangeEditProposal("a.txt", 1, 1, "A = 2"),
                LineRangeEditProposal("b.txt", 1, 1, "B = 2"),
            )
        ),
        _candidates(m02),
        m02,
        0,
    )
    line_result = _execute(m02, line.arguments) if line.arguments is not None else None
    cases.append(
        MultiFileMutationCase(
            "M02",
            line_result is not None and line_result.status is ToolResultStatus.SUCCESS,
            "two_file_line_range",
        )
    )

    m03 = _fixture(root, "M03")
    invalid = validate_multi_file_structured_edit(
        MultiFileStructuredEditProposal(
            (
                StructuredEditProposal("a.txt", "A = 1", "A = 2"),
                StructuredEditProposal("b.txt", "missing", "B = 2"),
            )
        ),
        _candidates(m03),
        m03,
        0,
    )
    cases.append(
        MultiFileMutationCase(
            "M03",
            invalid.failure is StructuredEditFailure.OLD_TEXT_NOT_FOUND
            and (m03 / "a.txt").read_text() == "A = 1\n",
            "second_validation_zero_writes",
        )
    )

    m04 = _fixture(root, "M04")
    stale = _validation(m04)
    assert stale.arguments is not None
    preview_multi_file_mutation(stale.arguments, ExecutionContext(m04))
    (m04 / "b.txt").write_text("external\n")
    stale_result = _execute(m04, stale.arguments)
    cases.append(
        MultiFileMutationCase(
            "M04",
            stale_result.status is ToolResultStatus.FAILURE
            and (m04 / "a.txt").read_text() == "A = 1\n",
            "stale_second_zero_writes",
        )
    )

    m05 = _fixture(root, "M05")
    noop = validate_multi_file_structured_edit(
        MultiFileStructuredEditProposal(
            (
                StructuredEditProposal("a.txt", "A = 1", "A = 2"),
                StructuredEditProposal("b.txt", "B = 1", "B = 1"),
            )
        ),
        _candidates(m05),
        m05,
        0,
    )
    cases.append(
        MultiFileMutationCase(
            "M05",
            noop.failure is StructuredEditFailure.NO_OP_EDIT,
            "grouped_noop_rejected",
        )
    )

    m06 = _fixture(root, "M06")
    approval_validation = _validation(m06)
    assert approval_validation.arguments is not None
    approved = ToolInvocation(
        "group", "repository.apply_multi_patch", approval_validation.arguments
    )
    changed = ToolInvocation(
        "group-changed", "repository.apply_multi_patch", approval_validation.arguments
    )
    cases.append(
        MultiFileMutationCase(
            "M06",
            not InvocationApproval.for_invocation(approved).matches(changed),
            "approval_exact_group",
        )
    )

    m07 = _fixture(root, "M07")
    rollback_validation = _validation(m07)
    assert rollback_validation.arguments is not None
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected")
        os.replace(source, target)

    rollback = _execute(
        m07,
        rollback_validation.arguments,
        MultiFilePatchTool(replace_operation=fail_second),
    )
    cases.append(
        MultiFileMutationCase(
            "M07",
            rollback.status is ToolResultStatus.FAILURE
            and rollback.output["mutation_group_rollback_result"] == "restored"
            and (m07 / "a.txt").read_text() == "A = 1\n",
            "handled_failure_rollback",
        )
    )

    m08 = _fixture(root, "M08")
    fatal_validation = _validation(m08)
    assert fatal_validation.arguments is not None
    calls = 0

    def fail_rollback(_source: Path, _target: Path) -> None:
        raise OSError("injected rollback")

    fatal = _execute(
        m08,
        fatal_validation.arguments,
        MultiFilePatchTool(
            replace_operation=fail_second, rollback_operation=fail_rollback
        ),
    )
    cases.append(
        MultiFileMutationCase(
            "M08",
            fatal.status is ToolResultStatus.FAILURE
            and fatal.output["workspace_integrity_failure"] is True,
            "fatal_integrity_status",
        )
    )

    state = CodingTaskState(0, transition_required=False)
    state.phase = CodingTaskPhase.AWAITING_MUTATION_APPROVAL
    state.mutation_succeeded(
        "repository.apply_multi_patch",
        {
            "paths": ("a.txt", "b.txt"),
            "changes": (
                {"path": "a.txt", "old_sha256": "a", "new_sha256": "b"},
                {"path": "b.txt", "old_sha256": "c", "new_sha256": "d"},
            ),
            "mutation_group_id": "0" * 64,
            "mutation_group_apply_result": "applied",
            "mutation_group_rollback_result": "not_needed",
        },
        1,
    )
    cases.append(
        MultiFileMutationCase(
            "M09",
            state.mutation_count == 1
            and state.generation == 1
            and tuple(state.changed_files) == ("a.txt", "b.txt"),
            "verification_ready_generation_transition",
        )
    )

    repair = CodingTaskState(0, repair_enabled=True, transition_required=False)
    repair.phase = CodingTaskPhase.AWAITING_MUTATION_APPROVAL
    output = {
        "paths": ("a.txt", "b.txt"),
        "changes": (
            {"path": "a.txt", "old_sha256": "a", "new_sha256": "b"},
            {"path": "b.txt", "old_sha256": "c", "new_sha256": "d"},
        ),
        "mutation_group_id": "1" * 64,
        "mutation_group_apply_result": "applied",
        "mutation_group_rollback_result": "not_needed",
    }
    repair.mutation_succeeded("repository.apply_multi_patch", output, 1)
    repair.repair_eligible = True
    repair.phase = CodingTaskPhase.AWAITING_REPAIR_APPROVAL
    repair.repair_attempted = True
    repair.mutation_succeeded("repository.apply_multi_patch", output, 2)
    cases.append(
        MultiFileMutationCase(
            "M10",
            repair.mutation_count == 2 and repair.generation == 2,
            "grouped_repair_two_mutation_ceiling",
        )
    )

    return MultiFileMutationRun(tuple(cases))
