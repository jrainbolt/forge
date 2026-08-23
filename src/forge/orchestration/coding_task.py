"""Explicit state and externally truthful results for one A11 coding task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class CodingTaskPhase(Enum):
    INSPECTING = "inspecting"
    AWAITING_MUTATION_APPROVAL = "awaiting_mutation_approval"
    MUTATED = "mutated"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class CodingTaskStatus(Enum):
    COMPLETED_VERIFIED = "completed_verified"
    COMPLETED_UNVERIFIED = "completed_unverified"
    COMPLETED_READ_ONLY = "completed_read_only"
    REJECTED = "rejected"
    FAILED_BEFORE_MUTATION = "failed_before_mutation"
    MUTATED_VERIFICATION_FAILED = "mutated_verification_failed"
    MUTATED_TASK_FAILED = "mutated_task_failed"


class VerificationDecision(Enum):
    NOT_DECIDED = "not_decided"
    REQUESTED = "requested"
    COMPLETED = "completed"
    DECLINED = "declined"


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    attempted: bool = False
    status: str = "not_run"
    exit_code: int | None = None
    timed_out: bool = False
    truncated: bool = False
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class CodingTaskResult:
    answer: str
    status: CodingTaskStatus
    mutation_count: int
    mutation_tool: str | None
    changed_files: tuple[str, ...]
    old_sha256: str | None
    new_sha256: str | None
    build: VerificationRecord
    test: VerificationRecord
    verification_generation: int
    tool_sequence: tuple[str, ...]

    @property
    def footer(self) -> str:
        change = "applied" if self.mutation_count else "not applied"
        return (
            f"Change: {change}\n"
            f"Build: {self.build.status}\n"
            f"Tests: {self.test.status}\n"
            f"Status: {self.status.value.upper()}"
        )


class CodingTaskState:
    """Small mutable transition authority scoped to one user request."""

    def __init__(self, generation: int) -> None:
        self.phase = CodingTaskPhase.INSPECTING
        self.mutation_count = 0
        self.mutation_tool: str | None = None
        self.changed_files: list[str] = []
        self.old_sha256: str | None = None
        self.new_sha256: str | None = None
        self.generation = generation
        self.build = VerificationRecord()
        self.test = VerificationRecord()
        self.verification_decision = VerificationDecision.NOT_DECIDED
        self.tool_sequence: list[str] = []
        self._terminal_status: CodingTaskStatus | None = None

    @property
    def terminal(self) -> bool:
        return self._terminal_status is not None

    @property
    def may_propose_mutation(self) -> bool:
        return self.phase is CodingTaskPhase.INSPECTING and not self.terminal

    @property
    def may_verify(self) -> bool:
        return not self.terminal

    def record_tool(self, name: str) -> None:
        self.tool_sequence.append(name)

    def mutation_proposed(self) -> bool:
        if not self.may_propose_mutation:
            self.fail_after_mutation()
            return False
        self.phase = CodingTaskPhase.AWAITING_MUTATION_APPROVAL
        return True

    def mutation_rejected(self) -> None:
        self.phase = CodingTaskPhase.REJECTED
        self._terminal_status = CodingTaskStatus.REJECTED

    def mutation_failed(self) -> None:
        if self.mutation_count:
            self.fail_after_mutation()
            return
        self.phase = CodingTaskPhase.FAILED
        self._terminal_status = CodingTaskStatus.FAILED_BEFORE_MUTATION

    def mutation_succeeded(
        self, tool: str, output: Mapping[str, object], generation: int
    ) -> None:
        if self.mutation_count:
            raise RuntimeError("coding task mutation limit exceeded")
        self.mutation_count = 1
        self.mutation_tool = tool
        path = output.get("path")
        if isinstance(path, str):
            self.changed_files.append(path)
        old_hash = output.get("old_sha256")
        new_hash = output.get("new_sha256")
        self.old_sha256 = old_hash if isinstance(old_hash, str) else None
        self.new_sha256 = new_hash if isinstance(new_hash, str) else None
        self.generation = generation
        self.build = _stale(self.build)
        self.test = _stale(self.test)
        self.verification_decision = VerificationDecision.NOT_DECIDED
        self.phase = CodingTaskPhase.MUTATED

    def verification_requested(self, operation: str) -> bool:
        if not self.may_verify:
            self.fail_after_mutation()
            return False
        record = self._verification(operation)
        if record.attempted and record.generation == self.generation:
            self.fail_after_mutation()
            return False
        self.phase = CodingTaskPhase.VERIFYING
        self.verification_decision = VerificationDecision.REQUESTED
        return True

    def verification_finished(
        self,
        operation: str,
        status: str,
        output: Mapping[str, object] | None,
    ) -> None:
        outcome = output.get("outcome") if output is not None else None
        if status == "success":
            label = "passed"
        elif outcome == "command_not_configured" or status == "approval_required":
            label = "not_run"
        else:
            label = "failed"
        exit_code = output.get("exit_code") if output is not None else None
        timed_out = output.get("timed_out") if output is not None else False
        stdout_truncated = (
            output.get("stdout_truncated") if output is not None else False
        )
        stderr_truncated = (
            output.get("stderr_truncated") if output is not None else False
        )
        record = VerificationRecord(
            attempted=status != "approval_required"
            and outcome != "command_not_configured",
            status=label,
            exit_code=exit_code if isinstance(exit_code, int) else None,
            timed_out=timed_out is True,
            truncated=stdout_truncated is True or stderr_truncated is True,
            generation=self.generation if label == "passed" else None,
        )
        if operation == "build":
            self.build = record
        else:
            self.test = record
        if label == "failed":
            self.verification_decision = VerificationDecision.COMPLETED
            self.phase = CodingTaskPhase.FAILED
            self._terminal_status = (
                CodingTaskStatus.MUTATED_VERIFICATION_FAILED
                if self.mutation_count
                else CodingTaskStatus.FAILED_BEFORE_MUTATION
            )
        elif label == "not_run":
            self.verification_decision = VerificationDecision.DECLINED
            self.phase = CodingTaskPhase.COMPLETED
            self._terminal_status = CodingTaskStatus.COMPLETED_UNVERIFIED
        else:
            self.verification_decision = VerificationDecision.COMPLETED
            self.phase = (
                CodingTaskPhase.MUTATED
                if self.mutation_count
                else CodingTaskPhase.INSPECTING
            )

    def fail_after_mutation(self) -> None:
        if self._terminal_status is not None:
            return
        self.phase = CodingTaskPhase.FAILED
        self._terminal_status = (
            CodingTaskStatus.MUTATED_TASK_FAILED
            if self.mutation_count
            else CodingTaskStatus.FAILED_BEFORE_MUTATION
        )

    def decline_verification(self) -> None:
        if self.verification_decision is VerificationDecision.NOT_DECIDED:
            self.verification_decision = VerificationDecision.DECLINED

    def finish(self, answer: str) -> CodingTaskResult:
        if self._terminal_status is None:
            if any(
                record.status == "passed" and record.generation == self.generation
                for record in (self.build, self.test)
            ):
                status = CodingTaskStatus.COMPLETED_VERIFIED
            elif not self.mutation_count:
                status = CodingTaskStatus.COMPLETED_READ_ONLY
            else:
                status = CodingTaskStatus.COMPLETED_UNVERIFIED
            self._terminal_status = status
            self.phase = CodingTaskPhase.COMPLETED
        return CodingTaskResult(
            answer,
            self._terminal_status,
            self.mutation_count,
            self.mutation_tool,
            tuple(self.changed_files),
            self.old_sha256,
            self.new_sha256,
            self.build,
            self.test,
            self.generation,
            tuple(self.tool_sequence),
        )

    def _verification(self, operation: str) -> VerificationRecord:
        if operation == "build":
            return self.build
        if operation == "test":
            return self.test
        raise ValueError("verification operation must be build or test")


def _stale(record: VerificationRecord) -> VerificationRecord:
    if record.status != "passed":
        return record
    return VerificationRecord(
        record.attempted,
        "stale",
        record.exit_code,
        record.timed_out,
        record.truncated,
        record.generation,
    )
