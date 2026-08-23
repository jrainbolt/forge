"""Bounded A12 agent-task state, metrics, and machine-readable termination."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from forge.orchestration.coding_task import (
    CodingTaskResult,
    MutationRecord,
    VerificationRecord,
)


class AgentPhase(Enum):
    ANALYZING = "analyzing"
    ACTING = "acting"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    OBSERVING = "observing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class AgentStopReason(Enum):
    COMPLETED = "completed"
    USER_REJECTED = "user_rejected"
    CANCELLED = "cancelled"
    TOOL_LIMIT = "tool_limit"
    ITERATION_LIMIT = "iteration_limit"
    MODEL_CALL_LIMIT = "model_call_limit"
    NO_PROGRESS = "no_progress"
    REPEATED_CALL = "repeated_call"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    VERIFICATION_FAILED = "verification_failed"
    SECOND_MUTATION_BLOCKED = "second_mutation_blocked"
    CONTEXT_LIMIT = "context_limit"
    PROTOCOL_ERROR = "protocol_error"
    REPAIR_REJECTED = "repair_rejected"
    REPAIR_VERIFICATION_FAILED = "repair_verification_failed"
    REPAIR_BUDGET_EXHAUSTED = "repair_budget_exhausted"
    REPAIR_NOT_ELIGIBLE = "repair_not_eligible"


class AgentCancelled(RuntimeError):
    """The foreground user cancelled an active agent task."""


@dataclass(frozen=True, slots=True)
class AgentTaskResult:
    final_answer: str
    status: str
    stop_reason: AgentStopReason
    iterations: int
    model_calls: int
    tool_calls: int
    unique_files_read: tuple[str, ...]
    mutation_count: int
    changed_files: tuple[str, ...]
    build: VerificationRecord
    test: VerificationRecord
    workspace_generation: int
    approval_requests: int
    approvals_granted: int
    approvals_rejected: int
    no_progress_cycles: int
    repair_enabled: bool = False
    repair_eligible: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_eligibility_outcome: str | None = None
    mutations: tuple[MutationRecord, ...] = ()
    build_attempts: tuple[VerificationRecord, ...] = ()
    test_attempts: tuple[VerificationRecord, ...] = ()

    @property
    def footer(self) -> str:
        if self.repair_enabled:
            initial = _verification_label(self.build_attempts, self.test_attempts, 0)
            repair = _verification_label(self.build_attempts, self.test_attempts, 1)
            initial_change = "applied" if self.mutation_count >= 1 else "not applied"
            repair_change = "applied" if self.mutation_count >= 2 else "not applied"
            return (
                f"Change #1: {initial_change}\n"
                f"Initial verification: {initial}\n"
                f"Repair #2: {repair_change}\n"
                f"Repair verification: {repair}\n"
                f"Status: {self.status.upper()}\n"
                f"Stop: {self.stop_reason.value.upper()}\n"
                f"Iterations: {self.iterations}\n"
                f"Tools: {self.tool_calls}"
            )
        return (
            f"Status: {self.status.upper()}\n"
            f"Stop: {self.stop_reason.value.upper()}\n"
            f"Iterations: {self.iterations}\n"
            f"Tools: {self.tool_calls}\n"
            f"Mutation: {self.mutation_count}\n"
            f"Tests: {self.test.status}"
        )


class AgentTaskState:
    """Controlled per-request counters and simple progress detection."""

    def __init__(self) -> None:
        self.phase = AgentPhase.ANALYZING
        self.iterations = 0
        self.model_calls = 0
        self.tool_calls = 0
        self.approval_requests = 0
        self.approvals_granted = 0
        self.approvals_rejected = 0
        self.no_progress_cycles = 0
        self.unique_files_read: set[str] = set()
        self._progress_keys: set[str] = set()

    def model_called(self) -> None:
        self.iterations += 1
        self.model_calls += 1
        self.phase = AgentPhase.ANALYZING

    def tool_requested(self) -> None:
        self.tool_calls += 1
        self.phase = AgentPhase.ACTING

    def approval_requested(self) -> None:
        self.approval_requests += 1
        self.phase = AgentPhase.WAITING_FOR_APPROVAL

    def approval_finished(self, approved: bool) -> None:
        if approved:
            self.approvals_granted += 1
        else:
            self.approvals_rejected += 1
        self.phase = AgentPhase.OBSERVING

    def observe(self, key: str | None, *, file_read: str | None = None) -> bool:
        if file_read is not None:
            self.unique_files_read.add(file_read)
        if key is not None and key not in self._progress_keys:
            self._progress_keys.add(key)
            self.no_progress_cycles = 0
            return True
        self.no_progress_cycles += 1
        return False

    def result(
        self,
        coding: CodingTaskResult,
        reason: AgentStopReason,
        *,
        answer: str,
    ) -> AgentTaskResult:
        self.phase = (
            AgentPhase.COMPLETED
            if reason is AgentStopReason.COMPLETED
            else AgentPhase.STOPPED
            if reason
            in {
                AgentStopReason.CANCELLED,
                AgentStopReason.USER_REJECTED,
                AgentStopReason.NO_PROGRESS,
                AgentStopReason.REPEATED_CALL,
                AgentStopReason.TOOL_LIMIT,
                AgentStopReason.ITERATION_LIMIT,
                AgentStopReason.MODEL_CALL_LIMIT,
            }
            else AgentPhase.FAILED
        )
        return AgentTaskResult(
            answer,
            coding.status.value,
            reason,
            self.iterations,
            self.model_calls,
            self.tool_calls,
            tuple(sorted(self.unique_files_read)),
            coding.mutation_count,
            coding.changed_files,
            coding.build,
            coding.test,
            coding.verification_generation,
            self.approval_requests,
            self.approvals_granted,
            self.approvals_rejected,
            self.no_progress_cycles,
            coding.repair_enabled,
            coding.repair_eligible,
            coding.repair_attempted,
            coding.repair_succeeded,
            coding.repair_eligibility_outcome,
            coding.mutations,
            coding.build_attempts,
            coding.test_attempts,
        )


def _verification_label(
    builds: tuple[VerificationRecord, ...],
    tests: tuple[VerificationRecord, ...],
    index: int,
) -> str:
    records = tuple(record for record in (*builds, *tests) if record.attempted)
    return records[index].status if len(records) > index else "not_run"
