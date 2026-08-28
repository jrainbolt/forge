"""Explicit state and externally truthful results for one A11 coding task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class CodingTaskPhase(Enum):
    INSPECTING = "inspecting"
    MUTATION_READY = "mutation_ready"
    AWAITING_MUTATION_APPROVAL = "awaiting_mutation_approval"
    MUTATED = "mutated"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    DIAGNOSING = "diagnosing"
    AWAITING_REPAIR_APPROVAL = "awaiting_repair_approval"
    REPAIRED = "repaired"
    VERIFYING_REPAIR = "verifying_repair"


class CodingTaskStatus(Enum):
    COMPLETED_VERIFIED = "completed_verified"
    COMPLETED_UNVERIFIED = "completed_unverified"
    COMPLETED_READ_ONLY = "completed_read_only"
    REJECTED = "rejected"
    FAILED_BEFORE_MUTATION = "failed_before_mutation"
    MUTATED_VERIFICATION_FAILED = "mutated_verification_failed"
    MUTATED_TASK_FAILED = "mutated_task_failed"
    COMPLETED_REPAIRED_VERIFIED = "completed_repaired_verified"
    REPAIR_REJECTED = "repair_rejected"
    REPAIR_UNVERIFIED = "repair_unverified"
    REPAIR_VERIFICATION_FAILED = "repair_verification_failed"
    REPAIR_FAILED = "repair_failed"


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
    outcome: str | None = None


@dataclass(frozen=True, slots=True)
class MutationRecord:
    tool: str
    path: str | None
    old_sha256: str | None
    new_sha256: str | None
    generation: int


@dataclass(frozen=True, slots=True)
class MutationCandidate:
    path: str
    sha256: str
    generation: int
    observation_id: str
    start_line: int | None = None
    end_line: int | None = None
    targeted_reread_available: bool = False


@dataclass(frozen=True, slots=True)
class MutationTransitionMetrics:
    entries: int = 0
    model_calls: int = 0
    proposals: int = 0
    premature_finals: int = 0
    post_ready_discovery_attempts: int = 0
    post_ready_discovery_executions: int = 0
    targeted_rereads: int = 0
    write_approvals: int = 0
    successful_mutations: int = 0
    tools_before_ready: int | None = None
    tools_before_proposal: int | None = None
    invalidations: int = 0


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
    repair_enabled: bool = False
    repair_eligible: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_eligibility_outcome: str | None = None
    mutations: tuple[MutationRecord, ...] = ()
    build_attempts: tuple[VerificationRecord, ...] = ()
    test_attempts: tuple[VerificationRecord, ...] = ()
    transition_metrics: MutationTransitionMetrics = MutationTransitionMetrics()

    @property
    def footer(self) -> str:
        if self.repair_enabled:
            initial = _attempt_label(self.build_attempts, self.test_attempts, 0)
            repair = _attempt_label(self.build_attempts, self.test_attempts, 1)
            initial_change = "applied" if self.mutation_count >= 1 else "not applied"
            repair_change = "applied" if self.mutation_count >= 2 else "not applied"
            return (
                f"Change #1: {initial_change}\n"
                f"Initial verification: {initial}\n"
                f"Repair #2: {repair_change}\n"
                f"Repair verification: {repair}\n"
                f"Status: {self.status.value.upper()}"
            )
        change = "applied" if self.mutation_count else "not applied"
        return (
            f"Change: {change}\n"
            f"Build: {self.build.status}\n"
            f"Tests: {self.test.status}\n"
            f"Status: {self.status.value.upper()}"
        )


class CodingTaskState:
    """Small mutable transition authority scoped to one user request."""

    def __init__(
        self,
        generation: int,
        *,
        repair_enabled: bool = False,
        transition_required: bool = True,
    ) -> None:
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
        self.repair_enabled = repair_enabled
        self.repair_eligible = False
        self.repair_attempted = False
        self.repair_eligibility_outcome: str | None = None
        self.mutations: list[MutationRecord] = []
        self.build_attempts: list[VerificationRecord] = []
        self.test_attempts: list[VerificationRecord] = []
        self._terminal_status: CodingTaskStatus | None = None
        self.mutation_candidates: list[MutationCandidate] = []
        self.transition_metrics = MutationTransitionMetrics()
        self._mutation_ready_correction_used = False
        self.transition_required = transition_required

    @property
    def terminal(self) -> bool:
        return self._terminal_status is not None

    @property
    def may_propose_mutation(self) -> bool:
        return not self.terminal and (
            self.phase is CodingTaskPhase.MUTATION_READY
            or (
                not self.transition_required
                and self.phase is CodingTaskPhase.INSPECTING
            )
            or (
                self.repair_enabled
                and self.repair_eligible
                and not self.repair_attempted
                and self.mutation_count == 1
                and self.phase is CodingTaskPhase.DIAGNOSING
            )
        )

    @property
    def mutation_ready(self) -> bool:
        return self.phase is CodingTaskPhase.MUTATION_READY and not self.terminal

    @property
    def inspecting(self) -> bool:
        return self.phase is CodingTaskPhase.INSPECTING and not self.terminal

    @property
    def mutation_candidate_paths(self) -> tuple[str, ...]:
        return tuple(candidate.path for candidate in self.mutation_candidates)

    def consider_source(
        self,
        path: str,
        sha256: str,
        generation: int,
        observation_id: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
        targeted_reread_available: bool = False,
    ) -> None:
        if self.terminal or self.mutation_count or generation != self.generation:
            return
        if not self.transition_required:
            return
        candidate = MutationCandidate(
            path,
            sha256,
            generation,
            observation_id,
            start_line,
            end_line,
            targeted_reread_available,
        )
        self.mutation_candidates = [
            item for item in self.mutation_candidates if item.path != path
        ]
        self.mutation_candidates.append(candidate)
        self.mutation_candidates = self.mutation_candidates[-4:]

    def enter_mutation_ready(self, tool_count: int | None = None) -> bool:
        if (
            self.terminal
            or self.mutation_count
            or not self.mutation_candidates
            or self.phase is not CodingTaskPhase.INSPECTING
        ):
            return False
        self.phase = CodingTaskPhase.MUTATION_READY
        self.transition_metrics = _transition_replace(
            self.transition_metrics,
            entries=self.transition_metrics.entries + 1,
            tools_before_ready=(
                len(self.tool_sequence) if tool_count is None else tool_count
            ),
        )
        return True

    def note_ready_model_call(self) -> None:
        self.transition_metrics = _transition_replace(
            self.transition_metrics,
            model_calls=self.transition_metrics.model_calls + 1,
        )

    def note_post_ready_discovery(self) -> bool:
        self.transition_metrics = _transition_replace(
            self.transition_metrics,
            post_ready_discovery_attempts=(
                self.transition_metrics.post_ready_discovery_attempts + 1
            ),
        )
        if self._mutation_ready_correction_used:
            return False
        self._mutation_ready_correction_used = True
        return True

    def note_premature_final(self) -> bool:
        self.transition_metrics = _transition_replace(
            self.transition_metrics,
            premature_finals=self.transition_metrics.premature_finals + 1,
        )
        if self._mutation_ready_correction_used:
            return False
        self._mutation_ready_correction_used = True
        return True

    def use_targeted_reread(self) -> bool:
        candidate = self.mutation_candidates[-1] if self.mutation_candidates else None
        if candidate is None or not candidate.targeted_reread_available:
            return False
        self.mutation_candidates[-1] = MutationCandidate(
            candidate.path,
            candidate.sha256,
            candidate.generation,
            candidate.observation_id,
            candidate.start_line,
            candidate.end_line,
            False,
        )
        self.transition_metrics = _transition_replace(
            self.transition_metrics,
            targeted_rereads=self.transition_metrics.targeted_rereads + 1,
        )
        return True

    def invalidate_mutation_ready(self, generation: int | None = None) -> None:
        if self.mutation_count:
            return
        self.mutation_candidates.clear()
        self.phase = CodingTaskPhase.INSPECTING
        if generation is not None:
            self.generation = generation
        self._mutation_ready_correction_used = False
        self.transition_metrics = _transition_replace(
            self.transition_metrics,
            invalidations=self.transition_metrics.invalidations + 1,
        )

    def mutation_blocked_by_policy(self) -> None:
        if self.mutation_count or self.terminal:
            return
        self.phase = CodingTaskPhase.FAILED
        self._terminal_status = CodingTaskStatus.FAILED_BEFORE_MUTATION

    @property
    def may_verify(self) -> bool:
        return not self.terminal

    def may_verify_operation(self, operation: str) -> bool:
        return self.may_verify and (
            not self.repair_enabled or len(self._attempts(operation)) < 2
        )

    def record_tool(self, name: str) -> None:
        self.tool_sequence.append(name)

    def mutation_proposed(self, tool_count: int | None = None) -> bool:
        if not self.may_propose_mutation:
            self.fail_after_mutation()
            return False
        if self.mutation_count == 1:
            self.repair_attempted = True
            self.phase = CodingTaskPhase.AWAITING_REPAIR_APPROVAL
        else:
            self.phase = CodingTaskPhase.AWAITING_MUTATION_APPROVAL
            self.transition_metrics = _transition_replace(
                self.transition_metrics,
                proposals=self.transition_metrics.proposals + 1,
                tools_before_proposal=(
                    max(0, len(self.tool_sequence) - 1)
                    if tool_count is None
                    else tool_count
                ),
            )
        return True

    def creation_proposed(self) -> bool:
        if self.terminal or self.mutation_count or not self.inspecting:
            return False
        self.phase = CodingTaskPhase.AWAITING_MUTATION_APPROVAL
        return True

    def mutation_rejected(self) -> None:
        if self.terminal:
            return
        self.phase = CodingTaskPhase.REJECTED
        self._terminal_status = (
            CodingTaskStatus.REPAIR_REJECTED
            if self.mutation_count == 1 and self.repair_enabled
            else CodingTaskStatus.REJECTED
        )

    def note_write_approval(self) -> None:
        self.transition_metrics = _transition_replace(
            self.transition_metrics,
            write_approvals=self.transition_metrics.write_approvals + 1,
        )

    def mutation_failed(self) -> None:
        if self.terminal:
            return
        if self.mutation_count:
            self.phase = CodingTaskPhase.FAILED
            self._terminal_status = (
                CodingTaskStatus.REPAIR_FAILED
                if self.repair_enabled and self.repair_attempted
                else CodingTaskStatus.MUTATED_TASK_FAILED
            )
            return
        self.phase = CodingTaskPhase.FAILED
        self._terminal_status = CodingTaskStatus.FAILED_BEFORE_MUTATION

    def mutation_succeeded(
        self, tool: str, output: Mapping[str, object], generation: int
    ) -> None:
        if self.mutation_count >= (2 if self.repair_enabled else 1):
            raise RuntimeError("coding task mutation limit exceeded")
        self.mutation_count += 1
        self.mutation_tool = tool
        path = output.get("path")
        if isinstance(path, str):
            self.changed_files.append(path)
        old_hash = output.get("old_sha256")
        new_hash = output.get("new_sha256")
        self.old_sha256 = old_hash if isinstance(old_hash, str) else None
        self.new_sha256 = new_hash if isinstance(new_hash, str) else None
        self.mutations.append(
            MutationRecord(
                tool,
                path if isinstance(path, str) else None,
                self.old_sha256,
                self.new_sha256,
                generation,
            )
        )
        self.generation = generation
        self.build = _stale(self.build)
        self.test = _stale(self.test)
        self.verification_decision = VerificationDecision.NOT_DECIDED
        self.repair_eligible = False
        self.phase = (
            CodingTaskPhase.REPAIRED
            if self.mutation_count == 2
            else CodingTaskPhase.MUTATED
        )
        self.mutation_candidates.clear()
        self.transition_metrics = _transition_replace(
            self.transition_metrics,
            successful_mutations=self.transition_metrics.successful_mutations + 1,
        )

    def verification_requested(self, operation: str) -> bool:
        if not self.may_verify_operation(operation):
            self.fail_after_mutation()
            return False
        record = self._verification(operation)
        if record.attempted and record.generation == self.generation:
            self.fail_after_mutation()
            return False
        self.phase = (
            CodingTaskPhase.VERIFYING_REPAIR
            if self.mutation_count == 2
            else CodingTaskPhase.VERIFYING
        )
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
            generation=(
                self.generation
                if status != "approval_required" and outcome != "command_not_configured"
                else None
            ),
            outcome=outcome if isinstance(outcome, str) else None,
        )
        self._attempts(operation).append(record)
        if operation == "build":
            self.build = record
        else:
            self.test = record
        if label == "failed":
            self.verification_decision = VerificationDecision.COMPLETED
            eligible = outcome in {"nonzero_exit", "timeout"}
            if self.repair_enabled and self.mutation_count == 1 and eligible:
                self.repair_eligible = True
                self.repair_eligibility_outcome = outcome
                self.phase = CodingTaskPhase.DIAGNOSING
            else:
                self.phase = CodingTaskPhase.FAILED
                self._terminal_status = (
                    CodingTaskStatus.REPAIR_VERIFICATION_FAILED
                    if self.mutation_count == 2 and self.repair_enabled
                    else CodingTaskStatus.MUTATED_VERIFICATION_FAILED
                    if self.mutation_count
                    else CodingTaskStatus.FAILED_BEFORE_MUTATION
                )
        elif label == "not_run":
            self.verification_decision = VerificationDecision.DECLINED
            self.phase = CodingTaskPhase.COMPLETED
            self._terminal_status = (
                CodingTaskStatus.REPAIR_UNVERIFIED
                if self.mutation_count == 2
                else CodingTaskStatus.COMPLETED_UNVERIFIED
            )
        else:
            self.verification_decision = VerificationDecision.COMPLETED
            self.phase = (
                CodingTaskPhase.REPAIRED
                if self.mutation_count == 2
                else CodingTaskPhase.MUTATED
                if self.mutation_count == 1
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
                status = (
                    CodingTaskStatus.COMPLETED_REPAIRED_VERIFIED
                    if self.mutation_count == 2
                    else CodingTaskStatus.COMPLETED_VERIFIED
                )
            elif not self.mutation_count:
                status = CodingTaskStatus.COMPLETED_READ_ONLY
            else:
                status = (
                    CodingTaskStatus.REPAIR_UNVERIFIED
                    if self.mutation_count == 2
                    else CodingTaskStatus.MUTATED_VERIFICATION_FAILED
                    if self.repair_eligible
                    else CodingTaskStatus.COMPLETED_UNVERIFIED
                )
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
            self.repair_enabled,
            self.repair_eligible,
            self.repair_attempted,
            self.mutation_count == 2,
            self.repair_eligibility_outcome,
            tuple(self.mutations),
            tuple(self.build_attempts),
            tuple(self.test_attempts),
            self.transition_metrics,
        )

    def _verification(self, operation: str) -> VerificationRecord:
        if operation == "build":
            return self.build
        if operation == "test":
            return self.test
        raise ValueError("verification operation must be build or test")

    def _attempts(self, operation: str) -> list[VerificationRecord]:
        if operation == "build":
            return self.build_attempts
        if operation == "test":
            return self.test_attempts
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
        record.outcome,
    )


def _transition_replace(
    metrics: MutationTransitionMetrics, **changes: object
) -> MutationTransitionMetrics:
    values = {
        field: getattr(metrics, field)
        for field in MutationTransitionMetrics.__dataclass_fields__
    }
    values.update(changes)
    return MutationTransitionMetrics(**values)  # type: ignore[arg-type]


def _attempt_label(
    builds: tuple[VerificationRecord, ...],
    tests: tuple[VerificationRecord, ...],
    index: int,
) -> str:
    records = tuple(record for record in (*builds, *tests) if record.attempted)
    return records[index].status if len(records) > index else "not_run"
