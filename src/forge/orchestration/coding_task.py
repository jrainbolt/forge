"""Explicit state and externally truthful results for one A11 coding task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from forge.orchestration.verification_attribution import (
    AttributionResult,
    VerificationAttribution,
    VerificationBaselineEvidence,
    VerificationPlanBaseline,
)


class CodingTaskPhase(Enum):
    INSPECTING = "inspecting"
    MUTATION_READY = "mutation_ready"
    AWAITING_MUTATION_APPROVAL = "awaiting_mutation_approval"
    MUTATED = "mutated"
    VERIFICATION_READY = "verification_ready"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    DIAGNOSING = "diagnosing"
    REPAIR_READY = "repair_ready"
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
    MUTATED_VERIFICATION_BLOCKED_BY_BASELINE_FAILURE = (
        "mutated_verification_blocked_by_baseline_failure"
    )
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
    duration_seconds: float = 0.0
    execution_isolation_mode: str = "none"
    execution_sandbox_adapter: str = "none"
    execution_sandbox_available: bool = False
    execution_environment_hardened: bool = False
    execution_home_redirected: bool = False
    execution_tmp_redirected: bool = False
    execution_isolation_failure: bool = False
    strict_policy_version: str = "none"
    strict_capabilities: tuple[str, ...] = ()
    strict_failure_class: str = "none"


@dataclass(frozen=True, slots=True)
class VerificationPlanRun:
    plan_id: str
    steps: tuple[tuple[str, VerificationRecord], ...]
    outcome: str
    failed_step: str | None
    generation: int
    required_steps: int

    @property
    def completed_steps(self) -> int:
        return sum(record.status == "passed" for _, record in self.steps)

    @property
    def duration_seconds(self) -> float:
        return sum(record.duration_seconds for _, record in self.steps)


@dataclass(frozen=True, slots=True)
class MutationRecord:
    tool: str
    path: str | None
    old_sha256: str | None
    new_sha256: str | None
    generation: int
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True, slots=True)
class RepairEvidence:
    verification_observation_id: str
    source_observation_id: str
    path: str
    generation: int
    mutation_index: int


@dataclass(frozen=True, slots=True)
class RepairGroundingMetrics:
    diagnosis_entries: int = 0
    diagnostics_registered: int = 0
    source_refreshes: int = 0
    ready_entries: int = 0
    proposals: int = 0
    previews: int = 0
    mutations: int = 0
    reverification_executed: int = 0
    reverification_result: str = "not_run"


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
class StructuredMutationMetrics:
    mutation_representation: str = "exact_text"
    attempts: int = 0
    mutation_intent_requests: int = 0
    structured_edit_attempts: int = 0
    no_op_edit_attempts: int = 0
    no_op_corrections: int = 0
    no_op_correction_successes: int = 0
    non_noop_proposals: int = 0
    materialized_deltas: int = 0
    valid: int = 0
    old_text_misses: int = 0
    ambiguous_old_text: int = 0
    stale_proposals: int = 0
    corrections: int = 0
    correction_successes: int = 0
    materialized_previews: int = 0
    approved_previews: int = 0
    line_range_attempts: int = 0
    line_range_valid: int = 0
    line_range_target_valid: int = 0
    line_range_materialized: int = 0
    line_range_corrections: int = 0
    line_range_preview_created: int = 0


@dataclass(frozen=True, slots=True)
class VerificationGateMetrics:
    ready_entries: int = 0
    required: bool = False
    provider: str | None = None
    permission: str | None = None
    approval_requested: int = 0
    approved: int = 0
    executed: int = 0
    result: str = "not_run"
    verification_tools: int = 0
    post_mutation_reads_before_verification: int = 0
    skipped: bool = False


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
    structured_mutation_metrics: StructuredMutationMetrics = StructuredMutationMetrics()
    verification_gate_metrics: VerificationGateMetrics = VerificationGateMetrics()
    repair_evidence: RepairEvidence | None = None
    repair_grounding_metrics: RepairGroundingMetrics = RepairGroundingMetrics()
    verification_baseline: VerificationBaselineEvidence | None = None
    verification_attribution: VerificationAttribution = VerificationAttribution()
    attribution_attempts: tuple[VerificationAttribution, ...] = ()
    verification_plan_runs: tuple[VerificationPlanRun, ...] = ()
    verification_plan_baseline: VerificationPlanBaseline | None = None
    configure: VerificationRecord = VerificationRecord()
    configure_attempts: tuple[VerificationRecord, ...] = ()

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
        mutation_representation: str = "exact_text",
    ) -> None:
        if mutation_representation not in {"exact_text", "line_range"}:
            raise ValueError("unsupported mutation representation")
        self.phase = CodingTaskPhase.INSPECTING
        self.mutation_count = 0
        self.mutation_tool: str | None = None
        self.changed_files: list[str] = []
        self.old_sha256: str | None = None
        self.new_sha256: str | None = None
        self.generation = generation
        self.build = VerificationRecord()
        self.test = VerificationRecord()
        self.configure = VerificationRecord()
        self.verification_decision = VerificationDecision.NOT_DECIDED
        self.tool_sequence: list[str] = []
        self.repair_enabled = repair_enabled
        self.repair_eligible = False
        self.repair_attempted = False
        self.repair_eligibility_outcome: str | None = None
        self.mutations: list[MutationRecord] = []
        self.build_attempts: list[VerificationRecord] = []
        self.test_attempts: list[VerificationRecord] = []
        self.configure_attempts: list[VerificationRecord] = []
        self._terminal_status: CodingTaskStatus | None = None
        self.mutation_candidates: list[MutationCandidate] = []
        self.transition_metrics = MutationTransitionMetrics()
        self.structured_mutation_metrics = StructuredMutationMetrics(
            mutation_representation=mutation_representation
        )
        self.verification_gate_metrics = VerificationGateMetrics()
        self.repair_evidence: RepairEvidence | None = None
        self.repair_grounding_metrics = RepairGroundingMetrics()
        self.verification_baseline: VerificationBaselineEvidence | None = None
        self.verification_attribution = VerificationAttribution()
        self.attribution_attempts: list[VerificationAttribution] = []
        self.verification_plan_runs: list[VerificationPlanRun] = []
        self.verification_plan_baseline: VerificationPlanBaseline | None = None
        self._plan_id: str | None = None
        self._plan_required_steps = 0
        self._plan_steps: list[tuple[str, VerificationRecord]] = []
        self._pending_mutation_range: tuple[int, int] | None = None
        self._repair_diagnostic_id: str | None = None
        self._mutation_ready_correction_used = False
        self._structured_edit_correction_used = False
        self._structured_edit_awaiting_correction = False
        self._no_op_awaiting_correction = False
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
                and self.phase is CodingTaskPhase.REPAIR_READY
            )
        )

    @property
    def mutation_ready(self) -> bool:
        return self.phase is CodingTaskPhase.MUTATION_READY and not self.terminal

    @property
    def repair_ready(self) -> bool:
        return self.phase is CodingTaskPhase.REPAIR_READY and not self.terminal

    @property
    def structured_edit_ready(self) -> bool:
        return self.mutation_ready or (
            not self.terminal
            and self.phase is CodingTaskPhase.REPAIR_READY
            and self.repair_enabled
            and self.repair_eligible
            and not self.repair_attempted
            and bool(self.mutation_candidates)
        )

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
        repair_source = (
            self.mutation_count == 1
            and self.repair_enabled
            and self.repair_eligible
            and self.phase is CodingTaskPhase.DIAGNOSING
        )
        if (
            self.terminal
            or (self.mutation_count and not repair_source)
            or generation != self.generation
        ):
            return
        if not self.transition_required and not repair_source:
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

    def note_mutation_intent_request(self) -> None:
        metrics = self.structured_mutation_metrics
        self.structured_mutation_metrics = _structured_replace(
            metrics,
            mutation_intent_requests=metrics.mutation_intent_requests + 1,
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
        self._structured_edit_awaiting_correction = False
        self._no_op_awaiting_correction = False
        self.transition_metrics = _transition_replace(
            self.transition_metrics,
            invalidations=self.transition_metrics.invalidations + 1,
        )

    def note_structured_edit(
        self, failure: str | None, *, representation: str = "exact_text"
    ) -> bool:
        """Record validation and return whether processing/correction may continue."""
        metrics = self.structured_mutation_metrics
        changes: dict[str, int] = {
            "attempts": metrics.attempts + 1,
            "structured_edit_attempts": metrics.structured_edit_attempts + 1,
        }
        if representation == "line_range":
            changes["line_range_attempts"] = metrics.line_range_attempts + 1
            target_valid = failure not in {
                "path_not_eligible",
                "stale_source",
                "out_of_range",
            }
            if target_valid:
                changes["line_range_target_valid"] = metrics.line_range_target_valid + 1
            if failure is None:
                changes["line_range_valid"] = metrics.line_range_valid + 1
                changes["line_range_materialized"] = metrics.line_range_materialized + 1
        no_op = failure in {"no_op_edit", "materialized_no_delta"}
        if no_op:
            changes["no_op_edit_attempts"] = metrics.no_op_edit_attempts + 1
        else:
            changes["non_noop_proposals"] = metrics.non_noop_proposals + 1
        if failure is None:
            changes["valid"] = metrics.valid + 1
            changes["materialized_deltas"] = metrics.materialized_deltas + 1
            if self._structured_edit_awaiting_correction:
                changes["correction_successes"] = metrics.correction_successes + 1
            if self._no_op_awaiting_correction:
                changes["no_op_correction_successes"] = (
                    metrics.no_op_correction_successes + 1
                )
            self._structured_edit_awaiting_correction = False
            self._no_op_awaiting_correction = False
            self.structured_mutation_metrics = _structured_replace(metrics, **changes)
            return True
        metric = {
            "old_text_not_found": "old_text_misses",
            "old_text_ambiguous": "ambiguous_old_text",
            "stale_source": "stale_proposals",
        }.get(failure)
        if metric is not None:
            changes[metric] = getattr(metrics, metric) + 1
        if failure == "stale_source" or self._structured_edit_correction_used:
            self.structured_mutation_metrics = _structured_replace(metrics, **changes)
            return False
        self._structured_edit_correction_used = True
        self._structured_edit_awaiting_correction = True
        changes["corrections"] = metrics.corrections + 1
        if representation == "line_range":
            changes["line_range_corrections"] = metrics.line_range_corrections + 1
        if no_op:
            self._no_op_awaiting_correction = True
            changes["no_op_corrections"] = metrics.no_op_corrections + 1
        self.structured_mutation_metrics = _structured_replace(metrics, **changes)
        return True

    def note_materialized_preview(self, *, approved: bool) -> None:
        metrics = self.structured_mutation_metrics
        changes: dict[str, int] = {
            "materialized_previews": metrics.materialized_previews + 1,
            "approved_previews": metrics.approved_previews + (1 if approved else 0),
        }
        if metrics.mutation_representation == "line_range":
            changes["line_range_preview_created"] = (
                metrics.line_range_preview_created + 1
            )
        self.structured_mutation_metrics = _structured_replace(
            metrics,
            **changes,
        )
        if self.mutation_count == 1:
            grounding = self.repair_grounding_metrics
            self.repair_grounding_metrics = _repair_grounding_replace(
                grounding, previews=grounding.previews + 1
            )

    def set_pending_mutation_range(self, start_line: int, end_line: int) -> None:
        self._pending_mutation_range = (start_line, end_line)

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
            grounding = self.repair_grounding_metrics
            self.repair_grounding_metrics = _repair_grounding_replace(
                grounding, proposals=grounding.proposals + 1
            )
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
                self._pending_mutation_range[0]
                if self._pending_mutation_range is not None
                else None,
                self._pending_mutation_range[1]
                if self._pending_mutation_range is not None
                else None,
            )
        )
        self._pending_mutation_range = None
        self.generation = generation
        self.build = _stale(self.build)
        self.test = _stale(self.test)
        self.configure = _stale(self.configure)
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
        if self.mutation_count == 2:
            grounding = self.repair_grounding_metrics
            self.repair_grounding_metrics = _repair_grounding_replace(
                grounding, mutations=grounding.mutations + 1
            )

    def register_repair_diagnostic(self, observation_id: str) -> None:
        if not self.repair_eligible or self.phase is not CodingTaskPhase.DIAGNOSING:
            return
        self._repair_diagnostic_id = observation_id
        metrics = self.repair_grounding_metrics
        self.repair_grounding_metrics = _repair_grounding_replace(
            metrics,
            diagnosis_entries=metrics.diagnosis_entries + 1,
            diagnostics_registered=metrics.diagnostics_registered + 1,
        )

    def repair_source_refreshed(
        self, *, observation_id: str, path: str, generation: int
    ) -> bool:
        candidate = next(
            (
                item
                for item in self.mutation_candidates
                if item.path == path and item.generation == generation
            ),
            None,
        )
        if (
            candidate is None
            or self._repair_diagnostic_id is None
            or generation != self.generation
            or self.phase is not CodingTaskPhase.DIAGNOSING
        ):
            return False
        self.repair_evidence = RepairEvidence(
            self._repair_diagnostic_id,
            observation_id,
            path,
            generation,
            self.mutation_count,
        )
        # Primary and repair proposals each receive one bounded structured-edit
        # recovery opportunity; a primary correction cannot consume the repair one.
        self._structured_edit_correction_used = False
        self._structured_edit_awaiting_correction = False
        self._no_op_awaiting_correction = False
        self.phase = CodingTaskPhase.REPAIR_READY
        metrics = self.repair_grounding_metrics
        self.repair_grounding_metrics = _repair_grounding_replace(
            metrics,
            source_refreshes=metrics.source_refreshes + 1,
            ready_entries=metrics.ready_entries + 1,
        )
        return True

    def verification_ready(self, operation: str) -> None:
        self.phase = CodingTaskPhase.VERIFICATION_READY
        metrics = self.verification_gate_metrics
        self.verification_gate_metrics = _verification_gate_replace(
            metrics,
            ready_entries=metrics.ready_entries + 1,
            required=True,
            provider=operation,
        )

    def plan_started(self, plan_id: str, required_steps: int) -> None:
        self._plan_id = plan_id
        self._plan_required_steps = required_steps
        self._plan_steps = []

    def plan_step_finished(self, operation: str) -> None:
        record = self._verification(operation)
        if record.generation != self.generation:
            record = VerificationRecord()
        self._plan_steps.append((f"project.{operation}", record))

    def plan_finished(self, outcome: str, failed_step: str | None = None) -> None:
        assert self._plan_id is not None
        self.verification_plan_runs.append(
            VerificationPlanRun(
                self._plan_id,
                tuple(self._plan_steps),
                outcome,
                failed_step,
                self.generation,
                self._plan_required_steps,
            )
        )
        if (
            outcome != "pass"
            and self._terminal_status is None
            and not self.repair_eligible
        ):
            self.phase = CodingTaskPhase.COMPLETED
            self._terminal_status = (
                CodingTaskStatus.REPAIR_UNVERIFIED
                if self.mutation_count == 2
                else CodingTaskStatus.COMPLETED_UNVERIFIED
            )

    def note_verification_gate(
        self,
        *,
        permission: str,
        approval_requested: bool = False,
        approved: bool = False,
        executed: bool = False,
        result: str = "not_run",
    ) -> None:
        metrics = self.verification_gate_metrics
        self.verification_gate_metrics = _verification_gate_replace(
            metrics,
            permission=permission,
            approval_requested=metrics.approval_requested + int(approval_requested),
            approved=metrics.approved + int(approved),
            executed=metrics.executed + int(executed),
            result=result,
            verification_tools=metrics.verification_tools + int(executed),
        )

    def verification_skipped(self) -> None:
        self.verification_gate_metrics = _verification_gate_replace(
            self.verification_gate_metrics, skipped=True, result="skipped"
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
        *,
        attribution: VerificationAttribution | None = None,
    ) -> None:
        if attribution is not None:
            self.verification_attribution = attribution
            self.attribution_attempts.append(attribution)
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
            duration_seconds=(
                float(output["duration_seconds"])
                if output is not None
                and isinstance(output.get("duration_seconds"), (int, float))
                else 0.0
            ),
            execution_isolation_mode=(
                str(output.get("execution_isolation_mode", "none"))
                if output is not None
                else "none"
            ),
            execution_sandbox_adapter=(
                str(output.get("execution_sandbox_adapter", "none"))
                if output is not None
                else "none"
            ),
            execution_sandbox_available=(
                output.get("execution_sandbox_available") is True
                if output is not None
                else False
            ),
            execution_environment_hardened=(
                output.get("execution_environment_hardened") is True
                if output is not None
                else False
            ),
            execution_home_redirected=(
                output.get("execution_home_redirected") is True
                if output is not None
                else False
            ),
            execution_tmp_redirected=(
                output.get("execution_tmp_redirected") is True
                if output is not None
                else False
            ),
            execution_isolation_failure=(
                output.get("execution_isolation_failure") is True
                if output is not None
                else False
            ),
            strict_policy_version=(
                str(output.get("strict_policy_version", "none"))
                if output is not None
                else "none"
            ),
            strict_capabilities=(
                tuple(output.get("strict_capabilities", ()))
                if output is not None
                and isinstance(output.get("strict_capabilities"), (tuple, list))
                else ()
            ),
            strict_failure_class=(
                str(output.get("strict_failure_class", "none"))
                if output is not None
                else "none"
            ),
        )
        self._attempts(operation).append(record)
        if operation == "configure":
            self.configure = record
        elif operation == "build":
            self.build = record
        else:
            self.test = record
        if label == "failed":
            self.verification_decision = VerificationDecision.COMPLETED
            eligible = operation != "configure" and outcome in {
                "nonzero_exit",
                "timeout",
            }
            preexisting = (
                self.verification_attribution.result
                is AttributionResult.PREEXISTING_OR_UNRELATED
            )
            if (
                self.repair_enabled
                and self.mutation_count == 1
                and eligible
                and not preexisting
            ):
                self.repair_eligible = True
                self.repair_eligibility_outcome = outcome
                self.phase = CodingTaskPhase.DIAGNOSING
            else:
                self.phase = CodingTaskPhase.FAILED
                self._terminal_status = (
                    CodingTaskStatus.REPAIR_VERIFICATION_FAILED
                    if self.mutation_count == 2 and self.repair_enabled
                    else (
                        CodingTaskStatus.MUTATED_VERIFICATION_BLOCKED_BY_BASELINE_FAILURE
                    )
                    if self.mutation_count and preexisting
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
        if self.mutation_count == 2:
            metrics = self.repair_grounding_metrics
            self.repair_grounding_metrics = _repair_grounding_replace(
                metrics,
                reverification_executed=(
                    metrics.reverification_executed + int(record.attempted)
                ),
                reverification_result=label,
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
            if (
                self._plan_id is not None
                and self.verification_plan_runs
                and self.verification_plan_runs[-1].generation == self.generation
                and self.verification_plan_runs[-1].outcome == "pass"
            ) or (
                self._plan_id is None
                and any(
                    record.status == "passed" and record.generation == self.generation
                    for record in (self.build, self.test)
                )
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
            self.structured_mutation_metrics,
            self.verification_gate_metrics,
            self.repair_evidence,
            self.repair_grounding_metrics,
            self.verification_baseline,
            self.verification_attribution,
            tuple(self.attribution_attempts),
            tuple(self.verification_plan_runs),
            self.verification_plan_baseline,
            self.configure,
            tuple(self.configure_attempts),
        )

    def _verification(self, operation: str) -> VerificationRecord:
        if operation == "configure":
            return self.configure
        if operation == "build":
            return self.build
        if operation == "test":
            return self.test
        raise ValueError("verification operation must be configure, build, or test")

    def _attempts(self, operation: str) -> list[VerificationRecord]:
        if operation == "configure":
            return self.configure_attempts
        if operation == "build":
            return self.build_attempts
        if operation == "test":
            return self.test_attempts
        raise ValueError("verification operation must be configure, build, or test")


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


def _structured_replace(
    metrics: StructuredMutationMetrics, **changes: object
) -> StructuredMutationMetrics:
    values = {
        field: getattr(metrics, field)
        for field in StructuredMutationMetrics.__dataclass_fields__
    }
    values.update(changes)
    return StructuredMutationMetrics(**values)


def _verification_gate_replace(
    metrics: VerificationGateMetrics, **changes: object
) -> VerificationGateMetrics:
    values = {
        field: getattr(metrics, field)
        for field in VerificationGateMetrics.__dataclass_fields__
    }
    values.update(changes)
    return VerificationGateMetrics(**values)  # type: ignore[arg-type]


def _repair_grounding_replace(
    metrics: RepairGroundingMetrics, **changes: object
) -> RepairGroundingMetrics:
    values = {
        field: getattr(metrics, field)
        for field in RepairGroundingMetrics.__dataclass_fields__
    }
    values.update(changes)
    return RepairGroundingMetrics(**values)  # type: ignore[arg-type]


def _attempt_label(
    builds: tuple[VerificationRecord, ...],
    tests: tuple[VerificationRecord, ...],
    index: int,
) -> str:
    records = tuple(record for record in (*builds, *tests) if record.attempted)
    return records[index].status if len(records) > index else "not_run"
