"""Reproducible real-repository evaluation on disposable workspaces."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from forge.embeddings import EmbeddingModel
from forge.interaction import AutonomyMode, resolve_interaction_policy
from forge.lexical_index import RepositoryLexicalIndex
from forge.models import (
    GenerationConfig,
    Model,
    ModelError,
    ModelUsage,
    MutationRepresentationPolicy,
)
from forge.orchestration import (
    DuplicateToolCallIdError,
    RepositoryChatSession,
    RepositoryOrchestrationError,
)
from forge.process_isolation import ExecutionIsolationPolicy
from forge.project_config import ProjectCommand, ProjectCommands, VerificationPlan
from forge.repository_index import RepositoryIndex
from forge.semantic_index import SemanticIndex
from forge.tools import (
    MultiFileMutationPreview,
    MutationPreview,
    PreparedProjectCommand,
    ToolInvocation,
    create_repository_registry,
)

REALWORLD_V1 = "realworld-v1"
REALWORLD_SUITE_VERSION = 1
REALWORLD_SCHEMA_VERSION = 1
DEFAULT_SEEDS = (7, 42)
_IGNORED_NAMES = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".forge-exec",
        "__pycache__",
        "build",
        "dist",
    }
)


class RealWorldLevel(Enum):
    REPOSITORY_REASONING = "repository_reasoning"
    SINGLE_CHANGE = "single_change"
    BOUNDED_REPAIR = "bounded_repair"


class RealWorldStatus(Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    UNSUPPORTED = "UNSUPPORTED"


class EvaluationOutcome(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


class RealWorldFailure(Enum):
    MODEL_QUALITY = "MODEL_QUALITY"
    RETRIEVAL = "RETRIEVAL"
    CONTEXT = "CONTEXT"
    PROTOCOL = "PROTOCOL"
    TOOL_LIMIT = "TOOL_LIMIT"
    MUTATION = "MUTATION"
    VERIFICATION = "VERIFICATION"
    REPAIR = "REPAIR"
    CAPABILITY_UNSUPPORTED = "CAPABILITY_UNSUPPORTED"
    INFRASTRUCTURE = "INFRASTRUCTURE"


@dataclass(frozen=True, slots=True)
class SetupReplacement:
    path: str
    expected: str
    replacement: str

    def __post_init__(self) -> None:
        if not self.path or not self.expected or self.expected == self.replacement:
            raise ValueError("setup replacement must be a non-empty exact change")


@dataclass(frozen=True, slots=True)
class RealWorldTask:
    task_id: str
    level: RealWorldLevel
    mode: AutonomyMode
    prompt: str
    expected_files: tuple[str, ...]
    allowed_paths: tuple[str, ...] = ()
    expected_changed_paths: tuple[str, ...] = ()
    required_candidate_paths: tuple[str, ...] = ()
    setup: tuple[SetupReplacement, ...] = ()
    setup_commands: tuple[tuple[str, ...], ...] = ()
    build_command: tuple[str, ...] | None = None
    test_command: tuple[str, ...] | None = None
    verification_plan: VerificationPlan | None = None
    oracle_commands: tuple[tuple[str, ...], ...] = ()
    seeds: tuple[int, ...] = (42,)
    max_mutations: int = 0
    unsupported_reason: str | None = None
    configure_command: tuple[str, ...] | None = None
    execution_isolation: ExecutionIsolationPolicy = ExecutionIsolationPolicy()

    def __post_init__(self) -> None:
        if not self.task_id or not self.prompt:
            raise ValueError("task ID and prompt must be non-empty")
        if not isinstance(self.execution_isolation, ExecutionIsolationPolicy):
            raise TypeError("execution_isolation must be ExecutionIsolationPolicy")
        for name in (
            "expected_files",
            "allowed_paths",
            "expected_changed_paths",
            "required_candidate_paths",
        ):
            values = tuple(getattr(self, name))
            if any(
                not value or Path(value).is_absolute() or ".." in Path(value).parts
                for value in values
            ):
                raise ValueError(f"{name} must contain confined relative paths")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "setup", tuple(self.setup))
        object.__setattr__(
            self, "setup_commands", tuple(tuple(c) for c in self.setup_commands)
        )
        object.__setattr__(
            self, "oracle_commands", tuple(tuple(c) for c in self.oracle_commands)
        )
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.max_mutations < 0:
            raise ValueError("max_mutations must not be negative")
        if self.unsupported_reason is not None and self.mode is not AutonomyMode.AGENT:
            raise ValueError("unsupported multi-file tasks use agent mode")


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    name: str
    identity: str
    language: str
    source_files: int
    test_files: int
    approximate_loc: int
    build_command: tuple[str, ...]
    test_command: tuple[str, ...]
    baseline_outcome: EvaluationOutcome
    baseline_duration_seconds: float


@dataclass(frozen=True, slots=True)
class RealWorldMetrics:
    model_calls: int = 0
    tool_executions: int = 0
    bootstrap_executions: int = 0
    bootstrap_provider: str = "none"
    expected_implementation_acquired: bool = False
    lexical_index_builds: int = 0
    lexical_index_refreshes: int = 0
    lexical_files_retokenized: int = 0
    lexical_index_duration_seconds: float = 0.0
    mutation_ready_reached: bool = False
    tools_before_mutation_ready: int | None = None
    mutation_proposed: bool = False
    tools_before_mutation_proposal: int | None = None
    post_ready_discovery_attempts: int = 0
    targeted_rereads: int = 0
    structured_mutation_attempts: int = 0
    structured_mutation_valid: int = 0
    structured_mutation_corrections: int = 0
    no_op_proposals: int = 0
    mutation_representation: str = "exact_text"
    line_range_attempts: int = 0
    line_range_valid: int = 0
    line_range_target_valid: int = 0
    line_range_materialized: int = 0
    line_range_corrections: int = 0
    line_range_preview_created: int = 0
    mutation_file_count: int = 0
    mutation_group_id: str | None = None
    mutation_group_validation_result: str = "not_run"
    mutation_group_preview_created: int = 0
    mutation_group_apply_result: str = "not_run"
    mutation_group_rollback_attempted: bool = False
    mutation_group_rollback_result: str = "not_run"
    required_candidate_count: int = 0
    required_sources_ready: int = 0
    required_sources_missing: int = 0
    deterministic_source_reads: int = 0
    model_source_reads: int = 0
    source_acquisition_failures: int = 0
    duplicate_tool_call_ids: int = 0
    protocol_corrections: int = 0
    source_acquisition_duration_seconds: float = 0.0
    structured_edit_recovery_used: bool = False
    actual_delta_proposed: bool = False
    preview_created: int = 0
    discovery_calls: int = 0
    source_reads: int = 0
    range_reads: int = 0
    whole_file_reads: int = 0
    files_inspected: tuple[str, ...] = ()
    context_peak_estimate: int = 0
    candidate_suppressions: int = 0
    mutations: int = 0
    verification_attempts: int = 0
    verification_ready_reached: bool = False
    verification_required: bool = False
    verification_provider: str = "none"
    verification_permission: str = "none"
    verification_approval_requested: int = 0
    verification_approved: int = 0
    verification_executed: int = 0
    verification_result: str = "not_run"
    verification_plan_id: str | None = None
    verification_plan_required_steps: int = 0
    verification_plan_steps_started: int = 0
    verification_plan_steps_passed: int = 0
    verification_plan_failed_step: str | None = None
    verification_plan_result: str = "not_run"
    verification_plan_tools: int = 0
    verification_plan_duration: float = 0.0
    verification_plan_model_calls: int = 0
    verification_build_duration: float = 0.0
    verification_test_duration: float = 0.0
    configure_required: bool = False
    configure_executed: bool = False
    configure_result: str = "not_run"
    configure_duration: float = 0.0
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
    baseline_verification_executed: bool = False
    baseline_duration_seconds: float = 0.0
    post_mutation_duration_seconds: float = 0.0
    attribution_result: str = "no_baseline"
    fingerprint_equal: bool | None = None
    extra_tool_executions: int = 0
    extra_elapsed_seconds: float = 0.0
    verification_tools: int = 0
    post_mutation_reads_before_verification: int = 0
    repair_diagnosis_entered: bool = False
    verification_diagnostic_registered: bool = False
    repair_source_refreshed: bool = False
    repair_ready_reached: bool = False
    repair_proposal_emitted: bool = False
    repair_preview_created: bool = False
    repair_mutation_executed: bool = False
    reverification_executed: bool = False
    reverification_result: str = "not_run"
    repair_attempts: int = 0
    post_coverage_tools: int = 0


@dataclass(frozen=True, slots=True)
class RealWorldTaskResult:
    task_id: str
    mode: str
    level: str
    seed: int
    status: RealWorldStatus
    infrastructure: EvaluationOutcome
    model: EvaluationOutcome
    oracle: EvaluationOutcome
    failure: RealWorldFailure | None
    failure_message: str | None
    final_status: str
    expected_files_found: tuple[str, ...]
    changed_paths: tuple[str, ...]
    unexpected_paths: tuple[str, ...]
    before_hashes: tuple[tuple[str, str], ...]
    after_hashes: tuple[tuple[str, str], ...]
    metrics: RealWorldMetrics
    usage: ModelUsage
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RealWorldSummary:
    runs: int
    passed: int
    partial: int
    failed: int
    unsupported: int
    read_only_completion_rate: float
    coding_completion_rate: float
    verification_pass_rate: float
    repair_success_rate: float
    grounded_answer_rate: float
    mean_tool_executions: float
    mean_source_reads: float
    mean_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RealWorldRun:
    suite: str
    suite_version: int
    schema_version: int
    model_profile: str
    repository: RepositorySnapshot
    results: tuple[RealWorldTaskResult, ...]
    summary: RealWorldSummary
    canonical_unchanged: bool


class TaskSetupError(RuntimeError):
    """A benchmark setup precondition did not match its declared baseline."""


class RealWorldEvaluationRunner:
    """Run production orchestration in fresh copies and score it independently."""

    def __init__(
        self,
        model_profile: str,
        model: Model,
        repository: Path,
        *,
        embedding_model: EmbeddingModel | None = None,
        mutation_representation: MutationRepresentationPolicy = (
            MutationRepresentationPolicy.EXACT_TEXT
        ),
        verification_baseline: bool = False,
    ) -> None:
        self._profile = model_profile
        self._model = model
        self._repository = repository.resolve(strict=True)
        self._embedding_model = embedding_model
        self._mutation_representation = mutation_representation
        self._verification_baseline = verification_baseline

    def run(
        self, tasks: Iterable[RealWorldTask], repository: RepositorySnapshot
    ) -> RealWorldRun:
        task_values = tuple(tasks)
        canonical_before = hash_workspace(self._repository)
        results: list[RealWorldTaskResult] = []
        for task in task_values:
            for seed in task.seeds:
                results.append(self._run_task(task, seed))
        canonical_unchanged = canonical_before == hash_workspace(self._repository)
        if not canonical_unchanged:
            raise RuntimeError(
                "canonical benchmark repository changed during evaluation"
            )
        values = tuple(results)
        return RealWorldRun(
            REALWORLD_V1,
            REALWORLD_SUITE_VERSION,
            REALWORLD_SCHEMA_VERSION,
            self._profile,
            repository,
            values,
            summarize_results(values),
            True,
        )

    def _run_task(self, task: RealWorldTask, seed: int) -> RealWorldTaskResult:
        started = time.perf_counter()
        if task.unsupported_reason is not None:
            return _unsupported_result(task, seed, started)
        with tempfile.TemporaryDirectory(prefix="forge-realworld-") as name:
            workspace = copy_repository(self._repository, Path(name) / "workspace")
            try:
                apply_task_setup(workspace, task.setup)
                if run_oracle(workspace, task.setup_commands) is EvaluationOutcome.FAIL:
                    raise TaskSetupError("evaluator-owned setup command failed")
                before = hash_workspace(workspace)
            except (OSError, TaskSetupError) as error:
                return _failed_result(
                    task, seed, started, RealWorldFailure.INFRASTRUCTURE, str(error)
                )
            commands = _project_commands(task)
            approvals = ExpectedApproval(task, workspace, commands)
            policy = resolve_interaction_policy(task.mode)
            index = RepositoryIndex(workspace)
            lexical_index = RepositoryLexicalIndex(
                workspace, cache_root=Path(name) / "cache"
            )
            semantic_index = None
            if self._embedding_model is not None:
                semantic_index = SemanticIndex(
                    workspace,
                    self._embedding_model,
                    cache_root=Path(name) / "cache",
                )
                semantic_index.build()
            activity: list[object] = []
            session = RepositoryChatSession(
                self._profile,
                self._model,
                workspace,
                mode=task.mode,
                generation=GenerationConfig(max_tokens=512, temperature=0.0, seed=seed),
                registry=create_repository_registry(
                    policy, commands, index, semantic_index, lexical_index
                ),
                interaction_policy=policy,
                approval_callback=approvals,
                repository_index=index,
                semantic_index=semantic_index,
                lexical_index=lexical_index,
                require_relevant_source=False,
                require_mutation_relevance=True,
                activity_callback=activity.append,
                mutation_representation=self._mutation_representation,
                verification_baseline=self._verification_baseline,
                verification_plan=commands.verification_plan,
                minimum_source_files=max(1, len(task.expected_changed_paths)),
                required_candidate_paths=task.required_candidate_paths,
            )
            response = None
            coding_result = None
            failure = None
            message = None
            try:
                response = session.ask(task.prompt)
            except (
                Exception
            ) as error:  # isolation: a model failure cannot abort a suite
                failure = classify_realworld_failure(error)
                message = str(error)[:1_000]
                coding_result = session.last_coding_task
            after = hash_workspace(workspace)
            changed = changed_paths(before, after)
            unexpected = tuple(sorted(set(changed) - set(task.allowed_paths)))
            oracle = run_oracle(workspace, task.oracle_commands)
            return score_task_result(
                task,
                seed,
                response,
                failure,
                message,
                before,
                after,
                changed,
                unexpected,
                oracle,
                approvals,
                time.perf_counter() - started,
                tuple(activity),
                lexical_index,
                coding_result,
            )


class ExpectedApproval:
    """Approve only task-declared paths and exact evaluator-owned commands."""

    def __init__(self, task: RealWorldTask, workspace: Path, commands: ProjectCommands):
        self.task = task
        self.workspace = workspace.resolve()
        self.commands = commands
        self.approved = 0
        self.rejected = 0

    def __call__(
        self,
        _invocation: ToolInvocation,
        preview: MutationPreview | MultiFileMutationPreview | PreparedProjectCommand,
    ) -> bool:
        approved = False
        if isinstance(preview, MultiFileMutationPreview):
            approved = set(preview.paths).issubset(self.task.allowed_paths)
        elif isinstance(preview, MutationPreview):
            approved = preview.path in self.task.allowed_paths
        elif isinstance(preview, PreparedProjectCommand):
            configured = getattr(self.commands, preview.operation, None)
            approved = (
                configured is not None
                and preview.workspace.resolve() == self.workspace
                and preview.argv == configured.argv
                and preview.timeout_seconds == configured.timeout_seconds
                and preview.isolation == self.commands.execution_isolation
            )
        if approved:
            self.approved += 1
        else:
            self.rejected += 1
        return approved


def copy_repository(source: Path, destination: Path) -> Path:
    source = source.resolve(strict=True)
    return Path(
        shutil.copytree(
            source,
            destination,
            ignore=lambda _path, names: sorted(set(names) & _IGNORED_NAMES),
        )
    )


def apply_task_setup(workspace: Path, changes: Sequence[SetupReplacement]) -> None:
    root = workspace.resolve(strict=True)
    for change in changes:
        path = (root / change.path).resolve()
        if root not in path.parents or not path.is_file():
            raise TaskSetupError(f"setup path unavailable: {change.path}")
        text = path.read_text(encoding="utf-8")
        count = text.count(change.expected)
        if count != 1:
            raise TaskSetupError(
                f"setup precondition for {change.path} matched {count} times"
            )
        path.write_text(
            text.replace(change.expected, change.replacement, 1), encoding="utf-8"
        )


def hash_workspace(workspace: Path) -> tuple[tuple[str, str], ...]:
    root = workspace.resolve(strict=True)
    values = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(
            part in _IGNORED_NAMES for part in path.relative_to(root).parts
        ):
            values.append(
                (
                    path.relative_to(root).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return tuple(values)


def changed_paths(
    before: Sequence[tuple[str, str]], after: Sequence[tuple[str, str]]
) -> tuple[str, ...]:
    left, right = dict(before), dict(after)
    return tuple(
        sorted(
            path for path in set(left) | set(right) if left.get(path) != right.get(path)
        )
    )


def run_oracle(workspace: Path, commands: Sequence[Sequence[str]]) -> EvaluationOutcome:
    if not commands:
        return EvaluationOutcome.NOT_RUN
    environment = {
        "CI": "1",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }
    import os

    environment = {**os.environ, **environment}
    for command in commands:
        try:
            completed = subprocess.run(
                tuple(command),
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=300,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return EvaluationOutcome.FAIL
        if completed.returncode != 0:
            return EvaluationOutcome.FAIL
    return EvaluationOutcome.PASS


def classify_realworld_failure(error: Exception) -> RealWorldFailure:
    from forge.conversation import ContextBudgetError

    if isinstance(error, ContextBudgetError):
        return RealWorldFailure.CONTEXT
    if isinstance(error, ModelError):
        return RealWorldFailure.MODEL_QUALITY
    if isinstance(error, DuplicateToolCallIdError):
        return RealWorldFailure.PROTOCOL
    if isinstance(error, RepositoryOrchestrationError):
        message = str(error).casefold()
        if "protocol" in message or "json" in message:
            return RealWorldFailure.PROTOCOL
        if "limit" in message or "repeated" in message:
            return RealWorldFailure.TOOL_LIMIT
        if "verification" in message:
            return RealWorldFailure.VERIFICATION
        if "repair" in message:
            return RealWorldFailure.REPAIR
        if "evidence" in message or "relevant" in message:
            return RealWorldFailure.RETRIEVAL
    return RealWorldFailure.MODEL_QUALITY


def score_task_result(
    task: RealWorldTask,
    seed: int,
    response: object | None,
    failure: RealWorldFailure | None,
    message: str | None,
    before: tuple[tuple[str, str], ...],
    after: tuple[tuple[str, str], ...],
    changed: tuple[str, ...],
    unexpected: tuple[str, ...],
    oracle: EvaluationOutcome,
    approvals: ExpectedApproval,
    elapsed: float,
    recorded_activity: tuple[object, ...] = (),
    lexical_index: RepositoryLexicalIndex | None = None,
    coding_result: object | None = None,
) -> RealWorldTaskResult:
    activities = tuple(getattr(response, "tool_activity", ())) or recorded_activity
    inspected = tuple(
        dict.fromkeys(a.path for a in activities if a.path and a.status == "success")
    )
    found = tuple(path for path in task.expected_files if path in inspected)
    coding = getattr(response, "coding_task", None) or coding_result
    agent = getattr(response, "agent_task", None)
    mutations = getattr(agent, "mutation_count", getattr(coding, "mutation_count", 0))
    attempts = (
        sum(
            record.attempted
            for record in (
                *getattr(agent or coding, "build_attempts", ()),
                *getattr(agent or coding, "test_attempts", ()),
                *getattr(agent or coding, "configure_attempts", ()),
            )
        )
        if agent is not None or coding is not None
        else 0
    )
    expected_changes = set(changed) == set(task.expected_changed_paths)
    grounded = set(task.expected_files).issubset(found)
    model_pass = response is not None and (
        grounded if task.max_mutations == 0 else expected_changes and not unexpected
    )
    transition = getattr(coding, "transition_metrics", None)
    structured = getattr(coding, "structured_mutation_metrics", None)
    acquisition = getattr(coding, "source_acquisition_metrics", None)
    verification_gate = getattr(coding, "verification_gate_metrics", None)
    repair_grounding = getattr(coding, "repair_grounding_metrics", None)
    oracle_pass = oracle in {EvaluationOutcome.PASS, EvaluationOutcome.NOT_RUN}
    if model_pass and oracle_pass:
        status = RealWorldStatus.PASS
    elif response is not None and (found or changed) and not unexpected:
        status = RealWorldStatus.PARTIAL
    else:
        status = RealWorldStatus.FAIL
    if failure is None and unexpected:
        failure = RealWorldFailure.MUTATION
    elif failure is None and getattr(transition, "proposals", 0) > 0 and mutations == 0:
        failure = RealWorldFailure.MODEL_QUALITY
    elif failure is None and not oracle_pass:
        failure = (
            RealWorldFailure.VERIFICATION
            if attempts
            else RealWorldFailure.MODEL_QUALITY
        )
    elif failure is None and not model_pass:
        failure = (
            RealWorldFailure.RETRIEVAL
            if not grounded
            else RealWorldFailure.MODEL_QUALITY
        )
    context = getattr(response, "context_metrics", None)
    retrieval = getattr(response, "retrieval_metrics", None)
    primary_attribution = next(iter(getattr(coding, "attribution_attempts", ())), None)
    plan_runs = getattr(coding, "verification_plan_runs", ())
    current_plan = plan_runs[-1] if plan_runs else None
    plan_baseline = getattr(coding, "verification_plan_baseline", None)
    plan_baseline_steps = getattr(plan_baseline, "steps", ())
    baseline_duration = (
        sum(step.duration_seconds for step in plan_baseline_steps)
        if plan_baseline is not None
        else getattr(
            getattr(coding, "verification_baseline", None), "duration_seconds", 0.0
        )
    )
    metrics = RealWorldMetrics(
        model_calls=(
            getattr(agent, "model_calls", getattr(response, "orchestration_steps", 0))
            if response is not None
            else len(activities)
        ),
        tool_executions=len(activities),
        bootstrap_executions=getattr(
            getattr(response, "bootstrap_metrics", None), "executions", 0
        ),
        bootstrap_provider=next(
            (
                "semantic"
                if activity.tool_name == "repository.semantic_search"
                else "lexical"
                for activity in activities
                if getattr(activity, "invocation_id", "").startswith("forge-bootstrap")
            ),
            "none",
        ),
        expected_implementation_acquired=grounded,
        lexical_index_builds=getattr(lexical_index, "builds", 0),
        lexical_index_refreshes=getattr(lexical_index, "refreshes", 0),
        lexical_files_retokenized=getattr(lexical_index, "total_files_retokenized", 0),
        lexical_index_duration_seconds=getattr(
            lexical_index, "total_duration_seconds", 0.0
        ),
        mutation_ready_reached=getattr(transition, "entries", 0) > 0,
        tools_before_mutation_ready=getattr(transition, "tools_before_ready", None),
        mutation_proposed=getattr(transition, "proposals", 0) > 0,
        tools_before_mutation_proposal=getattr(
            transition, "tools_before_proposal", None
        ),
        post_ready_discovery_attempts=getattr(
            transition, "post_ready_discovery_attempts", 0
        ),
        targeted_rereads=getattr(transition, "targeted_rereads", 0),
        structured_mutation_attempts=getattr(structured, "attempts", 0),
        structured_mutation_valid=getattr(structured, "valid", 0),
        structured_mutation_corrections=getattr(structured, "corrections", 0),
        no_op_proposals=getattr(structured, "no_op_edit_attempts", 0),
        mutation_representation=getattr(
            structured, "mutation_representation", "exact_text"
        ),
        line_range_attempts=getattr(structured, "line_range_attempts", 0),
        line_range_valid=getattr(structured, "line_range_valid", 0),
        line_range_target_valid=getattr(structured, "line_range_target_valid", 0),
        line_range_materialized=getattr(structured, "line_range_materialized", 0),
        line_range_corrections=getattr(structured, "line_range_corrections", 0),
        line_range_preview_created=getattr(structured, "line_range_preview_created", 0),
        mutation_file_count=getattr(structured, "mutation_file_count", 0),
        mutation_group_id=getattr(structured, "mutation_group_id", None),
        mutation_group_validation_result=getattr(
            structured, "mutation_group_validation_result", "not_run"
        ),
        mutation_group_preview_created=getattr(
            structured, "mutation_group_preview_created", 0
        ),
        mutation_group_apply_result=getattr(
            structured, "mutation_group_apply_result", "not_run"
        ),
        mutation_group_rollback_attempted=getattr(
            structured, "mutation_group_rollback_attempted", False
        ),
        mutation_group_rollback_result=getattr(
            structured, "mutation_group_rollback_result", "not_run"
        ),
        required_candidate_count=getattr(acquisition, "required_candidate_count", 0),
        required_sources_ready=getattr(acquisition, "required_sources_ready", 0),
        required_sources_missing=getattr(acquisition, "required_sources_missing", 0),
        deterministic_source_reads=getattr(
            acquisition, "deterministic_source_reads", 0
        ),
        model_source_reads=getattr(acquisition, "model_source_reads", 0),
        source_acquisition_failures=getattr(
            acquisition, "source_acquisition_failures", 0
        ),
        duplicate_tool_call_ids=getattr(acquisition, "duplicate_tool_call_ids", 0),
        protocol_corrections=getattr(acquisition, "protocol_corrections", 0),
        source_acquisition_duration_seconds=getattr(
            acquisition, "acquisition_duration_seconds", 0.0
        ),
        structured_edit_recovery_used=getattr(structured, "corrections", 0) > 0,
        actual_delta_proposed=getattr(structured, "non_noop_proposals", 0) > 0,
        preview_created=getattr(structured, "materialized_previews", 0),
        discovery_calls=sum(
            a.tool_name
            in {
                "repository.search_files",
                "repository.semantic_search",
                "repository.lexical_search",
                "repository.find_symbol",
                "repository.find_references",
            }
            for a in activities
        ),
        source_reads=sum(
            a.tool_name in {"repository.read_file", "repository.read_range"}
            for a in activities
        ),
        range_reads=sum(a.tool_name == "repository.read_range" for a in activities),
        whole_file_reads=sum(a.tool_name == "repository.read_file" for a in activities),
        files_inspected=inspected,
        context_peak_estimate=getattr(context, "estimated_context_peak", 0),
        candidate_suppressions=getattr(retrieval, "candidates_suppressed", 0),
        mutations=mutations,
        verification_attempts=attempts,
        verification_ready_reached=getattr(verification_gate, "ready_entries", 0) > 0,
        verification_required=getattr(verification_gate, "required", False),
        verification_provider=getattr(verification_gate, "provider", None) or "none",
        verification_permission=getattr(verification_gate, "permission", None)
        or "none",
        verification_approval_requested=getattr(
            verification_gate, "approval_requested", 0
        ),
        verification_approved=getattr(verification_gate, "approved", 0),
        verification_executed=getattr(verification_gate, "executed", 0),
        verification_result=getattr(verification_gate, "result", "not_run"),
        verification_plan_id=getattr(current_plan, "plan_id", None),
        verification_plan_required_steps=getattr(current_plan, "required_steps", 0),
        verification_plan_steps_started=len(getattr(current_plan, "steps", ())),
        verification_plan_steps_passed=getattr(current_plan, "completed_steps", 0),
        verification_plan_failed_step=getattr(current_plan, "failed_step", None),
        verification_plan_result=getattr(current_plan, "outcome", "not_run"),
        verification_plan_tools=len(getattr(current_plan, "steps", ())),
        verification_plan_duration=getattr(current_plan, "duration_seconds", 0.0),
        verification_plan_model_calls=0,
        verification_build_duration=getattr(
            getattr(coding, "build", None), "duration_seconds", 0.0
        ),
        verification_test_duration=getattr(
            getattr(coding, "test", None), "duration_seconds", 0.0
        ),
        configure_required=(
            task.verification_plan is not None
            and "project.configure" in task.verification_plan.steps
        ),
        configure_executed=getattr(
            getattr(coding, "configure", None), "attempted", False
        ),
        configure_result=getattr(
            getattr(coding, "configure", None), "status", "not_run"
        ),
        configure_duration=getattr(
            getattr(coding, "configure", None), "duration_seconds", 0.0
        ),
        execution_isolation_mode=getattr(
            getattr(coding, "configure", None),
            "execution_isolation_mode",
            task.execution_isolation.mode.value,
        ),
        execution_sandbox_adapter=getattr(
            getattr(coding, "configure", None), "execution_sandbox_adapter", "none"
        ),
        execution_sandbox_available=getattr(
            getattr(coding, "configure", None), "execution_sandbox_available", False
        ),
        execution_environment_hardened=getattr(
            getattr(coding, "configure", None), "execution_environment_hardened", False
        ),
        execution_home_redirected=getattr(
            getattr(coding, "configure", None), "execution_home_redirected", False
        ),
        execution_tmp_redirected=getattr(
            getattr(coding, "configure", None), "execution_tmp_redirected", False
        ),
        execution_isolation_failure=getattr(
            getattr(coding, "configure", None), "execution_isolation_failure", False
        ),
        strict_policy_version=getattr(
            getattr(coding, "configure", None), "strict_policy_version", "none"
        ),
        strict_capabilities=getattr(
            getattr(coding, "configure", None), "strict_capabilities", ()
        ),
        strict_failure_class=getattr(
            getattr(coding, "configure", None), "strict_failure_class", "none"
        ),
        baseline_verification_executed=(
            any(step.executed for step in plan_baseline_steps)
            if plan_baseline is not None
            else getattr(
                getattr(coding, "verification_baseline", None), "executed", False
            )
        ),
        baseline_duration_seconds=baseline_duration,
        post_mutation_duration_seconds=getattr(
            primary_attribution,
            "post_duration_seconds",
            0.0,
        ),
        attribution_result=getattr(
            getattr(primary_attribution, "result", None),
            "value",
            "no_baseline",
        ),
        fingerprint_equal=getattr(primary_attribution, "fingerprint_equal", None),
        extra_tool_executions=sum(
            getattr(activity, "evidence", "") == "baseline_verification"
            for activity in activities
        ),
        extra_elapsed_seconds=baseline_duration,
        verification_tools=getattr(verification_gate, "verification_tools", 0),
        post_mutation_reads_before_verification=getattr(
            verification_gate, "post_mutation_reads_before_verification", 0
        ),
        repair_diagnosis_entered=getattr(repair_grounding, "diagnosis_entries", 0) > 0,
        verification_diagnostic_registered=getattr(
            repair_grounding, "diagnostics_registered", 0
        )
        > 0,
        repair_source_refreshed=getattr(repair_grounding, "source_refreshes", 0) > 0,
        repair_ready_reached=getattr(repair_grounding, "ready_entries", 0) > 0,
        repair_proposal_emitted=getattr(repair_grounding, "proposals", 0) > 0,
        repair_preview_created=getattr(repair_grounding, "previews", 0) > 0,
        repair_mutation_executed=getattr(repair_grounding, "mutations", 0) > 0,
        reverification_executed=getattr(repair_grounding, "reverification_executed", 0)
        > 0,
        reverification_result=getattr(
            repair_grounding, "reverification_result", "not_run"
        ),
        repair_attempts=int(bool(getattr(agent, "repair_attempted", False))),
        post_coverage_tools=getattr(
            getattr(response, "finalization_metrics", None), "post_coverage_tools", 0
        ),
    )
    final = getattr(
        getattr(coding, "status", None),
        "value",
        "completed" if response is not None else "failed",
    )
    return RealWorldTaskResult(
        task.task_id,
        task.mode.value,
        task.level.value,
        seed,
        status,
        EvaluationOutcome.PASS,
        EvaluationOutcome.PASS if response is not None else EvaluationOutcome.FAIL,
        oracle,
        failure,
        message,
        final,
        found,
        changed,
        unexpected,
        before,
        after,
        metrics,
        getattr(response, "usage", ModelUsage()),
        elapsed,
    )


def summarize_results(results: Sequence[RealWorldTaskResult]) -> RealWorldSummary:
    count = len(results)
    read = [r for r in results if r.level == RealWorldLevel.REPOSITORY_REASONING.value]
    coding = [
        r
        for r in results
        if r.level != RealWorldLevel.REPOSITORY_REASONING.value
        and r.status is not RealWorldStatus.UNSUPPORTED
    ]
    verification = [r for r in coding if r.oracle is not EvaluationOutcome.NOT_RUN]
    repair = [r for r in coding if r.level == RealWorldLevel.BOUNDED_REPAIR.value]

    def rate(
        values: Sequence[RealWorldTaskResult],
        predicate: Callable[[RealWorldTaskResult], bool],
    ) -> float:
        return (
            sum(predicate(value) for value in values) / len(values) if values else 0.0
        )

    return RealWorldSummary(
        count,
        sum(r.status is RealWorldStatus.PASS for r in results),
        sum(r.status is RealWorldStatus.PARTIAL for r in results),
        sum(r.status is RealWorldStatus.FAIL for r in results),
        sum(r.status is RealWorldStatus.UNSUPPORTED for r in results),
        rate(read, lambda r: r.model is EvaluationOutcome.PASS),
        rate(coding, lambda r: r.status is RealWorldStatus.PASS),
        rate(verification, lambda r: r.oracle is EvaluationOutcome.PASS),
        rate(repair, lambda r: r.status is RealWorldStatus.PASS),
        rate(read, lambda r: len(r.expected_files_found) > 0),
        sum(r.metrics.tool_executions for r in results) / count if count else 0.0,
        sum(r.metrics.source_reads for r in results) / count if count else 0.0,
        sum(r.elapsed_seconds for r in results) / count if count else 0.0,
    )


def render_realworld_report(run: RealWorldRun) -> str:
    lines = ["Task  Mode    Seed  Status       Oracle   Tools  Time"]
    for result in run.results:
        lines.append(
            f"{result.task_id:<5} {result.mode:<7} {result.seed:<5} "
            f"{result.status.value:<12} {result.oracle.value:<8} "
            f"{result.metrics.tool_executions:<6} {result.elapsed_seconds:.1f}s"
        )
    summary = run.summary
    lines.append(
        f"Runs: {summary.runs}; PASS {summary.passed}; PARTIAL {summary.partial}; "
        f"FAIL {summary.failed}; UNSUPPORTED {summary.unsupported}"
    )
    return "\n".join(lines)


def realworld_run_to_dict(run: RealWorldRun) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(asdict(run))  # type: ignore[return-value]


def write_realworld_json(run: RealWorldRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(realworld_run_to_dict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inspect_repository(
    source: Path,
    *,
    name: str,
    language: str,
    source_suffixes: frozenset[str],
    build_command: tuple[str, ...],
    test_command: tuple[str, ...],
    setup_commands: tuple[tuple[str, ...], ...] = (),
) -> RepositorySnapshot:
    source = source.resolve(strict=True)
    source_paths = [
        p
        for p in source.rglob("*")
        if p.is_file() and p.suffix in source_suffixes and "tests" not in p.parts
    ]
    test_paths = [
        p
        for p in source.rglob("tests/*")
        if p.is_file() and p.suffix in source_suffixes
    ]
    loc = sum(
        len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        for p in (*source_paths, *test_paths)
    )
    identity = (
        _git_identity(source)
        or hashlib.sha256(repr(hash_workspace(source)).encode()).hexdigest()
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="forge-realworld-baseline-") as temporary:
        copy = copy_repository(source, Path(temporary) / "workspace")
        outcome = run_oracle(copy, (*setup_commands, build_command, test_command))
    return RepositorySnapshot(
        name,
        identity,
        language,
        len(source_paths),
        len(test_paths),
        loc,
        build_command,
        test_command,
        outcome,
        time.perf_counter() - started,
    )


def _git_identity(source: Path) -> str | None:
    completed = subprocess.run(
        ("git", "-C", str(source), "rev-parse", "HEAD"),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        shell=False,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _project_commands(task: RealWorldTask) -> ProjectCommands:
    return ProjectCommands(
        ProjectCommand(task.build_command, 300) if task.build_command else None,
        ProjectCommand(task.test_command, 300) if task.test_command else None,
        task.verification_plan,
        ProjectCommand(task.configure_command, 300) if task.configure_command else None,
        task.execution_isolation,
    )


def _unsupported_result(
    task: RealWorldTask, seed: int, started: float
) -> RealWorldTaskResult:
    return RealWorldTaskResult(
        task.task_id,
        task.mode.value,
        task.level.value,
        seed,
        RealWorldStatus.UNSUPPORTED,
        EvaluationOutcome.PASS,
        EvaluationOutcome.NOT_RUN,
        EvaluationOutcome.NOT_RUN,
        RealWorldFailure.CAPABILITY_UNSUPPORTED,
        task.unsupported_reason,
        "unsupported",
        (),
        (),
        (),
        (),
        (),
        RealWorldMetrics(),
        ModelUsage(),
        time.perf_counter() - started,
    )


def _failed_result(
    task: RealWorldTask,
    seed: int,
    started: float,
    failure: RealWorldFailure,
    message: str,
) -> RealWorldTaskResult:
    return RealWorldTaskResult(
        task.task_id,
        task.mode.value,
        task.level.value,
        seed,
        RealWorldStatus.FAIL,
        EvaluationOutcome.FAIL,
        EvaluationOutcome.NOT_RUN,
        EvaluationOutcome.NOT_RUN,
        failure,
        message[:1_000],
        "not_run",
        (),
        (),
        (),
        (),
        (),
        RealWorldMetrics(),
        ModelUsage(),
        time.perf_counter() - started,
    )
