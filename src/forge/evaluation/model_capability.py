"""Local-model comparison over unchanged realworld-v1 production tasks."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    RealWorldRun,
    RealWorldTask,
    RealWorldTaskResult,
    RepositorySnapshot,
    hash_workspace,
)
from forge.models import (
    GenerationConfig,
    Message,
    MessageRole,
    ModelCatalog,
    ModelProfile,
    ModelRequest,
)

MODEL_CAPABILITY_V1 = "model-capability-v1"
MODEL_CAPABILITY_SUITE_VERSION = 1
MODEL_CAPABILITY_SCHEMA_VERSION = 1


class ModelAvailability(Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ModelConfigurationSnapshot:
    profile: str
    backend: str
    model_id: str
    model_artifact: str | None
    context_size: int | None
    generation_max_tokens: int = 512
    generation_temperature: float = 0.0
    bootstrap_provider: str = "lexical-cold"


@dataclass(frozen=True, slots=True)
class ModelAvailabilityResult:
    status: ModelAvailability
    stage: str
    exception_category: str | None
    message: str | None
    load_seconds: float
    context_creation_success: bool
    smoke_generation_seconds: float | None


@dataclass(frozen=True, slots=True)
class ModelCapabilitySummary:
    runs: int = 0
    read_only_runs: int = 0
    grounded_read_only: int = 0
    coding_runs: int = 0
    valid_deltas: int = 0
    previews: int = 0
    mutations: int = 0
    verification_passes: int = 0
    oracle_passes: int = 0
    repair_runs: int = 0
    repair_successes: int = 0
    mean_tools: float = 0.0
    mean_model_calls: float = 0.0
    mean_context_peak: float = 0.0
    mean_elapsed_seconds: float = 0.0

    @property
    def valid_delta_rate(self) -> float:
        return _rate(self.valid_deltas, self.coding_runs)

    @property
    def preview_rate(self) -> float:
        return _rate(self.previews, self.coding_runs)

    @property
    def mutation_rate(self) -> float:
        return _rate(self.mutations, self.coding_runs)

    @property
    def verification_pass_rate(self) -> float:
        return _rate(self.verification_passes, self.coding_runs)

    @property
    def oracle_pass_rate(self) -> float:
        return _rate(self.oracle_passes, self.coding_runs)

    @property
    def repair_success_rate(self) -> float:
        return _rate(self.repair_successes, self.repair_runs)


@dataclass(frozen=True, slots=True)
class ModelCapabilityProfileResult:
    configuration: ModelConfigurationSnapshot
    availability: ModelAvailabilityResult
    realworld: RealWorldRun | None
    summary: ModelCapabilitySummary


@dataclass(frozen=True, slots=True)
class ModelCapabilityRun:
    suite: str
    suite_version: int
    schema_version: int
    repository: RepositorySnapshot
    task_ids: tuple[str, ...]
    seeds: tuple[tuple[str, tuple[int, ...]], ...]
    profiles: tuple[ModelCapabilityProfileResult, ...]
    canonical_unchanged: bool


type SeedPlan = dict[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class ModelMatrixCell:
    profile: str
    task_id: str
    seed: int


def build_model_matrix(
    profiles: tuple[str, ...], tasks: tuple[RealWorldTask, ...], seeds: SeedPlan
) -> tuple[ModelMatrixCell, ...]:
    selected = _task_plan(tasks, seeds)
    return tuple(
        ModelMatrixCell(profile, task.task_id, seed)
        for profile in profiles
        for task in selected
        for seed in task.seeds
    )


class ModelCapabilityRunner:
    """Load configured profiles sequentially and reuse realworld-v1 unchanged."""

    def __init__(
        self, catalog: ModelCatalog, repository: Path, snapshot: RepositorySnapshot
    ) -> None:
        self._catalog = catalog
        self._repository = repository.resolve(strict=True)
        self._snapshot = snapshot

    def run(
        self,
        profiles: tuple[str, ...],
        tasks: tuple[RealWorldTask, ...],
        seeds: SeedPlan,
    ) -> ModelCapabilityRun:
        if not profiles or len(set(profiles)) != len(profiles):
            raise ValueError("model matrix requires unique configured profiles")
        if not tasks:
            raise ValueError("model matrix requires existing realworld tasks")
        configurations = tuple(
            _configuration_snapshot(self._catalog.profile(name)) for name in profiles
        )
        context_sizes = {item.context_size for item in configurations}
        if len(context_sizes) > 1:
            raise ValueError("model matrix profiles must use one context size")
        selected = _task_plan(tasks, seeds)
        canonical_before = hash_workspace(self._repository)
        results = tuple(self._run_profile(name, selected) for name in profiles)
        canonical_unchanged = canonical_before == hash_workspace(self._repository)
        if not canonical_unchanged:
            raise RuntimeError("canonical benchmark repository changed")
        return ModelCapabilityRun(
            MODEL_CAPABILITY_V1,
            MODEL_CAPABILITY_SUITE_VERSION,
            MODEL_CAPABILITY_SCHEMA_VERSION,
            self._snapshot,
            tuple(task.task_id for task in selected),
            tuple((task.task_id, task.seeds) for task in selected),
            results,
            True,
        )

    def _run_profile(
        self, profile_name: str, tasks: tuple[RealWorldTask, ...]
    ) -> ModelCapabilityProfileResult:
        profile = self._catalog.profile(profile_name)
        configuration = _configuration_snapshot(profile)
        started = time.perf_counter()
        try:
            model = self._catalog.create(profile_name)
        except Exception as error:
            return ModelCapabilityProfileResult(
                configuration,
                _unavailable("load", error, time.perf_counter() - started),
                None,
                ModelCapabilitySummary(),
            )
        load_seconds = time.perf_counter() - started
        smoke_started = time.perf_counter()
        try:
            with model:
                model.generate(
                    ModelRequest(
                        (Message(MessageRole.USER, 'Reply with JSON: {"ok": true}'),),
                        GenerationConfig(max_tokens=16, temperature=0.0, seed=42),
                    )
                )
                smoke_seconds = time.perf_counter() - smoke_started
                realworld = RealWorldEvaluationRunner(
                    profile_name, model, self._repository
                ).run(tasks, self._snapshot)
        except Exception as error:
            model.close()
            return ModelCapabilityProfileResult(
                configuration,
                _unavailable(
                    "context_or_smoke",
                    error,
                    load_seconds,
                    time.perf_counter() - smoke_started,
                ),
                None,
                ModelCapabilitySummary(),
            )
        finally:
            model.close()
        return ModelCapabilityProfileResult(
            configuration,
            ModelAvailabilityResult(
                ModelAvailability.AVAILABLE,
                "complete",
                None,
                None,
                load_seconds,
                True,
                smoke_seconds,
            ),
            realworld,
            summarize_model(realworld.results),
        )


def summarize_model(results: tuple[RealWorldTaskResult, ...]) -> ModelCapabilitySummary:
    runs = len(results)
    read_only = tuple(
        result for result in results if result.level == "repository_reasoning"
    )
    coding = tuple(
        result for result in results if result.level != "repository_reasoning"
    )
    repair = tuple(result for result in coding if result.mode == "repair")
    return ModelCapabilitySummary(
        runs=runs,
        read_only_runs=len(read_only),
        grounded_read_only=sum(
            result.metrics.expected_implementation_acquired
            and result.model is EvaluationOutcome.PASS
            for result in read_only
        ),
        coding_runs=len(coding),
        valid_deltas=sum(result.metrics.actual_delta_proposed for result in coding),
        previews=sum(result.metrics.preview_created > 0 for result in coding),
        mutations=sum(result.metrics.mutations > 0 for result in coding),
        verification_passes=sum(
            result.metrics.verification_result == "passed" for result in coding
        ),
        oracle_passes=sum(result.oracle is EvaluationOutcome.PASS for result in coding),
        repair_runs=len(repair),
        repair_successes=sum(
            result.metrics.reverification_result == "passed" for result in repair
        ),
        mean_tools=_mean(result.metrics.tool_executions for result in results),
        mean_model_calls=_mean(result.metrics.model_calls for result in results),
        mean_context_peak=_mean(
            result.metrics.context_peak_estimate for result in results
        ),
        mean_elapsed_seconds=_mean(result.elapsed_seconds for result in results),
    )


def model_capability_to_dict(run: ModelCapabilityRun) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    payload = convert(asdict(run))
    assert isinstance(payload, dict)
    profiles = payload.get("profiles")
    assert isinstance(profiles, list)
    for item, result in zip(profiles, run.profiles, strict=True):
        assert isinstance(item, dict)
        item["rates"] = {
            "valid_delta": result.summary.valid_delta_rate,
            "preview": result.summary.preview_rate,
            "mutation": result.summary.mutation_rate,
            "verification_pass": result.summary.verification_pass_rate,
            "oracle_pass": result.summary.oracle_pass_rate,
            "repair_success": result.summary.repair_success_rate,
        }
    return payload


def write_model_capability_json(run: ModelCapabilityRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model_capability_to_dict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def merge_model_capability_runs(
    runs: tuple[ModelCapabilityRun, ...],
) -> ModelCapabilityRun:
    """Combine independently checkpointed profiles with identical invariants."""
    if not runs:
        raise ValueError("at least one model capability run is required")
    first = runs[0]
    for run in runs[1:]:
        if (
            run.repository != first.repository
            or run.task_ids != first.task_ids
            or run.seeds != first.seeds
            or run.suite_version != first.suite_version
            or run.schema_version != first.schema_version
        ):
            raise ValueError(
                "model capability runs have different benchmark invariants"
            )
    profiles = tuple(profile for run in runs for profile in run.profiles)
    names = tuple(profile.configuration.profile for profile in profiles)
    if len(names) != len(set(names)):
        raise ValueError("model capability runs contain duplicate profiles")
    return ModelCapabilityRun(
        first.suite,
        first.suite_version,
        first.schema_version,
        first.repository,
        first.task_ids,
        first.seeds,
        profiles,
        all(run.canonical_unchanged for run in runs),
    )


def render_model_capability_report(run: ModelCapabilityRun) -> str:
    rows = [
        (
            "Model",
            "Task",
            "Seed",
            "Source",
            "Delta",
            "Verify",
            "Repair",
            "Oracle",
            "Tools",
            "Time",
        )
    ]
    for profile in run.profiles:
        if profile.realworld is None:
            rows.append(
                (
                    profile.configuration.profile,
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "UNAVAILABLE",
                    "-",
                    f"{profile.availability.load_seconds:.1f}s",
                )
            )
            continue
        for result in profile.realworld.results:
            rows.append(
                (
                    profile.configuration.profile,
                    result.task_id,
                    str(result.seed),
                    _yes(result.metrics.expected_implementation_acquired),
                    _yes(result.metrics.actual_delta_proposed),
                    result.metrics.verification_result,
                    result.metrics.reverification_result,
                    result.oracle.value,
                    str(result.metrics.tool_executions),
                    f"{result.elapsed_seconds:.1f}s",
                )
            )
    output = [_table(rows), "", "Model aggregates"]
    aggregates = [
        (
            "Model",
            "Runs",
            "Grounded",
            "Delta",
            "Verify",
            "Oracle",
            "Repair",
            "Tools",
            "Mean time",
        )
    ]
    for profile in run.profiles:
        summary = profile.summary
        aggregates.append(
            (
                profile.configuration.profile,
                str(summary.runs),
                f"{summary.grounded_read_only}/{summary.read_only_runs}",
                f"{summary.valid_deltas}/{summary.coding_runs}",
                f"{summary.verification_passes}/{summary.coding_runs}",
                f"{summary.oracle_passes}/{summary.coding_runs}",
                f"{summary.repair_successes}/{summary.repair_runs}",
                f"{summary.mean_tools:.1f}",
                f"{summary.mean_elapsed_seconds:.1f}s",
            )
        )
    output.append(_table(aggregates))
    output.append(f"Canonical unchanged: {run.canonical_unchanged}")
    return "\n".join(output)


def _task_plan(
    tasks: tuple[RealWorldTask, ...], seeds: SeedPlan
) -> tuple[RealWorldTask, ...]:
    from dataclasses import replace

    by_id = {task.task_id: task for task in tasks}
    if set(by_id) != set(seeds):
        raise ValueError("seed plan must name every selected task exactly once")
    return tuple(replace(task, seeds=tuple(seeds[task.task_id])) for task in tasks)


def _configuration_snapshot(profile: ModelProfile) -> ModelConfigurationSnapshot:
    config = profile.backend_config
    path = getattr(config, "model_path", None)
    return ModelConfigurationSnapshot(
        profile=profile.name,
        backend=profile.backend_id,
        model_id=profile.model_id,
        model_artifact=Path(path).name if path is not None else None,
        context_size=getattr(config, "context_size", None),
    )


def _unavailable(
    stage: str,
    error: Exception,
    load_seconds: float,
    smoke_seconds: float | None = None,
) -> ModelAvailabilityResult:
    return ModelAvailabilityResult(
        ModelAvailability.UNAVAILABLE,
        stage,
        type(error).__name__,
        str(error)[:1_000],
        load_seconds,
        False,
        smoke_seconds,
    )


def _rate(successes: int, runs: int) -> float:
    return successes / runs if runs else 0.0


def _mean(values: Iterable[float]) -> float:
    numbers = tuple(values)
    return sum(numbers) / len(numbers) if numbers else 0.0


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
