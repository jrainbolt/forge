"""A61 evaluator boundary for honest real-repository discovery measurement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from benchmarks.real_repository_pilot_v1.suite import (
    AuthorityMode,
    PilotTask,
)
from benchmarks.real_repository_pilot_v1.suite import (
    tasks as a60_tasks,
)
from forge.evaluation.realworld import RealWorldTask

SUITE = "real-repository-discovery-boundary-v1"
VERSION = 1


class DiscoveryExpectationLeak(ValueError):
    """Hidden evaluator knowledge was offered as production authority."""

    code = "DISCOVERY_EXPECTATION_LEAK"


@dataclass(frozen=True, slots=True)
class ExpectedImplementationPath:
    """Evaluator-only location used after a run for discovery scoring."""

    value: str


@dataclass(frozen=True, slots=True)
class RequiredCandidatePath:
    """Explicit user path authority that may enter production setup."""

    value: str


class DiscoveryScore(StrEnum):
    EXPECTED_PATH_FOUND = "EXPECTED_PATH_FOUND"
    ALTERNATIVE_VALID_SOURCE_FOUND = "ALTERNATIVE_VALID_SOURCE_FOUND"
    INSUFFICIENT_SOURCE = "INSUFFICIENT_SOURCE"
    MISDIRECTED_SOURCE = "MISDIRECTED_SOURCE"


@dataclass(frozen=True, slots=True)
class BoundaryTask:
    task_id: str
    authority_mode: AuthorityMode
    production_task: RealWorldTask
    expected_implementation_paths: tuple[ExpectedImplementationPath, ...]

    @property
    def expected_values(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.expected_implementation_paths)


def _explicit_paths(prompt: str, expected: tuple[str, ...]) -> tuple[str, ...]:
    """Return expected paths actually stated by the simulated user."""
    return tuple(path for path in expected if path in prompt)


def _production_task(source: PilotTask) -> RealWorldTask:
    task = source.production_task
    expected = tuple(task.expected_changed_paths)
    if source.authority_mode is AuthorityMode.DISCOVERY_REQUIRED:
        # Expected paths, allowed paths, and source requirements are evaluator facts.
        # None is needed to exercise production discovery in the read-only smoke.
        return replace(
            task,
            expected_files=(),
            allowed_paths=(),
            expected_changed_paths=(),
            required_candidate_paths=(),
        )
    explicit = _explicit_paths(task.prompt, expected)
    if set(explicit) != set(expected):
        raise ValueError("path-known authority must originate in task text")
    return replace(task, required_candidate_paths=explicit)


def tasks() -> tuple[BoundaryTask, ...]:
    """Derive A61 definitions without changing the frozen A60 definitions."""
    return tuple(
        BoundaryTask(
            item.task_id,
            item.authority_mode,
            _production_task(item),
            tuple(ExpectedImplementationPath(path) for path in item.edit_paths),
        )
        for item in a60_tasks()
    )


def validate_production_boundary(task: BoundaryTask) -> None:
    """Fail closed if hidden expectation crosses into discovery authority."""
    if task.authority_mode is not AuthorityMode.DISCOVERY_REQUIRED:
        return
    production = task.production_task
    authority = {
        *production.expected_files,
        *production.allowed_paths,
        *production.expected_changed_paths,
        *production.required_candidate_paths,
        *production.create_candidate_paths,
    }
    leaked = authority.intersection(task.expected_values)
    if leaked:
        raise DiscoveryExpectationLeak(
            f"{DiscoveryExpectationLeak.code}: {', '.join(sorted(leaked))}"
        )


def production_inputs(task: BoundaryTask) -> dict[str, object]:
    """Return the evaluator-auditable values supplied to production setup."""
    validate_production_boundary(task)
    production = task.production_task
    return {
        "task_text": production.prompt,
        "required_candidate_paths": production.required_candidate_paths,
        "expected_changed_paths": production.expected_changed_paths,
        "allowed_paths": production.allowed_paths,
        "create_candidate_paths": production.create_candidate_paths,
    }


def score_acquired_source(
    task: BoundaryTask,
    acquired_paths: tuple[str, ...],
    *,
    sufficient: bool,
    alternative_valid: bool = False,
) -> DiscoveryScore:
    """Score only after trusted acquisition; valid alternatives are accepted."""
    if not acquired_paths:
        return DiscoveryScore.INSUFFICIENT_SOURCE
    if set(acquired_paths).intersection(task.expected_values):
        return (
            DiscoveryScore.EXPECTED_PATH_FOUND
            if sufficient
            else DiscoveryScore.INSUFFICIENT_SOURCE
        )
    if sufficient and alternative_valid:
        return DiscoveryScore.ALTERNATIVE_VALID_SOURCE_FOUND
    return DiscoveryScore.MISDIRECTED_SOURCE
