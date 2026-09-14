"""Immutable trusted configuration for named project verification commands."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from forge.process_isolation import ExecutionIsolationMode, ExecutionIsolationPolicy

DEFAULT_BUILD_TIMEOUT_SECONDS = 120.0
DEFAULT_CONFIGURE_TIMEOUT_SECONDS = 120.0
DEFAULT_TEST_TIMEOUT_SECONDS = 300.0
MAX_TIMEOUT_SECONDS = 3600.0
MAX_VERIFICATION_STEPS = 4
VERIFICATION_OPERATIONS = frozenset(
    {"project.configure", "project.build", "project.test"}
)


class ProjectConfigurationError(ValueError):
    """Project command configuration is malformed."""


@dataclass(frozen=True, slots=True)
class ProjectCommand:
    """One immutable argument-array command owned by trusted local config."""

    argv: tuple[str, ...]
    timeout_seconds: float

    def __post_init__(self) -> None:
        argv = tuple(self.argv)
        if not argv:
            raise ProjectConfigurationError("project command argv must not be empty")
        if any(not isinstance(item, str) or not item for item in argv):
            raise ProjectConfigurationError(
                "project command argv entries must be non-empty text"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not 0 < self.timeout_seconds <= MAX_TIMEOUT_SECONDS
        ):
            raise ProjectConfigurationError(
                f"project command timeout_seconds must be greater than zero and at "
                f"most {MAX_TIMEOUT_SECONDS:g}"
            )
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    """Trusted ordered references to configured A10 operations."""

    plan_id: str
    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if (
            not isinstance(self.plan_id, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", self.plan_id) is None
        ):
            raise ProjectConfigurationError(
                "verification plan id must be a non-empty identifier"
            )
        if not steps or len(steps) > MAX_VERIFICATION_STEPS:
            raise ProjectConfigurationError(
                "verification plan must contain 1 to 4 steps"
            )
        if any(
            not isinstance(step, str) or step not in VERIFICATION_OPERATIONS
            for step in steps
        ):
            raise ProjectConfigurationError(
                "verification plan references an unsupported operation"
            )
        if len(set(steps)) != len(steps):
            raise ProjectConfigurationError("verification plan steps must be unique")
        if steps == ("project.configure",):
            raise ProjectConfigurationError(
                "project.configure alone cannot establish verification success"
            )
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True, slots=True)
class ProjectCommands:
    build: ProjectCommand | None = None
    test: ProjectCommand | None = None
    verification_plan: VerificationPlan | None = None
    configure: ProjectCommand | None = None
    execution_isolation: ExecutionIsolationPolicy = ExecutionIsolationPolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.execution_isolation, ExecutionIsolationPolicy):
            raise ProjectConfigurationError(
                "execution_isolation must be an ExecutionIsolationPolicy"
            )
        if self.verification_plan is not None and not isinstance(
            self.verification_plan, VerificationPlan
        ):
            raise ProjectConfigurationError(
                "verification_plan must be a VerificationPlan"
            )
        if self.verification_plan is not None:
            for step in self.verification_plan.steps:
                if getattr(self, step.removeprefix("project.")) is None:
                    raise ProjectConfigurationError(
                        f"verification plan step {step} is not configured"
                    )


def parse_project_commands(document: Mapping[str, object]) -> ProjectCommands:
    """Parse the optional project table without executing any command."""
    raw_project = document.get("project")
    if raw_project is None:
        return ProjectCommands()
    if not isinstance(raw_project, dict):
        raise ProjectConfigurationError("project must be a TOML table")
    unknown = set(raw_project) - {"commands", "verification", "execution"}
    if unknown:
        raise ProjectConfigurationError(
            f"project has unknown keys: {_format_keys(unknown)}"
        )
    raw_commands = raw_project.get("commands")
    if raw_commands is None:
        raw_commands = {}
    if not isinstance(raw_commands, dict):
        raise ProjectConfigurationError("project.commands must be a TOML table")
    unknown = set(raw_commands) - {"configure", "build", "test"}
    if unknown:
        raise ProjectConfigurationError(
            f"project.commands has unknown keys: {_format_keys(unknown)}"
        )
    return ProjectCommands(
        build=_parse_command(raw_commands.get("build"), "build"),
        test=_parse_command(raw_commands.get("test"), "test"),
        verification_plan=_parse_verification_plan(raw_project.get("verification")),
        configure=_parse_command(raw_commands.get("configure"), "configure"),
        execution_isolation=_parse_execution_isolation(raw_project.get("execution")),
    )


def _parse_execution_isolation(raw: object) -> ExecutionIsolationPolicy:
    if raw is None:
        return ExecutionIsolationPolicy()
    if not isinstance(raw, dict):
        raise ProjectConfigurationError("project.execution must be a TOML table")
    unknown = set(raw) - {"isolation"}
    if unknown:
        raise ProjectConfigurationError(
            f"project.execution has unknown keys: {_format_keys(unknown)}"
        )
    value = raw.get("isolation", "none")
    try:
        return ExecutionIsolationPolicy(ExecutionIsolationMode(value))
    except (TypeError, ValueError) as error:
        raise ProjectConfigurationError(
            "project.execution.isolation must be none, controlled_env, or strict"
        ) from error


def _parse_verification_plan(raw: object) -> VerificationPlan | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProjectConfigurationError("project.verification must be a TOML table")
    unknown = set(raw) - {"id", "steps"}
    if unknown:
        raise ProjectConfigurationError(
            f"project.verification has unknown keys: {_format_keys(unknown)}"
        )
    steps = raw.get("steps")
    if not isinstance(steps, list) or any(not isinstance(step, str) for step in steps):
        raise ProjectConfigurationError(
            "project.verification.steps must be a text array"
        )
    return VerificationPlan(raw.get("id"), tuple(steps))  # type: ignore[arg-type]


def _parse_command(raw: object, operation: str) -> ProjectCommand | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProjectConfigurationError(
            f"project.commands.{operation} must be a TOML table"
        )
    unknown = set(raw) - {"argv", "timeout_seconds"}
    if unknown:
        raise ProjectConfigurationError(
            f"project.commands.{operation} has unknown keys: {_format_keys(unknown)}"
        )
    argv = raw.get("argv")
    if not isinstance(argv, list):
        raise ProjectConfigurationError(
            f"project.commands.{operation}.argv must be an argument array"
        )
    timeout = raw.get(
        "timeout_seconds",
        DEFAULT_CONFIGURE_TIMEOUT_SECONDS
        if operation == "configure"
        else DEFAULT_BUILD_TIMEOUT_SECONDS
        if operation == "build"
        else DEFAULT_TEST_TIMEOUT_SECONDS,
    )
    try:
        return ProjectCommand(tuple(argv), timeout)  # type: ignore[arg-type]
    except ProjectConfigurationError as error:
        raise ProjectConfigurationError(
            f"invalid project.commands.{operation}: {error}"
        ) from error


def _format_keys(keys: set[object]) -> str:
    return ", ".join(sorted(repr(key) for key in keys))
