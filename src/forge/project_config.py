"""Immutable trusted configuration for named project verification commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_BUILD_TIMEOUT_SECONDS = 120.0
DEFAULT_TEST_TIMEOUT_SECONDS = 300.0
MAX_TIMEOUT_SECONDS = 3600.0


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
class ProjectCommands:
    build: ProjectCommand | None = None
    test: ProjectCommand | None = None


def parse_project_commands(document: Mapping[str, object]) -> ProjectCommands:
    """Parse the optional project table without executing any command."""
    raw_project = document.get("project")
    if raw_project is None:
        return ProjectCommands()
    if not isinstance(raw_project, dict):
        raise ProjectConfigurationError("project must be a TOML table")
    unknown = set(raw_project) - {"commands"}
    if unknown:
        raise ProjectConfigurationError(
            f"project has unknown keys: {_format_keys(unknown)}"
        )
    raw_commands = raw_project.get("commands")
    if raw_commands is None:
        return ProjectCommands()
    if not isinstance(raw_commands, dict):
        raise ProjectConfigurationError("project.commands must be a TOML table")
    unknown = set(raw_commands) - {"build", "test"}
    if unknown:
        raise ProjectConfigurationError(
            f"project.commands has unknown keys: {_format_keys(unknown)}"
        )
    return ProjectCommands(
        build=_parse_command(raw_commands.get("build"), "build"),
        test=_parse_command(raw_commands.get("test"), "test"),
    )


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
        DEFAULT_BUILD_TIMEOUT_SECONDS
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
