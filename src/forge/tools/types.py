"""Immutable value objects for Forge tool requests and results."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
ARGUMENT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

type ScalarValue = str | int | float | bool | None
type StructuredValue = (
    ScalarValue | tuple[StructuredValue, ...] | Mapping[str, StructuredValue]
)


class ArgumentType(Enum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    TEXT_EDITS = "text_edits"


class ToolResultStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    APPROVAL_REQUIRED = "approval_required"


class ToolErrorKind(Enum):
    UNKNOWN_TOOL = "unknown_tool"
    VALIDATION = "validation"
    TOOL_FAILURE = "tool_failure"
    UNEXPECTED = "unexpected"


class PermissionDecision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ToolRisk(Enum):
    UNSPECIFIED = "unspecified"
    READ_ONLY = "read_only"
    WRITE = "write"
    EXECUTE = "execute"


class ToolCapability(Enum):
    UNCLASSIFIED = "unclassified"
    READ = "read"
    WRITE = "write"
    BUILD = "build"
    TEST = "test"


class ToolEvidence(Enum):
    NONE = "none"
    DISCOVERY = "discovery"
    SOURCE_CONTENT = "source_content"
    GIT_WORKING_STATE = "git_working_state"
    WRITE_SUCCESS = "write_success"
    PATCH_SUCCESS = "patch_success"
    BUILD_RESULT = "build_result"
    TEST_RESULT = "test_result"


class ToolValidationError(ValueError):
    """Tool metadata or invocation arguments violate their contract."""


@dataclass(frozen=True, slots=True)
class ArgumentSpec:
    name: str
    value_type: ArgumentType
    description: str
    required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not ARGUMENT_NAME_PATTERN.fullmatch(
            self.name
        ):
            raise ToolValidationError(f"invalid argument name: {self.name!r}")
        if not isinstance(self.value_type, ArgumentType):
            raise TypeError("value_type must be an ArgumentType")
        _validate_description(self.description, "argument description")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a Boolean")


@dataclass(frozen=True, slots=True)
class ArgumentSchema:
    arguments: tuple[ArgumentSpec, ...] = ()

    def __post_init__(self) -> None:
        try:
            arguments = tuple(self.arguments)
        except TypeError as error:
            raise TypeError("arguments must be iterable") from error
        if not all(isinstance(argument, ArgumentSpec) for argument in arguments):
            raise TypeError("arguments must contain only ArgumentSpec values")
        names = [argument.name for argument in arguments]
        if len(names) != len(set(names)):
            raise ToolValidationError("duplicate argument definitions are not allowed")
        object.__setattr__(self, "arguments", arguments)

    def validate(self, values: Mapping[str, object]) -> Mapping[str, object]:
        if not isinstance(values, Mapping):
            raise ToolValidationError("tool arguments must be a mapping")
        supplied = dict(values)
        specifications = {argument.name: argument for argument in self.arguments}
        unknown = set(supplied) - set(specifications)
        if unknown:
            rendered = ", ".join(sorted(repr(name) for name in unknown))
            raise ToolValidationError(f"unknown arguments: {rendered}")
        missing = [
            argument.name
            for argument in self.arguments
            if argument.required and argument.name not in supplied
        ]
        if missing:
            raise ToolValidationError(
                f"missing required arguments: {', '.join(sorted(missing))}"
            )
        validated: dict[str, object] = {}
        for name, value in supplied.items():
            specification = specifications[name]
            if not _matches_type(value, specification.value_type):
                raise ToolValidationError(
                    f"argument {name!r} must be {specification.value_type.value}"
                )
            validated[name] = _freeze_argument(value, specification.value_type)
        return MappingProxyType(validated)


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    name: str
    description: str
    argument_schema: ArgumentSchema
    risk: ToolRisk = ToolRisk.UNSPECIFIED
    evidence: ToolEvidence = ToolEvidence.NONE
    capability: ToolCapability = ToolCapability.UNCLASSIFIED

    def __post_init__(self) -> None:
        validate_tool_name(self.name)
        _validate_description(self.description, "tool description")
        if not isinstance(self.argument_schema, ArgumentSchema):
            raise TypeError("argument_schema must be an ArgumentSchema")
        if not isinstance(self.risk, ToolRisk):
            raise TypeError("risk must be a ToolRisk")
        if not isinstance(self.evidence, ToolEvidence):
            raise TypeError("evidence must be a ToolEvidence")
        if not isinstance(self.capability, ToolCapability):
            raise TypeError("capability must be a ToolCapability")


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    invocation_id: str
    tool_name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id.strip():
            raise ToolValidationError("invocation_id must be non-empty text")
        validate_tool_name(self.tool_name)
        if not isinstance(self.arguments, Mapping):
            raise ToolValidationError("arguments must be a mapping")
        frozen = freeze_structured_value(dict(self.arguments))
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "arguments", frozen)


@dataclass(frozen=True, slots=True)
class InvocationApproval:
    """Explicit approval for one exact invocation and argument set."""

    invocation_id: str
    tool_name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        invocation = ToolInvocation(self.invocation_id, self.tool_name, self.arguments)
        object.__setattr__(self, "arguments", invocation.arguments)

    @classmethod
    def for_invocation(cls, invocation: ToolInvocation) -> InvocationApproval:
        return cls(invocation.invocation_id, invocation.tool_name, invocation.arguments)

    def matches(self, invocation: ToolInvocation) -> bool:
        return (
            self.invocation_id == invocation.invocation_id
            and self.tool_name == invocation.tool_name
            and dict(self.arguments) == dict(invocation.arguments)
        )


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    workspace: Path

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Path):
            raise TypeError("workspace must be a Path")
        if not self.workspace.is_absolute():
            raise ValueError("workspace must be an absolute path")
        try:
            workspace = self.workspace.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"workspace does not exist: {self.workspace}") from error
        if not workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
        object.__setattr__(self, "workspace", workspace)


@dataclass(frozen=True, slots=True)
class ToolExecutionMetadata:
    permission_decision: PermissionDecision
    duration_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.permission_decision, PermissionDecision):
            raise TypeError("permission_decision must be a PermissionDecision")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must not be negative")


@dataclass(frozen=True, slots=True)
class ToolResult:
    invocation_id: str
    tool_name: str
    status: ToolResultStatus
    metadata: ToolExecutionMetadata
    output: StructuredValue = None
    error_kind: ToolErrorKind | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id.strip():
            raise ToolValidationError("invocation_id must be non-empty text")
        validate_tool_name(self.tool_name)
        if not isinstance(self.status, ToolResultStatus):
            raise TypeError("status must be a ToolResultStatus")
        if not isinstance(self.metadata, ToolExecutionMetadata):
            raise TypeError("metadata must be ToolExecutionMetadata")
        object.__setattr__(self, "output", freeze_structured_value(self.output))
        if self.error_kind is not None and not isinstance(
            self.error_kind, ToolErrorKind
        ):
            raise TypeError("error_kind must be ToolErrorKind or None")


def validate_tool_name(name: str) -> None:
    if not isinstance(name, str) or not TOOL_NAME_PATTERN.fullmatch(name):
        raise ToolValidationError(f"invalid tool name: {name!r}")


def _validate_description(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    if not value.strip():
        raise ToolValidationError(f"{label} must not be empty")


def _matches_type(value: object, value_type: ArgumentType) -> bool:
    if value_type is ArgumentType.STRING:
        return isinstance(value, str)
    if value_type is ArgumentType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type is ArgumentType.BOOLEAN:
        return isinstance(value, bool)
    if value_type is ArgumentType.TEXT_EDITS:
        return isinstance(value, (list, tuple)) and all(
            isinstance(edit, Mapping)
            and set(edit) == {"old", "new"}
            and isinstance(edit["old"], str)
            and isinstance(edit["new"], str)
            for edit in value
        )
    return False


def _freeze_argument(value: object, value_type: ArgumentType) -> object:
    if value_type is ArgumentType.TEXT_EDITS:
        assert isinstance(value, (list, tuple))
        return tuple(MappingProxyType(dict(edit)) for edit in value)
    return value


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def freeze_structured_value(value: object) -> StructuredValue:
    """Validate and recursively freeze a JSON-like structured value."""
    if _is_scalar(value):
        return value  # type: ignore[return-value]
    if isinstance(value, (list, tuple)):
        return tuple(freeze_structured_value(item) for item in value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("structured output mapping keys must be text")
        return MappingProxyType(
            {key: freeze_structured_value(item) for key, item in value.items()}
        )
    raise TypeError("output must contain only JSON-like structured values")
