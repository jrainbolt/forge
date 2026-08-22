"""Backend-independent value objects used by the Forge model API."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class MessageRole(Enum):
    """Roles supported by model conversation messages in A1."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """One immutable, non-empty text message."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be a MessageRole")
        if not isinstance(self.content, str):
            raise TypeError("content must be text")
        if not self.content.strip():
            raise ValueError("message content must not be empty")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Generic controls that influence text generation."""

    max_tokens: int = 256
    temperature: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise TypeError("temperature must be a number")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")


class ResponseFormat(Enum):
    """Backend-independent response representation requested by a caller."""

    TEXT = "text"
    JSON = "json"


type JsonValue = (
    str | int | bool | None | tuple[JsonValue, ...] | Mapping[str, JsonValue]
)


@dataclass(frozen=True, slots=True)
class OutputSpecification:
    """A generic output format with an optional standard JSON schema."""

    format: ResponseFormat = ResponseFormat.TEXT
    schema: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.format, ResponseFormat):
            raise TypeError("format must be a ResponseFormat")
        if self.format is ResponseFormat.TEXT and self.schema is not None:
            raise ValueError("text output cannot define a JSON schema")
        if self.schema is not None:
            if not isinstance(self.schema, Mapping):
                raise TypeError("schema must be a mapping or None")
            object.__setattr__(self, "schema", _freeze_json(self.schema))


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """An ordered conversation and its generation controls."""

    messages: tuple[Message, ...]
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputSpecification = field(default_factory=OutputSpecification)

    def __post_init__(self) -> None:
        try:
            messages = tuple(self.messages)
        except TypeError as error:
            raise TypeError(
                "messages must be an iterable of Message objects"
            ) from error
        if not messages:
            raise ValueError("a model request must contain at least one message")
        if not all(isinstance(message, Message) for message in messages):
            raise TypeError("messages must contain only Message objects")
        if not isinstance(self.generation, GenerationConfig):
            raise TypeError("generation must be a GenerationConfig")
        if not isinstance(self.output, OutputSpecification):
            raise TypeError("output must be an OutputSpecification")
        object.__setattr__(self, "messages", messages)


class FinishReason(Enum):
    """Generic reasons why text generation ended."""

    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Identify a language model separately from its execution backend."""

    model_id: str
    backend_id: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("model_id", self.model_id),
            ("backend_id", self.backend_id),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be text")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")


class ModelCapability(Enum):
    """Capabilities callers may explicitly discover in A1."""

    CHAT = "chat"
    SYSTEM_MESSAGES = "system_messages"
    SEEDED_GENERATION = "seeded_generation"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Immutable set of capabilities exposed by a model implementation."""

    values: frozenset[ModelCapability] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        try:
            values = frozenset(self.values)
        except TypeError as error:
            raise TypeError("capabilities must be iterable") from error
        if not all(isinstance(value, ModelCapability) for value in values):
            raise TypeError("capabilities must contain only ModelCapability values")
        object.__setattr__(self, "values", values)

    def supports(self, capability: ModelCapability) -> bool:
        """Return whether a capability was explicitly declared."""
        if not isinstance(capability, ModelCapability):
            raise TypeError("capability must be a ModelCapability")
        return capability in self.values


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Token counts when a model implementation can report them."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer or None")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A backend-independent result of model generation."""

    text: str
    finish_reason: FinishReason
    identity: ModelIdentity
    usage: ModelUsage = field(default_factory=ModelUsage)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.finish_reason, FinishReason):
            raise TypeError("finish_reason must be a FinishReason")
        if not isinstance(self.identity, ModelIdentity):
            raise TypeError("identity must be a ModelIdentity")
        if not isinstance(self.usage, ModelUsage):
            raise TypeError("usage must be ModelUsage")


def _freeze_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON mapping keys must be text")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    raise TypeError("JSON values must contain only scalar, sequence, or mapping data")
