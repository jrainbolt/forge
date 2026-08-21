"""Backend-independent value objects used by the Forge model API."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


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


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """An ordered conversation and its generation controls."""

    messages: tuple[Message, ...]
    generation: GenerationConfig = field(default_factory=GenerationConfig)

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
