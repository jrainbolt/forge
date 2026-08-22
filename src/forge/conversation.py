"""Explicit, bounded in-memory conversation state."""

from __future__ import annotations

import math
from dataclasses import dataclass

from forge.models import GenerationConfig, Message, MessageRole, ModelRequest


class ContextBudgetError(ValueError):
    """A required message cannot fit within the configured context budget."""


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One complete user/assistant exchange."""

    user: Message
    assistant: Message

    def __post_init__(self) -> None:
        if self.user.role is not MessageRole.USER:
            raise ValueError("turn user message must have the user role")
        if self.assistant.role is not MessageRole.ASSISTANT:
            raise ValueError("turn assistant message must have the assistant role")


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """A request plus transparent estimated-budget metadata."""

    request: ModelRequest
    estimated_input_tokens: int
    omitted_turns: int


class ConservativeTokenEstimator:
    """Conservatively estimate tokens without claiming tokenizer precision."""

    label = "estimated (UTF-8 bytes / 3 + message overhead)"

    def estimate(self, message: Message) -> int:
        return math.ceil(len(message.content.encode("utf-8")) / 3) + 4


class Conversation:
    """Own complete conversational turns and construct bounded requests."""

    def __init__(
        self,
        *,
        system_message: str | None = None,
        estimator: ConservativeTokenEstimator | None = None,
    ) -> None:
        self._system = (
            Message(MessageRole.SYSTEM, system_message)
            if system_message is not None
            else None
        )
        self._turns: list[ConversationTurn] = []
        self._estimator = estimator or ConservativeTokenEstimator()

    @property
    def system_message(self) -> Message | None:
        return self._system

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._turns)

    @property
    def messages(self) -> tuple[Message, ...]:
        messages: list[Message] = []
        if self._system is not None:
            messages.append(self._system)
        for turn in self._turns:
            messages.extend((turn.user, turn.assistant))
        return tuple(messages)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def estimator_label(self) -> str:
        return self._estimator.label

    def clear(self) -> None:
        """Clear complete turns while retaining the configured system message."""
        self._turns.clear()

    def discard_oldest_turns(self, count: int) -> None:
        """Remove a known number of oldest complete turns."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("discard count must be a non-negative integer")
        if count:
            del self._turns[:count]

    def commit(self, user_text: str, assistant_text: str) -> ConversationTurn:
        """Atomically append one complete turn."""
        turn = ConversationTurn(
            Message(MessageRole.USER, user_text),
            Message(MessageRole.ASSISTANT, assistant_text),
        )
        self._turns.append(turn)
        return turn

    def plan_request(
        self,
        user_text: str,
        generation: GenerationConfig,
        *,
        context_capacity: int | None,
        safety_reserve: int = 64,
        temporary_messages: tuple[Message, ...] = (),
    ) -> RequestPlan:
        """Build a request with bounded history and ephemeral turn messages."""
        if not isinstance(generation, GenerationConfig):
            raise TypeError("generation must be a GenerationConfig")
        try:
            temporary = tuple(temporary_messages)
        except TypeError as error:
            raise TypeError("temporary_messages must be iterable") from error
        if not all(isinstance(message, Message) for message in temporary):
            raise TypeError("temporary_messages must contain only Message objects")
        user = Message(MessageRole.USER, user_text)
        required = ([self._system] if self._system is not None else []) + [
            user,
            *temporary,
        ]

        effective_capacity = context_capacity or 4096
        if effective_capacity <= generation.max_tokens + safety_reserve:
            raise ContextBudgetError(
                "context capacity is too small for output and safety reserves"
            )
        input_budget = effective_capacity - generation.max_tokens - safety_reserve
        required_cost = sum(self._estimator.estimate(item) for item in required)
        if required_cost > input_budget:
            raise ContextBudgetError(
                "system and current user messages exceed the estimated input budget"
            )
        remaining = input_budget - required_cost
        newest_first: list[ConversationTurn] = []
        for turn in reversed(self._turns):
            cost = self._estimator.estimate(turn.user) + self._estimator.estimate(
                turn.assistant
            )
            if cost > remaining:
                break
            newest_first.append(turn)
            remaining -= cost
        selected = list(reversed(newest_first))

        messages: list[Message] = []
        if self._system is not None:
            messages.append(self._system)
        for turn in selected:
            messages.extend((turn.user, turn.assistant))
        messages.append(user)
        messages.extend(temporary)
        estimate = sum(self._estimator.estimate(item) for item in messages)
        return RequestPlan(
            ModelRequest(tuple(messages), generation),
            estimated_input_tokens=estimate,
            omitted_turns=len(self._turns) - len(selected),
        )
