"""Generic model-backed ephemeral chat session."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from forge.conversation import Conversation, RequestPlan
from forge.models import (
    GenerationConfig,
    Model,
    ModelCapability,
    ModelIdentity,
    ModelResponse,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_SYSTEM_MESSAGE = "You are Forge, a local AI assistant."


@dataclass(frozen=True, slots=True)
class SessionInfo:
    profile_name: str
    identity: ModelIdentity
    context_capacity: int | None
    message_count: int
    completed_turns: int
    last_estimated_input_tokens: int | None
    last_omitted_turns: int
    estimate_method: str


class ChatSession:
    """Own one model and transactional conversation execution."""

    def __init__(
        self,
        profile_name: str,
        model: Model,
        *,
        generation: GenerationConfig | None = None,
        system_message: str | None = DEFAULT_SYSTEM_MESSAGE,
    ) -> None:
        if not isinstance(model, Model):
            raise TypeError("model must implement Model")
        if not model.capabilities.supports(ModelCapability.CHAT):
            raise ValueError("selected model does not declare chat capability")
        supports_system = model.capabilities.supports(ModelCapability.SYSTEM_MESSAGES)
        if system_message is not None and not supports_system:
            LOGGER.info("Omitting system message: selected model does not support it")
        self._profile_name = profile_name
        self._model = model
        self._generation = generation or GenerationConfig(
            max_tokens=256, temperature=0.4
        )
        self._conversation = Conversation(
            system_message=system_message if supports_system else None
        )
        self._closed = False
        self._last_plan: RequestPlan | None = None

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def info(self) -> SessionInfo:
        plan = self._last_plan
        return SessionInfo(
            profile_name=self._profile_name,
            identity=self._model.identity,
            context_capacity=self._model.context_capacity,
            message_count=self._conversation.message_count,
            completed_turns=len(self._conversation.turns),
            last_estimated_input_tokens=(
                plan.estimated_input_tokens if plan is not None else None
            ),
            last_omitted_turns=plan.omitted_turns if plan is not None else 0,
            estimate_method=self._conversation.estimator_label,
        )

    def ask(self, user_text: str) -> ModelResponse:
        """Generate and commit a complete turn only after success."""
        if self._closed:
            raise RuntimeError("chat session is closed")
        plan = self._conversation.plan_request(
            user_text,
            self._generation,
            context_capacity=self._model.context_capacity,
        )
        LOGGER.debug(
            "Starting model request completed_turns=%d omitted_turns=%d",
            len(self._conversation.turns),
            plan.omitted_turns,
        )
        started = time.perf_counter()
        response = self._model.generate(plan.request)
        elapsed = time.perf_counter() - started
        self._conversation.discard_oldest_turns(plan.omitted_turns)
        self._conversation.commit(user_text, response.text)
        self._last_plan = plan
        LOGGER.debug("Completed model request in %.2f seconds", elapsed)
        return response

    def clear(self) -> None:
        self._conversation.clear()
        self._last_plan = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model.close()

    def __enter__(self) -> ChatSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
