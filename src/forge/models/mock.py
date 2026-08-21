"""Deterministic model implementation for Forge tests."""

from __future__ import annotations

from collections.abc import Iterable

from forge.models.model import Model, ModelError
from forge.models.types import (
    FinishReason,
    ModelCapabilities,
    ModelCapability,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
)

DEFAULT_MOCK_IDENTITY = ModelIdentity(model_id="deterministic", backend_id="mock")
DEFAULT_MOCK_CAPABILITIES = ModelCapabilities(
    frozenset(
        {
            ModelCapability.CHAT,
            ModelCapability.SYSTEM_MESSAGES,
            ModelCapability.SEEDED_GENERATION,
        }
    )
)


class MockModel(Model):
    """Return configured responses in order and retain exact requests."""

    def __init__(
        self,
        responses: Iterable[str],
        *,
        identity: ModelIdentity = DEFAULT_MOCK_IDENTITY,
        capabilities: ModelCapabilities = DEFAULT_MOCK_CAPABILITIES,
        context_capacity: int | None = 4096,
    ) -> None:
        if not isinstance(identity, ModelIdentity):
            raise TypeError("identity must be a ModelIdentity")
        if not isinstance(capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities")
        if context_capacity is not None and (
            isinstance(context_capacity, bool)
            or not isinstance(context_capacity, int)
            or context_capacity <= 0
        ):
            raise ValueError("context_capacity must be positive or None")
        try:
            response_values = tuple(responses)
        except TypeError as error:
            raise TypeError("responses must be an iterable of strings") from error
        if not response_values:
            raise ValueError("MockModel requires at least one response")
        if not all(isinstance(response, str) for response in response_values):
            raise TypeError("responses must contain only strings")

        self._responses = response_values
        self._identity = identity
        self._capabilities = capabilities
        self._context_capacity = context_capacity
        self._requests: list[ModelRequest] = []
        self._next_response = 0
        self._closed = False

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def context_capacity(self) -> int | None:
        return self._context_capacity

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        """Return an immutable snapshot of requests received so far."""
        return tuple(self._requests)

    @property
    def closed(self) -> bool:
        """Return whether this model has been closed."""
        return self._closed

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self._closed:
            raise ModelError("model is closed")
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        if self._next_response >= len(self._responses):
            raise ModelError("mock responses exhausted")

        self._requests.append(request)
        text = self._responses[self._next_response]
        self._next_response += 1
        return ModelResponse(
            text=text,
            finish_reason=FinishReason.STOP,
            identity=self.identity,
        )

    def close(self) -> None:
        self._closed = True
