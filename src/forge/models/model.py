"""Central synchronous model abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from forge.models.types import (
    ModelCapabilities,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
)


class ModelError(Exception):
    """A generic failure while using a model implementation."""


class Model(ABC):
    """An initialized model that can serve synchronous generation requests.

    Construction returns an initialized model. Implementations must perform any
    loading explicitly during construction or in their own factory and must
    release owned resources in :meth:`close`. Closing is idempotent.
    """

    @property
    @abstractmethod
    def identity(self) -> ModelIdentity:
        """Return the model and backend identity without implicit loading."""

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return explicitly declared model capabilities."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a complete response synchronously."""

    @abstractmethod
    def close(self) -> None:
        """Release resources owned by this model; repeated calls are safe."""

    def __enter__(self) -> Model:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
