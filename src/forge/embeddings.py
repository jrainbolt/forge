"""Generic, local-first embedding models used by semantic retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EmbeddingError(RuntimeError):
    """An embedding backend failed or returned an invalid vector."""


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    backend: str
    model: str

    def __post_init__(self) -> None:
        if not self.backend.strip() or not self.model.strip():
            raise ValueError("embedding identity values must be non-empty")


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("embedding vectors must be non-empty and finite")
        object.__setattr__(self, "values", values)


class EmbeddingModel(ABC):
    """Backend-independent synchronous embedding lifecycle."""

    @property
    @abstractmethod
    def identity(self) -> EmbeddingIdentity: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    def embed_text(self, text: str) -> EmbeddingVector:
        return self.embed_batch((text,))[0]

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]: ...

    def close(self) -> None:
        """Release backend resources; implementations must be idempotent."""
        return None

    def __enter__(self) -> EmbeddingModel:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class MockEmbeddingModel(EmbeddingModel):
    """Deterministic feature-hashing model for tests and offline evaluation."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions
        self.calls = 0
        self.texts_embedded = 0

    @property
    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity("mock", "deterministic-token-hash-v1")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_batch(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        self.calls += 1
        result = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError("embedding input must be non-empty text")
            values = [0.0] * self.dimensions
            tokens = re.findall(r"[a-z0-9]+", _split_identifiers(text))
            for token in tokens:
                digest = hashlib.sha256(token.encode()).digest()
                values[int.from_bytes(digest[:4], "little") % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            result.append(EmbeddingVector(tuple(value / norm for value in values)))
            self.texts_embedded += 1
        return tuple(result)


class LlamaCppEmbeddingModel(EmbeddingModel):
    """Optional llama.cpp adapter for a user-owned local GGUF embedding model."""

    def __init__(
        self,
        model_path: Path,
        *,
        dimensions: int,
        context_size: int = 2048,
        gpu_layers: int = 0,
    ) -> None:
        path = model_path.expanduser().resolve(strict=True)
        if not path.is_file() or dimensions <= 0:
            raise ValueError(
                "model_path must be a file and dimensions must be positive"
            )
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise EmbeddingError(
                "llama.cpp embeddings require the optional 'semantic' extra"
            ) from error
        self._identity = EmbeddingIdentity("llama.cpp", str(path))
        self._dimensions = dimensions
        self._model: Any = Llama(
            model_path=str(path),
            embedding=True,
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_batch(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        if self._model is None:
            raise EmbeddingError("embedding model is closed")
        if not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise EmbeddingError("embedding input must be non-empty text")
        try:
            response = self._model.create_embedding(input=list(texts))
            vectors = tuple(
                EmbeddingVector(tuple(item["embedding"])) for item in response["data"]
            )
        except Exception as error:
            raise EmbeddingError(f"local embedding failed: {error}") from error
        if len(vectors) != len(texts) or any(
            len(vector.values) != self.dimensions for vector in vectors
        ):
            raise EmbeddingError("embedding backend returned inconsistent dimensions")
        return vectors

    def close(self) -> None:
        model, self._model = self._model, None
        if model is not None and callable(getattr(model, "close", None)):
            model.close()


def _split_identifiers(text: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).lower()
