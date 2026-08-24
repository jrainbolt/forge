from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from forge.embeddings import (
    EmbeddingPurpose,
    EmbeddingVector,
    LlamaCppEmbeddingModel,
    MockEmbeddingModel,
)


def test_embedding_vector_is_immutable_and_validated() -> None:
    vector = EmbeddingVector((1, 2.5))
    assert vector.values == (1.0, 2.5)
    with pytest.raises((AttributeError, TypeError)):
        vector.values = (3.0,)  # type: ignore[misc]
    with pytest.raises(ValueError):
        EmbeddingVector(())
    with pytest.raises(ValueError):
        EmbeddingVector((math.inf,))


def test_mock_embedding_is_deterministic_and_batch_consistent() -> None:
    model = MockEmbeddingModel(32)
    one = model.embed_text("permission approval", purpose=EmbeddingPurpose.DOCUMENT)
    batch = model.embed_batch(
        ("permission approval", "repository index"),
        purpose=EmbeddingPurpose.DOCUMENT,
    )
    assert one == batch[0]
    assert len(batch[1].values) == 32
    assert model.identity.backend == "mock"
    assert one != model.embed_text(
        "permission approval", purpose=EmbeddingPurpose.QUERY
    )
    with pytest.raises(TypeError):
        model.embed_text("permission approval", purpose="query")  # type: ignore[arg-type]


def test_llama_adapter_owns_retrieval_prefix_formatting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = []

    class FakeLlama:
        def __init__(self, **_kwargs: object) -> None: ...

        def create_embedding(self, *, input: list[str]) -> dict[str, object]:
            inputs.append(input)
            return {"data": [{"embedding": [1.0] * 8} for _ in input]}

        def close(self) -> None: ...

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"fixture")
    model = LlamaCppEmbeddingModel(
        artifact,
        dimensions=8,
        document_prefix="search_document: ",
        query_prefix="search_query: ",
    )
    model.embed_text("source", purpose=EmbeddingPurpose.DOCUMENT)
    model.embed_text("question", purpose=EmbeddingPurpose.QUERY)
    assert inputs == [["search_document: source"], ["search_query: question"]]
