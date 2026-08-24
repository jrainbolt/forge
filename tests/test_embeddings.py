from __future__ import annotations

import math

import pytest

from forge.embeddings import EmbeddingVector, MockEmbeddingModel


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
    one = model.embed_text("permission approval")
    batch = model.embed_batch(("permission approval", "repository index"))
    assert one == batch[0]
    assert len(batch[1].values) == 32
    assert model.identity.backend == "mock"
