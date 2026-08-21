from __future__ import annotations

import os
from pathlib import Path

import pytest

from forge.models import (
    GenerationConfig,
    LlamaCppConfig,
    LlamaCppModel,
    Message,
    MessageRole,
    ModelRequest,
)


@pytest.mark.integration
def test_real_local_generation() -> None:
    configured_path = os.environ.get("FORGE_TEST_MODEL")
    if not configured_path:
        pytest.skip("FORGE_TEST_MODEL is not configured")

    request = ModelRequest(
        (Message(MessageRole.USER, "Reply with one short greeting."),),
        GenerationConfig(max_tokens=24, temperature=0.0, seed=42),
    )
    with LlamaCppModel(
        LlamaCppConfig(Path(configured_path), context_size=2048, gpu_layers=-1)
    ) as model:
        response = model.generate(request)

    assert response.text.strip()
    assert len(response.text) < 1000
