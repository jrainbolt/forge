"""Strict configuration for separately selected embedding profiles."""

from __future__ import annotations

import tomllib
from pathlib import Path

from forge.embeddings import EmbeddingModel, LlamaCppEmbeddingModel, MockEmbeddingModel


def load_embedding_profile(path: Path, profile: str) -> EmbeddingModel:
    try:
        document = tomllib.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot load embedding configuration: {error}") from error
    profiles = document.get("embeddings")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(profile), dict):
        raise ValueError(f"unknown embedding profile: {profile}")
    values = profiles[profile]
    allowed = {
        "backend",
        "model_path",
        "dimensions",
        "context_size",
        "gpu_layers",
        "document_prefix",
        "query_prefix",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown embedding settings: {', '.join(sorted(unknown))}")
    backend = values.get("backend")
    dimensions = values.get("dimensions", 128)
    if not isinstance(dimensions, int):
        raise ValueError("embedding dimensions must be an integer")
    if backend == "mock":
        return MockEmbeddingModel(dimensions)
    if backend != "llama.cpp":
        raise ValueError("embedding backend must be 'llama.cpp' (or 'mock' for tests)")
    model_path = values.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("llama.cpp embedding profile requires model_path")
    return LlamaCppEmbeddingModel(
        Path(model_path),
        dimensions=dimensions,
        context_size=_integer(values, "context_size", 2048),
        gpu_layers=_integer(values, "gpu_layers", 0),
        document_prefix=_string(values, "document_prefix", ""),
        query_prefix=_string(values, "query_prefix", ""),
    )


def _integer(values: dict[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"embedding {key} must be an integer")
    return value


def _string(values: dict[str, object], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"embedding {key} must be a string")
    return value
