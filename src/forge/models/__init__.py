"""Generic language-model API and deterministic test implementation."""

from forge.models.llama_cpp import LlamaCppConfig, LlamaCppModel
from forge.models.mock import MockModel
from forge.models.model import Model, ModelError
from forge.models.types import (
    FinishReason,
    GenerationConfig,
    Message,
    MessageRole,
    ModelCapabilities,
    ModelCapability,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)

__all__ = [
    "FinishReason",
    "GenerationConfig",
    "Message",
    "MessageRole",
    "LlamaCppConfig",
    "LlamaCppModel",
    "MockModel",
    "Model",
    "ModelCapabilities",
    "ModelCapability",
    "ModelError",
    "ModelIdentity",
    "ModelRequest",
    "ModelResponse",
    "ModelUsage",
]
