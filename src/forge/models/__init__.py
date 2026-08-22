"""Generic language-model API and deterministic test implementation."""

from forge.models.catalog import (
    BackendDefinition,
    BackendRegistry,
    ModelCatalog,
    ModelConfigurationError,
    ModelProfile,
    ModelSelectionError,
    default_backend_registry,
    load_model_catalog,
)
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
    OutputSpecification,
    ResponseFormat,
)

__all__ = [
    "BackendDefinition",
    "BackendRegistry",
    "FinishReason",
    "GenerationConfig",
    "Message",
    "MessageRole",
    "LlamaCppConfig",
    "LlamaCppModel",
    "MockModel",
    "Model",
    "ModelCatalog",
    "ModelCapabilities",
    "ModelCapability",
    "ModelError",
    "ModelConfigurationError",
    "ModelIdentity",
    "ModelRequest",
    "ModelResponse",
    "ModelProfile",
    "ModelSelectionError",
    "ModelUsage",
    "OutputSpecification",
    "ResponseFormat",
    "default_backend_registry",
    "load_model_catalog",
]
