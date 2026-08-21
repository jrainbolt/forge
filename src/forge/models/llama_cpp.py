"""llama.cpp adapter for the generic Forge model API."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.models.model import Model, ModelError
from forge.models.types import (
    FinishReason,
    ModelCapabilities,
    ModelCapability,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)

LOGGER = logging.getLogger(__name__)
BACKEND_ID = "llama.cpp"
LLAMA_CAPABILITIES = ModelCapabilities(
    frozenset(
        {
            ModelCapability.CHAT,
            ModelCapability.SYSTEM_MESSAGES,
            ModelCapability.SEEDED_GENERATION,
        }
    )
)

LlamaFactory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    """Execution settings owned by the llama.cpp backend."""

    model_path: Path
    model_id: str | None = None
    context_size: int = 4096
    gpu_layers: int = -1
    threads: int | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        try:
            model_path = Path(self.model_path).expanduser()
        except TypeError as error:
            raise TypeError("model_path must be path-like") from error
        if not model_path.is_file():
            raise ValueError(f"model_path is not a file: {model_path}")
        object.__setattr__(self, "model_path", model_path)

        if self.model_id is not None:
            if not isinstance(self.model_id, str):
                raise TypeError("model_id must be text or None")
            if not self.model_id.strip():
                raise ValueError("model_id must not be empty")
        if isinstance(self.context_size, bool) or not isinstance(
            self.context_size, int
        ):
            raise TypeError("context_size must be an integer")
        if self.context_size <= 0:
            raise ValueError("context_size must be greater than zero")
        if isinstance(self.gpu_layers, bool) or not isinstance(self.gpu_layers, int):
            raise TypeError("gpu_layers must be an integer")
        if self.gpu_layers < -1:
            raise ValueError("gpu_layers must be -1 or non-negative")
        if self.threads is not None:
            if isinstance(self.threads, bool) or not isinstance(self.threads, int):
                raise TypeError("threads must be an integer or None")
            if self.threads <= 0:
                raise ValueError("threads must be greater than zero")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a Boolean")

    @property
    def resolved_model_id(self) -> str:
        """Return the explicit identifier or the local filename stem."""
        return self.model_id or self.model_path.stem


def _load_llama_factory() -> LlamaFactory:
    try:
        from llama_cpp import Llama
    except ImportError as error:
        raise ModelError(
            "llama.cpp support is not installed; install Forge with the 'llama' extra"
        ) from error
    return Llama


class LlamaCppModel(Model):
    """Serve generic Forge requests with a local GGUF model via llama.cpp."""

    def __init__(
        self,
        config: LlamaCppConfig,
        *,
        _llama_factory: LlamaFactory | None = None,
    ) -> None:
        if not isinstance(config, LlamaCppConfig):
            raise TypeError("config must be a LlamaCppConfig")

        self._config = config
        self._identity = ModelIdentity(config.resolved_model_id, BACKEND_ID)
        self._llama: Any | None = None

        factory = _llama_factory or _load_llama_factory()
        load_options: dict[str, object] = {
            "model_path": str(config.model_path),
            "n_ctx": config.context_size,
            "n_gpu_layers": config.gpu_layers,
            "verbose": config.verbose,
        }
        if config.threads is not None:
            load_options["n_threads"] = config.threads

        LOGGER.info(
            "Loading model %s with backend=%s context_size=%d gpu_layers=%d",
            self.identity.model_id,
            self.identity.backend_id,
            config.context_size,
            config.gpu_layers,
        )
        try:
            llama = factory(**load_options)
        except Exception as error:
            raise ModelError(
                f"failed to load llama.cpp model from {config.model_path}: {error}"
            ) from error

        if not _has_chat_template(llama):
            _close_llama(llama)
            raise ModelError(
                "the GGUF model does not provide a chat template; Forge will not "
                "guess a model-specific prompt format"
            )
        self._llama = llama

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def capabilities(self) -> ModelCapabilities:
        return LLAMA_CAPABILITIES

    @property
    def context_capacity(self) -> int:
        return self._config.context_size

    @property
    def closed(self) -> bool:
        """Return whether this adapter has released its model reference."""
        return self._llama is None

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self._llama is None:
            raise ModelError("model is closed")
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")

        messages = [
            {"role": message.role.value, "content": message.content}
            for message in request.messages
        ]
        generation = request.generation
        try:
            native_response = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=generation.max_tokens,
                temperature=float(generation.temperature),
                seed=generation.seed,
                stream=False,
            )
            return _translate_response(native_response, self.identity)
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(f"llama.cpp generation failed: {error}") from error

    def close(self) -> None:
        llama = self._llama
        if llama is None:
            return
        self._llama = None
        try:
            _close_llama(llama)
        except Exception as error:
            raise ModelError(f"failed to close llama.cpp model: {error}") from error


def _has_chat_template(llama: Any) -> bool:
    metadata = getattr(llama, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    return any(
        key == "tokenizer.chat_template" or key.startswith("tokenizer.chat_template.")
        for key in metadata
        if isinstance(key, str)
    )


def _close_llama(llama: Any) -> None:
    close = getattr(llama, "close", None)
    if callable(close):
        close()


def _translate_response(response: object, identity: ModelIdentity) -> ModelResponse:
    if not isinstance(response, Mapping):
        raise ModelError("llama.cpp returned a malformed response")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ModelError("llama.cpp response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ModelError("llama.cpp returned a malformed response choice")
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ModelError("llama.cpp response choice has no text content")

    return ModelResponse(
        text=message["content"],
        finish_reason=_translate_finish_reason(choice.get("finish_reason")),
        identity=identity,
        usage=_translate_usage(response.get("usage")),
    )


def _translate_finish_reason(reason: object) -> FinishReason:
    if reason == "stop":
        return FinishReason.STOP
    if reason == "length":
        return FinishReason.MAX_TOKENS
    if reason is None:
        return FinishReason.UNKNOWN
    return FinishReason.UNKNOWN


def _translate_usage(usage: object) -> ModelUsage:
    if not isinstance(usage, Mapping):
        return ModelUsage()
    return ModelUsage(
        input_tokens=_optional_token_count(usage.get("prompt_tokens")),
        output_tokens=_optional_token_count(usage.get("completion_tokens")),
    )


def _optional_token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
