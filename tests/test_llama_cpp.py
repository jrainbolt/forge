from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from forge.models import (
    FinishReason,
    GenerationConfig,
    LlamaCppConfig,
    LlamaCppModel,
    Message,
    MessageRole,
    ModelCapability,
    ModelError,
    ModelRequest,
    ModelUsage,
    OutputSpecification,
    ResponseFormat,
)
from forge.models import llama_cpp as adapter

CHAT_METADATA = {"tokenizer.chat_template": "template"}


class FakeLlama:
    def __init__(
        self,
        *,
        response: object | None = None,
        metadata: object = CHAT_METADATA,
        generation_error: Exception | None = None,
        **load_options: object,
    ) -> None:
        self.load_options = load_options
        self.metadata = metadata
        self.response = successful_native_response() if response is None else response
        self.generation_error = generation_error
        self.calls: list[dict[str, object]] = []
        self.close_calls = 0

    def create_chat_completion(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.generation_error is not None:
            raise self.generation_error
        return self.response

    def close(self) -> None:
        self.close_calls += 1


class Factory:
    def __init__(self, llama: FakeLlama) -> None:
        self.llama = llama
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> FakeLlama:
        self.calls.append(kwargs)
        self.llama.load_options = kwargs
        return self.llama


def successful_native_response(
    *, finish_reason: object = "stop", usage: object = None
) -> dict[str, object]:
    response: dict[str, object] = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from llama"},
                "finish_reason": finish_reason,
            }
        ]
    }
    if usage is not None:
        response["usage"] = usage
    return response


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "fixture-model.gguf"
    path.touch()
    return path


def request() -> ModelRequest:
    return ModelRequest(
        (
            Message(MessageRole.SYSTEM, "Be concise"),
            Message(MessageRole.USER, "Say hello"),
        ),
        GenerationConfig(max_tokens=32, temperature=0.25, seed=7),
    )


def test_config_requires_existing_model_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a file"):
        LlamaCppConfig(tmp_path / "missing.gguf")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_size": 0}, "context_size"),
        ({"gpu_layers": -2}, "gpu_layers"),
        ({"threads": 0}, "threads"),
        ({"model_id": " "}, "model_id"),
    ],
)
def test_config_rejects_invalid_values(
    model_file: Path, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LlamaCppConfig(model_file, **kwargs)


def test_config_is_backend_local_and_derives_identity(model_file: Path) -> None:
    config = LlamaCppConfig(model_file, context_size=2048, gpu_layers=12, threads=4)
    assert config.resolved_model_id == "fixture-model"
    assert config.context_size == 2048
    assert config.gpu_layers == 12
    assert config.threads == 4


def test_missing_optional_dependency_has_install_guidance(
    model_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(ModelError, match="llama.*extra") as error_info:
        adapter._load_llama_factory()
    assert isinstance(error_info.value.__cause__, ImportError)


def test_model_maps_load_configuration(model_file: Path) -> None:
    fake = FakeLlama()
    factory = Factory(fake)
    config = LlamaCppConfig(
        model_file,
        model_id="explicit-model",
        context_size=8192,
        gpu_layers=-1,
        threads=6,
        verbose=True,
    )

    model = LlamaCppModel(config, _llama_factory=factory)

    assert factory.calls == [
        {
            "model_path": str(model_file),
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "verbose": True,
            "n_threads": 6,
        }
    ]
    assert model.identity.model_id == "explicit-model"
    assert model.identity.backend_id == "llama.cpp"
    assert model.context_capacity == 8192


def test_model_requires_metadata_chat_template(model_file: Path) -> None:
    fake = FakeLlama(metadata={})
    with pytest.raises(ModelError, match="does not provide a chat template"):
        LlamaCppModel(LlamaCppConfig(model_file), _llama_factory=Factory(fake))
    assert fake.close_calls == 1


def test_model_declares_only_implemented_capabilities(model_file: Path) -> None:
    model = LlamaCppModel(
        LlamaCppConfig(model_file), _llama_factory=Factory(FakeLlama())
    )
    assert model.capabilities.supports(ModelCapability.CHAT)
    assert model.capabilities.supports(ModelCapability.SYSTEM_MESSAGES)
    assert model.capabilities.supports(ModelCapability.SEEDED_GENERATION)
    assert model.capabilities.supports(ModelCapability.STRUCTURED_OUTPUT)


def test_request_translation_maps_messages_and_generation(model_file: Path) -> None:
    fake = FakeLlama()
    model = LlamaCppModel(LlamaCppConfig(model_file), _llama_factory=Factory(fake))

    model.generate(request())

    assert fake.calls == [
        {
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Say hello"},
            ],
            "max_tokens": 32,
            "temperature": 0.25,
            "seed": 7,
            "stream": False,
        }
    ]


def test_structured_output_translation_is_adapter_local(model_file: Path) -> None:
    fake = FakeLlama()
    model = LlamaCppModel(LlamaCppConfig(model_file), _llama_factory=Factory(fake))
    schema = {"type": "object", "required": ["answer"]}
    structured = ModelRequest(
        (Message(MessageRole.USER, "Answer as JSON"),),
        output=OutputSpecification(ResponseFormat.JSON, schema),
    )

    model.generate(structured)

    assert fake.calls[0]["response_format"] == {
        "type": "json_object",
        "schema": schema,
    }


@pytest.mark.parametrize(
    ("native_reason", "forge_reason"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.MAX_TOKENS),
        (None, FinishReason.UNKNOWN),
        ("backend-specific", FinishReason.UNKNOWN),
    ],
)
def test_finish_reason_translation(
    model_file: Path, native_reason: object, forge_reason: FinishReason
) -> None:
    fake = FakeLlama(response=successful_native_response(finish_reason=native_reason))
    model = LlamaCppModel(LlamaCppConfig(model_file), _llama_factory=Factory(fake))
    assert model.generate(request()).finish_reason is forge_reason


def test_response_translation_includes_trustworthy_usage(model_file: Path) -> None:
    native = successful_native_response(
        usage={"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
    )
    model = LlamaCppModel(
        LlamaCppConfig(model_file),
        _llama_factory=Factory(FakeLlama(response=native)),
    )

    response = model.generate(request())

    assert response.text == "Hello from llama"
    assert response.usage == ModelUsage(input_tokens=9, output_tokens=4)
    assert response.identity is model.identity


def test_response_translation_keeps_unavailable_usage_unknown(
    model_file: Path,
) -> None:
    native = successful_native_response(
        usage={"prompt_tokens": "unknown", "completion_tokens": -1}
    )
    model = LlamaCppModel(
        LlamaCppConfig(model_file),
        _llama_factory=Factory(FakeLlama(response=native)),
    )
    assert model.generate(request()).usage == ModelUsage()


@pytest.mark.parametrize("response", [[], {}, {"choices": []}])
def test_malformed_native_response_is_model_error(
    model_file: Path, response: object
) -> None:
    fake = FakeLlama(response=response)
    model = LlamaCppModel(LlamaCppConfig(model_file), _llama_factory=Factory(fake))
    with pytest.raises(ModelError, match="malformed|exactly one"):
        model.generate(request())


def test_load_failure_is_translated(model_file: Path) -> None:
    def failing_factory(**kwargs: object) -> None:
        raise RuntimeError("native load failed")

    with pytest.raises(ModelError, match="failed to load.*native load failed"):
        LlamaCppModel(LlamaCppConfig(model_file), _llama_factory=failing_factory)


def test_generation_failure_is_translated(model_file: Path) -> None:
    fake = FakeLlama(generation_error=RuntimeError("native generation failed"))
    model = LlamaCppModel(LlamaCppConfig(model_file), _llama_factory=Factory(fake))
    with pytest.raises(ModelError, match="generation failed.*native generation failed"):
        model.generate(request())


def test_close_is_idempotent_and_prevents_generation(model_file: Path) -> None:
    fake = FakeLlama()
    model = LlamaCppModel(LlamaCppConfig(model_file), _llama_factory=Factory(fake))
    model.close()
    model.close()

    assert model.closed
    assert fake.close_calls == 1
    with pytest.raises(ModelError, match="closed"):
        model.generate(request())
