from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from forge.models import (
    FinishReason,
    GenerationConfig,
    Message,
    MessageRole,
    MockModel,
    Model,
    ModelCapabilities,
    ModelCapability,
    ModelError,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)


def user_request(content: str = "Hello") -> ModelRequest:
    return ModelRequest((Message(MessageRole.USER, content),))


@pytest.mark.parametrize("role", list(MessageRole))
def test_message_supports_each_declared_role(role: MessageRole) -> None:
    assert Message(role, "content") == Message(role=role, content="content")


def test_message_is_immutable() -> None:
    message = Message(MessageRole.USER, "hello")
    with pytest.raises(FrozenInstanceError):
        message.content = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("content", ["", "   "])
def test_message_rejects_empty_content(content: str) -> None:
    with pytest.raises(ValueError, match="content"):
        Message(MessageRole.USER, content)


def test_message_rejects_arbitrary_role() -> None:
    with pytest.raises(TypeError, match="MessageRole"):
        Message("user", "hello")  # type: ignore[arg-type]


def test_generation_config_defaults() -> None:
    assert GenerationConfig() == GenerationConfig(
        max_tokens=256, temperature=0.0, seed=None
    )


def test_generation_config_accepts_explicit_boundary_values() -> None:
    assert GenerationConfig(max_tokens=1, temperature=0, seed=0).seed == 0


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_generation_config_rejects_non_positive_token_limit(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        GenerationConfig(max_tokens=max_tokens)


@pytest.mark.parametrize("temperature", [-0.1, float("inf"), float("nan")])
def test_generation_config_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(temperature=temperature)


def test_generation_config_rejects_non_integer_seed() -> None:
    with pytest.raises(TypeError, match="seed"):
        GenerationConfig(seed=1.5)  # type: ignore[arg-type]


def test_request_preserves_order_and_associates_configuration() -> None:
    messages = [
        Message(MessageRole.SYSTEM, "Be concise"),
        Message(MessageRole.USER, "Hello"),
        Message(MessageRole.ASSISTANT, "Hi"),
    ]
    generation = GenerationConfig(max_tokens=10, temperature=0.5, seed=7)

    request = ModelRequest(messages, generation)  # type: ignore[arg-type]

    assert request.messages == tuple(messages)
    assert request.generation is generation
    messages.reverse()
    assert request.messages[0].role is MessageRole.SYSTEM


def test_request_rejects_empty_messages() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ModelRequest(())


def test_request_rejects_non_message_member() -> None:
    with pytest.raises(TypeError, match="Message"):
        ModelRequest(("hello",))  # type: ignore[arg-type]


def test_response_contains_generic_result_data() -> None:
    identity = ModelIdentity("example-model", "example-backend")
    usage = ModelUsage(input_tokens=4, output_tokens=2)

    response = ModelResponse("hello", FinishReason.STOP, identity, usage)

    assert response.text == "hello"
    assert response.finish_reason is FinishReason.STOP
    assert response.identity is identity
    assert response.usage is usage


def test_response_usage_can_be_unavailable() -> None:
    response = ModelResponse(
        "hello", FinishReason.UNKNOWN, ModelIdentity("model", "backend")
    )
    assert response.usage == ModelUsage(input_tokens=None, output_tokens=None)


@pytest.mark.parametrize("count", [-1, -100])
def test_usage_rejects_impossible_counts(count: int) -> None:
    with pytest.raises(ValueError, match="negative"):
        ModelUsage(output_tokens=count)


@pytest.mark.parametrize(
    ("model_id", "backend_id"), [("", "backend"), ("model", "   ")]
)
def test_identity_rejects_empty_identifiers(model_id: str, backend_id: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ModelIdentity(model_id, backend_id)


def test_capabilities_are_declared_and_queryable() -> None:
    capabilities = ModelCapabilities(
        frozenset({ModelCapability.CHAT, ModelCapability.SYSTEM_MESSAGES})
    )

    assert capabilities.supports(ModelCapability.CHAT)
    assert capabilities.supports(ModelCapability.SYSTEM_MESSAGES)
    assert not capabilities.supports(ModelCapability.SEEDED_GENERATION)


def test_capabilities_reject_unknown_values() -> None:
    with pytest.raises(TypeError, match="ModelCapability"):
        ModelCapabilities(frozenset({"chat"}))  # type: ignore[arg-type]


def test_mock_model_conforms_and_returns_deterministic_responses() -> None:
    model: Model = MockModel(["first", "second"])
    request = user_request()

    assert model.generate(request).text == "first"
    assert model.generate(request).text == "second"


def test_mock_model_records_exact_request() -> None:
    model = MockModel(["response"])
    request = user_request("Exact request")

    model.generate(request)

    assert model.requests == (request,)
    assert model.requests[0] is request


def test_mock_model_exposes_identity_and_capabilities() -> None:
    identity = ModelIdentity("fixture-model", "fixture-backend")
    capabilities = ModelCapabilities(frozenset({ModelCapability.CHAT}))
    model = MockModel(["response"], identity=identity, capabilities=capabilities)

    assert model.identity is identity
    assert model.capabilities is capabilities


def test_mock_model_reports_exhaustion() -> None:
    model = MockModel(["only response"])
    model.generate(user_request())

    with pytest.raises(ModelError, match="exhausted"):
        model.generate(user_request())


def test_mock_model_lifecycle_is_explicit_and_idempotent() -> None:
    model = MockModel(["unused"])
    model.close()
    model.close()

    assert model.closed
    with pytest.raises(ModelError, match="closed"):
        model.generate(user_request())


def test_model_context_manager_closes_model() -> None:
    model = MockModel(["response"])
    with model as active_model:
        assert active_model.generate(user_request()).text == "response"

    assert model.closed


def test_caller_depends_only_on_generic_model_interface() -> None:
    def ask_model(model: Model) -> str:
        return model.generate(user_request("What is Forge?")).text

    model: Model = MockModel(["A local-first coding assistant."])

    assert ask_model(model) == "A local-first coding assistant."
