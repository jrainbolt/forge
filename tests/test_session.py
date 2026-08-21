from __future__ import annotations

import pytest

from forge.models import (
    GenerationConfig,
    MessageRole,
    MockModel,
    ModelCapabilities,
    ModelCapability,
    ModelError,
    ModelIdentity,
)
from forge.session import ChatSession


def test_session_exposes_identity_and_executes_repeated_turns() -> None:
    identity = ModelIdentity("fixture", "mock")
    model = MockModel(("one", "two"), identity=identity)
    session = ChatSession("profile", model)

    assert session.info.identity is identity
    assert session.ask("first").text == "one"
    assert session.ask("second").text == "two"
    assert [message.content for message in model.requests[1].messages] == [
        "You are Forge, a local AI assistant.",
        "first",
        "one",
        "second",
    ]
    assert session.info.completed_turns == 2


def test_generation_failure_rolls_back_entire_pending_turn() -> None:
    model = MockModel(("only",))
    session = ChatSession("profile", model)
    session.ask("committed")

    with pytest.raises(ModelError, match="exhausted"):
        session.ask("not committed")

    assert len(session.conversation.turns) == 1
    assert session.conversation.turns[0].user.content == "committed"


def test_clear_resets_turns_without_closing_model() -> None:
    model = MockModel(("answer", "after clear"))
    session = ChatSession("profile", model)
    session.ask("question")
    session.clear()
    assert session.info.completed_turns == 0
    assert not model.closed
    assert session.ask("new question").text == "after clear"


def test_session_context_manager_closes_model() -> None:
    model = MockModel(("unused",))
    with ChatSession("profile", model):
        pass
    assert model.closed


def test_system_message_is_omitted_for_unsupported_model() -> None:
    model = MockModel(
        ("answer",),
        capabilities=ModelCapabilities(frozenset({ModelCapability.CHAT})),
    )
    session = ChatSession("profile", model)
    session.ask("question")
    assert [message.role for message in model.requests[0].messages] == [
        MessageRole.USER
    ]


def test_non_chat_model_is_rejected() -> None:
    model = MockModel(("unused",), capabilities=ModelCapabilities())
    with pytest.raises(ValueError, match="chat capability"):
        ChatSession("profile", model)


def test_session_uses_supplied_generic_generation_config() -> None:
    generation = GenerationConfig(max_tokens=12, temperature=0.7, seed=4)
    model = MockModel(("answer",))
    ChatSession("profile", model, generation=generation).ask("question")
    assert model.requests[0].generation is generation


def test_successful_bounded_request_removes_omitted_stored_turns() -> None:
    model = MockModel(("A" * 90, "recent answer", "final answer"), context_capacity=160)
    session = ChatSession(
        "profile",
        model,
        generation=GenerationConfig(max_tokens=20),
        system_message=None,
    )
    session.ask("A" * 90)
    session.ask("recent user")
    session.ask("current")

    assert session.info.last_omitted_turns == 1
    assert [turn.user.content for turn in session.conversation.turns] == [
        "recent user",
        "current",
    ]
