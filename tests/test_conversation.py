from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from forge.conversation import (
    ContextBudgetError,
    Conversation,
    ConversationTurn,
)
from forge.models import GenerationConfig, Message, MessageRole


def test_empty_conversation_has_no_messages() -> None:
    conversation = Conversation()
    assert conversation.messages == ()
    assert conversation.turns == ()


def test_system_message_is_retained_when_clearing() -> None:
    conversation = Conversation(system_message="Be concise.")
    conversation.commit("Question", "Answer")
    conversation.clear()
    assert conversation.messages == (Message(MessageRole.SYSTEM, "Be concise."),)


def test_complete_turn_ordering_and_ownership_snapshot() -> None:
    conversation = Conversation(system_message="System")
    turn = conversation.commit("First user", "First assistant")
    snapshot = conversation.turns
    conversation.commit("Second user", "Second assistant")

    assert conversation.messages == (
        Message(MessageRole.SYSTEM, "System"),
        Message(MessageRole.USER, "First user"),
        Message(MessageRole.ASSISTANT, "First assistant"),
        Message(MessageRole.USER, "Second user"),
        Message(MessageRole.ASSISTANT, "Second assistant"),
    )
    assert snapshot == (turn,)


def test_turn_is_immutable_and_validates_roles() -> None:
    turn = ConversationTurn(
        Message(MessageRole.USER, "Question"),
        Message(MessageRole.ASSISTANT, "Answer"),
    )
    with pytest.raises(FrozenInstanceError):
        turn.user = Message(MessageRole.USER, "Changed")  # type: ignore[misc]
    with pytest.raises(ValueError, match="user role"):
        ConversationTurn(
            Message(MessageRole.ASSISTANT, "Wrong"),
            Message(MessageRole.ASSISTANT, "Answer"),
        )


def test_multi_turn_request_includes_prior_complete_turns() -> None:
    conversation = Conversation(system_message="System")
    conversation.commit("First", "Answer one")
    plan = conversation.plan_request(
        "Follow-up", GenerationConfig(max_tokens=20), context_capacity=1000
    )
    assert [message.content for message in plan.request.messages] == [
        "System",
        "First",
        "Answer one",
        "Follow-up",
    ]
    assert plan.omitted_turns == 0


def test_bounded_history_keeps_contiguous_recent_complete_turns() -> None:
    conversation = Conversation()
    conversation.commit("A" * 90, "B" * 90)
    conversation.commit("recent user", "recent assistant")
    plan = conversation.plan_request(
        "current",
        GenerationConfig(max_tokens=20),
        context_capacity=130,
        safety_reserve=20,
    )
    assert [message.content for message in plan.request.messages] == [
        "recent user",
        "recent assistant",
        "current",
    ]
    assert plan.omitted_turns == 1


def test_required_messages_that_do_not_fit_fail_clearly() -> None:
    conversation = Conversation(system_message="System")
    with pytest.raises(ContextBudgetError, match="exceed"):
        conversation.plan_request(
            "x" * 1000,
            GenerationConfig(max_tokens=20),
            context_capacity=100,
            safety_reserve=20,
        )


def test_estimate_is_explicitly_labeled_as_estimated() -> None:
    conversation = Conversation()
    plan = conversation.plan_request(
        "hello", GenerationConfig(max_tokens=20), context_capacity=1000
    )
    assert "estimated" in conversation.estimator_label
    assert plan.estimated_input_tokens > 0


def test_temporary_orchestration_messages_are_budgeted_but_not_persisted() -> None:
    conversation = Conversation(system_message="System")
    temporary = (
        Message(MessageRole.ASSISTANT, "tool request"),
        Message(MessageRole.USER, "tool result"),
    )
    plan = conversation.plan_request(
        "question",
        GenerationConfig(max_tokens=20),
        context_capacity=1000,
        temporary_messages=temporary,
    )
    assert plan.request.messages[-2:] == temporary
    assert conversation.turns == ()
    assert conversation.messages == (Message(MessageRole.SYSTEM, "System"),)
