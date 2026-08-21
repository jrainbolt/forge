"""Small injectable terminal loop for Forge chat."""

from __future__ import annotations

from collections.abc import Callable

from forge.models import ModelError
from forge.session import ChatSession

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]

HELP_TEXT = (
    "/help  Show commands\n"
    "/clear Clear conversation\n"
    "/info  Show session info\n"
    "/exit  Exit Forge"
)


def run_repl(
    session: ChatSession,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    """Run a synchronous chat loop against one already-loaded session."""
    info = session.info
    output_fn(f"Forge [{info.profile_name}]")
    output_fn("Type /help for commands.")
    while True:
        try:
            text = input_fn("\n> ").strip()
        except EOFError:
            output_fn("\nGoodbye.")
            return 0
        except KeyboardInterrupt:
            output_fn("\nGoodbye.")
            return 130
        if not text:
            continue
        if text.startswith("/"):
            if _run_command(text, session, output_fn):
                return 0
            continue
        try:
            response = session.ask(text)
        except (ModelError, ValueError, RuntimeError) as error:
            output_fn(f"Error: {error}")
            continue
        output_fn(f"\n{response.text}")


def _run_command(command: str, session: ChatSession, output_fn: OutputFunction) -> bool:
    if command == "/help":
        output_fn(HELP_TEXT)
    elif command == "/clear":
        session.clear()
        output_fn("Conversation cleared.")
    elif command == "/info":
        info = session.info
        capacity = (
            info.context_capacity if info.context_capacity is not None else "unknown"
        )
        estimate = (
            info.last_estimated_input_tokens
            if info.last_estimated_input_tokens is not None
            else "unavailable"
        )
        output_fn(
            f"profile: {info.profile_name}\n"
            f"model: {info.identity.model_id}\n"
            f"backend: {info.identity.backend_id}\n"
            f"context capacity: {capacity}\n"
            f"messages: {info.message_count}\n"
            f"completed turns: {info.completed_turns}\n"
            f"last input estimate: {estimate}\n"
            f"estimate method: {info.estimate_method}\n"
            f"last omitted turns: {info.last_omitted_turns}"
        )
    elif command == "/exit":
        output_fn("Goodbye.")
        return True
    else:
        output_fn(f"Unknown command: {command}. Type /help for commands.")
    return False
