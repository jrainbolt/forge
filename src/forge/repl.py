"""Small injectable terminal loop for Forge chat."""

from __future__ import annotations

from collections.abc import Callable

from forge.models import ModelError
from forge.orchestration import RepositoryChatSession, RepositoryResponse
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
    session: ChatSession | RepositoryChatSession,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    """Run a synchronous chat loop against one already-loaded session."""
    info = session.info
    output_fn(f"Forge [{info.profile_name}]")
    if isinstance(session, RepositoryChatSession):
        output_fn(f"Workspace: {info.workspace}")
        output_fn("Repository access: read-only")
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
        if isinstance(response, RepositoryResponse):
            for activity in response.tool_activity:
                path = f": {activity.path}" if activity.path is not None else ""
                output_fn(f"[tool] {activity.tool_name}{path} ({activity.status})")
        output_fn(f"\n{response.text}")


def _run_command(
    command: str,
    session: ChatSession | RepositoryChatSession,
    output_fn: OutputFunction,
) -> bool:
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
        rendered = (
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
        if isinstance(session, RepositoryChatSession):
            rendered += (
                "\nrepository mode: read-only"
                f"\nworkspace: {info.workspace}"
                f"\navailable tools: {info.available_tools}"
                f"\nlast tool count: {info.last_tool_count}"
            )
        output_fn(rendered)
    elif command == "/exit":
        output_fn("Goodbye.")
        return True
    else:
        output_fn(f"Unknown command: {command}. Type /help for commands.")
    return False
