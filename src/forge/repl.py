"""Small injectable terminal loop for Forge chat."""

from __future__ import annotations

from collections.abc import Callable

from forge.models import ModelError
from forge.orchestration import RepositoryChatSession, RepositoryResponse
from forge.session import ChatSession
from forge.tools import MutationPreview, PreparedProjectCommand, ToolInvocation

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
        mode = (
            "assist (writes and project execution require approval)"
            if info.assist_mode
            else "read-only"
        )
        output_fn(f"Repository access: {mode}")
        if info.assist_mode:
            session.set_approval_callback(
                _approval_prompt(input_fn=input_fn, output_fn=output_fn)
            )
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
            if isinstance(session, RepositoryChatSession):
                for activity in session.last_activity:
                    if activity.status == "success" and activity.evidence in {
                        "write_success",
                        "patch_success",
                    }:
                        output_fn(
                            "Mutation remains on disk despite conversation failure: "
                            f"{activity.path}. Code correctness was not tested."
                        )
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
            repository_mode = "assist" if info.assist_mode else "read-only"
            rendered += (
                f"\nrepository mode: {repository_mode}"
                f"\nworkspace: {info.workspace}"
                f"\navailable tools: {info.available_tools}"
                f"\nlast tool count: {info.last_tool_count}"
                f"\nbuild configured: {'yes' if info.build_configured else 'no'}"
                f"\ntest configured: {'yes' if info.test_configured else 'no'}"
                f"\nmutation generation: {info.mutation_generation}"
            )
        output_fn(rendered)
    elif command == "/exit":
        output_fn("Goodbye.")
        return True
    else:
        output_fn(f"Unknown command: {command}. Type /help for commands.")
    return False


def _approval_prompt(
    *, input_fn: InputFunction, output_fn: OutputFunction
) -> Callable[[ToolInvocation, MutationPreview | PreparedProjectCommand], bool]:
    def approve(
        invocation: ToolInvocation,
        preview: MutationPreview | PreparedProjectCommand,
    ) -> bool:
        if isinstance(preview, MutationPreview):
            output_fn(f"[proposed] {invocation.tool_name}: {preview.path}")
            output_fn(preview.diff)
            prompt = "Approve? [y/N] "
            rejection = "Write rejected."
        else:
            output_fn(f"[proposed] {invocation.tool_name}")
            output_fn(f"Workspace: {preview.workspace}")
            output_fn("Command argv:")
            for argument in preview.argv:
                output_fn(f"  {argument!r}")
            output_fn(f"Timeout: {preview.timeout_seconds:g} seconds")
            prompt = "Approve execution? [y/N] "
            rejection = "Execution rejected."
        try:
            answer = input_fn(prompt).strip().casefold()
        except (EOFError, KeyboardInterrupt):
            output_fn(rejection)
            return False
        if answer in {"y", "yes"}:
            return True
        output_fn(rejection)
        return False

    return approve
