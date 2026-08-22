from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from forge.models import MockModel
from forge.orchestration import RepositoryChatSession
from forge.repl import run_repl
from forge.session import ChatSession


def _input(values: Iterator[str]):
    def read(_prompt: str) -> str:
        return next(values)

    return read


def test_repl_commands_and_exit() -> None:
    output: list[str] = []
    model = MockModel(("answer",))
    session = ChatSession("fixture", model)
    result = run_repl(
        session,
        input_fn=_input(iter(("/help", "question", "/info", "/clear", "/exit"))),
        output_fn=output.append,
    )
    rendered = "\n".join(output)
    assert result == 0
    assert "/clear" in rendered
    assert "answer" in rendered
    assert "completed turns: 1" in rendered
    assert "Conversation cleared." in rendered
    assert session.info.completed_turns == 0


def test_repl_ignores_empty_input_and_handles_eof() -> None:
    output: list[str] = []
    values = iter(("",))

    def read(_prompt: str) -> str:
        try:
            return next(values)
        except StopIteration as error:
            raise EOFError from error

    assert (
        run_repl(
            ChatSession("fixture", MockModel(("unused",))),
            input_fn=read,
            output_fn=output.append,
        )
        == 0
    )
    assert "Goodbye." in "\n".join(output)


def test_repl_handles_keyboard_interrupt() -> None:
    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    output: list[str] = []
    result = run_repl(
        ChatSession("fixture", MockModel(("unused",))),
        input_fn=interrupt,
        output_fn=output.append,
    )
    assert result == 130
    assert "Goodbye." in "\n".join(output)


def test_repository_repl_shows_workspace_tools_and_extended_info(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    model = MockModel(
        (
            '<forge_tool_call>\n{"id":"one","tool":"repository.list_directory",'
            '"arguments":{"path":"."}}\n</forge_tool_call>',
            "The repository is empty.",
        )
    )
    session = RepositoryChatSession("fixture", model, workspace)
    output: list[str] = []
    result = run_repl(
        session,
        input_fn=_input(iter(("Inspect it", "/info", "/clear", "/exit"))),
        output_fn=output.append,
    )
    rendered = "\n".join(output)
    assert result == 0
    assert f"Workspace: {workspace.resolve()}" in rendered
    assert "Repository access: read-only" in rendered
    assert "[tool] repository.list_directory: . (success)" in rendered
    assert "repository mode: read-only" in rendered
    assert "available tools: 5" in rendered
    assert "last tool count: 1" in rendered
    assert session.info.completed_turns == 0
    assert session.info.workspace == workspace.resolve()
