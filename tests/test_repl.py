from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from forge.models import MockModel
from forge.orchestration import RepositoryChatSession
from forge.repl import run_repl
from forge.session import ChatSession
from forge.tools import (
    create_assist_repository_policy,
    create_assist_repository_registry,
)


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
            '{"type":"tool_call","id":"one","tool":"repository.read_file",'
            '"arguments":{"path":"evidence.txt"}}',
            '{"type":"final","answer":"The repository has evidence."}',
        )
    )
    (workspace / "evidence.txt").write_text("evidence")
    session = RepositoryChatSession(
        "fixture", model, workspace, require_relevant_source=False
    )
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
    assert "[tool] repository.read_file: evidence.txt (success)" in rendered
    assert "repository mode: read-only" in rendered
    assert "available tools: 5" in rendered
    assert "last tool count: 1" in rendered
    assert session.info.completed_turns == 0
    assert session.info.workspace == workspace.resolve()


def _assist_session(workspace: Path, model: MockModel) -> RepositoryChatSession:
    return RepositoryChatSession(
        "fixture",
        model,
        workspace,
        registry=create_assist_repository_registry(),
        policy=create_assist_repository_policy(),
        minimum_source_files=1,
        require_relevant_source=False,
    )


def _assist_responses() -> tuple[str, ...]:
    digest = hashlib.sha256(b"VALUE = 1\n").hexdigest()
    return (
        json.dumps(
            {
                "type": "tool_call",
                "id": "read",
                "tool": "repository.read_file",
                "arguments": {"path": "value.py"},
            }
        ),
        json.dumps(
            {
                "type": "tool_call",
                "id": "patch",
                "tool": "repository.apply_patch",
                "arguments": {
                    "path": "value.py",
                    "expected_sha256": digest,
                    "edits": [{"old": "VALUE = 1", "new": "VALUE = 2"}],
                },
            }
        ),
        json.dumps({"type": "final", "answer": "Mutation handling complete."}),
    )


def test_assist_repl_previews_then_explicit_yes_approves(tmp_path: Path) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    target = workspace / "value.py"
    target.write_bytes(b"VALUE = 1\n")
    output: list[str] = []
    session = _assist_session(workspace, MockModel(_assist_responses()))
    result = run_repl(
        session,
        input_fn=_input(iter(("Change VALUE", "y", "/exit"))),
        output_fn=output.append,
    )
    rendered = "\n".join(output)
    assert result == 0
    assert "Repository access: assist" in rendered
    assert "[proposed] repository.apply_patch: value.py" in rendered
    assert "-VALUE = 1" in rendered and "+VALUE = 2" in rendered
    assert "[tool] repository.apply_patch: value.py (success)" in rendered
    assert target.read_bytes() == b"VALUE = 2\n"


def test_assist_repl_default_no_rejects_without_mutation(tmp_path: Path) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    target = workspace / "value.py"
    target.write_bytes(b"VALUE = 1\n")
    output: list[str] = []
    session = _assist_session(workspace, MockModel(_assist_responses()))
    run_repl(
        session,
        input_fn=_input(iter(("Change VALUE", "", "/exit"))),
        output_fn=output.append,
    )
    assert "Write rejected." in output
    assert target.read_bytes() == b"VALUE = 1\n"


@pytest.mark.parametrize("exception", (EOFError, KeyboardInterrupt))
def test_assist_repl_eof_or_interrupt_during_approval_rejects(
    tmp_path: Path, exception: type[BaseException]
) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    target = workspace / "value.py"
    target.write_bytes(b"VALUE = 1\n")
    main_inputs = iter(("Change VALUE", "/exit"))

    def read(prompt: str) -> str:
        if prompt.startswith("Approve?"):
            raise exception
        return next(main_inputs)

    output: list[str] = []
    run_repl(
        _assist_session(workspace, MockModel(_assist_responses())),
        input_fn=read,
        output_fn=output.append,
    )
    assert "Write rejected." in output
    assert target.read_bytes() == b"VALUE = 1\n"


def test_assist_repl_reports_persisted_mutation_after_conversation_failure(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repository"
    workspace.mkdir()
    target = workspace / "value.py"
    target.write_bytes(b"VALUE = 1\n")
    responses = (*_assist_responses()[:2], "not json", "still not json")
    output: list[str] = []
    run_repl(
        _assist_session(workspace, MockModel(responses)),
        input_fn=_input(iter(("Change VALUE", "y", "/exit"))),
        output_fn=output.append,
    )
    assert target.read_bytes() == b"VALUE = 2\n"
    assert any("Mutation remains on disk" in line for line in output)
