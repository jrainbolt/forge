from __future__ import annotations

import io
import os
import sys
import threading

from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from forge.interactive_frontend import (
    CommandCompleter,
    InteractiveFrontend,
    frontend_style,
)
from forge.models import MockModel
from forge.repl import (
    COMMAND_REGISTRY,
    _toolbar_fragments,
    command_help,
    command_suggestions,
    run_repl,
)
from forge.session import ChatSession
from forge.terminal_ui import (
    ColorRole,
    LoadingIndicator,
    render_splash,
    resolve_theme,
    truncate_status,
)


def _input(values: tuple[str, ...]):
    iterator = iter(values)
    return lambda _prompt: next(iterator)


def test_command_registry_is_unique_alphabetical_and_drives_help() -> None:
    names = tuple(item.name for item in COMMAND_REGISTRY)
    assert names == tuple(sorted(names))
    assert len(names) == len(set(names))
    assert tuple(item.name for item in command_suggestions("/tr")) == ("trace",)
    assert "/trace" in command_help()
    assert "/theme default|ember|mono" in command_help()


def test_theme_honors_plain_output_and_no_color() -> None:
    assert not resolve_theme("default", interactive=False).color_enabled
    assert not resolve_theme(
        "ember", interactive=True, environment={"NO_COLOR": "1"}
    ).color_enabled
    assert not resolve_theme("mono", interactive=True).color_enabled
    assert resolve_theme("default", interactive=True, environment={}).color_enabled


def test_splash_has_brand_and_only_known_metadata() -> None:
    lines = render_splash(
        project="forge",
        model="qwen-large",
        mode="PLAN",
        display="NORMAL",
        access="PLAN",
        context=None,
        theme=resolve_theme("mono", interactive=True),
    )
    rendered = "\n".join(lines)
    assert "FORGE" in rendered
    assert "Local AI Agent" in rendered
    assert "Project   forge" in rendered
    assert "Context" not in rendered


def test_splash_uses_side_by_side_layout_when_wide() -> None:
    lines = render_splash(
        project="forge",
        model="qwen-large",
        mode="PLAN",
        display="NORMAL",
        access="READ ONLY",
        context="8.2k",
        theme=resolve_theme("mono", interactive=True),
        width=100,
    )
    assert len(lines) == 9
    assert " .'" in lines[0] and "FORGE" in lines[0]
    assert any("Project" in line and "forge" in line for line in lines)


def test_splash_stacks_when_narrow() -> None:
    lines = render_splash(
        project="forge",
        model="qwen-large",
        mode="PLAN",
        display="NORMAL",
        access="READ ONLY",
        context="8.2k",
        theme=resolve_theme("mono", interactive=True),
        width=40,
    )
    assert lines[0].strip() == ".'"
    assert "FORGE" in lines[8]
    assert any(line.startswith("Project") for line in lines[9:])


def test_status_truncation_respects_narrow_terminal() -> None:
    assert truncate_status("abcdefghij", width=7) == "abcd..."
    assert truncate_status("abc", width=7) == "abc"


def test_loading_flame_disabled_has_no_background_thread() -> None:
    indicator = LoadingIndicator(enabled=False, label="Loading fixture")
    indicator.start()
    indicator.stop()
    assert not indicator.running


def test_loading_indicator_is_one_line_and_has_no_vertical_cursor_motion(
    monkeypatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    indicator = LoadingIndicator(enabled=True, label="Loading fixture")
    indicator.start()
    indicator.stop()
    rendered = stream.getvalue()
    assert not indicator.running
    assert "Loading fixture" in rendered
    assert "\n" not in rendered
    assert not any(
        f"\033[{count}{direction}" in rendered
        for count in range(1, 20)
        for direction in "AB"
    )


def test_command_completer_reads_registry_and_filters_prefix() -> None:
    completer = CommandCompleter(COMMAND_REGISTRY)
    completions = tuple(completer.get_completions(Document("/tr"), None))
    assert tuple(item.text for item in completions) == ("/trace",)
    all_commands = tuple(completer.get_completions(Document("/"), None))
    assert tuple(item.text for item in all_commands) == tuple(
        f"/{item.name}" for item in COMMAND_REGISTRY
    )


def test_interactive_frontend_tab_completion_and_escape() -> None:
    with create_pipe_input() as pipe:
        frontend = InteractiveFrontend(
            COMMAND_REGISTRY,
            lambda: [("class:toolbar.mode", "PLAN")],
            theme_name="mono",
            color_enabled=False,
            input=pipe,
            output=DummyOutput(),
        )
        result: list[str] = []
        worker = threading.Thread(target=lambda: result.append(frontend.prompt()))
        worker.start()
        pipe.send_text("/\x1b\x15/tr\t\n")
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert result == ["/trace"]


def test_toolbar_uses_foreground_styles_only() -> None:
    for theme_name in ("default", "ember", "mono"):
        rules = dict(
            frontend_style(theme_name, color_enabled=theme_name != "mono").style_rules
        )
        toolbar_rules = {
            name: value
            for name, value in rules.items()
            if name == "bottom-toolbar" or name.startswith("toolbar.")
        }
        assert toolbar_rules.get("bottom-toolbar", "") == ""
        assert all("bg:" not in value for value in toolbar_rules.values())


def test_frontend_instances_keep_independent_theme_state() -> None:
    first = InteractiveFrontend(
        COMMAND_REGISTRY,
        lambda: [],
        theme_name="default",
        color_enabled=True,
        output=DummyOutput(),
    )
    second = InteractiveFrontend(
        COMMAND_REGISTRY,
        lambda: [],
        theme_name="mono",
        color_enabled=False,
        output=DummyOutput(),
    )
    first.set_theme("ember", color_enabled=True)
    assert first.theme_name == "ember"
    assert second.theme_name == "mono"


def test_toolbar_responsively_uses_semantic_spans(monkeypatch) -> None:
    session = ChatSession("qwen-large", MockModel(("unused",)))
    from forge.repl import DisplayLevel, _ReplState

    state = _ReplState(
        DisplayLevel.TRACE,
        "PLAN",
        resolve_theme("default", interactive=True, environment={}),
        True,
        True,
    )
    monkeypatch.setattr(
        "forge.repl.shutil.get_terminal_size",
        lambda fallback: os.terminal_size((90, 24)),
    )
    wide = _toolbar_fragments(session, state)
    assert ("class:toolbar.brand", "Forge") in wide
    assert ("class:toolbar.display", "TRACE") in wide
    assert ("class:toolbar.warm", "warm") in wide
    assert len({style for style, _ in wide}) > 4
    monkeypatch.setattr(
        "forge.repl.shutil.get_terminal_size",
        lambda fallback: os.terminal_size((30, 24)),
    )
    narrow = _toolbar_fragments(session, state)
    assert ("class:toolbar.mode", "PLAN") in narrow
    assert not any(style == "class:toolbar.display" for style, _ in narrow)


def test_normal_text_role_is_unstyled_and_semantic_roles_differ() -> None:
    theme = resolve_theme("default", interactive=True, environment={})
    assert theme.paint("assistant", ColorRole.NORMAL_TEXT) == "assistant"
    assert theme.paint("tool", ColorRole.TOOL) != "tool"
    assert theme.paint("path", ColorRole.PATH) != theme.paint(
        "path", ColorRole.METADATA
    )


def test_interactive_repl_renders_splash_palette_and_status() -> None:
    output: list[str] = []
    run_repl(
        ChatSession("fixture", MockModel(("unused",))),
        input_fn=_input(("/", "/theme mono", "/minimal", "/trace", "/exit")),
        output_fn=output.append,
        interactive=True,
    )
    rendered = "\n".join(output)
    assert "Local AI Agent" in rendered
    assert "/trace" in rendered
    assert "Theme: mono" in rendered
    assert "Display: MINIMAL" in rendered
    assert "Display: TRACE" in rendered
    assert "Forge · CHAT · TRACE" in rendered


def test_noninteractive_repl_uses_legacy_plain_header_without_status_bar() -> None:
    output: list[str] = []
    run_repl(
        ChatSession("fixture", MockModel(("unused",))),
        input_fn=_input(("/exit",)),
        output_fn=output.append,
        interactive=False,
    )
    rendered = "\n".join(output)
    assert "Forge ready." in rendered
    assert "Local AI Agent" not in rendered
    assert "Forge ·" not in rendered
    assert "\x1b[" not in rendered
    assert "/help commands · Ctrl-C exit" in rendered
    assert "Use /multi" not in rendered
    assert "Session history" not in rendered
