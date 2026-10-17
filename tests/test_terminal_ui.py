from __future__ import annotations

from forge.models import MockModel
from forge.repl import COMMAND_REGISTRY, command_help, command_suggestions, run_repl
from forge.session import ChatSession
from forge.terminal_ui import (
    LoadingFlame,
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


def test_status_truncation_respects_narrow_terminal() -> None:
    assert truncate_status("abcdefghij", width=7) == "abcd..."
    assert truncate_status("abc", width=7) == "abc"


def test_loading_flame_disabled_has_no_background_thread() -> None:
    flame = LoadingFlame(enabled=False, label="Loading fixture")
    flame.start()
    flame.stop()
    assert not flame.running


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
