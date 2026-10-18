"""Focused prompt-toolkit adapter for Forge's real interactive TTY."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import DynamicStyle, Style


class CommandEntry(Protocol):
    name: str
    description: str


class CommandCompleter(Completer):
    """Complete slash commands directly from the shared command registry."""

    def __init__(self, commands: Sequence[CommandEntry]) -> None:
        self.commands = tuple(commands)

    def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def]
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        prefix = text[1:].casefold()
        for command in self.commands:
            if command.name.startswith(prefix):
                yield Completion(
                    f"/{command.name}",
                    start_position=-len(text),
                    display=f"/{command.name}",
                    display_meta=command.description,
                )


class InteractiveFrontend:
    """Non-full-screen input UI; orchestration remains outside this class."""

    def __init__(
        self,
        commands: Sequence[CommandEntry],
        toolbar: Callable[[], StyleAndTextTuples],
        *,
        theme_name: str,
        color_enabled: bool,
        input: Any = None,
        output: Any = None,
    ) -> None:
        self._toolbar = toolbar
        self._theme_name = theme_name
        self._color_enabled = color_enabled
        bindings = KeyBindings()

        @bindings.add("escape")
        def _dismiss_completion(event) -> None:  # type: ignore[no-untyped-def]
            event.current_buffer.complete_state = None

        @bindings.add("tab")
        def _complete_command(event) -> None:  # type: ignore[no-untyped-def]
            buffer = event.current_buffer
            text = buffer.document.text_before_cursor
            matches = [
                command for command in commands if f"/{command.name}".startswith(text)
            ]
            if text.startswith("/") and " " not in text and len(matches) == 1:
                buffer.text = f"/{matches[0].name}"
                buffer.cursor_position = len(buffer.text)
                buffer.complete_state = None
            else:
                buffer.start_completion(select_first=True)

        self._session: PromptSession[str] = PromptSession(
            completer=CommandCompleter(commands),
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=6,
            style=DynamicStyle(
                lambda: frontend_style(
                    self._theme_name, color_enabled=self._color_enabled
                )
            ),
            key_bindings=bindings,
            input=input,
            output=output,
        )

    def prompt(self, _legacy_prompt: str = "") -> str:
        return self._session.prompt(
            [("class:separator", "─" * 32 + "\n"), ("class:prompt", "> ")],
            bottom_toolbar=self._toolbar,
        )

    def set_theme(self, theme_name: str, *, color_enabled: bool) -> None:
        self._theme_name = theme_name
        self._color_enabled = color_enabled

    @property
    def theme_name(self) -> str:
        return self._theme_name


def frontend_style(theme_name: str, *, color_enabled: bool) -> Style:
    if not color_enabled:
        return Style.from_dict({})
    brand = "ansibrightyellow" if theme_name == "default" else "ansiyellow bold"
    return Style.from_dict(
        {
            "prompt": f"{brand} bold",
            "separator": "ansibrightblack",
            "completion-menu.completion": "bg:#303030 #eeeeee",
            "completion-menu.completion.current": "bg:#875f00 #ffffff bold",
            "completion-menu.meta.completion": "bg:#303030 #aaaaaa",
            "completion-menu.meta.completion.current": "bg:#875f00 #ffffff",
            "bottom-toolbar": "",
            "toolbar.brand": brand,
            "toolbar.mode": brand,
            "toolbar.display": "ansicyan",
            "toolbar.model": "",
            "toolbar.warm": "ansigreen",
            "toolbar.access": "ansicyan",
            "toolbar.context": "ansibrightblack",
            "toolbar.separator": "ansibrightblack",
        }
    )
