"""Focused prompt-toolkit adapter for Forge's real interactive TTY."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, Window
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import DynamicStyle, Style


class CommandEntry(Protocol):
    name: str
    description: str
    options: tuple[str, ...]


class CommandCompleter(Completer):
    """Complete slash commands directly from the shared command registry."""

    def __init__(self, commands: Sequence[CommandEntry]) -> None:
        self.commands = tuple(commands)

    def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def]
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        if " " in text:
            command_text, value = text.split(" ", 1)
            command_name = command_text.removeprefix("/").casefold()
            command = next(
                (item for item in self.commands if item.name == command_name), None
            )
            if command is None or " " in value:
                return
            for option in command.options:
                if option.startswith(value.casefold()):
                    yield Completion(
                        option,
                        start_position=-len(value),
                        display=option,
                        display_meta=f"{command.name} value",
                    )
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
        rich_busy_supported: bool | None = None,
    ) -> None:
        self._toolbar = toolbar
        self._output = output
        self._theme_name = theme_name
        self._color_enabled = color_enabled
        self._busy_label = "Working..."
        self._busy_application: Application[None] | None = None
        self._busy_thread: threading.Thread | None = None
        self._busy_input = None
        self._busy_input_context = None
        # The busy application uses a pipe so it cannot queue user input. That
        # pipe also cannot receive terminal CPR replies, so rich rendering is
        # safe only when support was independently confirmed by the caller.
        self._rich_busy_supported = bool(rich_busy_supported)
        bindings = KeyBindings()

        @bindings.add("escape")
        def _dismiss_completion(event) -> None:  # type: ignore[no-untyped-def]
            event.current_buffer.complete_state = None

        @bindings.add("tab")
        def _complete_command(event) -> None:  # type: ignore[no-untyped-def]
            buffer = event.current_buffer
            text = buffer.document.text_before_cursor
            completions = tuple(
                CommandCompleter(commands).get_completions(Document(text), None)
            )
            if len(completions) == 1:
                completion = completions[0]
                start = buffer.cursor_position + completion.start_position
                buffer.text = buffer.text[:start] + completion.text
                buffer.cursor_position = start + len(completion.text)
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
        resume_busy = self.busy
        if resume_busy:
            self.stop_busy()
        try:
            return self._session.prompt(
                [("class:separator", "─" * 32 + "\n"), ("class:prompt", "> ")],
                bottom_toolbar=self._toolbar,
            )
        finally:
            if resume_busy:
                self.start_busy(self._busy_label)

    @property
    def busy(self) -> bool:
        return self._busy_thread is not None and self._busy_thread.is_alive()

    @property
    def rich_busy_supported(self) -> bool:
        return self._rich_busy_supported

    def start_busy(self, label: str = "Working...") -> None:
        if not self._rich_busy_supported:
            return
        if self.busy:
            self.update_busy(label)
            return
        self._busy_label = label
        self._busy_input_context = create_pipe_input()
        self._busy_input = self._busy_input_context.__enter__()
        control = FormattedTextControl(
            lambda: [("class:tool", f"⠹ {self._busy_label}")]
        )
        toolbar = FormattedTextControl(self._toolbar)
        self._busy_application = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(control, height=1, dont_extend_height=True),
                        Window(toolbar, height=1, dont_extend_height=True),
                    ]
                )
            ),
            style=DynamicStyle(
                lambda: frontend_style(
                    self._theme_name, color_enabled=self._color_enabled
                )
            ),
            full_screen=False,
            erase_when_done=True,
            refresh_interval=0.1,
            terminal_size_polling_interval=0.2,
            input=self._busy_input,
            output=self._output,
        )
        self._busy_thread = threading.Thread(
            target=self._busy_application.run,
            daemon=True,
        )
        self._busy_thread.start()

    def update_busy(self, label: str) -> None:
        self._busy_label = label
        if self._busy_application is not None:
            self._busy_application.invalidate()

    def stop_busy(self) -> None:
        application = self._busy_application
        thread = self._busy_thread
        if application is not None and application.loop is not None:
            application.loop.call_soon_threadsafe(application.exit)
        if thread is not None:
            thread.join(timeout=1)
        if self._busy_input_context is not None:
            self._busy_input_context.__exit__(None, None, None)
        self._busy_application = None
        self._busy_thread = None
        self._busy_input = None
        self._busy_input_context = None

    def set_theme(self, theme_name: str, *, color_enabled: bool) -> None:
        self._theme_name = theme_name
        self._color_enabled = color_enabled

    @property
    def theme_name(self) -> str:
        return self._theme_name


def frontend_style(theme_name: str, *, color_enabled: bool) -> Style:
    if not color_enabled:
        return Style.from_dict({})
    ember = theme_name == "ember"
    brand = "#ff5f00 bold" if ember else "ansibrightyellow"
    display = "#ff8700" if ember else "ansicyan"
    access = "#ffaf5f" if ember else "ansicyan"
    tool = "#ff8700" if ember else "ansimagenta"
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
            "toolbar.display": display,
            "toolbar.model": "",
            "toolbar.warm": "ansigreen",
            "toolbar.access": access,
            "toolbar.context": "ansibrightblack",
            "toolbar.separator": "ansibrightblack",
            "tool": tool,
        }
    )
