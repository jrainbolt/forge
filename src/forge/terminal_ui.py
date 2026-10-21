"""Small terminal presentation primitives with plain-text fallbacks."""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing import Event, Process, get_all_start_methods, get_context


class ColorRole(StrEnum):
    BRAND_PRIMARY = "brand_primary"
    BRAND_SECONDARY = "brand_secondary"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PATH = "path"
    TOOL = "tool"
    MODE = "mode"
    METADATA = "metadata"
    MUTED = "muted"
    PROMPT = "prompt"
    NORMAL_TEXT = "normal_text"


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    roles: Mapping[ColorRole, str]
    color_enabled: bool

    def paint(self, text: str, role: ColorRole) -> str:
        code = self.roles.get(role, "") if self.color_enabled else ""
        return f"{code}{text}\033[0m" if code else text


_DEFAULT_ROLES = {
    ColorRole.BRAND_PRIMARY: "\033[38;5;214m",
    ColorRole.BRAND_SECONDARY: "\033[38;5;202m",
    ColorRole.SUCCESS: "\033[32m",
    ColorRole.WARNING: "\033[33m",
    ColorRole.ERROR: "\033[31m",
    ColorRole.PATH: "\033[36m",
    ColorRole.TOOL: "\033[35m",
    ColorRole.MODE: "\033[38;5;214m",
    ColorRole.METADATA: "\033[2m",
    ColorRole.MUTED: "\033[2m",
    ColorRole.PROMPT: "\033[1m",
    ColorRole.NORMAL_TEXT: "",
}
_EMBER_ROLES = {
    **_DEFAULT_ROLES,
    ColorRole.BRAND_PRIMARY: "\033[38;5;202m",
    ColorRole.BRAND_SECONDARY: "\033[38;5;214m",
    ColorRole.PATH: "\033[38;5;215m",
    ColorRole.TOOL: "\033[38;5;208m",
    ColorRole.MODE: "\033[38;5;220m",
    ColorRole.METADATA: "\033[38;5;180m",
}
THEME_NAMES = ("default", "ember", "mono")


def resolve_theme(
    name: str = "default",
    *,
    interactive: bool,
    environment: Mapping[str, str] | None = None,
) -> Theme:
    values = os.environ if environment is None else environment
    selected = name.casefold()
    if selected not in THEME_NAMES:
        raise ValueError(f"unknown theme {name!r}")
    enabled = interactive and selected != "mono" and "NO_COLOR" not in values
    roles = _EMBER_ROLES if selected == "ember" else _DEFAULT_ROLES
    return Theme(selected, roles, enabled)


FORGE_MARK = (
    "        .'",
    "      .'  `.",
    "     /      \\",
    "    /  /\\    \\",
    "   /  /  \\    \\",
    "   \\  \\  /    /",
    "    \\  \\/    /",
    "     `.___.'",
    "      FORGE",
)


def render_splash(
    *,
    project: str | None,
    model: str,
    mode: str,
    display: str,
    access: str,
    context: str | None,
    theme: Theme,
    width: int | None = None,
) -> tuple[str, ...]:
    fields = (
        ("Project", project),
        ("Model", model),
        ("Mode", mode),
        ("Display", display),
        ("Access", access),
        ("Context", context),
    )
    metadata = ["FORGE", "Local AI Agent", ""]
    metadata.extend(
        f"{label:<9} {value}" for label, value in fields if value is not None
    )
    available = width or shutil.get_terminal_size(fallback=(80, 24)).columns
    if available >= 68:
        wide_mark = FORGE_MARK[:-1]
        height = max(len(wide_mark), len(metadata))
        rendered: list[str] = []
        for index in range(height):
            left = wide_mark[index] if index < len(wide_mark) else ""
            right = metadata[index] if index < len(metadata) else ""
            flame_role = (
                ColorRole.BRAND_SECONDARY
                if index in {3, 4}
                else ColorRole.BRAND_PRIMARY
            )
            left = theme.paint(f"{left:<27}", flame_role)
            role = (
                ColorRole.BRAND_PRIMARY
                if index == 0
                else ColorRole.BRAND_SECONDARY
                if index == 1
                else ColorRole.METADATA
            )
            rendered.append(left + theme.paint(right, role))
        return tuple(rendered)
    lines = [
        theme.paint(
            line,
            ColorRole.BRAND_SECONDARY if index in {3, 4} else ColorRole.BRAND_PRIMARY,
        )
        for index, line in enumerate(FORGE_MARK)
    ]
    lines.append(theme.paint("Local AI Agent", ColorRole.BRAND_SECONDARY))
    lines.extend(f"{label:<9} {value}" for label, value in fields if value is not None)
    return tuple(lines)


def truncate_status(text: str, *, width: int | None = None) -> str:
    available = width or shutil.get_terminal_size(fallback=(80, 24)).columns
    if available <= 4:
        return text[:available]
    if len(text) <= available:
        return text
    return text[: available - 3].rstrip() + "..."


class LoadingIndicator:
    """A single-line startup spinner that never owns the prompt area."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(
        self, *, enabled: bool, label: str, theme: Theme | None = None
    ) -> None:
        self.enabled = enabled
        self.label = label
        self.theme = theme or resolve_theme("mono", interactive=False)
        self._stop: Event | None = None
        self._process: Process | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._render_frame(0, elapsed=0.0)
        context = get_context("fork" if "fork" in get_all_start_methods() else "spawn")
        self._stop = context.Event()
        styled_label = self.theme.paint(self.label, ColorRole.BRAND_PRIMARY)
        self._process = context.Process(
            target=_animate_loading_line,
            args=(self._stop, styled_label),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        if self._stop is not None:
            self._stop.set()
        if self._process is not None:
            self._process.join(timeout=0.75)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.25)
        self._process = None
        self._stop = None
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _render_frame(self, index: int, *, elapsed: float) -> None:
        label = self.theme.paint(self.label, ColorRole.BRAND_PRIMARY)
        sys.stdout.write(f"\r\033[2K{self.FRAMES[index]} {label}... {elapsed:.1f}s")
        sys.stdout.flush()


def _animate_loading_line(stop: Event, styled_label: str) -> None:
    frames = LoadingIndicator.FRAMES
    index = 1
    started = time.perf_counter()
    while not stop.wait(0.1):
        elapsed = time.perf_counter() - started
        sys.stdout.write(
            f"\r\033[2K{frames[index % len(frames)]} {styled_label}... {elapsed:.1f}s"
        )
        sys.stdout.flush()
        index += 1
