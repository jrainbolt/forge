"""Small terminal presentation primitives with plain-text fallbacks."""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


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
    ColorRole.BRAND_PRIMARY: "\033[38;5;208m",
    ColorRole.BRAND_SECONDARY: "\033[38;5;196m",
    ColorRole.MODE: "\033[38;5;208m",
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
            left = theme.paint(f"{left:<27}", ColorRole.BRAND_PRIMARY)
            role = (
                ColorRole.BRAND_PRIMARY
                if index == 0
                else ColorRole.BRAND_SECONDARY
                if index == 1
                else ColorRole.METADATA
            )
            rendered.append(left + theme.paint(right, role))
        return tuple(rendered)
    lines = [theme.paint(line, ColorRole.BRAND_PRIMARY) for line in FORGE_MARK]
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


class LoadingFlame:
    """Finite-frame multi-line Forge flame animation."""

    FRAMES = (
        (
            "        .'",
            "      .'  `.",
            "     /      \\",
            "    /  /\\    \\",
            "   /  /  \\    \\",
            "   \\  \\  /    /",
            "    \\  \\/    /",
            "     `.___.'",
        ),
        (
            "       * .",
            "       .' `.",
            "      /     \\",
            "    /   /\\   \\",
            "   /   /  \\   \\",
            "   \\   \\ /   /",
            "    \\   \\/  /",
            "     `.___.'",
        ),
        (
            "         *",
            "        `.",
            "      .'  `.",
            "     /  /\\  \\",
            "    /  /  \\  \\",
            "    \\  \\  /  /",
            "     \\  \\/  /",
            "      `.__.'",
        ),
        (
            "       . '",
            "      .'  `.",
            "     /      \\",
            "    /  /\\   \\",
            "   /  /  \\   \\",
            "   \\   \\ /   /",
            "    \\   \\/  /",
            "     `.___.'",
        ),
    )

    def __init__(
        self, *, enabled: bool, label: str, theme: Theme | None = None
    ) -> None:
        self.enabled = enabled
        self.label = label
        self.theme = theme or resolve_theme("mono", interactive=False)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rendered = False

    def start(self) -> None:
        if not self.enabled:
            return
        self._stop.clear()
        self._render_frame(0, elapsed=0.0)
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None
        if self._rendered:
            sys.stdout.write(f"\033[{len(self.FRAMES[0]) + 1}A")
            for _ in range(len(self.FRAMES[0]) + 1):
                sys.stdout.write("\r\033[2K\n")
            sys.stdout.write(f"\033[{len(self.FRAMES[0]) + 1}A")
        sys.stdout.flush()
        self._rendered = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _animate(self) -> None:
        index = 0
        started = time.perf_counter()
        while not self._stop.wait(0.18):
            elapsed = time.perf_counter() - started
            self._render_frame(index % len(self.FRAMES), elapsed=elapsed)
            index += 1

    def _render_frame(self, index: int, *, elapsed: float) -> None:
        if self._rendered:
            sys.stdout.write(f"\033[{len(self.FRAMES[index]) + 1}A")
        for line_number, line in enumerate(self.FRAMES[index]):
            role = (
                ColorRole.BRAND_SECONDARY
                if "*" in line or line_number in {3, 4}
                else ColorRole.BRAND_PRIMARY
            )
            sys.stdout.write(f"\r\033[2K{self.theme.paint(line, role)}\n")
        label = self.theme.paint(f"{self.label} {elapsed:.1f}s", ColorRole.METADATA)
        sys.stdout.write(f"\r\033[2K{label}\n")
        sys.stdout.flush()
        self._rendered = True
