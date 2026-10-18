"""Small injectable terminal loop for Forge chat."""

from __future__ import annotations

import shutil
import sys
import termios
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from forge.ephemeral_acceptance import (
    EphemeralAcceptanceMode,
    EphemeralAcceptancePreview,
)
from forge.interactive_frontend import InteractiveFrontend
from forge.models import ModelError
from forge.orchestration import (
    AgentCancelled,
    RepositoryChatSession,
    RepositoryResponse,
)
from forge.orchestration.repository_session import requires_repository_grounding
from forge.session import ChatSession
from forge.terminal_ui import (
    ColorRole,
    Theme,
    render_splash,
    resolve_theme,
    truncate_status,
)
from forge.tools import (
    MultiFileMutationPreview,
    MutationPreview,
    PreparedProjectCommand,
    ToolInvocation,
)

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    description: str
    arguments: str = ""
    aliases: tuple[str, ...] = ()
    availability: str = "all"

    @property
    def invocation(self) -> str:
        suffix = f" {self.arguments}" if self.arguments else ""
        return f"/{self.name}{suffix}"


COMMAND_REGISTRY = tuple(
    sorted(
        (
            CommandSpec("chat", "Switch to conversational intent"),
            CommandSpec("clear", "Clear conversation"),
            CommandSpec("display", "Set terminal detail", "minimal|normal|trace"),
            CommandSpec("exit", "Exit Forge"),
            CommandSpec("help", "Show commands"),
            CommandSpec("info", "Show session info"),
            CommandSpec("minimal", "Use minimal terminal detail"),
            CommandSpec("mode", "Switch interaction intent", "chat|plan"),
            CommandSpec(
                "multi", "Enter multiline input; finish with a line containing only ."
            ),
            CommandSpec("normal", "Use normal terminal detail"),
            CommandSpec("plan", "Switch to read-only planning intent"),
            CommandSpec("theme", "Select terminal theme", "default|ember|mono"),
            CommandSpec("trace", "Use trace detail (on/off also accepted)"),
        ),
        key=lambda item: item.name,
    )
)


def command_suggestions(command: str) -> tuple[CommandSpec, ...]:
    prefix = command.strip().removeprefix("/").casefold()
    return tuple(item for item in COMMAND_REGISTRY if item.name.startswith(prefix))


def command_help() -> str:
    width = max(len(item.invocation) for item in COMMAND_REGISTRY)
    return "\n".join(
        f"{item.invocation:<{width}}  {item.description}" for item in COMMAND_REGISTRY
    )


HELP_TEXT = command_help()


class DisplayLevel(StrEnum):
    """User-controlled operational visibility, independent of authority."""

    MINIMAL = "minimal"
    NORMAL = "normal"
    TRACE = "trace"


@dataclass(slots=True)
class _ReplState:
    display: DisplayLevel
    user_mode: str
    theme: Theme
    interactive: bool
    warm: bool = False


class _Spinner:
    """Single-line TTY feedback; never used for redirected/test output."""

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._label = "Thinking..."
        self._started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._terminal_attributes: list[object] | None = None

    def start(self, label: str = "Thinking...") -> None:
        if not self._enabled:
            return
        self._label = label
        self._started = time.perf_counter()
        self._stop.clear()
        sys.stdout.write("Forge busy — input is not queued; Ctrl-C exits.\n")
        sys.stdout.flush()
        try:
            descriptor = sys.stdin.fileno()
            attributes = termios.tcgetattr(descriptor)
            busy_attributes = list(attributes)
            busy_attributes[3] &= ~termios.ECHO
            termios.tcsetattr(descriptor, termios.TCSANOW, busy_attributes)
            self._terminal_attributes = attributes
        except (AttributeError, OSError, termios.error):
            self._terminal_attributes = None
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def update(self, label: str) -> None:
        self._label = label

    def finish(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._terminal_attributes is not None:
            try:
                descriptor = sys.stdin.fileno()
                termios.tcflush(descriptor, termios.TCIFLUSH)
                termios.tcsetattr(
                    descriptor, termios.TCSANOW, self._terminal_attributes
                )
            except (AttributeError, OSError, termios.error):
                pass
            self._terminal_attributes = None
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    def _animate(self) -> None:
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        index = 0
        while not self._stop.wait(0.1):
            elapsed = time.perf_counter() - self._started
            sys.stdout.write(
                f"\r{frames[index % len(frames)]} {self._label} {elapsed:.1f}s"
            )
            sys.stdout.flush()
            index += 1


def run_repl(
    session: ChatSession | RepositoryChatSession,
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
    display_level: str = "normal",
    user_mode: str | None = None,
    interactive: bool | None = None,
    theme_name: str = "default",
) -> int:
    """Run a synchronous chat loop against one already-loaded session."""
    tty = (
        sys.stdin.isatty() and sys.stdout.isatty()
        if interactive is None
        else interactive
    )
    state = _ReplState(
        display=DisplayLevel(display_level),
        user_mode=user_mode or _default_user_mode(session),
        theme=resolve_theme(
            theme_name,
            interactive=tty and output_fn is print,
        ),
        interactive=tty,
    )
    spinner = _Spinner(tty and output_fn is print)
    info = session.info
    rich_frontend = tty and input_fn is input and output_fn is print
    frontend = (
        InteractiveFrontend(
            COMMAND_REGISTRY,
            lambda: _toolbar_fragments(session, state),
            theme_name=state.theme.name,
            color_enabled=state.theme.color_enabled,
        )
        if rich_frontend
        else None
    )
    read_input = frontend.prompt if frontend is not None else input_fn
    restore_completion = _install_command_completion(
        enabled=tty and frontend is None and input_fn is input
    )
    if tty:
        workspace = getattr(info, "workspace", None)
        project = Path(workspace).name if workspace is not None else None
        context = (
            _compact_number(info.context_capacity)
            if info.context_capacity is not None
            else None
        )
        for line in render_splash(
            project=project,
            model=info.profile_name,
            mode=state.user_mode,
            display=state.display.value.upper(),
            access=_access_badge(info, long=True),
            context=context,
            theme=state.theme,
            width=shutil.get_terminal_size(fallback=(80, 24)).columns,
        ):
            output_fn(line)
    else:
        output_fn("Forge ready.")
        output_fn(f"Forge [{info.profile_name}]")
    if isinstance(session, RepositoryChatSession):
        if not tty:
            output_fn(f"Workspace: {info.workspace}")
        legacy_access = (
            "agent repair"
            if info.repair_enabled
            else "agent"
            if info.agent_mode
            else "assist"
            if info.assist_mode
            else "read-only"
        )
        if not tty:
            output_fn(f"Repository access: {legacy_access}")
            output_fn(
                f"[{_permission_badge(info)}] {state.user_mode} · "
                f"display {state.display.value.upper()} · {_context_gauge(info)}"
            )
            output_fn(f"Mode: {info.autonomy_mode.value.upper()}")
            output_fn(f"Permissions: {info.permission_profile}")
            output_fn(f"Structural checks: {info.structural_completeness_mode.upper()}")
            output_fn(
                "Reads: "
                + _permission_label(info.read_permission)
                + " | Writes: "
                + _permission_label(info.write_permission)
                + " | Build/Test: "
                + _permission_label(info.build_permission)
                + "/"
                + _permission_label(info.test_permission)
            )
        if info.assist_mode:
            session.set_approval_callback(
                _approval_prompt(
                    input_fn=read_input,
                    output_fn=output_fn,
                    cancel_on_interrupt=info.agent_mode,
                )
            )
            if session.ephemeral_acceptance_mode is not EphemeralAcceptanceMode.OFF:
                session.set_ephemeral_review_callback(
                    _ephemeral_review_prompt(input_fn=read_input, output_fn=output_fn)
                )
        session.set_activity_callback(
            lambda activity: spinner.update(_activity_progress(activity))
        )
    elif not tty:
        output_fn(
            f"[CHAT] CHAT · display {state.display.value.upper()} · "
            f"{_context_gauge(info)}"
        )
    output_fn("/help commands · Ctrl-C exit")
    while True:
        if tty and frontend is None:
            width = min(48, shutil.get_terminal_size(fallback=(80, 24)).columns)
            output_fn(state.theme.paint("─" * max(1, width), ColorRole.MUTED))
            output_fn(
                state.theme.paint(
                    truncate_status(_status_line(session, state)),
                    ColorRole.METADATA,
                )
            )
        try:
            prompt = state.theme.paint("> ", ColorRole.PROMPT) if tty else "\n> "
            text = read_input(prompt)
        except EOFError:
            output_fn("\nGoodbye.")
            restore_completion()
            return 0
        except KeyboardInterrupt:
            output_fn("\nGoodbye.")
            restore_completion()
            return 130
        if not text.strip():
            continue
        if text.startswith("/"):
            command = text.strip()
            if command == "/multi":
                text = _read_multiline(input_fn, output_fn)
                if text is None:
                    continue
            elif command == "/" or (
                " " not in command
                and command_suggestions(command)
                and not any(command == f"/{item.name}" for item in COMMAND_REGISTRY)
            ):
                output_fn(_render_suggestions(command))
                continue
            elif _run_command(command, session, output_fn, state):
                restore_completion()
                return 0
            else:
                if frontend is not None:
                    frontend.set_theme(
                        state.theme.name,
                        color_enabled=state.theme.color_enabled,
                    )
                continue
        started = time.perf_counter()
        spinner.start("Thinking...")
        try:
            response = (
                session.ask_conversational(text)
                if isinstance(session, RepositoryChatSession)
                and state.user_mode == "CHAT"
                and not requires_repository_grounding(text)
                else session.ask(text)
            )
        except KeyboardInterrupt:
            spinner.finish()
            output_fn("\nForge interrupted. Exiting; no incomplete turn was committed.")
            if isinstance(session, RepositoryChatSession):
                completed_writes = tuple(
                    item.path
                    for item in session.last_activity
                    if item.status == "success"
                    and item.evidence in {"write_success", "patch_success"}
                )
                if completed_writes:
                    output_fn(
                        "Already completed approved mutations remain on disk: "
                        + ", ".join(path for path in completed_writes if path)
                    )
            restore_completion()
            return 130
        except AgentCancelled:
            spinner.finish()
            output_fn(state.theme.paint("Agent task cancelled.", ColorRole.WARNING))
            if session.last_agent_task is not None:
                output_fn(f"\n{session.last_agent_task.footer}")
            continue
        except (ModelError, ValueError, RuntimeError) as error:
            spinner.finish()
            output_fn(
                state.theme.paint(f"Error: {_present_error(error)}", ColorRole.ERROR)
            )
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
                if session.last_coding_task is not None:
                    output_fn(f"\n{session.last_coding_task.footer}")
            continue
        spinner.finish()
        state.warm = True
        if isinstance(response, RepositoryResponse):
            for activity in response.tool_activity:
                rendered = _render_activity(activity, state.display, state.theme)
                if rendered is not None:
                    output_fn(rendered)
        output_fn("\n" + state.theme.paint(response.text, ColorRole.NORMAL_TEXT))
        if isinstance(response, RepositoryResponse):
            sources = _trusted_sources(response)
            if sources and state.display is not DisplayLevel.MINIMAL:
                output_fn(_render_sources(sources, state.theme))
            if state.display is DisplayLevel.TRACE:
                output_fn(_task_checklist(response, session))
        footer = _performance_footer(response, time.perf_counter() - started)
        if footer and state.display is not DisplayLevel.MINIMAL:
            output_fn(state.theme.paint(footer, ColorRole.METADATA))
        if (
            isinstance(response, RepositoryResponse)
            and response.coding_task is not None
            and response.agent_task is None
        ):
            output_fn(f"\n{response.coding_task.footer}")
        if isinstance(response, RepositoryResponse) and response.agent_task is not None:
            output_fn(f"\n{response.agent_task.footer}")


def _run_command(
    command: str,
    session: ChatSession | RepositoryChatSession,
    output_fn: OutputFunction,
    state: _ReplState,
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
            repository_mode = (
                "agent repair"
                if info.repair_enabled
                else "agent"
                if info.agent_mode
                else "assist"
                if info.assist_mode
                else "read-only"
            )
            rendered += (
                f"\nrepository mode: {repository_mode}"
                f"\nworkspace: {info.workspace}"
                f"\navailable tools: {info.available_tools}"
                f"\nlast tool count: {info.last_tool_count}"
                f"\nbuild configured: {'yes' if info.build_configured else 'no'}"
                f"\ntest configured: {'yes' if info.test_configured else 'no'}"
                f"\nmutation generation: {info.mutation_generation}"
                f"\nstructural checks: {info.structural_completeness_mode}"
                f"\nwrites: {_permission_label(info.write_permission)}"
                f"\ntask mutation limit: "
                f"{2 if info.repair_enabled else 1 if info.assist_mode else 0}"
            )
            if info.agent_mode:
                rendered += (
                    f"\niteration limit: {info.iteration_limit}"
                    f"\ntool limit: {info.tool_limit}"
                    f"\nrepair enabled: {'yes' if info.repair_enabled else 'no'}"
                    f"\npermissions: {info.permission_profile}"
                    f"\nread: {info.read_permission.value}"
                    f"\nwrite: {info.write_permission.value}"
                    f"\nbuild: {info.build_permission.value}"
                    f"\ntest: {info.test_permission.value}"
                )
        output_fn(rendered)
    elif command == "/display":
        output_fn(f"Display: {state.display.value.upper()}")
    elif command.startswith("/display "):
        value = command.removeprefix("/display ").strip().casefold()
        try:
            state.display = DisplayLevel(value)
        except ValueError:
            output_fn("Display must be minimal, normal, or trace.")
        else:
            output_fn(f"Display: {state.display.value.upper()}")
    elif command in {"/minimal", "/normal"}:
        state.display = DisplayLevel(command.removeprefix("/"))
        output_fn(f"Display: {state.display.value.upper()}")
    elif command in {"/trace on", "/trace off"}:
        state.display = (
            DisplayLevel.TRACE if command.endswith("on") else DisplayLevel.NORMAL
        )
        output_fn(f"Display: {state.display.value.upper()}")
    elif command == "/trace":
        state.display = DisplayLevel.TRACE
        output_fn("Display: TRACE")
    elif command == "/theme":
        output_fn(f"Theme: {state.theme.name}")
    elif command.startswith("/theme "):
        value = command.removeprefix("/theme ").strip().casefold()
        try:
            state.theme = resolve_theme(value, interactive=state.interactive)
        except ValueError:
            output_fn("Theme must be default, ember, or mono.")
        else:
            output_fn(f"Theme: {state.theme.name}")
    elif command == "/mode":
        output_fn(f"Mode: {state.user_mode}")
    elif command in {"/plan", "/chat"}:
        _switch_runtime_mode(
            "/mode " + command.removeprefix("/"), session, output_fn, state
        )
    elif command.startswith("/mode "):
        _switch_runtime_mode(command, session, output_fn, state)
    elif command == "/exit":
        output_fn("Goodbye.")
        return True
    else:
        output_fn(f"Unknown command: {command}. Type /help for commands.")
    return False


def _read_multiline(input_fn: InputFunction, output_fn: OutputFunction) -> str | None:
    output_fn("Multiline input: enter text exactly as intended; '.' submits.")
    lines: list[str] = []
    while True:
        try:
            line = input_fn("... ")
        except (EOFError, KeyboardInterrupt):
            output_fn("Multiline input cancelled.")
            return None
        if line == ".":
            return "\n".join(lines)
        lines.append(line)


def _switch_runtime_mode(
    command: str,
    session: ChatSession | RepositoryChatSession,
    output_fn: OutputFunction,
    state: _ReplState,
) -> None:
    requested = command.removeprefix("/mode ").strip().casefold()
    if requested not in {"chat", "plan", "dry-run", "manual", "auto"}:
        output_fn("Mode must be chat, plan, dry-run, manual, or auto.")
        return
    if requested == "dry-run":
        output_fn("DRY_RUN is deferred; preview-only execution is not safely isolated.")
        return
    if not isinstance(session, RepositoryChatSession):
        if requested == "chat":
            state.user_mode = "CHAT"
            output_fn("[CHAT] Mode: CHAT")
        else:
            output_fn("That mode requires restarting Forge in a project workspace.")
        return
    if session.info.autonomy_mode.value == "read" and requested in {"chat", "plan"}:
        state.user_mode = requested.upper()
        badge = "PLAN" if requested == "plan" else "READ"
        output_fn(f"[{badge}] Mode: {state.user_mode} (zero write authority)")
        return
    current = state.user_mode
    output_fn(
        f"Mode remains {current}. Restart with --mode {requested}; "
        "runtime authority changes never broaden or simulate revocation."
    )


def _default_user_mode(session: ChatSession | RepositoryChatSession) -> str:
    if not isinstance(session, RepositoryChatSession):
        return "CHAT"
    return {
        "read": "PLAN",
        "assist": "MANUAL",
        "agent": "AUTO",
        "repair": "AUTO",
    }[session.info.autonomy_mode.value]


def _permission_badge(info: object) -> str:
    mode = getattr(getattr(info, "autonomy_mode", None), "value", "chat")
    return {
        "read": "PLAN",
        "assist": "ASK WRITE",
        "agent": "AUTO",
        "repair": "AUTO",
    }.get(mode, "READ")


def _access_badge(info: object, *, long: bool = False) -> str:
    mode = getattr(getattr(info, "autonomy_mode", None), "value", "chat")
    if mode == "read":
        return "READ ONLY" if long else "READ"
    return {"assist": "ASK WRITE", "agent": "AUTO", "repair": "AUTO"}.get(mode, "CHAT")


def _context_gauge(info: object) -> str:
    used = getattr(info, "last_estimated_input_tokens", None)
    capacity = getattr(info, "context_capacity", None)
    if capacity is None:
        return "ctx unavailable"
    if used is None:
        return f"ctx ? / {_compact_number(capacity)}"
    return f"ctx {_compact_number(used)} / {_compact_number(capacity)}"


def _status_line(
    session: ChatSession | RepositoryChatSession, state: _ReplState
) -> str:
    info = session.info
    residency = "warm" if state.warm else "ready"
    fields = [
        f"Forge · {state.user_mode} · {state.display.value.upper()} · "
        f"{info.profile_name} {residency}",
        _access_badge(info),
    ]
    used = getattr(info, "last_estimated_input_tokens", None)
    capacity = getattr(info, "context_capacity", None)
    if used is not None and capacity is not None:
        fields.append(f"ctx {_compact_number(used)} / {_compact_number(capacity)}")
    return " · ".join(fields)


def _toolbar_fragments(
    session: ChatSession | RepositoryChatSession, state: _ReplState
) -> list[tuple[str, str]]:
    info = session.info
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    used = getattr(info, "last_estimated_input_tokens", None)
    capacity = getattr(info, "context_capacity", None)
    context = None
    percentage = None
    if used is not None and capacity:
        context = f"ctx {_compact_number(used)}/{_compact_number(capacity)}"
        percentage = f"{round(100 * used / capacity)}%"
    residency = "warm" if state.warm else "ready"
    if width >= 72:
        values = (
            ("class:toolbar.brand", "Forge"),
            ("class:toolbar.mode", state.user_mode),
            ("class:toolbar.display", state.display.value.upper()),
            ("class:toolbar.model", info.profile_name),
            ("class:toolbar.warm", residency),
            ("class:toolbar.access", _access_badge(info)),
            ("class:toolbar.context", context),
        )
    elif width >= 48:
        values = (
            ("class:toolbar.mode", state.user_mode),
            ("class:toolbar.display", state.display.value.upper()),
            ("class:toolbar.model", info.profile_name),
            ("class:toolbar.access", _access_badge(info)),
            ("class:toolbar.context", f"{percentage} ctx" if percentage else None),
        )
    else:
        values = (
            ("class:toolbar.mode", state.user_mode),
            ("class:toolbar.model", info.profile_name),
            ("class:toolbar.context", percentage),
        )
    fragments: list[tuple[str, str]] = []
    for style, value in values:
        if value is None:
            continue
        if fragments:
            fragments.append(("class:toolbar.separator", " | "))
        fragments.append((style, value))
    return fragments


def _render_suggestions(command: str) -> str:
    matches = command_suggestions(command)
    width = max((len(item.name) for item in matches), default=0)
    return "\n".join(f"/{item.name:<{width}}  {item.description}" for item in matches)


def _install_command_completion(*, enabled: bool) -> Callable[[], None]:
    if not enabled:
        return lambda: None
    try:
        import readline
    except ImportError:
        return lambda: None
    previous = readline.get_completer()

    def complete(text: str, state: int) -> str | None:
        matches = [f"/{item.name} " for item in command_suggestions(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")

    def restore() -> None:
        readline.set_completer(previous)

    return restore


def _compact_number(value: int) -> str:
    if value < 1000:
        return str(value)
    rendered = f"{value / 1000:.1f}".rstrip("0").rstrip(".")
    return f"{rendered}k"


def _activity_progress(activity: object) -> str:
    tool = getattr(activity, "tool_name", "tool")
    path = getattr(activity, "path", None)
    if tool.endswith("search"):
        return "Searching project..."
    if tool.endswith("read_file"):
        return f"Reading {path}..." if path else "Reading source..."
    if tool in {"project.test", "project.build"}:
        return "Running tests..." if tool.endswith("test") else "Running build..."
    if "patch" in tool or "write" in tool:
        return "Generating change..."
    return f"Running {tool}..."


def _render_activity(
    activity: object, display: DisplayLevel, theme: Theme
) -> str | None:
    if display is DisplayLevel.MINIMAL:
        return None
    tool = getattr(activity, "tool_name", "tool")
    status = getattr(activity, "status", "unknown")
    path = getattr(activity, "path", None)
    if display is DisplayLevel.TRACE:
        suffix = f": {path}" if path else ""
        evidence = getattr(activity, "evidence", "unknown")
        rendered = theme.paint(f"[tool] {tool}", ColorRole.TOOL)
        if suffix:
            rendered += theme.paint(suffix, ColorRole.PATH)
        return rendered + theme.paint(f" ({status}; {evidence})", ColorRole.METADATA)
    mark = "✓" if status == "success" else "!"
    action = {
        "repository.read_file": "Read",
        "repository.search": "Searched repository",
        "project.test": "Ran tests",
        "project.build": "Ran build",
    }.get(tool, tool)
    role = ColorRole.SUCCESS if status == "success" else ColorRole.WARNING
    rendered = theme.paint(mark, role) + " " + theme.paint(action, ColorRole.TOOL)
    if path:
        rendered += " " + theme.paint(path, ColorRole.PATH)
    return rendered


def _render_sources(sources: tuple[str, ...], theme: Theme) -> str:
    label = theme.paint("Sources:", ColorRole.MUTED)
    paths = ", ".join(theme.paint(path, ColorRole.PATH) for path in sources)
    return f"\n{label} {paths}"


def _trusted_sources(response: RepositoryResponse) -> tuple[str, ...]:
    sources: list[str] = []
    for activity in response.tool_activity:
        if (
            activity.status == "success"
            and activity.current_source
            and activity.relevant_source
            and activity.path is not None
            and activity.path not in sources
        ):
            sources.append(activity.path)
    return tuple(sources[:5])


def _performance_footer(response: object, elapsed: float) -> str:
    fields = [f"{elapsed:.1f}s"]
    activities = getattr(response, "tool_activity", ())
    if activities:
        fields.append(f"{len(activities)} tool calls")
        read_files = {
            item.path
            for item in activities
            if item.tool_name == "repository.read_file" and item.path is not None
        }
        if read_files:
            fields.append(f"{len(read_files)} files read")
    usage = getattr(response, "usage", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is not None:
        fields.append(f"{output_tokens} output tokens")
    return " · ".join(fields)


def _task_checklist(
    response: RepositoryResponse, session: RepositoryChatSession
) -> str:
    activities = response.tool_activity
    searched = any(item.tool_name == "repository.search" for item in activities)
    read = any(item.tool_name == "repository.read_file" for item in activities)
    changed = any(
        item.evidence in {"write_success", "patch_success"} for item in activities
    )
    verified = any(
        item.current_verification and item.status == "success" for item in activities
    )
    lines = [
        "Operational progress:",
        f"{'✓' if searched else '○'} Search repository",
        f"{'✓' if read else '○'} Inspect relevant source",
    ]
    if session.info.assist_mode:
        lines.extend(
            (
                f"{'✓' if changed else '○'} Apply approved change",
                f"{'✓' if verified else '○'} Verify",
            )
        )
    else:
        lines.append("✓ Synthesize answer or plan")
    return "\n".join(lines)


def _permission_label(decision: object) -> str:
    value = getattr(decision, "value", "deny")
    return {"allow": "automatic", "ask": "approval required"}.get(value, "denied")


def _approval_prompt(
    *,
    input_fn: InputFunction,
    output_fn: OutputFunction,
    cancel_on_interrupt: bool = False,
) -> Callable[
    [
        ToolInvocation,
        MutationPreview | MultiFileMutationPreview | PreparedProjectCommand,
    ],
    bool,
]:
    def approve(
        invocation: ToolInvocation,
        preview: MutationPreview | MultiFileMutationPreview | PreparedProjectCommand,
    ) -> bool:
        if isinstance(preview, MultiFileMutationPreview):
            output_fn(f"[proposed] {invocation.tool_name}: {preview.proposal_identity}")
            output_fn("Approval required: this action changes multiple files.")
            output_fn(preview.diff)
            prompt = "Approve grouped write? [y/N] "
            rejection = "Grouped write rejected."
        elif isinstance(preview, MutationPreview):
            output_fn(f"[proposed] {invocation.tool_name}: {preview.path}")
            output_fn("Approval required: this action changes a repository file.")
            output_fn(preview.diff)
            prompt = "Approve? [y/N] "
            rejection = "Write rejected."
        else:
            output_fn(f"[proposed] {invocation.tool_name}")
            output_fn("Approval required: this action runs a project command.")
            output_fn(f"Workspace: {preview.workspace}")
            output_fn("Command argv:")
            for argument in preview.argv:
                output_fn(f"  {argument!r}")
            output_fn(f"Timeout: {preview.timeout_seconds:g} seconds")
            prompt = "Approve execution? [y/N] "
            rejection = "Execution rejected."
        try:
            answer = input_fn(prompt).strip().casefold()
        except KeyboardInterrupt as error:
            if cancel_on_interrupt:
                raise AgentCancelled("agent task cancelled by user") from error
            output_fn(rejection)
            return False
        except EOFError:
            output_fn(rejection)
            return False
        if answer in {"y", "yes"}:
            return True
        output_fn(rejection)
        return False

    return approve


def _present_error(error: Exception) -> str:
    message = str(error)
    presentations = {
        "DUPLICATE_TOOL_CALL_ID": (
            "The model repeated an internal action identifier in this turn. "
            "No repeated action was executed."
        ),
        "STRUCTURAL_COMPLETENESS_REJECTED": (
            "The proposed change did not satisfy the required file structure."
        ),
        "EXECUTABLE_CHECK_FAILURE": (
            "The change failed a trusted behavior check after normal verification."
        ),
        "repeated identical tool-call limit exceeded": (
            "Forge could not make further progress without repeating the same lookup."
        ),
        "model response is not one valid JSON object": (
            "Forge could not determine a safe repository action. Please clarify "
            "what you want to inspect or change."
        ),
    }
    for classification, rendered in presentations.items():
        if classification in message:
            return rendered
    return message


def _ephemeral_review_prompt(
    *, input_fn: InputFunction, output_fn: OutputFunction
) -> Callable[[EphemeralAcceptancePreview], bool]:
    def review(preview: EphemeralAcceptancePreview) -> bool:
        output_fn("GENERATED ACCEPTANCE TEST APPROVAL — separate from source writes")
        output_fn(f"Task: {preview.task}")
        output_fn(f"Candidate: {preview.test_name} ({preview.language})")
        output_fn("Exact generated test source:")
        output_fn(preview.test_source)
        output_fn(f"Execution: {preview.execution_description}")
        output_fn(f"Execution isolation: {preview.isolation_description}")
        output_fn(f"Writable location: {preview.writable_location}")
        output_fn(f"Network: {preview.network_description}")
        output_fn(f"Project source writes: {preview.project_writes_description}")
        output_fn(f"Real HOME: {preview.real_home_description}")
        output_fn(f"Baseline: {preview.baseline_result}")
        output_fn(f"Baseline failure output:\n{preview.baseline_failure_output}")
        output_fn(f"Context files: {', '.join(preview.context_files)}")
        output_fn(preview.warning)
        try:
            answer = input_fn("Approve this exact ephemeral test? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().casefold() in {"y", "yes"}

    return review
