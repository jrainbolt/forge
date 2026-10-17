"""Command-line interface for Forge."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from forge import __version__
from forge.config import ForgeConfig
from forge.embedding_config import load_embedding_profile
from forge.ephemeral_acceptance import EphemeralAcceptanceMode
from forge.evaluation import (
    SUITE_VERSION,
    EvaluationRunner,
    fixture_workspace,
    load_suite,
    render_terminal_report,
    write_json_report,
)
from forge.interaction import (
    BUILTIN_PERMISSION_PROFILES,
    AutonomyMode,
    resolve_interaction_policy,
)
from forge.lexical_index import RepositoryLexicalIndex
from forge.logging import configure_logging
from forge.models import (
    GenerationConfig,
    ModelConfigurationError,
    ModelError,
    ModelRole,
    ModelSelectionError,
    MutationRepresentationPolicy,
    default_backend_registry,
    load_model_catalog,
)
from forge.orchestration import RepositoryChatSession
from forge.project_config import ProjectCommands
from forge.repl import run_repl
from forge.repository_index import RepositoryIndex, RepositoryIndexError
from forge.semantic_index import SemanticIndex, SemanticIndexError
from forge.session import DEFAULT_SYSTEM_MESSAGE, ChatSession
from forge.terminal_ui import THEME_NAMES, LoadingFlame
from forge.tools import (
    create_repository_registry,
)

LOGGER = logging.getLogger(__name__)

USER_MODES = ("chat", "plan", "dry-run", "manual", "auto")
DISPLAY_LEVELS = ("minimal", "normal", "trace")
PROJECT_MARKERS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "CMakeLists.txt",
    "Makefile",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Forge argument parser without parsing process state."""
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Forge local-first AI coding assistant",
        epilog=(
            "Preferred shortcuts: forge repo [PATH], forge assist [PATH], "
            "forge agent [PATH]. A bare forge uses the current project when detected."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    commands = parser.add_subparsers(dest="command")
    chat = commands.add_parser("chat", help="start an interactive local chat")
    chat.add_argument(
        "--model",
        help=(
            "configured model profile name (coding modes use their configured default)"
        ),
    )
    chat.add_argument(
        "--config",
        type=Path,
        help="model configuration path (or set FORGE_CONFIG)",
    )
    chat.add_argument(
        "--embedding-config", type=Path, help="separate embedding configuration path"
    )
    chat.add_argument(
        "--embedding-profile", help="configured local embedding profile name"
    )
    chat.add_argument("--max-tokens", type=int, default=256)
    chat.add_argument("--temperature", type=float, default=0.4)
    chat.add_argument("--seed", type=int)
    chat.add_argument(
        "--workspace",
        type=Path,
        help=(
            "select an explicit repository workspace (read-only unless --assist "
            "or --agent)"
        ),
    )
    chat.add_argument(
        "--no-system",
        action="store_true",
        help="omit Forge's default system message",
    )
    autonomy = chat.add_mutually_exclusive_group()
    autonomy.add_argument(
        "--assist",
        action="store_true",
        help=(
            "enable user-approved repository writes and configured build/test execution"
        ),
    )
    chat.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in AutonomyMode) + USER_MODES[1:],
        help=(
            "interaction mode: chat, plan, manual, auto, or legacy "
            "read/assist/agent/repair (dry-run is reserved but deferred)"
        ),
    )
    chat.add_argument(
        "--display",
        choices=DISPLAY_LEVELS,
        default="normal",
        help="terminal detail level: minimal, normal, or trace",
    )
    chat.add_argument(
        "--theme",
        choices=THEME_NAMES,
        default=None,
        help="terminal theme (default, ember, or mono; also FORGE_THEME)",
    )
    chat.add_argument(
        "--no-animation",
        action="store_true",
        help="disable the interactive model-loading animation",
    )
    chat.add_argument(
        "--permissions",
        choices=tuple(BUILTIN_PERMISSION_PROFILES),
        help="permission profile: safe, confirm, or trusted-exec",
    )
    chat.add_argument(
        "--repair",
        action="store_true",
        help="allow one approval-gated repair after qualifying verification failure",
    )
    chat.add_argument(
        "--ephemeral-acceptance",
        choices=tuple(mode.value for mode in EphemeralAcceptanceMode),
        default="off",
        help="experimental human-reviewed Python acceptance check (default: off)",
    )
    chat.add_argument(
        "--acceptance-context-file",
        action="append",
        default=[],
        help="trusted relevant source or visible test path; supply 2-4",
    )
    chat.add_argument(
        "--acceptance-import-root",
        default="",
        help="trusted Python package root allowed in generated assertions",
    )
    chat.add_argument(
        "--create-path",
        action="append",
        default=[],
        help="explicitly authorize one absent new text-file path (maximum two)",
    )
    chat.add_argument(
        "--edit-path",
        action="append",
        default=[],
        help=(
            "require one current existing-file edit candidate "
            "(maximum four combined with creates)"
        ),
    )
    autonomy.add_argument(
        "--agent",
        action="store_true",
        help=(
            "enable bounded multi-step reasoning with approval-gated writes and "
            "execution"
        ),
    )
    evaluation = commands.add_parser(
        "eval", help="run a controlled read-only coding evaluation"
    )
    evaluation.add_argument("--model", required=True, help="configured model profile")
    evaluation.add_argument(
        "--config", type=Path, help="model configuration path (or set FORGE_CONFIG)"
    )
    evaluation.add_argument("--suite", default="coding-v1")
    evaluation.add_argument("--output", type=Path, help="explicit JSON report path")
    evaluation.add_argument(
        "--verbose", action="store_true", help="include answers and files in report"
    )
    evaluation.add_argument("--workspace", type=Path)
    evaluation.add_argument("--embedding-config", type=Path)
    evaluation.add_argument("--embedding-profile")
    index = commands.add_parser(
        "index", help="inspect or maintain the local repository index"
    )
    index_commands = index.add_subparsers(dest="index_command", required=True)
    for name in ("status", "build", "refresh", "clear"):
        operation = index_commands.add_parser(name)
        operation.add_argument("--workspace", type=Path, default=Path.cwd())
    for name in (
        "semantic-status",
        "semantic-build",
        "semantic-refresh",
        "semantic-clear",
    ):
        operation = index_commands.add_parser(name)
        operation.add_argument("--workspace", type=Path, default=Path.cwd())
        operation.add_argument("--embedding-config", type=Path, required=True)
        operation.add_argument("--embedding-profile", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Forge CLI and return its process exit status."""
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    automatic_interactive_startup = not raw_arguments and sys.stdin.isatty()
    if not raw_arguments:
        if not sys.stdin.isatty():
            build_parser().print_help()
            return 2
        raw_arguments = _interactive_startup()
        if not raw_arguments:
            return 0
    args = build_parser().parse_args(_normalize_primary_command(raw_arguments))
    config = ForgeConfig.from_environment()
    if args.verbose:
        config = config.with_verbose(True)
    configure_logging(verbose=config.verbose)
    LOGGER.debug("Forge CLI initialized")
    if args.command == "index":
        try:
            if args.index_command.startswith("semantic-"):
                embedding = load_embedding_profile(
                    args.embedding_config, args.embedding_profile
                )
                semantic = SemanticIndex(args.workspace, embedding)
                operation = args.index_command.removeprefix("semantic-")
                if operation == "status":
                    result = semantic.status()
                elif operation == "build":
                    result = semantic.build()
                elif operation == "refresh":
                    result = semantic.refresh()
                else:
                    semantic.clear()
                    result = {"state": "cleared", "path": str(semantic.database_path)}
                semantic.close()
                print(result)
                return 0
            index = RepositoryIndex(args.workspace)
            if args.index_command == "status":
                result = index.status()
            elif args.index_command == "build":
                result = index.build()
            elif args.index_command == "refresh":
                result = index.refresh()
            else:
                index.clear()
                result = {"state": "cleared", "path": str(index.database_path)}
            print(result)
            return 0
        except (OSError, RepositoryIndexError, SemanticIndexError, ValueError) as error:
            LOGGER.error("%s", error)
            return 2
    if args.command == "chat":
        config_path = args.config or config.model_config_path
        if config_path is None:
            if automatic_interactive_startup:
                print(
                    "Forge needs a model configuration before opening this project.\n"
                    "Set FORGE_CONFIG=/path/to/forge.toml or run:\n"
                    "  forge chat --config /path/to/forge.toml --workspace . "
                    "--mode plan"
                )
                return 2
            LOGGER.error("chat requires --config or FORGE_CONFIG")
            return 2
        try:
            mode = _resolve_chat_mode(args)
            user_mode = _user_mode_name(args, mode)
            ephemeral_mode = EphemeralAcceptanceMode(args.ephemeral_acceptance)
            if ephemeral_mode is not EphemeralAcceptanceMode.OFF and (
                not mode.coding_mode
                or not 2 <= len(args.acceptance_context_file) <= 4
                or not args.acceptance_import_root.isidentifier()
            ):
                raise ValueError(
                    "ephemeral acceptance requires coding mode, 2-4 context files, "
                    "and a Python import root"
                )
            interaction = resolve_interaction_policy(mode, args.permissions)
            if args.workspace is not None and args.no_system:
                raise ValueError("--no-system cannot be used with repository chat")
            if mode.repository_mode and args.workspace is None:
                if args.assist:
                    raise ValueError("--assist requires --workspace")
                if args.agent:
                    raise ValueError("--agent requires --workspace")
                raise ValueError(f"--mode {mode.value} requires --workspace")
            if mode is AutonomyMode.CHAT and args.workspace is not None:
                raise ValueError("--mode chat cannot be used with --workspace")
            generation = GenerationConfig(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
            catalog = load_model_catalog(config_path, default_backend_registry())
            profile_name = _resolve_model_profile_interactive(
                catalog,
                args.model,
                mode,
                allow_prompt=automatic_interactive_startup,
            )
            LOGGER.info("Selected model profile %s", profile_name)
            load_started = time.perf_counter()
            presentation_tty = sys.stdin.isatty() and sys.stdout.isatty()
            animation_enabled = (
                presentation_tty
                and not args.no_animation
                and "FORGE_NO_ANIMATION" not in os.environ
            )
            flame = LoadingFlame(
                enabled=animation_enabled,
                label=f"Loading {profile_name}",
            )
            if presentation_tty and not animation_enabled:
                print(f"Loading {profile_name}...")
            flame.start()
            try:
                model = catalog.create(profile_name)
            finally:
                flame.stop()
            load_elapsed = time.perf_counter() - load_started
            if bool(args.embedding_config) != bool(args.embedding_profile):
                raise ValueError(
                    "--embedding-config and --embedding-profile must be "
                    "supplied together"
                )
            semantic_index = None
            if args.embedding_config is not None:
                if args.workspace is None:
                    raise ValueError("semantic retrieval requires --workspace")
                semantic_index = SemanticIndex(
                    args.workspace,
                    load_embedding_profile(
                        args.embedding_config, args.embedding_profile
                    ),
                )
            LOGGER.debug(
                "Loaded model profile %s in %.2f seconds",
                profile_name,
                load_elapsed,
            )
            if mode is AutonomyMode.CHAT:
                session = ChatSession(
                    profile_name,
                    model,
                    generation=generation,
                    system_message=(None if args.no_system else DEFAULT_SYSTEM_MESSAGE),
                )
            else:
                repository_index = RepositoryIndex(args.workspace)
                lexical_index = RepositoryLexicalIndex(args.workspace)
                profile = (
                    catalog.profile(profile_name)
                    if hasattr(catalog, "profile")
                    else None
                )
                session = RepositoryChatSession(
                    profile_name,
                    model,
                    args.workspace,
                    generation=generation,
                    registry=(
                        create_repository_registry(
                            interaction,
                            getattr(catalog, "project_commands", ProjectCommands()),
                            repository_index,
                            semantic_index,
                            lexical_index,
                            include_creation=bool(args.create_path),
                        )
                    ),
                    policy=interaction,
                    mode=mode,
                    interaction_policy=interaction,
                    repository_index=repository_index,
                    semantic_index=semantic_index,
                    lexical_index=lexical_index,
                    mutation_representation=getattr(
                        profile,
                        "mutation_representation",
                        MutationRepresentationPolicy.EXACT_TEXT,
                    ),
                    verification_plan=getattr(
                        getattr(catalog, "project_commands", ProjectCommands()),
                        "verification_plan",
                        None,
                    ),
                    structural_completeness_mode=(
                        getattr(catalog, "project_commands", None) or ProjectCommands()
                    ).structural_completeness_mode,
                    ephemeral_acceptance_mode=ephemeral_mode,
                    ephemeral_acceptance_paths=tuple(args.acceptance_context_file),
                    ephemeral_acceptance_import_root=args.acceptance_import_root,
                    create_candidate_paths=tuple(args.create_path),
                    required_candidate_paths=tuple(args.edit_path),
                    mixed_file_operations=bool(args.edit_path and args.create_path),
                )
            with model, session:
                result = run_repl(
                    session,
                    display_level=args.display,
                    user_mode=user_mode,
                    interactive=presentation_tty,
                    theme_name=args.theme or os.environ.get("FORGE_THEME", "default"),
                )
            if semantic_index is not None:
                semantic_index.close()
            return result
        except (
            ModelConfigurationError,
            ModelSelectionError,
            ModelError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            LOGGER.error("%s", error)
            if config.verbose:
                LOGGER.debug("Chat startup failed", exc_info=True)
            return 2
    if args.command == "eval":
        config_path = args.config or config.model_config_path
        if config_path is None:
            LOGGER.error("eval requires --config or FORGE_CONFIG")
            return 2
        try:
            tasks = load_suite(args.suite)
            workspace = (
                args.workspace.resolve(strict=True)
                if args.workspace is not None
                else fixture_workspace().resolve(strict=True)
            )
            if args.suite == "semantic-v1" and args.workspace is None:
                raise ValueError("semantic-v1 requires --workspace")
            if bool(args.embedding_config) != bool(args.embedding_profile):
                raise ValueError(
                    "--embedding-config and --embedding-profile must be "
                    "supplied together"
                )
            semantic_index = None
            if args.embedding_config is not None:
                semantic_index = SemanticIndex(
                    workspace,
                    load_embedding_profile(
                        args.embedding_config, args.embedding_profile
                    ),
                )
            print(f"Evaluation workspace: {workspace}")
            catalog = load_model_catalog(config_path, default_backend_registry())
            model = catalog.create(args.model)
            with model:
                runner = (
                    EvaluationRunner(args.model, model, workspace)
                    if semantic_index is None
                    else EvaluationRunner(
                        args.model,
                        model,
                        workspace,
                        semantic_index=semantic_index,
                    )
                )
                result = runner.run(args.suite, tasks, suite_version=SUITE_VERSION)
            print(render_terminal_report(result, verbose=args.verbose))
            if args.output is not None:
                write_json_report(result, args.output)
                print(f"JSON report: {args.output}")
            if semantic_index is not None:
                semantic_index.close()
            return 0
        except (
            ModelConfigurationError,
            ModelSelectionError,
            ModelError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            LOGGER.error("%s", error)
            if config.verbose:
                LOGGER.debug("Evaluation failed", exc_info=True)
            return 2
    return 0


def _normalize_primary_command(arguments: list[str]) -> list[str]:
    """Map concise UX aliases onto the single existing chat implementation."""
    arguments = list(arguments)
    prefix: list[str] = []
    while arguments and arguments[0] in {"-v", "--verbose"}:
        prefix.append(arguments.pop(0))
    if not arguments or arguments[0] not in {"repo", "assist", "agent"}:
        return [*prefix, *arguments]
    command = arguments[0]
    remaining = arguments[1:]
    workspace = "."
    if remaining and not remaining[0].startswith("-"):
        workspace = remaining.pop(0)
    mode = {"repo": "read", "assist": "assist", "agent": "agent"}[command]
    return [
        *prefix,
        "chat",
        "--mode",
        mode,
        "--workspace",
        workspace,
        *remaining,
    ]


def _interactive_startup(*, cwd: Path | None = None) -> list[str]:
    project = _detect_current_project(cwd or Path.cwd())
    if project is not None:
        print(f"Forge detected project: {project}")
        return ["chat", "--mode", "plan", "--workspace", str(project)]
    print("Forge — choose a mode")
    print("  1. Chat")
    print("  2. Repository chat (read-only)")
    print("  3. Coding assist (approval required)")
    print("  4. Agent (bounded, approval required)")
    try:
        choice = input("Mode [1-4, q to quit]: ").strip().casefold()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return []
    if choice in {"q", "quit", "exit"}:
        return []
    modes = {"1": "chat", "2": "repo", "3": "assist", "4": "agent"}
    command = modes.get(choice)
    if command is None:
        print("Invalid selection. Run `forge --help` for commands.")
        return []
    if command == "chat":
        return ["chat"]
    try:
        workspace = input("Repository path [.]: ").strip() or "."
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return []
    return [command, workspace]


def _detect_current_project(path: Path) -> Path | None:
    """Return the current directory when it has a common project marker."""
    try:
        candidate = path.resolve(strict=True)
    except OSError:
        return None
    if not candidate.is_dir():
        return None
    return (
        candidate
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS)
        else None
    )


def _resolve_chat_mode(args: argparse.Namespace) -> AutonomyMode:
    legacy_selected = args.assist or args.agent or args.repair
    if args.mode is not None and legacy_selected:
        raise ValueError(
            "--mode cannot be combined with --assist, --agent, or --repair"
        )
    if args.mode is not None:
        aliases = {
            "plan": AutonomyMode.READ,
            "manual": AutonomyMode.ASSIST,
            "auto": AutonomyMode.AGENT,
        }
        if args.mode == "dry-run":
            raise ValueError(
                "DRY_RUN is deferred: preview-only transaction execution is not "
                "yet safely isolated; use --mode plan"
            )
        if args.mode in aliases:
            return aliases[args.mode]
        return AutonomyMode(args.mode)
    if args.repair and not args.agent:
        raise ValueError("--repair requires --agent")
    if args.repair:
        return AutonomyMode.REPAIR
    if args.agent:
        return AutonomyMode.AGENT
    if args.assist:
        return AutonomyMode.ASSIST
    return AutonomyMode.READ if args.workspace is not None else AutonomyMode.CHAT


def _user_mode_name(args: argparse.Namespace, mode: AutonomyMode) -> str:
    if args.mode in USER_MODES:
        return args.mode.upper().replace("-", "_")
    return {
        AutonomyMode.CHAT: "CHAT",
        AutonomyMode.READ: "PLAN",
        AutonomyMode.ASSIST: "MANUAL",
        AutonomyMode.AGENT: "AUTO",
        AutonomyMode.REPAIR: "AUTO",
    }[mode]


def _resolve_model_profile(
    catalog: object, explicit: str | None, mode: AutonomyMode
) -> str:
    if explicit is not None:
        return explicit
    role = (
        ModelRole.CODING
        if mode.coding_mode
        else ModelRole.REPOSITORY
        if mode.repository_mode
        else ModelRole.CHAT
    )
    resolver = getattr(catalog, "resolve_profile", None)
    if resolver is None:
        raise ModelSelectionError(
            f"no default model profile is configured for {role.value}; use --model"
        )
    return resolver(None, role)


def _resolve_model_profile_interactive(
    catalog: object,
    explicit: str | None,
    mode: AutonomyMode,
    *,
    allow_prompt: bool,
) -> str:
    """Require a deliberate TTY choice when the requested role has no default."""
    try:
        return _resolve_model_profile(catalog, explicit, mode)
    except ModelSelectionError:
        names = getattr(catalog, "profile_names", ())
        if explicit is not None or not allow_prompt or not names:
            raise
        print("No repository default model is configured.")
        for index, name in enumerate(names, 1):
            print(f"  {index}. {name}")
        try:
            selected = input("Select model profile [number, q to quit]: ").strip()
        except (EOFError, KeyboardInterrupt) as error:
            raise ModelSelectionError("model selection cancelled") from error
        if selected.casefold() in {"q", "quit", "exit"}:
            raise ModelSelectionError("model selection cancelled") from None
        try:
            return names[int(selected) - 1]
        except (ValueError, IndexError) as error:
            raise ModelSelectionError("invalid model profile selection") from error
