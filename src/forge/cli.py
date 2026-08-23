"""Command-line interface for Forge."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from forge import __version__
from forge.config import ForgeConfig
from forge.evaluation import (
    SUITE_VERSION,
    EvaluationRunner,
    fixture_workspace,
    load_suite,
    render_terminal_report,
    write_json_report,
)
from forge.logging import configure_logging
from forge.models import (
    GenerationConfig,
    ModelConfigurationError,
    ModelError,
    ModelSelectionError,
    default_backend_registry,
    load_model_catalog,
)
from forge.orchestration import RepositoryChatSession
from forge.repl import run_repl
from forge.session import DEFAULT_SYSTEM_MESSAGE, ChatSession

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the Forge argument parser without parsing process state."""
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Forge local-first AI coding assistant",
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
    chat.add_argument("--model", required=True, help="configured model profile name")
    chat.add_argument(
        "--config",
        type=Path,
        help="model configuration path (or set FORGE_CONFIG)",
    )
    chat.add_argument("--max-tokens", type=int, default=256)
    chat.add_argument("--temperature", type=float, default=0.4)
    chat.add_argument("--seed", type=int)
    chat.add_argument(
        "--workspace",
        type=Path,
        help="enable read-only repository chat for this explicit workspace",
    )
    chat.add_argument(
        "--no-system",
        action="store_true",
        help="omit Forge's default system message",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Forge CLI and return its process exit status."""
    args = build_parser().parse_args(argv)
    config = ForgeConfig.from_environment()
    if args.verbose:
        config = config.with_verbose(True)
    configure_logging(verbose=config.verbose)
    LOGGER.debug("Forge CLI initialized")
    if args.command == "chat":
        config_path = args.config or config.model_config_path
        if config_path is None:
            LOGGER.error("chat requires --config or FORGE_CONFIG")
            return 2
        try:
            if args.workspace is not None and args.no_system:
                raise ValueError("--no-system cannot be used with repository chat")
            generation = GenerationConfig(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
            catalog = load_model_catalog(config_path, default_backend_registry())
            LOGGER.info("Selected model profile %s", args.model)
            load_started = time.perf_counter()
            model = catalog.create(args.model)
            LOGGER.debug(
                "Loaded model profile %s in %.2f seconds",
                args.model,
                time.perf_counter() - load_started,
            )
            if args.workspace is None:
                session = ChatSession(
                    args.model,
                    model,
                    generation=generation,
                    system_message=(None if args.no_system else DEFAULT_SYSTEM_MESSAGE),
                )
            else:
                session = RepositoryChatSession(
                    args.model,
                    model,
                    args.workspace,
                    generation=generation,
                )
            with model, session:
                return run_repl(session)
        except (
            ModelConfigurationError,
            ModelSelectionError,
            ModelError,
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
            workspace = fixture_workspace().resolve(strict=True)
            print(f"Evaluation workspace: {workspace}")
            catalog = load_model_catalog(config_path, default_backend_registry())
            model = catalog.create(args.model)
            with model:
                result = EvaluationRunner(args.model, model, workspace).run(
                    args.suite, tasks, suite_version=SUITE_VERSION
                )
            print(render_terminal_report(result, verbose=args.verbose))
            if args.output is not None:
                write_json_report(result, args.output)
                print(f"JSON report: {args.output}")
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
