"""Command-line interface for Forge."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from forge import __version__
from forge.config import ForgeConfig
from forge.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the Forge argument parser without parsing process state."""
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Forge local-first AI coding assistant (project bootstrap)",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Forge CLI and return its process exit status."""
    args = build_parser().parse_args(argv)
    config = ForgeConfig.from_environment()
    if args.verbose:
        config = config.with_verbose(True)
    configure_logging(verbose=config.verbose)
    LOGGER.debug("Forge CLI initialized")
    return 0
