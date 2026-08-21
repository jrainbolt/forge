"""Logging initialization for Forge applications."""

from __future__ import annotations

import logging

DEFAULT_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def configure_logging(*, verbose: bool = False) -> None:
    """Configure process logging explicitly for a Forge entry point."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
