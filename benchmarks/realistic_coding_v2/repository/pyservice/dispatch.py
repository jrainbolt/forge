"""Dispatch boundary; action mapping lives in a separate rule module."""

from __future__ import annotations

from pyservice.dispatch_rules import choose_handler


def dispatch(action: str | None) -> str:
    if action is None or not action.strip():
        raise ValueError("missing action")
    return choose_handler(action.strip().casefold())
