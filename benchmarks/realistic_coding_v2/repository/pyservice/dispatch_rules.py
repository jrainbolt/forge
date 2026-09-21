"""Dispatch targets for supported request actions."""

_HANDLERS = {"read": "reader", "write": "writer", "delete": "remover"}


def choose_handler(action: str) -> str:
    if action not in _HANDLERS:
        raise ValueError("unsupported action")
    return _HANDLERS[action]
