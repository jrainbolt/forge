from __future__ import annotations

import re

_HEADER_NAME = re.compile(r"^[A-Za-z0-9-]+$")


def normalize_header_name(value: str) -> str:
    candidate = value.strip()
    if not candidate or _HEADER_NAME.fullmatch(candidate) is None:
        raise ValueError("invalid header name")
    return candidate.casefold()


def get_header(headers: dict[str, str], name: str) -> str | None:
    return headers.get(normalize_header_name(name))
