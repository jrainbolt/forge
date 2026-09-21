"""Priority normalization shared by scheduling clients."""

_RANKS = {"high": 0, "normal": 1, "low": 2}


def priority_rank(value: str) -> int:
    """Rank known labels case-insensitively; reject unknown labels."""
    normalized = value.strip().casefold()
    if normalized not in _RANKS:
        raise ValueError("unknown priority")
    return _RANKS[normalized]
