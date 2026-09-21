"""Deterministic cents formatting without floating-point rounding."""


def format_cents(amount_cents: int) -> str:
    whole, remainder = divmod(amount_cents, 100)
    return f"${whole}.{remainder:02d}"
