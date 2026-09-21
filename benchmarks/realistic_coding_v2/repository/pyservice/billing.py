"""Billing validation and user-facing amount formatting."""

from pyservice.currency import format_cents


def invoice_line(amount_cents: int) -> str:
    if amount_cents < 0:
        raise ValueError("negative amount")
    return "TOTAL " + format_cents(amount_cents)
