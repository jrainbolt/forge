"""Queue scheduling boundary; ranking policy lives in a separate module."""

from pyservice.priority import priority_rank


def schedule(requests: list[tuple[str, str]]) -> list[str]:
    """Return request IDs in stable high-to-low priority order."""
    ordered = sorted(requests, key=lambda item: priority_rank(item[1]))
    return [request_id for request_id, _priority in ordered]
