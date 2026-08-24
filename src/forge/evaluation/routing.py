"""Fixed deterministic retrieval-routing evaluation definitions."""

from __future__ import annotations

from dataclasses import dataclass

ROUTING_V1 = "routing-v1"
ROUTING_SUITE_VERSION = 1


@dataclass(frozen=True, slots=True)
class RoutingTask:
    task_id: str
    description: str
    expected_transition: str


ROUTING_V1_TASKS = (
    RoutingTask("R01", "semantic candidate then targeted read", "source_acquired"),
    RoutingTask("R02", "exact symbol suppresses broad search", "target_identified"),
    RoutingTask("R03", "repeated candidate set has no novelty", "candidates_available"),
    RoutingTask("R04", "failed candidate read reopens discovery", "discovering"),
    RoutingTask("R05", "multi-file candidates permit two reads", "source_acquired"),
    RoutingTask("R06", "sufficient source suppresses broad search", "source_acquired"),
)


def load_routing_suite(name: str) -> tuple[RoutingTask, ...]:
    if name != ROUTING_V1:
        raise ValueError(f"unknown routing suite {name!r}; available: {ROUTING_V1}")
    return ROUTING_V1_TASKS
