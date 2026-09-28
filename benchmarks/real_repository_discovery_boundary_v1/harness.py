"""Deterministic A61 discovery traces and observability classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EventKind(StrEnum):
    SEARCH = "search"
    SYMBOL_QUERY = "symbol_query"
    INDEX_LOOKUP = "index_lookup"
    SOURCE_HINT = "source_hint"
    TRUSTED_READ = "trusted_read"
    CONTEXT_SELECTION = "context_selection"
    MUTATION_CANDIDATE = "mutation_candidate"


@dataclass(frozen=True, slots=True)
class DiscoveryEvent:
    kind: EventKind
    path: str | None = None
    authoritative: bool = False


@dataclass(frozen=True, slots=True)
class DiscoveryTrace:
    events: tuple[DiscoveryEvent, ...]

    @property
    def discovery_calls(self) -> int:
        kinds = {EventKind.SEARCH, EventKind.SYMBOL_QUERY, EventKind.INDEX_LOOKUP}
        return sum(item.kind in kinds for item in self.events)

    @property
    def search_calls(self) -> int:
        return sum(item.kind is EventKind.SEARCH for item in self.events)

    @property
    def index_calls(self) -> int:
        return sum(item.kind is EventKind.INDEX_LOOKUP for item in self.events)

    @property
    def source_reads(self) -> int:
        return sum(item.kind is EventKind.TRUSTED_READ for item in self.events)

    @property
    def context_files(self) -> tuple[str, ...]:
        return tuple(
            item.path
            for item in self.events
            if item.kind is EventKind.CONTEXT_SELECTION and item.path is not None
        )

    @property
    def mutation_candidates(self) -> tuple[str, ...]:
        return tuple(
            item.path
            for item in self.events
            if item.kind is EventKind.MUTATION_CANDIDATE and item.path is not None
        )


def deterministic_trace(discovered_path: str) -> DiscoveryTrace:
    """Model-free task -> search -> hint -> read -> candidate proof."""
    return DiscoveryTrace(
        (
            DiscoveryEvent(EventKind.SEARCH),
            DiscoveryEvent(EventKind.SOURCE_HINT, discovered_path, False),
            DiscoveryEvent(EventKind.TRUSTED_READ, discovered_path, True),
            DiscoveryEvent(EventKind.CONTEXT_SELECTION, discovered_path, True),
            DiscoveryEvent(EventKind.MUTATION_CANDIDATE, discovered_path, True),
        )
    )


def validate_trace(trace: DiscoveryTrace, *, discovery_required: bool) -> None:
    """Enforce hint/read authority and suspicious zero-discovery detection."""
    if discovery_required and trace.discovery_calls == 0:
        raise ValueError("ZERO_DISCOVERY_ACTIVITY")
    trusted = {
        item.path
        for item in trace.events
        if item.kind is EventKind.TRUSTED_READ and item.authoritative
    }
    candidates = set(trace.mutation_candidates)
    if not candidates.issubset(trusted):
        raise ValueError("UNTRUSTED_MUTATION_CANDIDATE")
