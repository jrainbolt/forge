"""Frozen A75 corpus reusing the authoritative A74 cases unchanged."""

from __future__ import annotations

from enum import StrEnum

from benchmarks.grounded_mutation_planning_v1.suite import (
    CASES,
    CONTEXT_SIZE,
    PROFILES,
    SEED,
    corpus_identity,
    task_map,
    validate_corpus,
)

SUITE = "grounded-mutation-contract-v1"
VERSION = 1
SCHEMA_VERSION = 1
RUN_ID = "a75-grounding-mutation-contract-v1-final"


class Condition(StrEnum):
    M0 = "M0_DIRECT_MUTATION"
    M1 = "M1_CONTRACT_AUGMENTED_MUTATION"


__all__ = [
    "CASES",
    "CONTEXT_SIZE",
    "PROFILES",
    "SEED",
    "Condition",
    "corpus_identity",
    "task_map",
    "validate_corpus",
]
