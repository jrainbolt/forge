"""Frozen A80 operational corpus derived from accepted A78 tasks."""

from __future__ import annotations

import hashlib
import json
from enum import Enum

from benchmarks.default_candidate_confirmation_v1.suite import (
    CONTEXT_SIZE,
    REPRESENTATION,
    SEED,
)
from benchmarks.default_candidate_confirmation_v1.suite import (
    corpus_identity as parent_corpus_identity,
)
from benchmarks.default_candidate_confirmation_v1.suite import (
    tasks as parent_tasks,
)

SUITE = "coding-default-operational-v1"
VERSION = 1
RUN_ID = "a80-qwen-large-coding-default-operational-v1"
CHECKPOINT_NAMESPACE = "a80-qwen-large-coding-default-operational-v1"
OUTPUT_TOKENS = 512
TASK_IDS = ("H01", "H02", "H04", "H05", "H07", "H08", "H09", "H10", "H11", "H12")
REPRESENTATIVE_TASK_IDS = ("H01", "H04", "H07", "H10")


class SelectionCondition(Enum):
    DEFAULT = "D0_DEFAULT_CODING"
    EXPLICIT_LARGE = "D1_EXPLICIT_QWEN_LARGE"
    EXPLICIT_SMALL = "D2_EXPLICIT_QWEN_SMALL_SMOKE"


def tasks():  # type: ignore[no-untyped-def]
    by_id = {item.task_id: item for item in parent_tasks()}
    return tuple(by_id[task_id] for task_id in TASK_IDS)


def corpus_identity() -> str:
    payload = {
        "suite": SUITE,
        "version": VERSION,
        "parent_corpus_identity": parent_corpus_identity(),
        "task_ids": TASK_IDS,
        "representative_task_ids": REPRESENTATIVE_TASK_IDS,
        "settings": {
            "seed": SEED,
            "temperature": 0,
            "context": CONTEXT_SIZE,
            "output": OUTPUT_TOKENS,
        },
        "representation": REPRESENTATION.value,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_corpus() -> None:
    values = tasks()
    counts: dict[str, int] = {}
    for item in values:
        counts[item.operation_class.value] = (
            counts.get(item.operation_class.value, 0) + 1
        )
    expected = {
        "EDIT_SINGLE": 2,
        "EDIT_MULTI": 2,
        "CREATE": 3,
        "MIXED_EDIT_CREATE": 3,
    }
    if len(values) != 10 or counts != expected:
        raise RuntimeError(f"invalid A80 corpus: {counts}")
