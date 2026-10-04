"""Frozen A71 corpus: A56 C01-C12 plus compatible frozen R02/R06 controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from benchmarks.realistic_coding_v2.suite import (
    AuthorityMode,
    FrozenTask,
    OperationClass,
)
from benchmarks.realistic_coding_v2.suite import (
    tasks as a56_tasks,
)
from benchmarks.realistic_semantic_v1.suite import realistic_semantic_tasks

SUITE = "transaction-readiness-v1"
VERSION = 1
SCHEMA_VERSION = 1
SEED = 42
PROFILES = ("qwen-small", "qwen-large", "codestral-22b")


def tasks() -> tuple[FrozenTask, ...]:
    legacy = {item.metadata.task_id: item for item in realistic_semantic_tasks((SEED,))}
    additions = []
    for task_id, operation, authority in (
        ("R02", OperationClass.EDIT_SINGLE, AuthorityMode.DISCOVERY_REQUIRED),
        ("R06", OperationClass.EDIT_MULTI, AuthorityMode.PATH_KNOWN),
    ):
        source = legacy[task_id]
        production = replace(source.metadata.production_task, seeds=(SEED,))
        additions.append(
            FrozenTask(
                task_id,
                source.metadata.task_version,
                source.metadata.language,
                operation,
                authority,
                production,
                source.wrong,
            )
        )
    return (*a56_tasks(), *additions)


def corpus_identity(definitions: tuple[FrozenTask, ...]) -> str:
    payload = [
        {
            "id": item.versioned_id,
            "class": item.operation_class.value,
            "authority": item.authority_mode.value,
            "prompt": item.production_task.prompt,
            "edits": item.edit_paths,
            "creates": item.create_paths,
            "setup": [
                (change.path, change.expected, change.replacement)
                for change in item.production_task.setup
            ],
        }
        for item in definitions
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_corpus(definitions: tuple[FrozenTask, ...]) -> None:
    if len(definitions) < 12 or len({item.task_id for item in definitions}) != len(
        definitions
    ):
        raise ValueError("A71 requires at least 12 unique frozen tasks")
    counts = {
        operation: sum(item.operation_class is operation for item in definitions)
        for operation in OperationClass
    }
    required = {
        OperationClass.EDIT_SINGLE: 4,
        OperationClass.EDIT_MULTI: 4,
        OperationClass.CREATE: 2,
        OperationClass.MIXED_EDIT_CREATE: 2,
    }
    if any(counts[operation] < minimum for operation, minimum in required.items()):
        raise ValueError(f"A71 corpus class coverage incomplete: {counts}")
