"""Frozen A72 matrix identity over the unchanged A71 task corpus."""

from __future__ import annotations

import hashlib
import json

from benchmarks.realistic_coding_v2.suite import FrozenTask
from benchmarks.transaction_readiness_v1.suite import (
    PROFILES,
    SEED,
    tasks,
)
from benchmarks.transaction_readiness_v1.suite import (
    validate_corpus as _validate_corpus,
)

SUITE = "transaction-readiness-v2"
VERSION = 2
SCHEMA_VERSION = 2
EVALUATOR_IDENTITY = "transaction-readiness-v2-association-v2"
RUN_IDENTITY = "a72-authoritative-20261005"
CONTEXT_SIZE = 8192
MAX_TOKENS = 512
TEMPERATURE = 0.0
SOURCE_IDENTITY_POLICY = "a66-source-only-v1"


def validate_corpus(definitions: tuple[FrozenTask, ...]) -> None:
    """Apply the unchanged A71 corpus coverage contract."""
    _validate_corpus(definitions)


def frozen_matrix_identity() -> str:
    """Bind tasks, oracles, verification plans, settings, and identity policy."""
    payload = []
    for definition in tasks():
        task = definition.production_task
        payload.append(
            {
                "id": definition.versioned_id,
                "operation_class": definition.operation_class.value,
                "authority_mode": definition.authority_mode.value,
                "prompt": task.prompt,
                "allowed_paths": task.allowed_paths,
                "required_candidate_paths": task.required_candidate_paths,
                "create_candidate_paths": task.create_candidate_paths,
                "expected_changed_paths": task.expected_changed_paths,
                "setup": [
                    (item.path, item.expected, item.replacement) for item in task.setup
                ],
                "setup_absent_paths": task.setup_absent_paths,
                "verification_plan": repr(task.verification_plan),
                "oracle_commands": task.oracle_commands,
            }
        )
    frozen = {
        "suite": SUITE,
        "version": VERSION,
        "evaluator_identity": EVALUATOR_IDENTITY,
        "run_identity": RUN_IDENTITY,
        "profiles": PROFILES,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "context_size": CONTEXT_SIZE,
        "max_tokens": MAX_TOKENS,
        "source_identity_policy": SOURCE_IDENTITY_POLICY,
        "tasks": payload,
    }
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
