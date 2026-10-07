"""Frozen A73 diagnostic corpus selected from authoritative A72 outcomes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from benchmarks.transaction_readiness_v1.suite import tasks as a72_tasks

SUITE = "verification-attribution-v1"
VERSION = 1
SCHEMA_VERSION = 1
SEED = 42
CONTEXT_SIZE = 8192
PROFILES = ("qwen-small", "qwen-large", "codestral-22b")


@dataclass(frozen=True, slots=True)
class DiagnosticCase:
    case_id: str
    profile: str
    task_id: str
    a72_outcome: str


CASES = (
    DiagnosticCase("A73-F01", "qwen-small", "C02", "verification_fail"),
    DiagnosticCase("A73-F02", "qwen-small", "C04", "verification_fail"),
    DiagnosticCase("A73-F03", "qwen-small", "C07", "verification_fail"),
    DiagnosticCase("A73-F04", "qwen-small", "C08", "verification_fail"),
    DiagnosticCase("A73-F05", "qwen-small", "C10", "verification_fail"),
    DiagnosticCase("A73-F06", "qwen-small", "R06", "verification_fail"),
    DiagnosticCase("A73-F07", "qwen-large", "C07", "verification_fail"),
    DiagnosticCase("A73-F08", "qwen-large", "C09", "verification_fail"),
    DiagnosticCase("A73-F09", "codestral-22b", "C01", "verification_fail"),
    DiagnosticCase("A73-F10", "codestral-22b", "C08", "verification_fail"),
    DiagnosticCase("A73-F11", "codestral-22b", "C10", "verification_fail"),
    DiagnosticCase("A73-F12", "codestral-22b", "R06", "verification_fail"),
    DiagnosticCase("A73-P01", "qwen-small", "C01", "verification_pass"),
    DiagnosticCase("A73-P02", "qwen-large", "C04", "verification_pass"),
    DiagnosticCase("A73-P03", "codestral-22b", "C09", "verification_pass"),
    DiagnosticCase("A73-P04", "codestral-22b", "C11", "verification_pass"),
)


def task_map():  # type: ignore[no-untyped-def]
    return {item.task_id: item for item in a72_tasks()}


def corpus_identity() -> str:
    definitions = task_map()
    payload = {
        "suite": SUITE,
        "version": VERSION,
        "seed": SEED,
        "context": CONTEXT_SIZE,
        "output": 512,
        "temperature": 0,
        "source_identity_policy": "a66-source-only-v1",
        "cases": [
            {
                "case_id": case.case_id,
                "profile": case.profile,
                "task": case.task_id,
                "a72_outcome": case.a72_outcome,
                "task_version": definitions[case.task_id].version,
                "verification_plan": repr(
                    definitions[case.task_id].production_task.verification_plan
                ),
                "oracle": definitions[case.task_id].production_task.oracle_commands,
            }
            for case in CASES
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_corpus() -> None:
    if len(CASES) != 16 or len({item.case_id for item in CASES}) != 16:
        raise ValueError("A73 requires 16 unique cases")
    if sum(item.a72_outcome == "verification_fail" for item in CASES) != 12:
        raise ValueError("A73 requires 12 verification failures")
    if sum(item.a72_outcome == "verification_pass" for item in CASES) != 4:
        raise ValueError("A73 requires four pass controls")
    if {item.profile for item in CASES} != set(PROFILES):
        raise ValueError("A73 requires all three profiles")
