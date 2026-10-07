"""Frozen paired A74 corpus of model/task cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from benchmarks.transaction_readiness_v1.suite import tasks as frozen_tasks

SUITE = "grounded-mutation-planning-v1"
VERSION = 1
SCHEMA_VERSION = 1
SEED = 42
CONTEXT_SIZE = 8192
PROFILES = ("qwen-small", "qwen-large", "codestral-22b")


class Condition(StrEnum):
    P0 = "P0_DIRECT_MUTATION"
    P1 = "P1_PLAN_THEN_MUTATE"


@dataclass(frozen=True, slots=True)
class PlanningCase:
    case_id: str
    profile: str
    task_id: str
    rationale: str


CASES = (
    PlanningCase("A74-C01", "qwen-small", "C07", "create_wrong_behavior"),
    PlanningCase("A74-C02", "qwen-small", "C08", "create_wrong_behavior"),
    PlanningCase("A74-C03", "qwen-large", "C09", "create_wrong_behavior"),
    PlanningCase("A74-C04", "codestral-22b", "C07", "create_control"),
    PlanningCase("A74-M01", "qwen-small", "C04", "partial_multi_file"),
    PlanningCase("A74-M02", "qwen-small", "R06", "partial_multi_file"),
    PlanningCase("A74-M03", "qwen-large", "C04", "multi_file_control"),
    PlanningCase("A74-M04", "codestral-22b", "R06", "repair_control"),
    PlanningCase("A74-S01", "qwen-small", "C02", "build_blocking"),
    PlanningCase("A74-S02", "codestral-22b", "C01", "wrong_behavior"),
    PlanningCase("A74-X01", "qwen-small", "C10", "partial_mixed"),
    PlanningCase("A74-X02", "codestral-22b", "C11", "mixed_control"),
)


def task_map():  # type: ignore[no-untyped-def]
    return {item.task_id: item for item in frozen_tasks()}


def corpus_identity() -> str:
    definitions = task_map()
    payload = {
        "suite": SUITE,
        "version": VERSION,
        "settings": {"seed": SEED, "temperature": 0, "context": 8192, "output": 512},
        "cases": [
            {
                "case": item.case_id,
                "profile": item.profile,
                "task": item.task_id,
                "rationale": item.rationale,
                "task_version": definitions[item.task_id].version,
                "prompt": definitions[item.task_id].production_task.prompt,
                "paths": definitions[item.task_id].production_task.allowed_paths,
                "oracle": definitions[item.task_id].production_task.oracle_commands,
            }
            for item in CASES
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_corpus() -> None:
    if len(CASES) != 12 or len({item.case_id for item in CASES}) != 12:
        raise ValueError("A74 requires 12 unique cases")
    definitions = task_map()
    counts = {
        name: 0 for name in ("CREATE", "EDIT_MULTI", "EDIT_SINGLE", "MIXED_EDIT_CREATE")
    }
    for case in CASES:
        counts[definitions[case.task_id].operation_class.value] += 1
    if counts != {
        "CREATE": 4,
        "EDIT_MULTI": 4,
        "EDIT_SINGLE": 2,
        "MIXED_EDIT_CREATE": 2,
    }:
        raise ValueError(f"A74 class distribution invalid: {counts}")
    if {item.profile for item in CASES} != set(PROFILES):
        raise ValueError("A74 requires all model profiles")
