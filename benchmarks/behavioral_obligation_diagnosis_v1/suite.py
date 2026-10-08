"""Frozen A76 diagnostic cases and production-visible obligations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from benchmarks.grounded_mutation_planning_v1.suite import (
    CASES as A74_CASES,
)
from benchmarks.grounded_mutation_planning_v1.suite import (
    PROFILES,
    SEED,
    task_map,
)
from forge.evidence_coverage import EvidenceGoal, decompose_evidence_plan

SUITE = "behavioral-obligation-diagnosis-v1"
VERSION = 1
SCHEMA_VERSION = 1
RUN_ID = "a76-behavioral-obligation-diagnosis-v1"
CONTEXT_SIZE = 8192


class Condition(StrEnum):
    F0 = "F0_FULL_TASK"
    F1 = "F1_ISOLATED_OBLIGATION"


@dataclass(frozen=True, slots=True)
class DiagnosticCase:
    case_id: str
    profile: str
    task_id: str
    operation_class: str
    rationale: str


@dataclass(frozen=True, slots=True)
class Obligation:
    goal_id: str
    description: str
    oracle_identity: str
    isolatable: bool
    non_isolatable_reason: str | None = None


_SELECTED = (
    "A74-C01",
    "A74-C02",
    "A74-C03",
    "A74-C04",
    "A74-M01",
    "A74-M02",
    "A74-M04",
    "A74-S01",
    "A74-S02",
    "A74-X01",
)


def cases() -> tuple[DiagnosticCase, ...]:
    definitions = task_map()
    by_id = {case.case_id: case for case in A74_CASES}
    return tuple(
        DiagnosticCase(
            case_id,
            by_id[case_id].profile,
            by_id[case_id].task_id,
            definitions[by_id[case_id].task_id].operation_class.value,
            by_id[case_id].rationale,
        )
        for case_id in _SELECTED
    )


def obligations(case: DiagnosticCase) -> tuple[Obligation, ...]:
    definition = task_map()[case.task_id]
    goals = tuple(
        goal
        for goal in decompose_evidence_plan(definition.production_task.prompt).goals
        if goal.required
    )
    if len(goals) != 1:
        return tuple(
            Obligation(
                goal.goal_id,
                goal.description,
                _obligation_oracle_identity(
                    goal, definition.production_task.oracle_commands
                ),
                False,
                "NO_PRODUCTION_VISIBLE_PER_GOAL_ORACLE_OR_AUTHORITY_MAPPING",
            )
            for goal in goals
        )
    return tuple(
        Obligation(
            goal.goal_id,
            goal.description,
            _obligation_oracle_identity(
                goal, definition.production_task.oracle_commands
            ),
            True,
        )
        for goal in goals
    )


def _obligation_oracle_identity(goal: EvidenceGoal, commands: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"goal_id": goal.goal_id, "commands": commands}, sort_keys=True
        ).encode()
    ).hexdigest()


def corpus_identity() -> str:
    payload = {
        "suite": SUITE,
        "settings": {"seed": SEED, "temperature": 0, "context": 8192, "output": 512},
        "cases": [
            {
                "case_id": case.case_id,
                "profile": case.profile,
                "task_id": case.task_id,
                "operation_class": case.operation_class,
                "goals": [
                    {
                        "goal_id": item.goal_id,
                        "description": item.description,
                        "oracle_identity": item.oracle_identity,
                        "isolatable": item.isolatable,
                    }
                    for item in obligations(case)
                ],
            }
            for case in cases()
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_corpus() -> None:
    selected = cases()
    if len(selected) != 10 or len({case.case_id for case in selected}) != 10:
        raise ValueError("A76 requires 10 unique cases")
    counts = {
        operation: sum(case.operation_class == operation for case in selected)
        for operation in {case.operation_class for case in selected}
    }
    if counts != {
        "CREATE": 4,
        "EDIT_MULTI": 3,
        "EDIT_SINGLE": 2,
        "MIXED_EDIT_CREATE": 1,
    }:
        raise ValueError(f"A76 class distribution invalid: {counts}")
    if {case.profile for case in selected} != set(PROFILES):
        raise ValueError("A76 requires all three model profiles")


def goal_origin(goal: EvidenceGoal, case: DiagnosticCase) -> bool:
    plan = decompose_evidence_plan(task_map()[case.task_id].production_task.prompt)
    return goal in plan.goals


def narrow_authority(
    original: tuple[str, ...], requested: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    if requested is None:
        return original
    if not set(requested).issubset(original):
        raise ValueError("isolated obligation authority cannot broaden")
    return requested
