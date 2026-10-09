"""Frozen A77 atomic task and four-profile matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from benchmarks.behavioral_obligation_diagnosis_v1.suite import (
    CONTEXT_SIZE,
    SEED,
    DiagnosticCase,
    Obligation,
    task_map,
)
from benchmarks.behavioral_obligation_diagnosis_v1.suite import (
    cases as a76_cases,
)
from benchmarks.behavioral_obligation_diagnosis_v1.suite import (
    obligations as a76_obligations,
)
from forge.evidence_coverage import decompose_evidence_plan
from forge.models import MutationRepresentationPolicy

SUITE = "atomic-coding-capability-v1"
VERSION = 1
SCHEMA_VERSION = 1
RUN_ID = "a77-atomic-coding-capability-v1"
PROFILES = ("qwen-small", "qwen-large", "codestral-22b", "deepseek-coder-lite")
REPRESENTATION = MutationRepresentationPolicy.LINE_RANGE


@dataclass(frozen=True, slots=True)
class AtomicTask:
    atomic_id: str
    source_case_id: str
    task_id: str
    operation_class: str
    obligation: Obligation
    control: bool = False


def tasks() -> tuple[AtomicTask, ...]:
    frozen = []
    for case in a76_cases():
        items = tuple(item for item in a76_obligations(case) if item.isolatable)
        if not items:
            continue
        if len(items) != 1:
            raise ValueError("A77 only admits exactly one frozen A76 obligation")
        frozen.append(
            AtomicTask(
                f"A77-{case.case_id.removeprefix('A74-')}",
                case.case_id,
                case.task_id,
                case.operation_class,
                items[0],
            )
        )
    definitions = task_map()
    for index, task_id in enumerate(("C03", "C05", "C06"), 1):
        definition = definitions[task_id]
        goals = tuple(
            goal
            for goal in decompose_evidence_plan(definition.production_task.prompt).goals
            if goal.required
        )
        if len(goals) != 1:
            raise ValueError("A77 controls require one production-visible goal")
        goal = goals[0]
        oracle = hashlib.sha256(
            json.dumps(
                {
                    "goal_id": goal.goal_id,
                    "commands": definition.production_task.oracle_commands,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        frozen.append(
            AtomicTask(
                f"A77-K{index:02d}",
                f"CONTROL-{task_id}",
                task_id,
                definition.operation_class.value,
                Obligation(goal.goal_id, goal.description, oracle, True),
                True,
            )
        )
    return tuple(frozen)


def diagnostic_case(task: AtomicTask, profile: str) -> DiagnosticCase:
    return DiagnosticCase(
        task.atomic_id,
        profile,
        task.task_id,
        task.operation_class,
        "atomic_capability",
    )


def corpus_identity() -> str:
    payload = {
        "suite": SUITE,
        "settings": {"seed": SEED, "temperature": 0, "context": 8192, "output": 512},
        "tasks": [
            {
                "atomic_id": task.atomic_id,
                "source_case_id": task.source_case_id,
                "task_id": task.task_id,
                "operation_class": task.operation_class,
                "goal_id": task.obligation.goal_id,
                "description": task.obligation.description,
                "oracle_identity": task.obligation.oracle_identity,
                "control": task.control,
            }
            for task in tasks()
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_matrix() -> None:
    frozen = tasks()
    if len(frozen) != 12 or len({item.atomic_id for item in frozen}) != 12:
        raise ValueError("A77 requires 12 unique atomic task cells per profile")
    if len(PROFILES) != 4 or "deepseek-coder-lite" not in PROFILES:
        raise ValueError("A77 requires four profiles including DeepSeek")
    if any(not item.obligation.isolatable for item in frozen):
        raise ValueError("A77 task is not isolatable")


__all__ = [
    "CONTEXT_SIZE",
    "PROFILES",
    "REPRESENTATION",
    "RUN_ID",
    "SCHEMA_VERSION",
    "SEED",
    "SUITE",
    "VERSION",
    "AtomicTask",
    "corpus_identity",
    "diagnostic_case",
    "task_map",
    "tasks",
    "validate_matrix",
]
