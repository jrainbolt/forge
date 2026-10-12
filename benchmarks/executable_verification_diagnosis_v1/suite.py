"""Frozen production-visible A90 behavioral obligation inventory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from benchmarks.default_candidate_confirmation_v1.suite import tasks as parent_tasks
from benchmarks.semantic_obligation_completeness_v1.suite import (
    OBLIGATIONS as A81_OBLIGATIONS,
)

RUN_ID = "a90-executable-verification-coverage-diagnosis-v2"
TASK_IDS = (
    "H07",
    "H08",
    "H09",
    "H10",
    "H11",
    "H12",
    "H01",
    "H04",
    "H02",
    "H03",
    "H05",
    "H06",
)


class Coverage(StrEnum):
    COVERS = "VERIFICATION_COVERS"
    PARTIAL = "VERIFICATION_PARTIALLY_COVERS"
    DOES_NOT_COVER = "VERIFICATION_DOES_NOT_COVER"
    UNKNOWN = "VERIFICATION_COVERAGE_UNKNOWN"


class Derivability(StrEnum):
    DERIVABLE = "EXECUTABLE_CHECK_DERIVABLE"
    PARTIAL = "EXECUTABLE_CHECK_PARTIALLY_DERIVABLE"
    NOT_DERIVABLE = "EXECUTABLE_CHECK_NOT_DERIVABLE"


@dataclass(frozen=True, slots=True)
class VerificationObligation:
    obligation_id: str
    check_id: str | None
    component: str
    coverage: Coverage
    derivability: Derivability
    source_class: str = "EXPLICIT_TASK_AND_PUBLIC_API"

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "id": self.obligation_id,
                    "check": self.check_id,
                    "component": self.component,
                    "coverage": self.coverage.value,
                    "derivability": self.derivability.value,
                    "source": self.source_class,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()


def _from_a81(task_id: str) -> tuple[VerificationObligation, ...]:
    return tuple(
        VerificationObligation(
            item.obligation_id,
            item.check_id,
            item.component,
            Coverage.COVERS
            if item.coverage.value == "VERIFICATION_COVERS_OBLIGATION"
            else Coverage.DOES_NOT_COVER,
            Derivability.DERIVABLE if item.check_id else Derivability.NOT_DERIVABLE,
        )
        for item in A81_OBLIGATIONS[task_id]
    )


def _o(
    task: str, name: str, component: str, coverage: Coverage
) -> VerificationObligation:
    return VerificationObligation(
        f"{task}-{name}",
        f"{task}-{name}",
        component,
        coverage,
        Derivability.DERIVABLE,
    )


OBLIGATIONS = {
    **{task: _from_a81(task) for task in TASK_IDS[:8]},
    "H02": (_o("H02", "storage-replace", "EDIT", Coverage.COVERS),),
    "H03": (
        _o("H03", "clock-forward", "EDIT", Coverage.COVERS),
        _o("H03", "clock-backward", "EDIT", Coverage.COVERS),
    ),
    "H05": (
        _o("H05", "queue-fifo", "EDIT", Coverage.COVERS),
        _o("H05", "missing-before-empty", "EDIT", Coverage.COVERS),
    ),
    "H06": (
        _o("H06", "port-zero", "EDIT", Coverage.COVERS),
        _o("H06", "quota-exact-fill", "EDIT", Coverage.COVERS),
    ),
}


def tasks():  # type: ignore[no-untyped-def]
    values = {item.task_id: item for item in parent_tasks()}
    return tuple(values[task_id] for task_id in TASK_IDS)


def corpus_identity() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "run": RUN_ID,
                "tasks": TASK_IDS,
                "obligations": {
                    task: [item.identity for item in items]
                    for task, items in OBLIGATIONS.items()
                },
                "inventory_frozen_before_model": True,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def validate_corpus() -> None:
    if len(TASK_IDS) != 12 or set(TASK_IDS) != set(OBLIGATIONS):
        raise RuntimeError("A90 requires exactly twelve frozen workflows")
    if any(not OBLIGATIONS[task] for task in TASK_IDS):
        raise RuntimeError("A90 task missing behavioral obligations")
