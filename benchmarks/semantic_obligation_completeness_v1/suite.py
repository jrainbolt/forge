"""Frozen production-visible obligations for A81."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from benchmarks.coding_default_operational_v1.suite import (
    REPRESENTATION,
)
from benchmarks.coding_default_operational_v1.suite import (
    corpus_identity as parent_corpus_identity,
)
from benchmarks.default_candidate_confirmation_v1.suite import tasks as parent_tasks

SUITE = "semantic-obligation-completeness-v1"
VERSION = 3
RUN_ID = "a81-semantic-obligation-completeness-v3-verification-aligned"
TASK_IDS = ("H07", "H08", "H09", "H10", "H11", "H12", "H01", "H04")
CHECK_SET_IDENTITY = hashlib.sha256(
    Path(__file__).with_name("checks.py").read_bytes()
).hexdigest()


class ObligationOrigin(Enum):
    REQUIRED_BEHAVIOR_GOAL = "REQUIRED_BEHAVIOR_GOAL"
    REQUIRED_COMPONENT_ROLE = "REQUIRED_COMPONENT_ROLE"
    REQUIRED_RELATIONSHIP = "REQUIRED_RELATIONSHIP"
    REQUIRED_INVARIANT = "REQUIRED_INVARIANT"
    REQUIRED_STATE_TRANSITION = "REQUIRED_STATE_TRANSITION"


class VerificationCoverage(Enum):
    COVERS = "VERIFICATION_COVERS_OBLIGATION"
    DOES_NOT_COVER = "VERIFICATION_DOES_NOT_COVER_OBLIGATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Obligation:
    obligation_id: str
    check_id: str | None
    origin: ObligationOrigin
    component: str
    coverage: VerificationCoverage
    relationship: bool = False

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "id": self.obligation_id,
                    "check": self.check_id,
                    "origin": self.origin.value,
                    "component": self.component,
                    "relationship": self.relationship,
                    "check_set_identity": CHECK_SET_IDENTITY,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()


def _o(
    task: str,
    name: str,
    origin: ObligationOrigin,
    component: str,
    coverage: VerificationCoverage,
    *,
    relationship: bool = False,
    check: bool = True,
) -> Obligation:
    return Obligation(
        f"{task}-{name}",
        f"{task}-{name}" if check else None,
        origin,
        component,
        coverage,
        relationship,
    )


P = ObligationOrigin
V = VerificationCoverage
OBLIGATIONS = {
    "H07": (
        _o("H07", "initial-zero", P.REQUIRED_INVARIANT, "CREATE", V.DOES_NOT_COVER),
        _o("H07", "accumulate", P.REQUIRED_BEHAVIOR_GOAL, "CREATE", V.DOES_NOT_COVER),
        _o("H07", "reject-negative", P.REQUIRED_INVARIANT, "CREATE", V.DOES_NOT_COVER),
    ),
    "H08": (
        _o("H08", "header-interface", P.REQUIRED_COMPONENT_ROLE, "CREATE", V.COVERS),
        _o(
            "H08",
            "rolling-algorithm",
            P.REQUIRED_BEHAVIOR_GOAL,
            "CREATE",
            V.DOES_NOT_COVER,
        ),
        _o("H08", "all-bytes", P.REQUIRED_INVARIANT, "CREATE", V.DOES_NOT_COVER),
    ),
    "H09": (
        _o("H09", "boolean-forms", P.REQUIRED_BEHAVIOR_GOAL, "CREATE", V.COVERS),
        _o("H09", "absent-default", P.REQUIRED_BEHAVIOR_GOAL, "CREATE", V.COVERS),
        _o("H09", "positive-timeout", P.REQUIRED_INVARIANT, "CREATE", V.COVERS),
        _o(
            "H09",
            "reject-nonpositive",
            P.REQUIRED_INVARIANT,
            "CREATE",
            V.DOES_NOT_COVER,
        ),
    ),
    "H10": (
        _o("H10", "queued-running", P.REQUIRED_STATE_TRANSITION, "EDIT", V.COVERS),
        _o(
            "H10",
            "queued-cancelled",
            P.REQUIRED_STATE_TRANSITION,
            "EDIT",
            V.DOES_NOT_COVER,
        ),
        _o(
            "H10",
            "counter-accumulate",
            P.REQUIRED_BEHAVIOR_GOAL,
            "CREATE",
            V.DOES_NOT_COVER,
        ),
        _o("H10", "counter-negative", P.REQUIRED_INVARIANT, "CREATE", V.DOES_NOT_COVER),
    ),
    "H11": (
        _o("H11", "positive-limit", P.REQUIRED_INVARIANT, "EDIT", V.COVERS),
        _o("H11", "strictly-below", P.REQUIRED_INVARIANT, "EDIT", V.COVERS),
        _o(
            "H11",
            "checksum-interface",
            P.REQUIRED_RELATIONSHIP,
            "RELATIONSHIP",
            V.COVERS,
            relationship=True,
        ),
        _o(
            "H11",
            "checksum-algorithm",
            P.REQUIRED_BEHAVIOR_GOAL,
            "CREATE",
            V.DOES_NOT_COVER,
        ),
    ),
    "H12": (
        _o(
            "H12", "storage-replace", P.REQUIRED_BEHAVIOR_GOAL, "EDIT", V.DOES_NOT_COVER
        ),
        _o("H12", "boolean-forms", P.REQUIRED_BEHAVIOR_GOAL, "CREATE", V.COVERS),
        _o("H12", "absent-default", P.REQUIRED_BEHAVIOR_GOAL, "CREATE", V.COVERS),
        _o("H12", "positive-timeout", P.REQUIRED_INVARIANT, "CREATE", V.COVERS),
        _o(
            "H12",
            "reject-nonpositive",
            P.REQUIRED_INVARIANT,
            "CREATE",
            V.DOES_NOT_COVER,
        ),
    ),
    "H01": (
        _o("H01", "accept-zero", P.REQUIRED_BEHAVIOR_GOAL, "EDIT", V.DOES_NOT_COVER),
        _o("H01", "reject-negative", P.REQUIRED_INVARIANT, "EDIT", V.DOES_NOT_COVER),
        _o("H01", "accumulate", P.REQUIRED_BEHAVIOR_GOAL, "EDIT", V.DOES_NOT_COVER),
    ),
    "H04": (
        _o("H04", "positive-timeout", P.REQUIRED_INVARIANT, "EDIT", V.COVERS),
        _o("H04", "header-normalize", P.REQUIRED_BEHAVIOR_GOAL, "EDIT", V.COVERS),
        _o("H04", "header-validate", P.REQUIRED_INVARIANT, "EDIT", V.COVERS),
    ),
}


def tasks():  # type: ignore[no-untyped-def]
    by_id = {item.task_id: item for item in parent_tasks()}
    return tuple(by_id[task_id] for task_id in TASK_IDS)


def corpus_identity() -> str:
    payload = {
        "suite": SUITE,
        "version": VERSION,
        "parent": parent_corpus_identity(),
        "tasks": TASK_IDS,
        "obligations": {
            task: [item.identity for item in values]
            for task, values in OBLIGATIONS.items()
        },
        "representation": REPRESENTATION.value,
        "check_set_identity": CHECK_SET_IDENTITY,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_corpus() -> None:
    if set(OBLIGATIONS) != set(TASK_IDS):
        raise RuntimeError("A81 obligation/task mismatch")
    if any(not values for values in OBLIGATIONS.values()):
        raise RuntimeError("A81 task has no production-visible obligation")
    if len(
        {item.obligation_id for values in OBLIGATIONS.values() for item in values}
    ) != sum(len(values) for values in OBLIGATIONS.values()):
        raise RuntimeError("A81 obligation IDs are not unique")
