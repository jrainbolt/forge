"""Frozen A83 shadow corpus and production-visible structural checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from benchmarks.default_candidate_confirmation_v1.suite import tasks as a78_tasks
from benchmarks.production_detectable_completeness_v1.suite import (
    CASES as A82_CASES,
)
from benchmarks.production_detectable_completeness_v1.suite import (
    REPRESENTATION,
    CheckKind,
    CompletenessCheck,
)
from benchmarks.realistic_coding_v2.suite import tasks as realistic_tasks

SUITE = "structural-completeness-shadow-v1"
VERSION = 1
RUN_ID = "a83-structural-completeness-shadow-v1"
MAX_MEDIAN_CHECKER_LATENCY_SECONDS = 0.25
MAX_WORST_CHECKER_LATENCY_SECONDS = 2.0
TASK_IDS = (
    "H11",
    "C11",
    "H01",
    "H03",
    "H04",
    "H05",
    "C07",
    "H08",
    "H07",
    "H09",
    "C09",
    "C08",
    "C10",
    "C12",
)


@dataclass(frozen=True, slots=True)
class ShadowCase:
    task_id: str
    truth: str
    checks: tuple[CompletenessCheck, ...]
    determinism_control: bool = False


def _check(
    task: str,
    name: str,
    kind: CheckKind,
    source: str,
    *,
    target: str | None = None,
    symbol: str | None = None,
    caller: str | None = None,
    scope: tuple[str, ...],
) -> CompletenessCheck:
    return CompletenessCheck(
        f"{task}-{name}", kind, source, target, symbol, caller, scope
    )


K = CheckKind
A82 = {case.task_id: case for case in A82_CASES}
ADDITIONAL = {
    "H01": (
        _check(
            "H01",
            "counter-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/metrics.py",
            symbol="Counter",
            scope=("pyservice/metrics.py",),
        ),
        _check(
            "H01",
            "counter-role",
            K.REQUIRED_COMPONENT_ROLE,
            "pyservice/metrics.py",
            symbol="Counter",
            scope=("pyservice/metrics.py",),
        ),
    ),
    "H03": (
        _check(
            "H03",
            "clock-declaration-implementation",
            K.DECLARATION_IMPLEMENTATION_RELATION,
            "cengine/clock.h",
            target="cengine/clock.c",
            symbol="clock_elapsed",
            scope=("cengine/clock.h", "cengine/clock.c"),
        ),
        _check(
            "H03",
            "clock-include",
            K.IMPORT_INCLUDE_RELATION,
            "cengine/clock.c",
            target="cengine/clock.h",
            scope=("cengine/clock.h", "cengine/clock.c"),
        ),
        _check(
            "H03",
            "clock-role",
            K.REQUIRED_COMPONENT_ROLE,
            "cengine/clock.c",
            target="cengine/clock.h",
            symbol="clock_elapsed",
            scope=("cengine/clock.h", "cengine/clock.c"),
        ),
    ),
    "H04": (
        _check(
            "H04",
            "timeout-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/config.py",
            symbol="parse_timeout",
            scope=("pyservice/config.py",),
        ),
        _check(
            "H04",
            "headers-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/headers.py",
            symbol="normalize_header_name",
            scope=("pyservice/headers.py",),
        ),
        _check(
            "H04",
            "config-role",
            K.REQUIRED_COMPONENT_ROLE,
            "pyservice/config.py",
            symbol="parse_timeout",
            scope=("pyservice/config.py",),
        ),
        _check(
            "H04",
            "headers-role",
            K.REQUIRED_COMPONENT_ROLE,
            "pyservice/headers.py",
            symbol="normalize_header_name",
            scope=("pyservice/headers.py",),
        ),
    ),
    "H05": (
        _check(
            "H05",
            "queue-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/queue.py",
            symbol="RequestQueue",
            scope=("pyservice/queue.py",),
        ),
        _check(
            "H05",
            "service-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/service.py",
            symbol="handle",
            scope=("pyservice/service.py",),
        ),
        _check(
            "H05",
            "queue-role",
            K.REQUIRED_COMPONENT_ROLE,
            "pyservice/queue.py",
            symbol="RequestQueue",
            scope=("pyservice/queue.py",),
        ),
        _check(
            "H05",
            "service-role",
            K.REQUIRED_COMPONENT_ROLE,
            "pyservice/service.py",
            symbol="handle",
            scope=("pyservice/service.py",),
        ),
    ),
    "H09": (
        _check(
            "H09",
            "enabled-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/config.py",
            symbol="parse_enabled",
            scope=("pyservice/config.py",),
        ),
        _check(
            "H09",
            "timeout-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/config.py",
            symbol="parse_timeout",
            scope=("pyservice/config.py",),
        ),
        _check(
            "H09",
            "enabled-role",
            K.REQUIRED_COMPONENT_ROLE,
            "pyservice/config.py",
            symbol="parse_enabled",
            scope=("pyservice/config.py",),
        ),
        _check(
            "H09",
            "timeout-role",
            K.REQUIRED_COMPONENT_ROLE,
            "pyservice/config.py",
            symbol="parse_timeout",
            scope=("pyservice/config.py",),
        ),
    ),
}


def _a82(task: str) -> tuple[CompletenessCheck, ...]:
    return A82[task].checks


CASES = (
    ShadowCase("H11", "KNOWN_STRUCTURAL_MISS", _a82("H11"), True),
    ShadowCase("C11", "KNOWN_STRUCTURAL_MISS", _a82("C11"), True),
    ShadowCase("H01", "KNOWN_GOOD_STRUCTURAL_CONTROL", ADDITIONAL["H01"]),
    ShadowCase("H03", "KNOWN_GOOD_STRUCTURAL_CONTROL", ADDITIONAL["H03"], True),
    ShadowCase("H04", "KNOWN_GOOD_STRUCTURAL_CONTROL", ADDITIONAL["H04"]),
    ShadowCase("H05", "KNOWN_GOOD_STRUCTURAL_CONTROL", ADDITIONAL["H05"]),
    ShadowCase("C07", "KNOWN_GOOD_STRUCTURAL_CONTROL", _a82("C07")),
    ShadowCase("H08", "KNOWN_GOOD_STRUCTURAL_CONTROL", _a82("H08"), True),
    ShadowCase("H07", "SEMANTIC_NEGATIVE_STRUCTURAL_PASS", _a82("H07"), True),
    ShadowCase("H09", "SEMANTIC_NEGATIVE_STRUCTURAL_PASS", ADDITIONAL["H09"]),
    ShadowCase("C09", "SEMANTIC_NEGATIVE_STRUCTURAL_PASS", _a82("C09"), True),
    ShadowCase("C08", "RELATIONSHIP_TARGET", _a82("C08")),
    ShadowCase("C10", "RELATIONSHIP_TARGET", _a82("C10")),
    ShadowCase("C12", "RELATIONSHIP_TARGET", _a82("C12"), True),
)


def tasks():  # type: ignore[no-untyped-def]
    definitions = {item.task_id: item for item in (*a78_tasks(), *realistic_tasks())}
    return tuple(definitions[task_id] for task_id in TASK_IDS)


def corpus_identity() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "suite": SUITE,
                "version": VERSION,
                "tasks": TASK_IDS,
                "cases": [
                    {
                        "task": case.task_id,
                        "truth": case.truth,
                        "checks": [check.identity for check in case.checks],
                        "determinism": case.determinism_control,
                    }
                    for case in CASES
                ],
                "representation": REPRESENTATION.value,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def validate_corpus() -> None:
    if tuple(case.task_id for case in CASES) != TASK_IDS:
        raise RuntimeError("A83 case/task mismatch")
    if not 12 <= len(CASES) <= 16:
        raise RuntimeError("A83 requires 12-16 cases")
    if len({check.check_id for case in CASES for check in case.checks}) != sum(
        len(case.checks) for case in CASES
    ):
        raise RuntimeError("A83 check IDs are not unique")
    if {check.kind for case in CASES for check in case.checks} != set(CheckKind):
        raise RuntimeError("A83 must cover all six accepted check kinds")
