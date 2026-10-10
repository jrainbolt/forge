"""Frozen A84 model and operational-policy readiness corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from benchmarks.default_candidate_confirmation_v1.suite import tasks as a78_tasks
from benchmarks.production_detectable_completeness_v1.suite import (
    REPRESENTATION,
    CheckKind,
    CompletenessCheck,
)
from benchmarks.realistic_coding_v2.suite import tasks as realistic_tasks
from benchmarks.structural_completeness_shadow_v1.suite import CASES as A83_CASES

SUITE = "structural-completeness-readiness-v1"
VERSION = 5
RUN_ID = "a84-structural-completeness-readiness-v5-checker-correction"
MAX_MEDIAN_CHECKER_LATENCY_SECONDS = 0.25
MAX_WORST_CHECKER_LATENCY_SECONDS = 2.0
COMPILER_TIMEOUT_SECONDS = 2


@dataclass(frozen=True, slots=True)
class ReadinessCase:
    case_id: str
    task_id: str | None
    truth: str
    checks: tuple[CompletenessCheck, ...]
    model_call: bool = True
    control_scenario: str | None = None
    determinism_control: bool = True


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
ADDITIONAL = {
    "H02": (),
    "H06": (
        _check(
            "H06",
            "parser-role",
            K.REQUIRED_COMPONENT_ROLE,
            "cengine/parser.c",
            target="cengine/parser.h",
            symbol="parse_port",
            scope=("cengine/parser.h", "cengine/parser.c"),
        ),
        _check(
            "H06",
            "quota-role",
            K.REQUIRED_COMPONENT_ROLE,
            "cengine/quota.c",
            target="cengine/quota.h",
            symbol="quota_reserve",
            scope=("cengine/quota.h", "cengine/quota.c"),
        ),
        _check(
            "H06",
            "parser-include",
            K.IMPORT_INCLUDE_RELATION,
            "cengine/parser.c",
            target="cengine/parser.h",
            scope=("cengine/parser.h", "cengine/parser.c"),
        ),
        _check(
            "H06",
            "quota-include",
            K.IMPORT_INCLUDE_RELATION,
            "cengine/quota.c",
            target="cengine/quota.h",
            scope=("cengine/quota.h", "cengine/quota.c"),
        ),
    ),
    "H10": (
        _check(
            "H10",
            "counter-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/metrics.py",
            symbol="Counter",
            scope=("pyservice/metrics.py",),
        ),
        _check(
            "H10",
            "counter-role",
            K.REQUIRED_COMPONENT_ROLE,
            "pyservice/metrics.py",
            symbol="Counter",
            scope=("pyservice/metrics.py",),
        ),
        _check(
            "H10",
            "state-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/state.py",
            symbol="can_transition",
            scope=("pyservice/state.py",),
        ),
    ),
    "H12": (
        _check(
            "H12",
            "storage-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/storage.py",
            symbol="ResponseStore",
            scope=("pyservice/storage.py",),
        ),
        _check(
            "H12",
            "enabled-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/config.py",
            symbol="parse_enabled",
            scope=("pyservice/config.py",),
        ),
        _check(
            "H12",
            "timeout-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/config.py",
            symbol="parse_timeout",
            scope=("pyservice/config.py",),
        ),
        _check(
            "H12",
            "config-role",
            K.REQUIRED_COMPONENT_ROLE,
            "pyservice/config.py",
            symbol="parse_enabled",
            scope=("pyservice/config.py",),
        ),
    ),
    "C01": (),
    "C02": (
        _check(
            "C02",
            "parser-declaration-implementation",
            K.DECLARATION_IMPLEMENTATION_RELATION,
            "cengine/parser.h",
            target="cengine/parser.c",
            symbol="parse_port",
            scope=("cengine/parser.h", "cengine/parser.c"),
        ),
        _check(
            "C02",
            "parser-include",
            K.IMPORT_INCLUDE_RELATION,
            "cengine/parser.c",
            target="cengine/parser.h",
            scope=("cengine/parser.h", "cengine/parser.c"),
        ),
        _check(
            "C02",
            "parser-role",
            K.REQUIRED_COMPONENT_ROLE,
            "cengine/parser.c",
            target="cengine/parser.h",
            symbol="parse_port",
            scope=("cengine/parser.h", "cengine/parser.c"),
        ),
    ),
    "C03": (
        _check(
            "C03",
            "quota-declaration-implementation",
            K.DECLARATION_IMPLEMENTATION_RELATION,
            "cengine/quota.h",
            target="cengine/quota.c",
            symbol="quota_release",
            scope=("cengine/quota.h", "cengine/quota.c"),
        ),
        _check(
            "C03",
            "quota-include",
            K.IMPORT_INCLUDE_RELATION,
            "cengine/quota.c",
            target="cengine/quota.h",
            scope=("cengine/quota.h", "cengine/quota.c"),
        ),
        _check(
            "C03",
            "quota-role",
            K.REQUIRED_COMPONENT_ROLE,
            "cengine/quota.c",
            target="cengine/quota.h",
            symbol="quota_release",
            scope=("cengine/quota.h", "cengine/quota.c"),
        ),
    ),
    "C04": (
        _check(
            "C04",
            "state-test-import",
            K.IMPORT_INCLUDE_RELATION,
            "tests/test_state.py",
            target="pyservice/state.py",
            symbol="can_transition",
            scope=("pyservice/state.py", "tests/test_state.py"),
        ),
        _check(
            "C04",
            "state-test-use",
            K.REGISTRATION_OR_USAGE_RELATION,
            "tests/test_state.py",
            target="pyservice/state.py",
            symbol="can_transition",
            scope=("pyservice/state.py", "tests/test_state.py"),
        ),
        _check(
            "C04",
            "state-symbol",
            K.REQUIRED_SYMBOL_PRESENCE,
            "pyservice/state.py",
            symbol="can_transition",
            scope=("pyservice/state.py",),
        ),
    ),
    "C05": (
        _check(
            "C05",
            "window-test-include",
            K.IMPORT_INCLUDE_RELATION,
            "cengine/tests/test_window.c",
            target="cengine/window.h",
            scope=("cengine/window.h", "cengine/tests/test_window.c"),
        ),
        _check(
            "C05",
            "window-test-call",
            K.CALLER_CALLEE_RELATION,
            "cengine/tests/test_window.c",
            target="cengine/window.c",
            symbol="window_accepts",
            scope=("cengine/window.c", "cengine/tests/test_window.c"),
        ),
        _check(
            "C05",
            "window-role",
            K.REQUIRED_COMPONENT_ROLE,
            "cengine/window.c",
            target="cengine/window.h",
            symbol="window_accepts",
            scope=("cengine/window.h", "cengine/window.c"),
        ),
    ),
    "C06": (
        _check(
            "C06",
            "status-test-include",
            K.IMPORT_INCLUDE_RELATION,
            "cengine/tests/test_status.c",
            target="cengine/status.h",
            scope=("cengine/status.h", "cengine/tests/test_status.c"),
        ),
        _check(
            "C06",
            "status-test-call",
            K.CALLER_CALLEE_RELATION,
            "cengine/tests/test_status.c",
            target="cengine/status.c",
            symbol="normalize_dependency_status",
            scope=("cengine/status.c", "cengine/tests/test_status.c"),
        ),
        _check(
            "C06",
            "status-role",
            K.REQUIRED_COMPONENT_ROLE,
            "cengine/status.c",
            target="cengine/status.h",
            symbol="normalize_dependency_status",
            scope=("cengine/status.h", "cengine/status.c"),
        ),
    ),
}


def _truth(task_id: str, prior: str) -> str:
    if task_id in {"H11", "C11"}:
        return "KNOWN_STRUCTURAL_MISS"
    if task_id == "H12":
        return "SEMANTIC_NEGATIVE_STRUCTURAL_MISS"
    if task_id in {"H07", "H09", "H10", "C09"}:
        return "SEMANTIC_NEGATIVE_STRUCTURAL_PASS"
    if task_id in {"H02", "C01"}:
        return "NO_STRUCTURAL_OBLIGATION"
    return "KNOWN_GOOD_STRUCTURAL_CONTROL"


A83_MODEL_CASES = tuple(
    ReadinessCase(
        case.task_id, case.task_id, _truth(case.task_id, case.truth), case.checks
    )
    for case in A83_CASES
)
FRESH_MODEL_CASES = tuple(
    ReadinessCase(task, task, _truth(task, ""), ADDITIONAL[task])
    for task in ("H02", "H06", "H10", "H12", "C01", "C02", "C03", "C04", "C05", "C06")
)
OPERATIONAL_CASES = (
    ReadinessCase("OP01", None, "OPERATIONAL_PARTIAL", (), False, "PARSER_FAILURE"),
    ReadinessCase(
        "OP02", None, "OPERATIONAL_PARTIAL", (), False, "UNSUPPORTED_LANGUAGE"
    ),
    ReadinessCase(
        "OP03", None, "OPERATIONAL_PARTIAL", (), False, "CHECKER_INTERNAL_ERROR"
    ),
    ReadinessCase(
        "OP04", None, "NO_STRUCTURAL_OBLIGATION", (), False, "NOT_APPLICABLE"
    ),
)
CASES = (*A83_MODEL_CASES, *FRESH_MODEL_CASES, *OPERATIONAL_CASES)
MODEL_TASK_IDS = tuple(case.task_id for case in CASES if case.model_call)


def tasks():  # type: ignore[no-untyped-def]
    definitions = {item.task_id: item for item in (*a78_tasks(), *realistic_tasks())}
    return tuple(definitions[task_id] for task_id in MODEL_TASK_IDS if task_id)


def corpus_identity() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "suite": SUITE,
                "version": VERSION,
                "cases": [
                    {
                        "case": case.case_id,
                        "task": case.task_id,
                        "truth": case.truth,
                        "checks": [check.identity for check in case.checks],
                        "model_call": case.model_call,
                        "scenario": case.control_scenario,
                    }
                    for case in CASES
                ],
                "representation": REPRESENTATION.value,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def validate_corpus() -> None:
    if not 24 <= len(CASES) <= 30:
        raise RuntimeError("A84 requires 24-30 cases")
    if len(set(MODEL_TASK_IDS)) != 24 or len(tasks()) != 24:
        raise RuntimeError("A84 requires all 24 unique model tasks")
    ids = [check.check_id for case in CASES for check in case.checks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("A84 check IDs are not unique")
    if {check.kind for case in CASES for check in case.checks} != set(CheckKind):
        raise RuntimeError("A84 must preserve all six accepted check kinds")
