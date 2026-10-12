"""Frozen A91 production-visible executable-check corpus."""

from __future__ import annotations

import hashlib
import json

from benchmarks.default_candidate_confirmation_v1.suite import tasks as a78_tasks
from forge.executable_verification import (
    CByteArrayInvocation,
    CheckProvenance,
    CInvocation,
    ExecutableCheck,
    ExecutableCheckKind,
    ExecutableVerificationPlan,
    PythonInvocation,
    PythonMethodCall,
    PythonObjectScenario,
    derive_python_check,
    derive_python_object_check,
    identity,
)

RUN_ID = "a91-bounded-executable-verification-v1"
CELLS = tuple((task, 42) for task in (f"H{i:02d}" for i in range(1, 13))) + (
    ("H07", 43),
    ("H09", 43),
    ("H10", 43),
    ("H11", 43),
)
NON_DERIVABLE = frozenset({"H02", "H05", "H06", "H08", "H12"})


def tasks():  # type: ignore[no-untyped-def]
    values = {item.task_id: item for item in a78_tasks()}
    return values


def _object(task_id: str, module: str) -> ExecutableVerificationPlan:
    scenario = PythonObjectScenario(
        module,
        "Counter",
        calls=(
            PythonMethodCall("__getattribute__", ("value",), expected=0),
            PythonMethodCall("increment", (0,), expected=None),
            PythonMethodCall("increment", (), expected=None),
            PythonMethodCall("increment", (4,), expected=None),
            PythonMethodCall("__getattribute__", ("value",), expected=5),
            PythonMethodCall("increment", (-1,), expected_exception="ValueError"),
        ),
    )
    check, failure = derive_python_object_check(
        check_id=f"{task_id}-counter-contract",
        kind=ExecutableCheckKind.INVARIANT_ASSERTION,
        provenance=CheckProvenance.EXPLICIT_INVARIANT,
        component_identity=f"{module}.Counter",
        scenario=scenario,
    )
    assert failure is None and check is not None
    return ExecutableVerificationPlan(f"a91-{task_id.lower()}", (check,))


def _config(task_id: str, include_header: bool) -> ExecutableVerificationPlan:
    cases = (
        PythonInvocation(
            "pyservice.config",
            "parse_enabled",
            ({"X": " YES "}, "X"),
            (("default", False),),
            True,
        ),
        PythonInvocation(
            "pyservice.config",
            "parse_enabled",
            ({"X": "off"}, "X"),
            (("default", True),),
            False,
        ),
        PythonInvocation(
            "pyservice.config",
            "parse_timeout",
            ({"T": "0"}, "T", 1),
            expected_exception="ValueError",
        ),
    )
    if include_header:
        cases += (
            PythonInvocation(
                "pyservice.headers",
                "normalize_header_name",
                (" X-Trace ",),
                expected="x-trace",
            ),
        )
    check, failure = derive_python_check(
        check_id=f"{task_id}-configuration-contract",
        kind=ExecutableCheckKind.BOUNDARY_ASSERTION,
        provenance=CheckProvenance.EXPLICIT_BOUNDARY_REQUIREMENT,
        component_identity="pyservice.config",
        cases=cases,
    )
    assert failure is None and check is not None
    return ExecutableVerificationPlan(f"a91-{task_id.lower()}", (check,))


def plan_for(task_id: str) -> ExecutableVerificationPlan | None:
    if task_id in {"H01", "H07", "H10"}:
        return _object(task_id, "pyservice.metrics")
    if task_id in {"H04", "H09"}:
        return _config(task_id, task_id == "H04")
    if task_id == "H03":
        case = CInvocation(
            "cengine/clock.h", ("cengine/clock.c",), "clock_elapsed", (8, 3), 0
        )
        check = ExecutableCheck(
            "H03-clock-boundary",
            ExecutableCheckKind.BOUNDARY_ASSERTION,
            CheckProvenance.EXPLICIT_BOUNDARY_REQUIREMENT,
            "cengine.clock_elapsed",
            identity({"args": case.arguments}),
            identity({"expected": case.expected}),
            5,
            c_case=case,
        )
        return ExecutableVerificationPlan("a91-h03", (check,))
    if task_id == "H11":
        case = CByteArrayInvocation(
            "cengine/checksum.h",
            ("cengine/checksum.c",),
            "checksum_bytes",
            (1, 2, 3),
            1152,
        )
        check = ExecutableCheck(
            "H11-checksum-contract",
            ExecutableCheckKind.COMPILE_OR_USE_PROBE,
            CheckProvenance.PUBLIC_API_CONTRACT,
            "cengine.checksum_bytes",
            identity({"values": case.values}),
            identity({"expected": case.expected}),
            5,
            c_byte_array_case=case,
        )
        return ExecutableVerificationPlan("a91-h11", (check,))
    return None


def corpus_identity() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "run": RUN_ID,
                "cells": CELLS,
                "plans": {
                    task: [check.check_id for check in plan.checks]
                    if (plan := plan_for(task)) is not None
                    else "CHECK_NOT_DERIVABLE"
                    for task in sorted(tasks())
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def validate_corpus() -> None:
    if len(CELLS) != 16 or len(set(CELLS)) != 16:
        raise RuntimeError("A91 requires sixteen fresh workflow cells")
    if set(task for task, _seed in CELLS) != set(tasks()):
        raise RuntimeError("A91 must cover H01-H12")
