"""Frozen A88 authoritative corpus and threshold."""

from __future__ import annotations

import hashlib
import json

from benchmarks.structural_completeness_readiness_v1.suite import CASES as A84_CASES

RUN_ID = "a88-structural-obligation-derivation-expansion-v1"
CASES = tuple(case for case in A84_CASES if case.model_call)
MIN_WORKFLOW_COVERAGE = 0.60
MIN_TASK_CLASSES = 3


def corpus_identity() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "run": RUN_ID,
                "cases": [
                    {
                        "id": case.case_id,
                        "truth": case.truth,
                        "checks": [check.identity for check in case.checks],
                    }
                    for case in CASES
                ],
                "threshold": [MIN_WORKFLOW_COVERAGE, MIN_TASK_CLASSES],
                "mode": "ordinary_default_shadow_no_evaluator_injection",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def validate_corpus() -> None:
    expected = {
        *(f"H{index:02d}" for index in range(1, 13)),
        *(f"C{index:02d}" for index in range(1, 13)),
    }
    if len(CASES) != 24 or {case.case_id for case in CASES} != expected:
        raise RuntimeError("A88 requires the frozen 24-workflow corpus")
