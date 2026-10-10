"""Frozen ordinary-workflow corpus for A86."""

from __future__ import annotations

import hashlib
import json

from benchmarks.structural_completeness_readiness_v1.suite import CASES as A84_CASES

RUN_ID = "a86-structural-enforcement-default-review-v1"
CASES = tuple(case for case in A84_CASES if case.model_call)


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
                "mode": "implicit_default_shadow",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def validate_corpus() -> None:
    if len(CASES) != 24 or {case.case_id for case in CASES} != {
        *(f"H{index:02d}" for index in range(1, 13)),
        *(f"C{index:02d}" for index in range(1, 13)),
    }:
        raise RuntimeError("A86 requires the 24 frozen ordinary workflows")
