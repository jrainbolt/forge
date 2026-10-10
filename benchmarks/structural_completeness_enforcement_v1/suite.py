"""Frozen A85 enforcement corpus."""

from __future__ import annotations

import hashlib
import json

from benchmarks.structural_completeness_readiness_v1.suite import CASES as A84_CASES

RUN_ID = "a85-structural-completeness-enforcement-v1"
MODEL_CASE_IDS = (
    "H11",
    "C11",
    "H12",
    "C05",
    "C06",
    "H01",
    "C07",
    "H07",
    "H09",
    "C09",
)
CONTROL_CASE_IDS = ("PARTIAL", "NOT_APPLICABLE")
_A84_BY_ID = {case.case_id: case for case in A84_CASES}
CASES = tuple(_A84_BY_ID[case_id] for case_id in MODEL_CASE_IDS)


def corpus_identity() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "run": RUN_ID,
                "model_cases": [
                    {
                        "id": case.case_id,
                        "truth": case.truth,
                        "checks": [check.identity for check in case.checks],
                    }
                    for case in CASES
                ],
                "controls": CONTROL_CASE_IDS,
                "mode": "enforce",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def validate_corpus() -> None:
    if tuple(case.case_id for case in CASES) != MODEL_CASE_IDS:
        raise RuntimeError("A85 frozen model cases changed")
    if len(CASES) + len(CONTROL_CASE_IDS) != 12:
        raise RuntimeError("A85 requires 12 frozen cases")
