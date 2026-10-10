"""Freeze A84 readiness cases, policy, and ceilings before execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.production_detectable_completeness_v1.checker import (
    ALLOWED_EXECUTABLES,
)
from benchmarks.structural_completeness_readiness_v1.suite import (
    CASES,
    COMPILER_TIMEOUT_SECONDS,
    MAX_MEDIAN_CHECKER_LATENCY_SECONDS,
    MAX_WORST_CHECKER_LATENCY_SECONDS,
    RUN_ID,
    corpus_identity,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_corpus()
    atomic_checkpoint(
        args.output,
        {
            "run_identity": RUN_ID,
            "corpus_identity": corpus_identity(),
            "cases": [case.case_id for case in CASES],
            "truth": {case.case_id: case.truth for case in CASES},
            "check_identities": {
                case.case_id: [check.identity for check in case.checks]
                for case in CASES
            },
            "policy": {
                "STRUCTURAL_SHADOW_PASS": "ALLOW",
                "STRUCTURAL_SHADOW_FAIL": "BLOCK",
                "STRUCTURAL_SHADOW_PARTIAL": "CONTINUE_WITH_DIAGNOSTIC",
                "STRUCTURAL_SHADOW_NOT_APPLICABLE": "ALLOW_NOT_APPLICABLE",
                "CHECKER_INTERNAL_ERROR": "FAIL_OPEN_CONTINUE_WITH_DIAGNOSTIC",
            },
            "checks_frozen_before_execution": True,
            "blocking_active": False,
            "production_visible_only": True,
            "hidden_oracle_data": False,
            "allowed_executables": sorted(ALLOWED_EXECUTABLES),
            "compiler_timeout_seconds": COMPILER_TIMEOUT_SECONDS,
            "max_median_checker_latency_seconds": MAX_MEDIAN_CHECKER_LATENCY_SECONDS,
            "max_worst_checker_latency_seconds": MAX_WORST_CHECKER_LATENCY_SECONDS,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
