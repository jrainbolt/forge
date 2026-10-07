"""Summarize the complete source-free A73 diagnostic corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from benchmarks.verification_attribution_v1.suite import CASES, SUITE
from forge.evaluation.mutation_ready import atomic_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = tuple(
        args.checkpoint_dir / "cases" / f"{case.case_id}.json" for case in CASES
    )
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"A73 incomplete: {missing}")
    cases = tuple(json.loads(path.read_text()) for path in paths)
    proposals = tuple(proposal for case in cases for proposal in case["proposals"])
    summary = {
        "suite": SUITE,
        "cases": len(cases),
        "applied_mutations": len(proposals),
        "quadrants": dict(
            sorted(Counter(item["quadrant"] for item in proposals).items())
        ),
        "verification_failures": dict(
            sorted(
                Counter(
                    item["verification_failure"]
                    for item in proposals
                    if item["verification_failure"]
                ).items()
            )
        ),
        "semantic_failures": dict(
            sorted(
                Counter(
                    item["semantic_failure"]
                    for item in proposals
                    if item["semantic_failure"]
                ).items()
            )
        ),
        "by_model": {
            profile: dict(
                sorted(
                    Counter(
                        proposal["quadrant"]
                        for case in cases
                        if case["profile"] == profile
                        for proposal in case["proposals"]
                    ).items()
                )
            )
            for profile in sorted({case["profile"] for case in cases})
        },
        "by_task_class": {
            operation: dict(
                sorted(
                    Counter(
                        proposal["quadrant"]
                        for case in cases
                        if case["operation_class"] == operation
                        for proposal in case["proposals"]
                    ).items()
                )
            )
            for operation in sorted({case["operation_class"] for case in cases})
        },
        "identity_mismatches": sum(case["identity_mismatch_count"] for case in cases),
        "baseline_valid": sum(bool(case["baseline_valid"]) for case in cases),
    }
    if not standard_result_is_source_free(summary):
        raise RuntimeError("A73 summary is not source-free")
    atomic_checkpoint(args.checkpoint_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
