"""Run fresh A88 precision controls through autonomous SHADOW evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from benchmarks.obligation_derivation_precision_v1.suite import (
    CASES,
    RUN_ID,
    corpus_identity,
)
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.structural_completeness import (
    StructuralCompletenessMode,
    evaluate_structural_completeness,
)
from forge.structural_obligations import (
    derive_structural_obligations,
    evaluate_obligations,
)


def _key(item) -> tuple[object, ...]:  # type: ignore[no-untyped-def]
    return (item.kind, item.source_path, item.target_path, item.symbol)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(
        args.output_dir / "manifest.json",
        {
            "run_identity": RUN_ID,
            "corpus_identity": corpus_identity(),
            "cases": [case.case_id for case in CASES],
            "truth_frozen_before_execution": True,
            "mode": "shadow",
            "evaluator_obligation_injection": False,
        },
    )
    cells = []
    for case in CASES:
        with tempfile.TemporaryDirectory(prefix="forge-a88-precision-") as temporary:
            workspace = Path(temporary)
            for relative, content in case.initial_files:
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            first = derive_structural_obligations(
                case.task, workspace, mutation_authority=case.authority
            )
            second = derive_structural_obligations(
                case.task, workspace, mutation_authority=case.authority
            )
            deterministic = (
                first.identity == second.identity
                and tuple(item.kind for item in first.obligations)
                == tuple(item.kind for item in second.obligations)
                and tuple(item.provenance_identity for item in first.obligations)
                == tuple(item.provenance_identity for item in second.obligations)
            )
            for relative, content in case.materialized_files:
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            def evaluate_derived(
                root: Path,
                obligation_set=first,  # type: ignore[no-untyped-def]
            ):  # type: ignore[no-untyped-def]
                return evaluate_obligations(obligation_set, root)

            shadow = evaluate_structural_completeness(
                StructuralCompletenessMode.SHADOW,
                workspace,
                evaluate_derived,
            )
        derived = {_key(item) for item in first.obligations}
        truth = set(case.truth)
        payload = {
            "case_id": case.case_id,
            "operation_class": case.operation_class,
            "language": case.language,
            "role": case.role,
            "obligation_count": len(derived),
            "true_obligations": len(derived & truth),
            "missed_obligations": len(truth - derived),
            "spurious_obligations": len(derived - truth),
            "kinds": sorted(item[0].value for item in derived),
            "structural_decision": shadow.decision,
            "failed_check_ids": shadow.failed_check_ids,
            "deterministic": deterministic,
            "derivation_latency_seconds": first.derivation_latency_seconds,
            "check_latency_seconds": shadow.latency_seconds,
            "source_operations": first.source_operations,
            "verification_pass": shadow.decision in {"pass", "not_applicable"},
            "semantic_pass": True,
        }
        atomic_checkpoint(args.output_dir / f"{case.case_id}.json", payload)
        cells.append(payload)
    covered = [cell for cell in cells if cell["true_obligations"]]
    summary = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cells": len(cells),
        "true_obligations": sum(int(cell["true_obligations"]) for cell in cells),
        "missed_obligations": sum(int(cell["missed_obligations"]) for cell in cells),
        "spurious_obligations": sum(
            int(cell["spurious_obligations"]) for cell in cells
        ),
        "covered_workflows": len(covered),
        "coverage": len(covered) / len(cells),
        "coverage_by_class": dict(Counter(cell["operation_class"] for cell in covered)),
        "kind_distribution": dict(
            Counter(kind for cell in cells for kind in cell["kinds"])
        ),
        "structural_decisions": dict(
            Counter(cell["structural_decision"] for cell in cells)
        ),
        "established_false_fails": [
            cell["case_id"]
            for cell in cells
            if cell["structural_decision"] == "fail" and cell["semantic_pass"]
        ],
        "deterministic_repeats": sum(bool(cell["deterministic"]) for cell in cells),
        "median_derivation_latency_seconds": statistics.median(
            float(cell["derivation_latency_seconds"]) for cell in cells
        ),
        "worst_derivation_latency_seconds": max(
            float(cell["derivation_latency_seconds"]) for cell in cells
        ),
        "median_check_latency_seconds": statistics.median(
            float(cell["check_latency_seconds"]) for cell in cells
        ),
        "worst_check_latency_seconds": max(
            float(cell["check_latency_seconds"]) for cell in cells
        ),
        "source_operations": sum(int(cell["source_operations"]) for cell in cells),
        "subprocess_count": 0,
    }
    atomic_checkpoint(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
