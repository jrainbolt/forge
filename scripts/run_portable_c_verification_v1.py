"""Run the frozen deterministic A92 portable C validation corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from benchmarks.portable_c_verification_v1.suite import (
    CASES,
    RUN_ID,
    corpus_identity,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint
from forge.executable_verification import (
    CByteArrayInvocation,
    CheckProvenance,
    CInvocation,
    ExecutableVerificationPlan,
    derive_c_check,
    source_free_record,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_corpus()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="forge-a92-") as temporary:
        root = Path(temporary)
        for case in CASES:
            workspace = root / case.case_id
            workspace.mkdir()
            (workspace / "api.h").write_text(case.header)
            (workspace / "candidate.c").write_text(case.source)
            invocation = _invocation(case.invocation)
            check, failure = derive_c_check(
                workspace,
                check_id=case.case_id,
                kind=case.kind,
                provenance=CheckProvenance.TRUSTED_INTERFACE_CONTRACT,
                component_identity="api.public",
                invocation=invocation,
            )
            if check is None:
                outcome = failure.value if failure is not None else "UNKNOWN"
                record = None
            else:
                executed = ExecutableVerificationPlan(case.case_id, (check,)).execute(
                    workspace, 1
                )
                record = source_free_record(executed)
                outcome = executed.results[0].outcome
            cell = {
                "case_id": case.case_id,
                "kind": case.kind.value,
                "semantic_correct": case.semantic_correct,
                "expected_outcome": case.expected_outcome,
                "outcome": outcome,
                "matched": outcome == case.expected_outcome,
                "record": record,
            }
            atomic_checkpoint(args.output_dir / f"{case.case_id}.json", cell)
            cells.append(cell)
    latency_fields = (
        "derivation_latency_seconds",
        "construction_latency_seconds",
        "compile_latency_seconds",
        "link_latency_seconds",
        "execution_latency_seconds",
        "duration_seconds",
    )
    latencies = {
        field: [
            float(result[field])
            for cell in cells
            if isinstance(cell["record"], dict)
            for result in cell["record"]["results"]  # type: ignore[index,union-attr]
        ]
        for field in latency_fields
    }
    results = [str(cell["outcome"]) for cell in cells]
    summary = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "frozen_before_execution": True,
        "cells": len(cells),
        "all_expected": all(bool(cell["matched"]) for cell in cells),
        "outcomes": dict(Counter(results)),
        "false_check_failures": [
            cell["case_id"]
            for cell in cells
            if cell["semantic_correct"] is True and cell["outcome"] == "CHECK_FAIL"
        ],
        "known_wrong_caught": [
            cell["case_id"]
            for cell in cells
            if cell["semantic_correct"] is False and cell["outcome"] == "CHECK_FAIL"
        ],
        "latency_seconds": {
            field: {
                "median": statistics.median(values),
                "p95": _percentile(values, 0.95),
                "worst": max(values, default=0.0),
            }
            for field, values in latencies.items()
        },
        "source_free": True,
    }
    atomic_checkpoint(
        args.output_dir / "manifest.json",
        {
            "run_identity": RUN_ID,
            "corpus_identity": corpus_identity(),
            "case_ids": [case.case_id for case in CASES],
        },
    )
    atomic_checkpoint(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_expected"] else 1


def _invocation(shape: str) -> CInvocation | CByteArrayInvocation:
    if shape == "bytes":
        return CByteArrayInvocation(
            "api.h", ("candidate.c",), "checksum_bytes", (1, 2, 3), 1026
        )
    if shape == "invariant":
        return CInvocation("api.h", ("candidate.c",), "identity", (17,), 17)
    if shape == "unsupported":
        return CInvocation("api.h", ("candidate.c",), "inspect", (1,), 1)
    arguments = (0, 0) if shape == "boundary" else (2, 3)
    expected = 0 if shape == "boundary" else 5
    return CInvocation("api.h", ("candidate.c",), "add", arguments, expected)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * percentile))]


if __name__ == "__main__":
    raise SystemExit(main())
