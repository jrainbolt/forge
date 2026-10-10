"""Summarize A83 shadow accuracy, determinism, and overhead."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from benchmarks.structural_completeness_shadow_v1.suite import (
    MAX_MEDIAN_CHECKER_LATENCY_SECONDS,
    MAX_WORST_CHECKER_LATENCY_SECONDS,
    RUN_ID,
    TASK_IDS,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    for task in TASK_IDS:
        cell = resume_checkpoint(args.checkpoint_dir / "cells" / f"{task}.json")
        if cell is None or cell.get("run_identity") != RUN_ID:
            raise RuntimeError(f"A83 missing or invalid checkpoint: {task}")
        cells.append(cell)
    decisions = [cell["primary_shadow"]["decision"] for cell in cells]
    known_misses = [
        cell for cell in cells if cell["structural_truth"] == "KNOWN_STRUCTURAL_MISS"
    ]
    controls = [
        cell
        for cell in cells
        if cell["structural_truth"] == "KNOWN_GOOD_STRUCTURAL_CONTROL"
    ]
    negatives = [
        cell
        for cell in cells
        if cell["structural_truth"] == "SEMANTIC_NEGATIVE_STRUCTURAL_PASS"
    ]
    misses_detected = sum(
        cell["primary_shadow"]["decision"] == "STRUCTURAL_SHADOW_FAIL"
        for cell in known_misses
    )
    false_rejections = sum(
        cell["primary_shadow"]["decision"] != "STRUCTURAL_SHADOW_PASS"
        for cell in controls
    )
    negative_passes = sum(
        cell["primary_shadow"]["decision"] == "STRUCTURAL_SHADOW_PASS"
        and not cell["semantic_pass"]
        for cell in negatives
    )
    deterministic = [
        cell["primary_shadow"]["deterministic_repeat"]
        for cell in cells
        if cell["primary_shadow"]["deterministic_repeat"] is not None
    ]
    latencies = [cell["primary_shadow"]["wall_time_seconds"] for cell in cells]
    median_latency = statistics.median(latencies)
    worst_latency = max(latencies)
    runtime_acceptable = (
        median_latency <= MAX_MEDIAN_CHECKER_LATENCY_SECONDS
        and worst_latency <= MAX_WORST_CHECKER_LATENCY_SECONDS
    )
    shadow_only = sum(
        decision == "STRUCTURAL_SHADOW_FAIL" and cell["verification_pass"]
        for decision, cell in zip(decisions, cells, strict=True)
    )
    verification_only = sum(
        decision == "STRUCTURAL_SHADOW_PASS" and not cell["verification_pass"]
        for decision, cell in zip(decisions, cells, strict=True)
    )
    overlap = sum(
        decision == "STRUCTURAL_SHADOW_FAIL" and not cell["verification_pass"]
        for decision, cell in zip(decisions, cells, strict=True)
    )
    all_applied = all(cell["transaction_applied"] for cell in cells)
    blocking_review = (
        false_rejections == 0
        and misses_detected == len(known_misses)
        and negative_passes == len(negatives)
        and all(deterministic)
        and runtime_acceptable
        and all_applied
    )
    summary = {
        "run_identity": RUN_ID,
        "cells": len(cells),
        "shadow_decisions": dict(Counter(decisions)),
        "verification_pass": sum(cell["verification_pass"] for cell in cells),
        "semantic_pass": sum(cell["semantic_pass"] for cell in cells),
        "known_structural_misses": len(known_misses),
        "known_structural_misses_detected": misses_detected,
        "correct_structural_controls": len(controls),
        "correct_control_false_rejections": false_rejections,
        "semantic_negative_controls": len(negatives),
        "semantic_negative_structural_passes": negative_passes,
        "shadow_only_catches": shadow_only,
        "verification_only_catches": verification_only,
        "shadow_verification_overlap": overlap,
        "determinism_repeats": len(deterministic),
        "determinism_matches": sum(bool(value) for value in deterministic),
        "median_checker_latency_seconds": median_latency,
        "worst_checker_latency_seconds": worst_latency,
        "compiler_subprocesses": sum(
            cell["primary_shadow"]["compiler_subprocesses"] for cell in cells
        ),
        "runtime_acceptable": runtime_acceptable,
        "all_transactions_applied": all_applied,
        "repair_attempted": sum(cell["repair_shadow"] is not None for cell in cells),
        "repair_change": dict(Counter(cell["repair_change"] for cell in cells)),
        "future_blocking_gate_review_justified": blocking_review,
    }
    atomic_checkpoint(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
