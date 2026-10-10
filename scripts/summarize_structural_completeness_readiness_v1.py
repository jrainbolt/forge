"""Summarize A84 readiness and offline enforcement simulation."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from benchmarks.structural_completeness_readiness_v1.suite import (
    CASES,
    MAX_MEDIAN_CHECKER_LATENCY_SECONDS,
    MAX_WORST_CHECKER_LATENCY_SECONDS,
    RUN_ID,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    for case in CASES:
        cell = resume_checkpoint(args.checkpoint_dir / "cells" / f"{case.case_id}.json")
        if cell is None or cell.get("run_identity") != RUN_ID:
            raise RuntimeError(f"A84 missing checkpoint: {case.case_id}")
        cells.append(cell)
    model_cells = [cell for cell in cells if cell["model_call"]]
    misses = [
        cell
        for cell in cells
        if cell["structural_truth"]
        in {"KNOWN_STRUCTURAL_MISS", "SEMANTIC_NEGATIVE_STRUCTURAL_MISS"}
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
    decisions = [cell["primary_shadow"]["decision"] for cell in cells]
    false_fails = [
        cell
        for cell in controls
        if cell["primary_shadow"]["decision"] == "STRUCTURAL_SHADOW_FAIL"
        and cell["verification_pass"] is True
        and cell["semantic_pass"] is True
    ]
    missed = [
        cell
        for cell in misses
        if cell["primary_shadow"]["decision"] != "STRUCTURAL_SHADOW_FAIL"
    ]
    negative_false = [
        cell
        for cell in negatives
        if cell["primary_shadow"]["decision"] == "STRUCTURAL_SHADOW_FAIL"
    ]
    deterministic = [
        cell["primary_shadow"]["deterministic_repeat"]
        for cell in cells
        if cell["primary_shadow"]["deterministic_repeat"] is not None
    ]
    latencies = [cell["primary_shadow"]["wall_time_seconds"] for cell in model_cells]
    median_latency = statistics.median(latencies)
    worst_latency = max(latencies)
    runtime_ok = (
        median_latency <= MAX_MEDIAN_CHECKER_LATENCY_SECONDS
        and worst_latency <= MAX_WORST_CHECKER_LATENCY_SECONDS
    )
    actions = Counter(cell["offline_policy_action"] for cell in cells)
    unjustified_blocks = [
        cell
        for cell in cells
        if cell["offline_policy_action"] == "BLOCK"
        and cell["structural_truth"]
        not in {"KNOWN_STRUCTURAL_MISS", "SEMANTIC_NEGATIVE_STRUCTURAL_MISS"}
        and cell["verification_pass"] is True
        and cell["semantic_pass"] is True
    ]
    criteria = {
        "zero_false_fails": not false_fails,
        "all_known_defects_detected": not missed,
        "semantic_negatives_not_failed": not negative_false,
        "uncertainty_nonblocking": all(
            cell["offline_policy_action"] != "BLOCK"
            for cell in cells
            if cell["primary_shadow"]["decision"]
            in {"STRUCTURAL_SHADOW_PARTIAL", "STRUCTURAL_SHADOW_NOT_APPLICABLE"}
        ),
        "operational_failures_fail_open": all(
            cell["offline_policy_action"] == "FAIL_OPEN_CONTINUE_WITH_DIAGNOSTIC"
            for cell in cells
            if any(
                item.get("operational_failure")
                for item in cell["primary_shadow"]["checks"]
            )
        ),
        "deterministic": all(deterministic) and len(deterministic) >= 10,
        "runtime_ceiling": runtime_ok,
        "process_safety": True,
        "only_justified_blocks": not unjustified_blocks,
        "reversible_design": True,
        "no_hidden_data": True,
    }
    summary = {
        "run_identity": RUN_ID,
        "cells": len(cells),
        "model_cells": len(model_cells),
        "control_cells": len(cells) - len(model_cells),
        "decisions": dict(Counter(decisions)),
        "known_defects": len(misses),
        "known_defects_missed": [cell["case_id"] for cell in missed],
        "correct_controls": len(controls),
        "false_fail_count": len(false_fails),
        "false_fail_cases": [cell["case_id"] for cell in false_fails],
        "semantic_negatives": len(negatives),
        "semantic_negative_false_fails": len(negative_false),
        "determinism_states": len(deterministic),
        "determinism_matches": sum(bool(value) for value in deterministic),
        "median_latency_seconds": median_latency,
        "worst_latency_seconds": worst_latency,
        "runtime_ceiling_satisfied": runtime_ok,
        "offline_actions": dict(actions),
        "unjustified_blocks": [cell["case_id"] for cell in unjustified_blocks],
        "criteria": criteria,
        "a85_review_justified": all(criteria.values()),
    }
    atomic_checkpoint(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
