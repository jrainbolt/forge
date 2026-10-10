"""Summarize completed source-free A82 structural checks."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from benchmarks.production_detectable_completeness_v1.suite import RUN_ID, TASK_IDS
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
            raise RuntimeError(f"A82 missing or invalid checkpoint: {task}")
        cells.append(cell)
    checks = [item for cell in cells for item in cell["primary_checks"]]
    incomplete = [item for item in checks if item["status"] == "INCOMPLETE"]
    correct_controls = [
        cell
        for cell in cells
        if cell["corpus_role"] == "KNOWN_GOOD_CONTROL" and cell["semantic_pass"]
    ]
    false_rejections = sum(
        cell["primary_structural_classification"] != "STRUCTURALLY_COMPLETE"
        for cell in correct_controls
    )
    semantic_fail_complete = sum(
        not cell["semantic_pass"]
        and cell["primary_structural_classification"] == "STRUCTURALLY_COMPLETE"
        for cell in cells
    )
    verification_miss_caught = sum(
        cell["verification_pass"]
        and not cell["semantic_pass"]
        and cell["primary_structural_classification"] == "STRUCTURAL_OBLIGATION_MISSING"
        for cell in cells
    )
    negative_controls_ok = all(
        cell["primary_structural_classification"] == "STRUCTURALLY_COMPLETE"
        and not cell["semantic_pass"]
        for cell in cells
        if cell["corpus_role"] == "NEGATIVE_SEMANTIC_CONTROL"
    )
    threshold = (
        len(incomplete) >= 2
        and false_rejections == 0
        and negative_controls_ok
        and all(item["status"] in {"COMPLETE", "INCOMPLETE"} for item in checks)
    )
    summary = {
        "run_identity": RUN_ID,
        "cells": len(cells),
        "checks": len(checks),
        "check_status": dict(Counter(item["status"] for item in checks)),
        "structural_classification": dict(
            Counter(cell["primary_structural_classification"] for cell in cells)
        ),
        "verification_pass": sum(cell["verification_pass"] for cell in cells),
        "semantic_pass": sum(cell["semantic_pass"] for cell in cells),
        "incomplete_checks": len(incomplete),
        "incomplete_check_ids": [item["check_id"] for item in incomplete],
        "verification_misses_structural_checker_catches": verification_miss_caught,
        "structurally_complete_semantic_failures": semantic_fail_complete,
        "correct_control_false_rejections": false_rejections,
        "negative_semantic_controls_correct": negative_controls_ok,
        "repair_attempted": sum(cell["repair_checks"] is not None for cell in cells),
        "production_gate_review_justified": threshold,
    }
    atomic_checkpoint(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
