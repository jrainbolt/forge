"""Validate and summarize source-free A81 checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from benchmarks.semantic_obligation_completeness_v1.suite import RUN_ID, TASK_IDS
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
            raise RuntimeError(f"A81 missing or invalid checkpoint: {task}")
        cells.append(cell)
    obligations = [item for cell in cells for item in cell["primary_obligations"]]
    missed = [
        item for item in obligations if item["satisfaction"] != "OBLIGATION_SATISFIED"
    ]
    uncovered_missed = [
        item
        for item in missed
        if item["verification_coverage"] == "VERIFICATION_DOES_NOT_COVER_OBLIGATION"
    ]
    summary = {
        "run_identity": RUN_ID,
        "cells": len(cells),
        "tasks": list(TASK_IDS),
        "semantic_pass": sum(cell["full_semantic_pass"] for cell in cells),
        "verification_pass": sum(cell["verification_pass"] for cell in cells),
        "obligations": len(obligations),
        "obligations_satisfied": len(obligations) - len(missed),
        "obligations_missed": len(missed),
        "missed_obligation_ids": [item["obligation_id"] for item in missed],
        "missed_by_origin": dict(Counter(item["origin"] for item in missed)),
        "missed_by_component": dict(Counter(item["component"] for item in missed)),
        "uncovered_missed_obligations": len(uncovered_missed),
        "classifications": dict(Counter(cell["semantic_taxonomy"] for cell in cells)),
        "completeness": dict(Counter(cell["primary_completeness"] for cell in cells)),
        "relationship": dict(
            Counter(cell["relationship_classification"] for cell in cells)
        ),
        "repair": dict(Counter(cell["repair_comparison"] for cell in cells)),
        "production_state_absent": sum(
            item["production_state"] != "OBLIGATION_PRESENT_IN_PRODUCTION_STATE"
            for item in obligations
        ),
        "check_unavailable": sum(
            item["satisfaction"] == "OBLIGATION_CHECK_UNAVAILABLE"
            for item in obligations
        ),
        "completeness_gate_feasibility": "PARTIALLY_FEASIBLE",
    }
    atomic_checkpoint(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
