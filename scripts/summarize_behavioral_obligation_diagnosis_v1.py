"""Summarize the complete A76 obligation diagnosis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.behavioral_obligation_diagnosis_v1.runner import checkpoint_path
from benchmarks.behavioral_obligation_diagnosis_v1.suite import (
    RUN_ID,
    SUITE,
    Condition,
    cases,
    obligations,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint


def _classification(
    full: dict[str, object],
    obligation_cells: tuple[dict[str, object], ...],
    isolatable: bool,
) -> str:
    if full["full_task_semantic_pass"]:
        return "FULL_TASK_PASS"
    if not isolatable:
        return "OBLIGATION_NOT_ISOLATABLE"
    passes = [bool(cell["obligation_oracle_pass"]) for cell in obligation_cells]
    if not passes:
        return "UNKNOWN"
    if all(passes):
        return "COORDINATION_FAILURE"
    if any(passes):
        return "MIXED_CAPABILITY_FAILURE"
    return "ATOMIC_CAPABILITY_FAILURE"


def _funnel(cells: tuple[dict[str, object], ...]) -> dict[str, int]:
    proposals = [cell["proposal"] for cell in cells]
    return {
        "cells": len(cells),
        "model_output": sum(item["model_output"] for item in proposals),
        "schema_valid": sum(item["schema_valid"] for item in proposals),
        "materializable": sum(
            item["mechanically_materializable"] for item in proposals
        ),
        "production_validatable": sum(
            item["production_validatable"] for item in proposals
        ),
        "transaction_applied": sum(item["transaction_applied"] for item in proposals),
        "obligation_oracle_pass": sum(
            cell["obligation_oracle_pass"] is True for cell in cells
        ),
        "full_task_semantic_pass": sum(
            cell["full_task_semantic_pass"] for cell in cells
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    loaded = {}
    missing = []
    for case in cases():
        cell_ids = [f"{case.case_id}-{Condition.F0.name}"]
        cell_ids.extend(
            f"{case.case_id}-{Condition.F1.name}-{item.goal_id}"
            for item in obligations(case)
            if item.isolatable
        )
        for cell_id in cell_ids:
            path = checkpoint_path(args.checkpoint_dir, cell_id)
            if not path.is_file():
                missing.append(path.name)
            else:
                loaded[cell_id] = json.loads(path.read_text())
    if missing:
        raise RuntimeError(f"A76 incomplete: {missing}")
    diagnoses = []
    for case in cases():
        obligation_defs = obligations(case)
        obligation_cells = tuple(
            loaded[f"{case.case_id}-{Condition.F1.name}-{item.goal_id}"]
            for item in obligation_defs
            if item.isolatable
        )
        classification = _classification(
            loaded[f"{case.case_id}-{Condition.F0.name}"],
            obligation_cells,
            bool(obligation_defs) and all(item.isolatable for item in obligation_defs),
        )
        diagnoses.append(
            {
                "case_id": case.case_id,
                "profile": case.profile,
                "operation_class": case.operation_class,
                "classification": classification,
                "required_obligations": len(obligation_defs),
                "isolated_obligations": len(obligation_cells),
                "isolated_passes": sum(
                    cell["obligation_oracle_pass"] is True for cell in obligation_cells
                ),
            }
        )
    f0 = tuple(
        cell for cell in loaded.values() if cell["condition"] == Condition.F0.value
    )
    f1 = tuple(
        cell for cell in loaded.values() if cell["condition"] == Condition.F1.value
    )
    summary = {
        "suite": SUITE,
        "run_identity": RUN_ID,
        "cases": len(cases()),
        "cells": len(loaded),
        "f0": _funnel(f0),
        "f1": _funnel(f1),
        "classifications": dict(
            sorted(Counter(item["classification"] for item in diagnoses).items())
        ),
        "by_task_class": {
            operation: dict(
                sorted(
                    Counter(
                        item["classification"]
                        for item in diagnoses
                        if item["operation_class"] == operation
                    ).items()
                )
            )
            for operation in sorted({item["operation_class"] for item in diagnoses})
        },
        "by_model": {
            profile: {
                "obligations": sum(
                    item["isolated_obligations"]
                    for item in diagnoses
                    if item["profile"] == profile
                ),
                "passes": sum(
                    item["isolated_passes"]
                    for item in diagnoses
                    if item["profile"] == profile
                ),
                "classifications": dict(
                    sorted(
                        Counter(
                            item["classification"]
                            for item in diagnoses
                            if item["profile"] == profile
                        ).items()
                    )
                ),
            }
            for profile in sorted({item["profile"] for item in diagnoses})
        },
        "diagnoses": diagnoses,
    }
    if not standard_result_is_source_free(summary):
        raise RuntimeError("A76 summary is not source-free")
    atomic_checkpoint(args.checkpoint_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
