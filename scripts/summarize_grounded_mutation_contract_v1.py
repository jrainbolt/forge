"""Summarize the authoritative paired A75 experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.grounded_mutation_contract_v1.runner import checkpoint_path
from benchmarks.grounded_mutation_contract_v1.suite import (
    CASES,
    RUN_ID,
    SUITE,
    Condition,
)
from benchmarks.grounded_mutation_planning_v1.paired_identity import (
    compare_paired_inputs,
)
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint


def _metrics(cells: tuple[dict[str, object], ...]) -> dict[str, object]:
    outcomes = [cell["primary"] for cell in cells]
    return {
        "cases": len(cells),
        "valid_contracts": sum(
            cell["contract"] is not None
            and cell["contract"]["classification"] == "VALID"
            for cell in cells
        ),
        "model_output": sum(item["model_output"] for item in outcomes),
        "schema_valid": sum(item["schema_valid"] for item in outcomes),
        "materializable": sum(item["mechanically_materializable"] for item in outcomes),
        "production_validatable": sum(
            item["production_validatable"] for item in outcomes
        ),
        "transaction_ready": sum(item["transaction_ready"] for item in outcomes),
        "applied": sum(item["transaction_applied"] for item in outcomes),
        "verification_pass": sum(item["verification_pass"] for item in outcomes),
        "semantic_pass": sum(item["semantic_pass"] for item in outcomes),
        "failures": dict(sorted(Counter(cell["failure"] for cell in cells).items())),
        "semantic_failures": dict(
            sorted(
                Counter(
                    item["semantic_failure"]
                    for item in outcomes
                    if item["semantic_failure"]
                ).items()
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        (case.case_id, condition): checkpoint_path(
            args.checkpoint_dir, case.case_id, condition
        )
        for case in CASES
        for condition in Condition
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"A75 incomplete: {missing}")
    cells = {key: json.loads(path.read_text()) for key, path in paths.items()}
    mismatches = []
    for case in CASES:
        direct = cells[(case.case_id, Condition.M0)]
        augmented = cells[(case.case_id, Condition.M1)]
        reasons = compare_paired_inputs(
            direct["paired_input_identity"], augmented["paired_input_identity"]
        )
        mismatches.extend(f"{case.case_id}:{reason.value}" for reason in reasons)
    if mismatches:
        raise RuntimeError(f"A75 paired-input mismatch: {mismatches}")
    grouped = {
        condition.value: tuple(cells[(case.case_id, condition)] for case in CASES)
        for condition in Condition
    }
    summary = {
        "suite": SUITE,
        "run_identity": RUN_ID,
        "authoritative_pairs": len(CASES),
        "identity_mismatches": 0,
        "cells": len(cells),
        "conditions": {name: _metrics(items) for name, items in grouped.items()},
        "by_task_class": {
            operation: {
                name: _metrics(
                    tuple(
                        item for item in items if item["operation_class"] == operation
                    )
                )
                for name, items in grouped.items()
            }
            for operation in sorted(
                {cell["operation_class"] for cell in cells.values()}
            )
        },
        "by_model": {
            profile: {
                name: _metrics(
                    tuple(item for item in items if item["profile"] == profile)
                )
                for name, items in grouped.items()
            }
            for profile in sorted({cell["profile"] for cell in cells.values()})
        },
        "repairs": {
            name: {
                "attempted": sum(item["repair"] is not None for item in items),
                "semantic_pass": sum(
                    item["repair"] is not None and item["repair"]["semantic_pass"]
                    for item in items
                ),
            }
            for name, items in grouped.items()
        },
        "contract_diagnosis": dict(
            sorted(
                Counter(
                    item["contract_diagnosis"]
                    for item in grouped[Condition.M1.value]
                    if item["contract_diagnosis"]
                ).items()
            )
        ),
    }
    m0 = summary["conditions"][Condition.M0.value]
    m1 = summary["conditions"][Condition.M1.value]
    delta = m1["semantic_pass"] - m0["semantic_pass"]
    improved_classes = sum(
        values[Condition.M1.value]["semantic_pass"]
        > values[Condition.M0.value]["semantic_pass"]
        for values in summary["by_task_class"].values()
    )
    improved_models = sum(
        values[Condition.M1.value]["semantic_pass"]
        > values[Condition.M0.value]["semantic_pass"]
        for values in summary["by_model"].values()
    )
    reliability_regression = (
        m1["schema_valid"] < m0["schema_valid"]
        or m1["materializable"] < m0["materializable"]
    )
    create_or_multi_improved = (
        summary["by_task_class"]["CREATE"][Condition.M1.value]["semantic_pass"]
        > summary["by_task_class"]["CREATE"][Condition.M0.value]["semantic_pass"]
        or summary["by_task_class"]["EDIT_MULTI"][Condition.M1.value]["semantic_pass"]
        > summary["by_task_class"]["EDIT_MULTI"][Condition.M0.value]["semantic_pass"]
    )
    summary["adoption_threshold"] = {
        "semantic_delta": delta,
        "improved_task_classes": improved_classes,
        "improved_model_profiles": improved_models,
        "reliability_regression": reliability_regression,
        "create_or_multi_improved": create_or_multi_improved,
        "future_review_justified": delta >= 3
        and improved_classes >= 2
        and improved_models >= 2
        and not reliability_regression
        and create_or_multi_improved,
    }
    if not standard_result_is_source_free(summary):
        raise RuntimeError("A75 summary is not source-free")
    atomic_checkpoint(args.checkpoint_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
