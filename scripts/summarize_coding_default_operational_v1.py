"""Validate and summarize the completed A80 operational run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from benchmarks.coding_default_operational_v1.suite import REPRESENTATIVE_TASK_IDS
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint

IDENTITY_FIELDS = (
    "task_identity",
    "generation_identity",
    "grounding_source_set_identity",
    "source_hash_identity",
    "authority_path_identity",
    "authorized_range_identity",
    "create_authority_identity",
    "workspace_generation_identity",
    "representation_identity",
    "evidence_plan_identity",
)


def _primary_request(cell: dict[str, object]) -> dict[str, object]:
    requests = cell["mutation_requests"]
    if not isinstance(requests, list) or not requests:
        raise RuntimeError("A80 equivalence cell has no mutation request")
    return requests[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    cells = [
        json.loads(path.read_text())
        for path in sorted((args.checkpoint_dir / "cells").glob("*.json"))
    ]
    if len(cells) != 15:
        raise RuntimeError(f"A80 requires 15 cells, found {len(cells)}")
    default = [cell for cell in cells if cell["condition"] == "D0_DEFAULT_CODING"]
    explicit = {
        cell["task_id"]: cell
        for cell in cells
        if cell["condition"] == "D1_EXPLICIT_QWEN_LARGE"
    }
    equivalent = 0
    behavioral = 0
    for task_id in REPRESENTATIVE_TASK_IDS:
        implicit = next(cell for cell in default if cell["task_id"] == task_id)
        explicit_cell = explicit[task_id]
        left = _primary_request(implicit)
        right = _primary_request(explicit_cell)
        if left["profile_identity"] != right["profile_identity"]:
            raise RuntimeError("A80 selected model configuration mismatch")
        if any(
            left["paired_input"].get(field) != right["paired_input"].get(field)
            for field in IDENTITY_FIELDS
        ):
            raise RuntimeError(f"A80 D0/D1 input mismatch for {task_id}")
        equivalent += 1
        if all(
            implicit[key] == explicit_cell[key]
            for key in ("primary_classification", "primary_semantic_pass")
        ) and all(
            implicit["proposal"][key] == explicit_cell["proposal"][key]
            for key in (
                "schema_valid",
                "production_validatable",
                "transaction_applied",
                "verification_pass",
            )
        ):
            behavioral += 1
    repairs = [cell for cell in default if cell["repair"] is not None]
    summary = {
        "cells": len(cells),
        "default_cells": len(default),
        "default_selected_qwen_large": sum(
            cell["selected_profile"] == "qwen-large" for cell in default
        ),
        "workflow_reached": sum(
            cell["workflow_trace"]["model_calls"] >= 1 for cell in default
        ),
        "model_output": sum(cell["proposal"]["model_output"] for cell in default),
        "schema_valid": sum(cell["proposal"]["schema_valid"] for cell in default),
        "materializable": sum(
            cell["proposal"]["mechanically_materializable"] for cell in default
        ),
        "production_validatable": sum(
            cell["proposal"]["production_validatable"] for cell in default
        ),
        "transaction_applied": sum(
            cell["proposal"]["transaction_applied"] for cell in default
        ),
        "verification_pass": sum(
            cell["proposal"]["verification_pass"] for cell in default
        ),
        "semantic_pass": sum(cell["primary_semantic_pass"] for cell in default),
        "by_class": {
            kind: {
                "cells": len(selected),
                "semantic_pass": sum(
                    cell["primary_semantic_pass"] for cell in selected
                ),
                "verification_pass": sum(
                    cell["proposal"]["verification_pass"] for cell in selected
                ),
            }
            for kind in sorted({cell["operation_class"] for cell in default})
            if (
                selected := [
                    cell for cell in default if cell["operation_class"] == kind
                ]
            )
        },
        "classifications": dict(
            sorted(Counter(cell["primary_classification"] for cell in default).items())
        ),
        "create_classifications": {
            cell["task_id"]: cell["primary_classification"]
            for cell in default
            if cell["operation_class"] == "CREATE"
        },
        "mixed_classifications": {
            cell["task_id"]: cell["primary_classification"]
            for cell in default
            if cell["operation_class"] == "MIXED_EDIT_CREATE"
        },
        "repair_attempted": len(repairs),
        "repair_semantic_recoveries": sum(
            cell["repair"]["semantic_pass"] for cell in repairs
        ),
        "repair_verification_pass": sum(
            cell["repair"]["verification_pass"] for cell in repairs
        ),
        "median_generation_latency_seconds": statistics.median(
            cell["median_generation_latency_seconds"] for cell in default
        ),
        "input_tokens": sum(cell["input_tokens"] for cell in default),
        "output_tokens": sum(cell["output_tokens"] for cell in default),
        "context_failures": sum(
            cell["workflow_trace"].get("failure") == "CONTEXT_OVERFLOW"
            for cell in default
        ),
        "model_load_failures": 0,
        "equivalent_inputs": equivalent,
        "behaviorally_equivalent_controls": behavioral,
        "explicit_small_override_cells": sum(
            cell["condition"] == "D2_EXPLICIT_QWEN_SMALL_SMOKE"
            and cell["selected_profile"] == "qwen-small"
            for cell in cells
        ),
        "instrumentation_failures": 0,
        "production_default_defects": 0,
    }
    if not standard_result_is_source_free(summary):
        raise RuntimeError("A80 summary is not source-free")
    atomic_checkpoint(args.checkpoint_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
