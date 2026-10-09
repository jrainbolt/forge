"""Summarize the complete A77 capability matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from benchmarks.atomic_coding_capability_v1.runner import checkpoint_path
from benchmarks.atomic_coding_capability_v1.suite import PROFILES, RUN_ID, SUITE, tasks
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint

BASELINES = ("qwen-small", "qwen-large", "codestral-22b")
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


def _funnel(cells: list[dict[str, object]]) -> dict[str, int]:
    proposals = [cell["proposal"] for cell in cells]
    return {
        "cells": len(cells),
        "model_output": sum(bool(item["model_output"]) for item in proposals),
        "schema_valid": sum(bool(item["schema_valid"]) for item in proposals),
        "mechanically_materializable": sum(
            bool(item["mechanically_materializable"]) for item in proposals
        ),
        "production_validatable": sum(
            bool(item["production_validatable"]) for item in proposals
        ),
        "transaction_applied": sum(
            bool(item["transaction_applied"]) for item in proposals
        ),
        "verification_pass": sum(bool(item["verification_pass"]) for item in proposals),
        "semantic_pass": sum(bool(cell["obligation_oracle_pass"]) for cell in cells),
    }


def _identity_projection(cell: dict[str, object]) -> tuple[object, ...]:
    paired = cell["paired_input_identity"]
    return tuple(paired.get(field) for field in IDENTITY_FIELDS)


def summarize(cells: list[dict[str, object]]) -> dict[str, object]:
    for task in tasks():
        selected = [cell for cell in cells if cell["atomic_id"] == task.atomic_id]
        if len({_identity_projection(cell) for cell in selected}) != 1:
            raise RuntimeError(f"A77 paired input mismatch for {task.atomic_id}")
    by_profile = {}
    for profile in PROFILES:
        selected = [cell for cell in cells if cell["profile"] == profile]
        by_profile[profile] = {
            "funnel": _funnel(selected),
            "by_operation_class": {
                operation: _funnel(
                    [cell for cell in selected if cell["operation_class"] == operation]
                )
                for operation in sorted({cell["operation_class"] for cell in selected})
            },
            "failure_taxonomy": dict(
                sorted(Counter(str(cell["failure"]) for cell in selected).items())
            ),
            "repair_semantic_recoveries": sum(
                not cell["obligation_oracle_pass"]
                and cell["repair"] is not None
                and bool(cell["repair"].get("semantic_pass"))
                for cell in selected
            ),
            "generation_calls": sum(int(cell["generation_calls"]) for cell in selected),
            "input_tokens": sum(int(cell["input_tokens"]) for cell in selected),
            "output_tokens": sum(int(cell["output_tokens"]) for cell in selected),
            "median_cell_generation_latency_seconds": statistics.median(
                float(cell["median_generation_latency_seconds"])
                for cell in selected
                if cell["median_generation_latency_seconds"] is not None
            ),
        }
    strongest = max(
        BASELINES,
        key=lambda name: by_profile[name]["funnel"]["semantic_pass"],
    )
    candidate = "deepseek-coder-lite"
    delta = (
        by_profile[candidate]["funnel"]["semantic_pass"]
        - by_profile[strongest]["funnel"]["semantic_pass"]
    )
    category_improvements = sum(
        by_profile[candidate]["by_operation_class"][operation]["semantic_pass"]
        > by_profile[strongest]["by_operation_class"][operation]["semantic_pass"]
        for operation in by_profile[candidate]["by_operation_class"]
    )
    materially_better = delta >= 3 and category_improvements >= 2
    return {
        "suite": SUITE,
        "run_identity": RUN_ID,
        "cells": len(cells),
        "task_count": len(tasks()),
        "profile_count": len(PROFILES),
        "input_equivalence": True,
        "by_profile": by_profile,
        "strongest_baseline": strongest,
        "candidate_semantic_delta": delta,
        "candidate_category_improvements": category_improvements,
        "material_improvement": materially_better,
        "conclusion": (
            "CURRENT_MODEL_SET_LIMITED"
            if materially_better
            else "LOCAL_MODEL_CLASS_LIMITED"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    cells = []
    missing = []
    for task in tasks():
        for profile in PROFILES:
            path = checkpoint_path(args.checkpoint_dir, task.atomic_id, profile)
            if not path.is_file():
                missing.append(path.name)
            else:
                cells.append(json.loads(path.read_text()))
    if missing:
        raise RuntimeError(f"A77 incomplete: {missing}")
    summary = summarize(cells)
    if not standard_result_is_source_free(summary):
        raise RuntimeError("A77 summary is not source-free")
    atomic_checkpoint(args.checkpoint_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
