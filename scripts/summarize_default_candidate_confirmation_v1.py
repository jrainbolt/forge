"""Summarize the complete A78 confirmation matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import checkpoint_path
from benchmarks.default_candidate_confirmation_v1.suite import (
    PRIMARY_PROFILES,
    PROFILES,
    RUN_ID,
    SUITE,
    tasks,
)
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


def _funnel(cells):  # type: ignore[no-untyped-def]
    proposals = [cell["proposal"] for cell in cells]
    return {
        "cells": len(cells),
        "model_output": sum(bool(x["model_output"]) for x in proposals),
        "schema_valid": sum(bool(x["schema_valid"]) for x in proposals),
        "mechanically_materializable": sum(
            bool(x["mechanically_materializable"]) for x in proposals
        ),
        "production_validatable": sum(
            bool(x["production_validatable"]) for x in proposals
        ),
        "transaction_applied": sum(bool(x["transaction_applied"]) for x in proposals),
        "verification_pass": sum(bool(x["verification_pass"]) for x in proposals),
        "semantic_pass": sum(bool(x["primary_semantic_pass"]) for x in cells),
    }


def summarize(cells):  # type: ignore[no-untyped-def]
    for definition in tasks():
        selected = [x for x in cells if x["task_id"] == definition.task_id]
        projections = {
            tuple(x["paired_input_identity"].get(field) for field in IDENTITY_FIELDS)
            for x in selected
        }
        if len(projections) != 1:
            raise RuntimeError(f"A78 paired input mismatch for {definition.task_id}")
    by_profile = {}
    for profile in PROFILES:
        selected = [x for x in cells if x["profile"] == profile]
        by_profile[profile] = {
            "funnel": _funnel(selected),
            "by_task_class": {
                kind: _funnel([x for x in selected if x["operation_class"] == kind])
                for kind in sorted({x["operation_class"] for x in selected})
            },
            "failures": dict(sorted(Counter(x["failure"] for x in selected).items())),
            "repair_semantic_recoveries": sum(
                not x["primary_semantic_pass"]
                and x["repair"] is not None
                and bool(x["repair"].get("semantic_pass"))
                for x in selected
            ),
            "median_cell_generation_latency_seconds": statistics.median(
                x["median_generation_latency_seconds"]
                for x in selected
                if x["median_generation_latency_seconds"] is not None
            ),
            "input_tokens": sum(x["input_tokens"] for x in selected),
            "output_tokens": sum(x["output_tokens"] for x in selected),
            "generation_calls": sum(x["generation_calls"] for x in selected),
        }
    small, large = (by_profile[name] for name in PRIMARY_PROFILES)
    delta = large["funnel"]["semantic_pass"] - small["funnel"]["semantic_pass"]
    category_improvements = sum(
        large["by_task_class"][kind]["semantic_pass"]
        > small["by_task_class"][kind]["semantic_pass"]
        for kind in large["by_task_class"]
    )
    no_reliability_regression = (
        large["funnel"]["schema_valid"] >= small["funnel"]["schema_valid"]
        and large["funnel"]["transaction_applied"]
        >= small["funnel"]["transaction_applied"]
    )
    latency_ratio = (
        large["median_cell_generation_latency_seconds"]
        / small["median_cell_generation_latency_seconds"]
    )
    threshold = (
        delta >= 3
        and category_improvements >= 2
        and no_reliability_regression
        and latency_ratio <= 2.0
    )
    return {
        "suite": SUITE,
        "run_identity": RUN_ID,
        "cells": len(cells),
        "input_equivalence": True,
        "by_profile": by_profile,
        "qwen_large_semantic_delta": delta,
        "qwen_large_category_improvements": category_improvements,
        "no_protocol_transaction_regression": no_reliability_regression,
        "qwen_large_latency_ratio": latency_ratio,
        "default_candidate_threshold_pass": threshold,
        "robustness_confirmation_required": threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    cells, missing = [], []
    for definition in tasks():
        for profile in PROFILES:
            path = checkpoint_path(args.checkpoint_dir, definition.task_id, profile)
            (cells if path.is_file() else missing).append(
                json.loads(path.read_text()) if path.is_file() else path.name
            )
    if missing:
        raise RuntimeError(f"A78 incomplete: {missing}")
    summary = summarize(cells)
    if not standard_result_is_source_free(summary):
        raise RuntimeError("A78 summary is not source-free")
    atomic_checkpoint(args.checkpoint_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
