"""Run the frozen A91 qwen-large production validation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from benchmarks.bounded_executable_verification_v1.suite import (
    CELLS,
    NON_DERIVABLE,
    RUN_ID,
    corpus_identity,
    plan_for,
    tasks,
    validate_corpus,
)
from benchmarks.default_candidate_confirmation_v1.suite import REPOSITORY
from benchmarks.realistic_coding_v2.runner import snapshot
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.evaluation.realworld import (
    EvaluationOutcome,
    RealWorldEvaluationRunner,
    run_oracle,
)
from forge.evaluation.replay import source_state_identity
from forge.models import ModelRole, default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_corpus()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qualification = resume_checkpoint(args.qualification)
    if qualification is None:
        raise RuntimeError("A91 qualification missing")
    catalog = load_model_catalog(args.config, default_backend_registry())
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    if selected != "qwen-large":
        raise RuntimeError("A91 requires unchanged qwen-large coding default")
    inventory = qualification["inventory"][selected]
    config_identity = hashlib.sha256(
        repr(catalog.profile(selected)).encode()
    ).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A91 model configuration changed")
    manifest = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cells": CELLS,
        "checks_frozen_before_model_execution": True,
        "production_visible_sources_only": True,
        "non_derivable_tasks": sorted(NON_DERIVABLE),
        "oracle_used_for_derivation": False,
    }
    atomic_checkpoint(args.output_dir / "manifest.json", manifest)
    definitions = tasks()
    repository_identity = source_state_identity(REPOSITORY)
    model = catalog.create(selected)
    cells: list[dict[str, object]] = []
    try:
        for task_id, seed in CELLS:
            definition = definitions[task_id]
            primary_semantic: list[bool] = []

            def primary_callback(  # type: ignore[no-untyped-def]
                task, workspace, capture=primary_semantic
            ):
                capture.append(
                    run_oracle(workspace, task.oracle_commands)
                    is EvaluationOutcome.PASS
                )

            started = time.monotonic()
            result = (
                RealWorldEvaluationRunner(
                    selected,
                    model,
                    REPOSITORY,
                    primary_mutation_callback=primary_callback,
                    executable_verification_plan_factory=lambda task: plan_for(
                        task.task_id
                    ),
                )
                .run(
                    (replace(definition.production_task, seeds=(seed,)),),
                    snapshot(repository_identity),
                )
                .results[0]
            )
            records = result.metrics.executable_verification
            primary_record = records[0] if records else None
            v0_pass = (
                primary_record is not None
                or result.metrics.verification_plan_result == "pass"
            )
            v1_failure = bool(
                primary_record
                and primary_record.get("outcome") == "EXECUTABLE_CHECK_FAILURE"
            )
            semantic = primary_semantic[0] if primary_semantic else False
            outcomes = [
                str(item.get("outcome"))
                for record in records
                for item in record.get("results", ())  # type: ignore[union-attr]
                if isinstance(item, dict)
            ]
            cell = {
                "cell_id": f"{task_id}-seed-{seed}",
                "task_id": task_id,
                "seed": seed,
                "operation_class": definition.operation_class.value,
                "language": definition.language,
                "model_profile": selected,
                "model_config_identity": config_identity,
                "repository_identity": repository_identity,
                "corpus_identity": corpus_identity(),
                "v0_existing_verification_pass": v0_pass,
                "v1_executable_failure": v1_failure,
                "v1_verification_pass": v0_pass and not v1_failure,
                "primary_semantic_pass": semantic,
                "final_semantic_pass": result.oracle is EvaluationOutcome.PASS,
                "check_derivation": (
                    "CHECK_NOT_DERIVABLE" if task_id in NON_DERIVABLE else "DERIVED"
                ),
                "executable_verification": records,
                "check_outcomes": outcomes,
                "repair_attempted": result.metrics.repair_attempts > 0,
                "repair_recovered": (
                    not semantic and result.oracle is EvaluationOutcome.PASS
                ),
                "elapsed_seconds": time.monotonic() - started,
            }
            atomic_checkpoint(args.output_dir / f"{task_id}-seed-{seed}.json", cell)
            cells.append(cell)
            print(
                json.dumps(
                    {
                        "cell": cell["cell_id"],
                        "v0": v0_pass,
                        "v1_failure": v1_failure,
                        "semantic": semantic,
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
    semantic_failures = [cell for cell in cells if not cell["primary_semantic_pass"]]
    caught_v0 = [
        cell for cell in semantic_failures if not cell["v0_existing_verification_pass"]
    ]
    additional = [
        cell
        for cell in semantic_failures
        if cell["v0_existing_verification_pass"] and cell["v1_executable_failure"]
    ]
    false_failures = [
        cell
        for cell in cells
        if cell["primary_semantic_pass"] and cell["v1_executable_failure"]
    ]
    durations = [
        float(record["duration_seconds"])
        for cell in cells
        for record in cell["executable_verification"]  # type: ignore[union-attr]
    ]
    outcomes = Counter(
        outcome
        for cell in cells
        for outcome in cell["check_outcomes"]  # type: ignore[union-attr]
    )
    summary = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cells": len(cells),
        "v0_semantic_failures_caught": [cell["cell_id"] for cell in caught_v0],
        "v1_additional_semantic_failures_caught": [
            cell["cell_id"] for cell in additional
        ],
        "correct_candidates_falsely_failed": [
            cell["cell_id"] for cell in false_failures
        ],
        "non_derivable_cells": [
            cell["cell_id"]
            for cell in cells
            if cell["check_derivation"] == "CHECK_NOT_DERIVABLE"
        ],
        "check_outcomes": dict(outcomes),
        "repair_recoveries": [
            cell["cell_id"] for cell in cells if cell["repair_recovered"]
        ],
        "median_execution_latency_seconds": statistics.median(durations)
        if durations
        else 0.0,
        "p95_execution_latency_seconds": _percentile(durations, 0.95),
        "worst_execution_latency_seconds": max(durations, default=0.0),
        "subprocess_count": sum(
            int(record["subprocess_count"])
            for cell in cells
            for record in cell["executable_verification"]  # type: ignore[union-attr]
        ),
        "timeout_rate": outcomes["CHECK_TIMEOUT"] / max(1, sum(outcomes.values())),
        "canonical_unchanged": source_state_identity(REPOSITORY) == repository_identity,
        "production_impact": "BOUNDED_EXECUTABLE_VERIFICATION_ONLY",
    }
    atomic_checkpoint(args.output_dir / "summary.json", summary)
    return 0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * percentile))]


if __name__ == "__main__":
    raise SystemExit(main())
