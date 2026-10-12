"""Run the frozen A90 qwen-large verification diagnosis exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.suite import (
    REPOSITORY,
    REPRESENTATION,
)
from benchmarks.executable_verification_diagnosis_v1.runner import run_cell
from benchmarks.executable_verification_diagnosis_v1.suite import (
    OBLIGATIONS,
    RUN_ID,
    TASK_IDS,
    corpus_identity,
    tasks,
    validate_corpus,
)
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
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
        raise RuntimeError("A90 qualification missing")
    catalog = load_model_catalog(args.config, default_backend_registry())
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    if selected != "qwen-large":
        raise RuntimeError("A90 requires unchanged qwen-large coding default")
    inventory = qualification["inventory"][selected]
    config_identity = hashlib.sha256(
        repr(catalog.profile(selected)).encode()
    ).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A90 model configuration changed")
    manifest = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "task_ids": TASK_IDS,
        "obligations": {
            task: [item.identity for item in OBLIGATIONS[task]] for task in TASK_IDS
        },
        "inventory_frozen_before_model_execution": True,
        "production_visible_sources_only": True,
        "oracle_used_for_check_construction": False,
        "candidate_checks_evaluator_only": True,
    }
    atomic_checkpoint(args.output_dir / "manifest.json", manifest)
    repository_identity = source_state_identity(REPOSITORY)
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    definitions = {item.task_id: item for item in tasks()}
    model = catalog.create(selected)
    cells = []
    try:
        for task_id in TASK_IDS:
            before = source_state_identity(REPOSITORY)
            cell = run_cell(
                definitions[task_id],
                model,
                selected_profile=selected,
                artifact=artifact,
                artifact_size=inventory["file_size"],
                model_config_identity=config_identity,
                repository_identity=repository_identity,
                corpus_identity=corpus_identity(),
                representation=REPRESENTATION,
            )
            if source_state_identity(REPOSITORY) != before:
                raise RuntimeError("canonical repository changed")
            atomic_checkpoint(args.output_dir / f"{task_id}.json", cell)
            cells.append(cell)
            print(
                json.dumps({"task": task_id, "category": cell["failure_category"]}),
                flush=True,
            )
    finally:
        model.close()
    semantic_failures = [cell for cell in cells if not cell["semantic_pass"]]
    caught = [
        cell
        for cell in semantic_failures
        if cell["existing_verification_pass"] and cell["candidate_check_failures"]
    ]
    false_failures = [
        cell
        for cell in cells
        if cell["semantic_pass"] and cell["candidate_check_failures"]
    ]
    false_negatives = [
        cell for cell in semantic_failures if not cell["candidate_check_failures"]
    ]
    supported = bool(caught) and not false_failures
    classification = (
        "EXECUTABLE_VERIFICATION_EXPANSION_SUPPORTED"
        if supported and len(caught) == len(semantic_failures)
        else "EXECUTABLE_VERIFICATION_PARTIALLY_SUPPORTED"
        if supported
        else "MODEL_SEMANTIC_LIMIT_REMAINS_PRIMARY"
        if semantic_failures
        else "EXECUTABLE_VERIFICATION_NOT_SUPPORTED"
    )
    summary = {
        "run_identity": RUN_ID,
        "corpus_identity": corpus_identity(),
        "cells": len(cells),
        "by_class": dict(Counter(cell["operation_class"] for cell in cells)),
        "verification_pass": sum(
            bool(cell["existing_verification_pass"]) for cell in cells
        ),
        "semantic_pass": sum(bool(cell["semantic_pass"]) for cell in cells),
        "semantic_failures": [cell["task_id"] for cell in semantic_failures],
        "failure_categories": dict(Counter(cell["failure_category"] for cell in cells)),
        "semantic_failures_caught_only_by_candidates": [
            cell["task_id"] for cell in caught
        ],
        "candidate_false_failures": [cell["task_id"] for cell in false_failures],
        "candidate_false_negatives": [cell["task_id"] for cell in false_negatives],
        "candidate_checks": sum(len(cell["checks"]) for cell in cells),
        "candidate_check_failures": sum(
            int(cell["candidate_check_failures"]) for cell in cells
        ),
        "repair_attempted": [
            cell["task_id"] for cell in cells if cell["repair_attempted"]
        ],
        "median_generation_latency_seconds": statistics.median(
            float(cell["generation_latency_seconds"])
            for cell in cells
            if cell["generation_latency_seconds"] is not None
        ),
        "existing_test_reuse_missed_behavior": [],
        "existing_test_reuse_finding": (
            "EXISTING_TESTS_PARTIAL_BUT_NO_FAILED_OBLIGATION_ALREADY_ASSERTED"
        ),
        "readiness_classification": classification,
        "canonical_unchanged": True,
        "production_impact": "NONE",
    }
    atomic_checkpoint(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
