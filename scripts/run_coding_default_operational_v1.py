"""Run one exactly-once A80 selection condition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.coding_default_operational_v1.runner import commit_cell, run_cell
from benchmarks.coding_default_operational_v1.suite import (
    REPRESENTATION,
    REPRESENTATIVE_TASK_IDS,
    RUN_ID,
    TASK_IDS,
    SelectionCondition,
    corpus_identity,
    tasks,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY
from forge.evaluation.mutation_ready import resume_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.models import ModelRole, default_backend_registry, load_model_catalog


def _validate_lineage(cell) -> None:  # type: ignore[no-untyped-def]
    workflow = cell.workflow_attempt
    attempt = workflow.get("workflow_attempt_id")
    if not attempt or workflow.get("outcome") == "WORKFLOW_IN_FLIGHT":
        raise RuntimeError("A80 workflow instrumentation failure")
    request_ids = {item.get("mutation_request_id") for item in cell.mutation_requests}
    if None in request_ids or any(
        item.get("workflow_attempt_id") != attempt for item in cell.mutation_requests
    ):
        raise RuntimeError("A80 request/workflow lineage failure")
    if set(workflow.get("mutation_request_ids", ())) != request_ids:
        raise RuntimeError("A80 workflow request-set failure")
    if any(
        value not in request_ids for value in cell.proposal_request_lineage.values()
    ):
        raise RuntimeError("A80 proposal/request lineage failure")
    if any(
        key not in cell.proposal_request_lineage or not values
        for key, values in cell.proposal_transaction_lineage.items()
    ):
        raise RuntimeError("A80 transaction/proposal lineage failure")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=tuple(item.value for item in SelectionCondition),
        required=True,
    )
    args = parser.parse_args()
    manifest = resume_checkpoint(args.manifest)
    if (
        manifest is None
        or manifest.get("run_identity") != RUN_ID
        or manifest.get("corpus_identity") != corpus_identity()
    ):
        raise RuntimeError("A80 frozen manifest mismatch")
    qualification = resume_checkpoint(args.qualification)
    if qualification is None:
        raise RuntimeError("A80 qualification missing")
    condition = SelectionCondition(args.condition)
    catalog = load_model_catalog(args.config, default_backend_registry())
    explicit = {
        SelectionCondition.DEFAULT: None,
        SelectionCondition.EXPLICIT_LARGE: "qwen-large",
        SelectionCondition.EXPLICIT_SMALL: "qwen-small",
    }[condition]
    selected = catalog.resolve_profile(explicit, ModelRole.CODING)
    expected = (
        "qwen-small" if condition is SelectionCondition.EXPLICIT_SMALL else "qwen-large"
    )
    if selected != expected:
        raise RuntimeError(f"A80 selection regression: {selected}")
    inventory = qualification["inventory"][selected]
    profile = catalog.profile(selected)
    config_identity = hashlib.sha256(repr(profile).encode()).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A80 model configuration changed")
    selected_ids = (
        TASK_IDS
        if condition is SelectionCondition.DEFAULT
        else REPRESENTATIVE_TASK_IDS
        if condition is SelectionCondition.EXPLICIT_LARGE
        else (REPRESENTATIVE_TASK_IDS[0],)
    )
    definitions = {item.task_id: item for item in tasks()}
    repository_identity = source_state_identity(REPOSITORY)
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    model = catalog.create(selected)
    try:
        for task_id in selected_ids:
            path = args.checkpoint_dir / "cells" / f"{condition.name}-{task_id}.json"
            if path.exists():
                raise RuntimeError(f"A80 exactly-once violation: {path}")
            before = source_state_identity(REPOSITORY)
            cell = run_cell(
                definitions[task_id],
                condition,
                selected,
                model,
                artifact=artifact,
                artifact_size=inventory["file_size"],
                model_config_identity=config_identity,
                repository_identity=repository_identity,
                corpus_identity=corpus_identity(),
                representation=REPRESENTATION,
            )
            _validate_lineage(cell)
            if source_state_identity(REPOSITORY) != before:
                raise RuntimeError("canonical source identity changed")
            commit_cell(path, cell)
            print(
                json.dumps(
                    {
                        "condition": condition.value,
                        "task": task_id,
                        "classification": cell.primary_classification,
                        "semantic_pass": cell.primary_semantic_pass,
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
