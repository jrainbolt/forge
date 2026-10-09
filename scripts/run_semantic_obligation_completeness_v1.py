"""Run the frozen A81 obligation diagnosis exactly once per task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.suite import REPOSITORY
from benchmarks.semantic_obligation_completeness_v1.runner import commit_cell, run_cell
from benchmarks.semantic_obligation_completeness_v1.suite import (
    REPRESENTATION,
    RUN_ID,
    TASK_IDS,
    corpus_identity,
    tasks,
)
from forge.evaluation.mutation_ready import resume_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.models import ModelRole, default_backend_registry, load_model_catalog


def _validate_lineage(cell) -> None:  # type: ignore[no-untyped-def]
    workflow = cell.workflow_attempt
    attempt = workflow.get("workflow_attempt_id")
    if not attempt or workflow.get("outcome") == "WORKFLOW_IN_FLIGHT":
        raise RuntimeError("A81 workflow instrumentation failure")
    request_ids = {item.get("mutation_request_id") for item in cell.mutation_requests}
    if None in request_ids or any(
        item.get("workflow_attempt_id") != attempt for item in cell.mutation_requests
    ):
        raise RuntimeError("A81 request/workflow lineage failure")
    if set(workflow.get("mutation_request_ids", ())) != request_ids:
        raise RuntimeError("A81 workflow request-set failure")
    if any(
        value not in request_ids for value in cell.proposal_request_lineage.values()
    ):
        raise RuntimeError("A81 proposal/request lineage failure")
    if any(
        proposal not in cell.proposal_request_lineage or not transactions
        for proposal, transactions in cell.proposal_transaction_lineage.items()
    ):
        raise RuntimeError("A81 transaction/proposal lineage failure")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = resume_checkpoint(args.manifest)
    if (
        manifest is None
        or manifest.get("run_identity") != RUN_ID
        or manifest.get("corpus_identity") != corpus_identity()
        or not manifest.get("obligations_frozen_before_execution")
    ):
        raise RuntimeError("A81 frozen manifest mismatch")
    qualification = resume_checkpoint(args.qualification)
    if qualification is None:
        raise RuntimeError("A81 qualification missing")
    catalog = load_model_catalog(args.config, default_backend_registry())
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    if selected != "qwen-large":
        raise RuntimeError(f"A81 production coding default changed: {selected}")
    inventory = qualification["inventory"][selected]
    profile = catalog.profile(selected)
    config_identity = hashlib.sha256(repr(profile).encode()).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A81 model configuration changed")
    repository_identity = source_state_identity(REPOSITORY)
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    definitions = {item.task_id: item for item in tasks()}
    model = catalog.create(selected)
    try:
        for task_id in TASK_IDS:
            path = args.checkpoint_dir / "cells" / f"{task_id}.json"
            if path.exists():
                raise RuntimeError(f"A81 exactly-once violation: {path}")
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
            _validate_lineage(cell)
            if source_state_identity(REPOSITORY) != before:
                raise RuntimeError("canonical source identity changed")
            commit_cell(path, cell)
            print(
                json.dumps(
                    {
                        "task": task_id,
                        "completeness": cell.primary_completeness,
                        "semantic_pass": cell.full_semantic_pass,
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
