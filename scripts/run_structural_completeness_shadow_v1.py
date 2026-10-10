"""Run A83 qwen-large shadow cells with checkpoint-safe resume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.structural_completeness_shadow_v1.runner import commit_cell, run_cell
from benchmarks.structural_completeness_shadow_v1.suite import (
    CASES,
    REPRESENTATION,
    RUN_ID,
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
        raise RuntimeError("A83 workflow instrumentation failure")
    requests = {item.get("mutation_request_id") for item in cell.mutation_requests}
    if None in requests or any(
        item.get("workflow_attempt_id") != attempt for item in cell.mutation_requests
    ):
        raise RuntimeError("A83 request/workflow lineage failure")
    if set(workflow.get("mutation_request_ids", ())) != requests:
        raise RuntimeError("A83 workflow request-set failure")
    if any(value not in requests for value in cell.proposal_request_lineage.values()):
        raise RuntimeError("A83 proposal/request lineage failure")


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
        or not manifest.get("shadow_mode")
        or manifest.get("blocking")
        or manifest.get("repair_triggering")
    ):
        raise RuntimeError("A83 frozen manifest mismatch")
    qualification = resume_checkpoint(args.qualification)
    if qualification is None:
        raise RuntimeError("A83 model qualification missing")
    catalog = load_model_catalog(args.config, default_backend_registry())
    selected = catalog.resolve_profile(None, ModelRole.CODING)
    if selected != "qwen-large":
        raise RuntimeError(f"A83 coding default changed: {selected}")
    inventory = qualification["inventory"][selected]
    profile = catalog.profile(selected)
    config_identity = hashlib.sha256(repr(profile).encode()).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A83 model configuration changed")
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    repository_identity = source_state_identity(REPOSITORY)
    definitions = {item.task_id: item for item in tasks()}
    model = catalog.create(selected)
    try:
        for case in CASES:
            path = args.checkpoint_dir / "cells" / f"{case.task_id}.json"
            if path.exists():
                prior = resume_checkpoint(path)
                if (
                    prior is None
                    or prior.get("run_identity") != RUN_ID
                    or prior.get("corpus_identity") != corpus_identity()
                    or prior.get("task_id") != case.task_id
                ):
                    raise RuntimeError(f"A83 invalid resume checkpoint: {path}")
                print(json.dumps({"task": case.task_id, "resume": True}), flush=True)
                continue
            before = source_state_identity(REPOSITORY)
            cell = run_cell(
                case,
                definitions[case.task_id],
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
                        "task": case.task_id,
                        "shadow": cell.primary_shadow.decision,
                        "verification": cell.verification_pass,
                        "semantic": cell.semantic_pass,
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
