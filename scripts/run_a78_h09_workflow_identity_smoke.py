"""Run the one authorized real H09 complete-workflow instrumentation smoke."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import run_cell
from benchmarks.default_candidate_confirmation_v1.suite import (
    QUALIFICATION_RUN_ID,
    REPRESENTATION,
    corpus_identity,
    tasks,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.transaction_readiness_v1.runner import standard_result_is_source_free
from forge.evaluation.mutation_ready import atomic_checkpoint, resume_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog

SMOKE_RUN_ID = "a78-h09-complete-workflow-identity-smoke-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("H09 smoke checkpoint already exists")
    qualification = resume_checkpoint(args.qualification)
    corpus = corpus_identity()
    if (
        qualification is None
        or qualification.get("run_identity") != QUALIFICATION_RUN_ID
    ):
        raise RuntimeError("A78 qualification identity mismatch")
    if qualification.get("corpus_identity") != corpus:
        raise RuntimeError("A78 frozen corpus identity mismatch")
    inventory = qualification["inventory"]["qwen-small"]
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile("qwen-small")
    config = profile.backend_config
    if not isinstance(config, LlamaCppConfig):
        raise RuntimeError("H09 smoke requires local llama.cpp")
    config_identity = hashlib.sha256(repr(profile).encode()).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("H09 smoke model configuration changed")
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    repository_identity = source_state_identity(REPOSITORY)
    definition = next(item for item in tasks() if item.task_id == "H09")
    model = catalog.create("qwen-small")
    try:
        result = run_cell(
            definition,
            "qwen-small",
            model,
            artifact=artifact,
            artifact_size=inventory["file_size"],
            model_config_identity=config_identity,
            repository_identity=repository_identity,
            corpus_identity=corpus,
            representation=REPRESENTATION,
            run_identity=SMOKE_RUN_ID,
        )
    finally:
        model.close()
    payload = asdict(result)
    workflow = payload["workflow_attempt"]
    if workflow["outcome"] == "WORKFLOW_IN_FLIGHT":
        raise RuntimeError("H09 smoke workflow outcome is incomplete")
    if not workflow["workflow_attempt_id"] or not workflow["equivalence_identity"]:
        raise RuntimeError("H09 smoke workflow identity is incomplete")
    if not standard_result_is_source_free(payload):
        raise RuntimeError("H09 smoke is not source-free")
    atomic_checkpoint(args.output, payload)
    if source_state_identity(REPOSITORY) != repository_identity:
        raise RuntimeError("canonical source identity changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
