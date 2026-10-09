"""Run or resume one profile of the frozen A78 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.default_candidate_confirmation_v1.runner import (
    checkpoint_path,
    commit_cell,
    read_cell,
    run_cell,
)
from benchmarks.default_candidate_confirmation_v1.suite import (
    CONTEXT_SIZE,
    PROFILES,
    QUALIFICATION_RUN_ID,
    REPRESENTATION,
    RUN_ID,
    corpus_identity,
    tasks,
    validate_corpus,
)
from benchmarks.realistic_coding_v2.suite import REPOSITORY
from forge.evaluation.mutation_ready import resume_checkpoint
from forge.evaluation.replay import source_state_identity
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()
    validate_corpus()
    qualification = resume_checkpoint(args.qualification)
    if qualification is None:
        raise RuntimeError("A78 qualification checkpoint is missing")
    corpus = corpus_identity()
    if (
        qualification.get("run_identity") != QUALIFICATION_RUN_ID
        or qualification.get("corpus_identity") != corpus
    ):
        raise RuntimeError("A78 qualification identity mismatch")
    inventory = qualification["inventory"][args.profile]
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.profile)
    config = profile.backend_config
    if not isinstance(config, LlamaCppConfig) or config.context_size != CONTEXT_SIZE:
        raise RuntimeError("A78 requires local llama.cpp context 8192")
    config_identity = hashlib.sha256(repr(profile).encode()).hexdigest()
    if config_identity != inventory["config_identity"]:
        raise RuntimeError("A78 model configuration changed after qualification")
    artifact = f"{inventory['gguf_filename']}:sha256:{inventory['sha256']}"
    repository_identity = source_state_identity(REPOSITORY)
    pending = []
    for definition in tasks():
        expected = {
            "run_identity": RUN_ID,
            "cell_id": f"A78-{definition.task_id}-{args.profile}",
            "corpus_identity": corpus,
            "repository_identity": repository_identity,
            "model_artifact": artifact,
            "model_config_identity": config_identity,
        }
        path = checkpoint_path(args.checkpoint_dir, definition.task_id, args.profile)
        if read_cell(path, expected) is None:
            pending.append(definition)
        else:
            print(f"skip committed A78-{definition.task_id}-{args.profile}", flush=True)
    if not pending:
        return 0
    canonical_before = source_state_identity(REPOSITORY)
    model = catalog.create(args.profile)
    try:
        for definition in pending:
            result = run_cell(
                definition,
                args.profile,
                model,
                artifact=artifact,
                artifact_size=inventory["file_size"],
                model_config_identity=config_identity,
                repository_identity=repository_identity,
                corpus_identity=corpus,
                representation=REPRESENTATION,
            )
            commit_cell(
                checkpoint_path(args.checkpoint_dir, definition.task_id, args.profile),
                result,
            )
            print(
                json.dumps(
                    {
                        "cell": result.cell_id,
                        "failure": result.failure,
                        "semantic_pass": result.primary_semantic_pass,
                    }
                ),
                flush=True,
            )
    finally:
        model.close()
    if source_state_identity(REPOSITORY) != canonical_before:
        raise RuntimeError("canonical source identity changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
