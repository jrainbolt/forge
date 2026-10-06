"""Run or resume the frozen A72 transaction-readiness matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.transaction_readiness_v2.runner import (
    checkpoint_path,
    commit_cell,
    read_cell,
    run_cell,
)
from benchmarks.transaction_readiness_v2.suite import (
    CONTEXT_SIZE,
    EVALUATOR_IDENTITY,
    PROFILES,
    RUN_IDENTITY,
    SEED,
    SUITE,
    frozen_matrix_identity,
    tasks,
    validate_corpus,
)
from forge.evaluation.replay import source_state_identity
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


def _file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"{path.name}:sha256:{digest.hexdigest()}"


def _config_identity(profile: object) -> str:
    return hashlib.sha256(repr(profile).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--tasks", default="all")
    args = parser.parse_args()
    definitions = tasks()
    validate_corpus(definitions)
    selected = {item.task_id for item in definitions}
    if args.tasks != "all":
        selected = {value.strip() for value in args.tasks.split(",") if value.strip()}
    if not selected or not selected.issubset({item.task_id for item in definitions}):
        parser.error("--tasks contains an unknown or empty task selection")
    repository_identity = source_state_identity(REPOSITORY)
    matrix_identity = frozen_matrix_identity()
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.profile)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A72 requires local llama.cpp profiles")
    if profile.backend_config.context_size != CONTEXT_SIZE:
        raise RuntimeError(f"A72 requires context {CONTEXT_SIZE}")
    artifact = _file_identity(profile.backend_config.model_path)
    config_identity = _config_identity(profile)
    pending = []
    for definition in definitions:
        if definition.task_id not in selected:
            continue
        path = checkpoint_path(args.checkpoint_dir, args.profile, definition.task_id)
        existing = read_cell(
            path,
            {
                "suite": SUITE,
                "evaluator_identity": EVALUATOR_IDENTITY,
                "run_identity": RUN_IDENTITY,
                "task_id": definition.task_id,
                "model_profile": args.profile,
                "model_artifact": artifact,
                "model_config_identity": config_identity,
                "seed": SEED,
                "repository_identity": repository_identity,
                "frozen_matrix_identity": matrix_identity,
            },
        )
        if existing is None:
            pending.append(definition)
        else:
            print(f"skip committed {args.profile} {definition.task_id}", flush=True)
    if not pending:
        return 0
    canonical_before = source_state_identity(REPOSITORY)
    model = catalog.create(args.profile)
    try:
        for definition in pending:
            cell = run_cell(
                definition,
                model,
                profile=args.profile,
                artifact=artifact,
                model_config_identity=config_identity,
                repository_identity=repository_identity,
                matrix_identity=matrix_identity,
                representation=profile.mutation_representation,
            )
            commit_cell(
                checkpoint_path(args.checkpoint_dir, args.profile, definition.task_id),
                cell,
            )
            print(
                json.dumps(
                    {"task": definition.task_id, "proposals": len(cell.proposals)}
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
