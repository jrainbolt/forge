"""Run or resume the frozen A71 transaction-readiness matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.realistic_coding_v2.suite import REPOSITORY
from benchmarks.transaction_readiness_v1.runner import (
    checkpoint_path,
    commit_cell,
    read_cell,
    run_cell,
)
from benchmarks.transaction_readiness_v1.suite import (
    PROFILES,
    SEED,
    corpus_identity,
    tasks,
    validate_corpus,
)
from forge.evaluation.replay import source_state_identity
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


def _artifact(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"{path.name}:sha256:{digest.hexdigest()}"


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
    identity = source_state_identity(REPOSITORY)
    corpus = corpus_identity(definitions)
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.profile)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A71 requires local llama.cpp profiles")
    if profile.backend_config.context_size != 8192:
        raise RuntimeError("A71 requires context 8192")
    artifact = _artifact(profile.backend_config.model_path)
    pending = []
    for definition in definitions:
        if definition.task_id not in selected:
            continue
        path = checkpoint_path(args.checkpoint_dir, args.profile, definition.task_id)
        existing = read_cell(
            path,
            {
                "task_id": definition.task_id,
                "model_profile": args.profile,
                "seed": SEED,
                "repository_identity": identity,
                "corpus_identity": corpus,
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
                repository_identity=identity,
                corpus_identity=corpus,
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
