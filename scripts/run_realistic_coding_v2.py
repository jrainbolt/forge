"""Freeze, run, and exactly-once resume realistic-coding-v2 model cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.realistic_coding_v2.runner import (
    checkpoint_path,
    commit_cell,
    freeze_manifest,
    read_checkpoint,
    run_cell,
)
from benchmarks.realistic_coding_v2.suite import (
    REPOSITORY,
    manifest_identity,
    tasks,
    validate_integrity,
)
from forge.evaluation.realworld import hash_workspace
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog


def _artifact_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return f"{path.name}:sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default="all")
    args = parser.parse_args()
    definitions = tasks()
    integrity = validate_integrity(definitions)
    frozen = freeze_manifest(args.checkpoint_dir, definitions, integrity)
    if args.integrity_only:
        print(
            json.dumps(
                {
                    "suite": frozen["suite"],
                    "manifest_identity": manifest_identity(definitions),
                    "repository_identity": frozen["repository_identity"],
                    "eligible": sum(item.eligible for item in integrity),
                    "tasks": len(integrity),
                    "verification_baseline_pass": sum(
                        item.verification_baseline_pass for item in integrity
                    ),
                    "verification_reference_pass": sum(
                        item.verification_reference_pass for item in integrity
                    ),
                    "verification_wrong_pass": sum(
                        item.verification_wrong_pass for item in integrity
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.config is None or args.profile is None:
        parser.error("--config and --profile are required outside --integrity-only")
    selected = tuple(item.task_id for item in definitions)
    if args.tasks != "all":
        selected = tuple(
            value.strip() for value in args.tasks.split(",") if value.strip()
        )
        if not selected or len(set(selected)) != len(selected):
            parser.error("--tasks requires unique IDs")
        known = {item.task_id for item in definitions}
        if any(task_id not in known for task_id in selected):
            parser.error("unknown A56 task ID")
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.profile)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A56 requires configured local llama.cpp profiles")
    if profile.backend_config.context_size != 8192:
        raise RuntimeError("A56 requires the unchanged 8192-token profile")
    artifact = _artifact_identity(profile.backend_config.model_path)
    planned = tuple(item for item in definitions if item.task_id in selected)
    missing = []
    for definition in planned:
        path = checkpoint_path(
            args.checkpoint_dir, args.profile, args.seed, definition.task_id
        )
        existing = read_checkpoint(
            path,
            profile=args.profile,
            seed=args.seed,
            definition=definition,
            artifact=artifact,
            manifest=frozen,
        )
        if existing is None:
            missing.append(definition)
        else:
            print(
                f"skip committed {args.profile} seed{args.seed} {definition.task_id}",
                flush=True,
            )
    if not missing:
        print("all requested cells already committed", flush=True)
        return 0
    canonical_before = hash_workspace(REPOSITORY)
    model = catalog.create(args.profile)
    if model.context_capacity != 8192:
        model.close()
        raise RuntimeError("loaded model context does not match A56 profile")
    try:
        for definition in missing:
            cell = run_cell(
                definition,
                model,
                profile=args.profile,
                artifact=artifact,
                seed=args.seed,
                repository_identity=str(frozen["repository_identity"]),
                manifest=str(frozen["manifest_identity"]),
                representation=profile.mutation_representation,
            )
            if canonical_before != hash_workspace(REPOSITORY):
                raise RuntimeError("canonical A56 repository changed during model cell")
            destination = checkpoint_path(
                args.checkpoint_dir, args.profile, args.seed, definition.task_id
            )
            commit_cell(destination, cell)
            print(
                f"checkpointed {args.profile} seed{args.seed} {definition.task_id}: "
                f"{cell.failure_layer} semantic={cell.final_semantic}",
                flush=True,
            )
    finally:
        model.close()
    if canonical_before != hash_workspace(REPOSITORY):
        raise RuntimeError("canonical A56 repository changed after model matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
