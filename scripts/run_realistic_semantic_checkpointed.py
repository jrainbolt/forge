"""Run frozen realistic-semantic-v1 while checkpointing every model cell."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.realistic_semantic_v1.suite import (
    REPOSITORY,
    realistic_semantic_tasks,
    validate_integrity,
)
from forge.evaluation import (
    FORGE_MILESTONE,
    REALISTIC_SEMANTIC_SCHEMA_VERSION,
    REALISTIC_SEMANTIC_SUITE_VERSION,
    REALISTIC_SEMANTIC_V1,
    RealisticSemanticRun,
    aggregate_realistic_results,
    hash_workspace,
    run_realistic_semantic_v1,
    write_realistic_semantic_json,
)
from forge.models import LlamaCppConfig, default_backend_registry, load_model_catalog
from scripts.run_realistic_semantic_v1 import _artifact_identity, _snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--tasks",
        default="R01,R02,R03,R04,R05,R06,R07,R08",
        help="comma-separated non-duplicated cells to execute",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    if not requested or len(requested) != len(set(requested)):
        parser.error("--tasks requires unique comma-separated task IDs")
    definitions = tuple(
        item
        for item in realistic_semantic_tasks((args.seed,))
        if item.metadata.task_id in requested
    )
    if tuple(item.metadata.task_id for item in definitions) != requested:
        parser.error("--tasks must use ordered realistic-semantic-v1 task IDs")
    integrity = validate_integrity(definitions)
    if not all(item.semantic_score_eligible for item in integrity):
        raise RuntimeError("benchmark integrity failed before model execution")
    snapshot = _snapshot()
    canonical_before = hash_workspace(REPOSITORY)
    catalog = load_model_catalog(args.config, default_backend_registry())
    profile = catalog.profile(args.model)
    if not isinstance(profile.backend_config, LlamaCppConfig):
        raise RuntimeError("A46 candidates require the existing llama.cpp backend")
    artifact = _artifact_identity(profile.backend_config.model_path)
    model = catalog.create(args.model)
    if model.context_capacity != 8192:
        model.close()
        raise RuntimeError("realistic-semantic-v1 requires context 8192")

    completed = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        for definition, task_integrity in zip(definitions, integrity, strict=True):
            run = run_realistic_semantic_v1(
                args.model,
                model,
                REPOSITORY,
                snapshot,
                (definition.metadata,),
                (task_integrity,),
                model_artifact=artifact,
                mutation_representation=profile.mutation_representation,
            )
            checkpoint = args.checkpoint_dir / (
                f"{args.model}-seed{args.seed}-{definition.metadata.task_id}.json"
            )
            write_realistic_semantic_json(run, checkpoint)
            completed.extend(run.results)
            print(
                f"checkpointed {definition.metadata.task_id}: "
                f"{run.results[0].failure_layer}",
                flush=True,
            )
    finally:
        model.close()

    results = tuple(completed)
    combined = RealisticSemanticRun(
        REALISTIC_SEMANTIC_V1,
        REALISTIC_SEMANTIC_SUITE_VERSION,
        REALISTIC_SEMANTIC_SCHEMA_VERSION,
        FORGE_MILESTONE,
        snapshot.identity,
        args.model,
        artifact,
        8192,
        512,
        0.0,
        integrity,
        results,
        (aggregate_realistic_results(args.model, args.seed, results),),
        canonical_before == hash_workspace(REPOSITORY),
    )
    if not combined.canonical_unchanged:
        raise RuntimeError("canonical realistic semantic repository changed")
    write_realistic_semantic_json(combined, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
